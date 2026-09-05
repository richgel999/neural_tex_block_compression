# CUDA_PLAN_B.md

# CUDA port of ntc, plan B: a first working version built on a texture-set structure

Status: proposal, not implemented. Companion to CUDA_PLAN_A.md (the minimal one-to-one port). Everything here was derived from `main.cpp` at commit a7fea48, METHOD.md, the reference log `out_model_512c2q31_128c4_6k.log`, and the toolchains actually installed on this machine.

## 0. Decisions at a glance

| Topic | Decision |
|---|---|
| Build | CMake -> "Visual Studio 17 2022" solution, CUDA 13.1 selected explicitly (`-T cuda=13.1` via a CMake preset), `CUDA_ARCHITECTURES 120`, MSVC 14.44 host compiler. `option(NTC_CUDA)` defaults OFF; the CPU-only build is byte-for-byte the current one. CUDA code lives in a separate static library target so `main.cpp` stays a plain C++ translation unit. |
| Selection | `--cuda` on the existing command line. Same model file, same outputs, same stats line. Without `NTC_CUDA` the flag prints "built without CUDA" and exits 1. |
| Ownership | With `--cuda` the device owns `z`, MLP weights and both Adam states; the host `Decoder` is a mirror refreshed at print/save time so `save_model`, `bitrate_stats`, `latent_stats`, `save_png` run unchanged. |
| Shared math | Phase 1 (recommended, optional): move the decoder core (tap, sample, positional encode, MLP forward, qat grid) out of `main.cpp` into `ntc_model.h`, annotated `NTC_HD` (`__host__ __device__` under nvcc, nothing otherwise), with POD shape structs. One implementation of the decoder for CPU and GPU. Guarded by the 23.83 dB regression and a byte-compare of `model.bin`. A "B-lite" fallback duplicates ~200 lines in the `.cuh` instead. |
| RNG | Counter-based, stateless, `__host__ __device__` hash -> Box-Muller. Seeded from `--seed`; keyed by (stream, iteration, pair, flat index). Host `mt19937` still does the latent/MLP init so iteration 0 is identical to the CPU run. No cuRAND dependency. |
| Reductions | No float atomics anywhere. Per-texel gather (one thread per texel), fixed-order block reductions in double, tiny fixed-order finish kernels. Same seed -> byte-identical `model.bin` on the same GPU/driver. |
| Precision | fp32 everywhere in the MLP, `--fmad=true` (default), NO `--use_fast_math`. Loss sums: float per thread, double across threads. |
| v1 scope | Nearest levels (1 or 2), `--qat` selector search, level-1 latent ES (`--lat-alt` included), MLP ES (minibatch and `--mlp-full`), FD, decode/loss/eval/stats, materials T <= 4, all table-free positional kinds. Refuses `--deblock`, bilinear levels, `onehot`, `bc7part`, MLPs above 12288 weights. |
| Structure for later | Every kernel is written against a `DevTexSet` (N textures, one shared decoder format) and launched with `blockIdx.y` = texture index; decoder weights are always addressed through a weight-set index. v1 runs with N = 1 and one weight set. This is what keeps the universal-decoder, batched-textures and transcode-in-the-loop paths open without rewriting kernels. |
| Expected speed | Typical config (512x512, 512x512x2 qat 3,1 + 128x128x4 nearest, 8->36->36->3, 64 MLP pairs, 4 latent pairs): about 2.3 ms/iter in the ES phase (~430 it/s vs 8.4 on 32 threads, ~50x) and ~8 ms/iter in the FD phase (~130 it/s vs 5.2, ~25x). The 6000-iteration reference run drops from 1150 s to roughly 30 s. Assumes ~10 TFLOPS effective; details in section 8. |
| Effort | Critical path to a full typical run: 2.5-3 days. With Phase 1 (shared header), the test harness and the performance pass: 5-6 days. |

## 1. Toolchain facts and the build

### 1.1 What is installed (verified read-only on this machine)

- `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\nvcc.exe` reports release 13.1, V13.1.80. `C:\...\CUDA\v12.4\bin\nvcc` is first on PATH; CUDA 12.4 cannot target sm_120 (that needs >= 12.8).
- `v13.1\include\crt\host_config.h` contains `#if _MSC_VER < 1920 || _MSC_VER >= 1950 #error ... Only the versions between 2019 and 2022 (inclusive) are supported`. VS 2022 Community ships MSVC 14.44 (`_MSC_VER` 1944): supported. VS 2026 Community (`C:\Program Files\Microsoft Visual Studio\18`) ships 14.51 (`_MSC_VER` 1951): rejected by nvcc unless `-allow-unsupported-compiler`, and its `MSBuild\Microsoft\VC\*\BuildCustomizations` has no CUDA props at all.
- The CUDA 13.1 MSBuild integration IS installed for VS 2022: `C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Microsoft\VC\v170\BuildCustomizations\CUDA 13.1.{props,targets,xml}` (next to CUDA 12.4's).
- The existing `build\CMakeCache.txt` uses generator "Visual Studio 17 2022", platform x64, empty toolset. CMake is 4.1.1.

Conclusion: generator "Visual Studio 17 2022", toolset `cuda=13.1,host=x64`. Do not use the VS 2026 generator for the CUDA build until NVIDIA ships a toolkit whose `host_config.h` accepts `_MSC_VER` 195x and installs its props under `v180`.

### 1.2 File layout

```
C:\dev\neural\
  CMakeLists.txt            modified: option NTC_CUDA, CXX flags scoped to CXX, optional cuda/ subdir
  CMakePresets.json         new: "cpu" and "cuda" configure presets (generator, toolset, CUDAToolkit_ROOT)
  main.cpp                  modified: --cuda flag, backend dispatch in the training loop, host mirror sync
  ntc_model.h               new (Phase 1): decoder core shared by CPU and GPU (see section 2.2)
  cuda/CMakeLists.txt       new: static library ntc_cuda (CUDA language), sm_120, ptxas -v
  cuda/ntc_cuda.h           new: host-facing API (struct CudaTrainer), plain C++ header, no CUDA types
  cuda/ntc_cuda.cu          new: device buffers, host wrappers, launches, stream handling
  cuda/ntc_kernels.cuh      new: device structs (DevLevel, DevTexture, DevTexSet, MlpShape, PosSpec), device functions, kernels
  cuda/ntc_noise.h          new: counter-based Gaussian, __host__ __device__, also used by the test harness
  cuda/ntc_cuda_test.cu     new: verification harness (target ntc_cuda_test, section 10)
```

### 1.3 CMakeLists.txt changes (top level)

```cmake
cmake_minimum_required(VERSION 3.20)
project(ntc LANGUAGES CXX)
option(NTC_CUDA "Build the CUDA backend (needs CUDA >= 12.8, sm_120)" OFF)
...
add_executable(ntc main.cpp)
if(MSVC)
  # Scope to CXX: nvcc must never see /fp:fast /arch:AVX2 (only matters when ntc links a CUDA target)
  target_compile_options(ntc PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W3 /O2 /fp:fast /arch:AVX2>)
else() ... endif()
if(NTC_CUDA)
  add_subdirectory(cuda)
  target_link_libraries(ntc PRIVATE ntc_cuda)
  target_compile_definitions(ntc PRIVATE NTC_CUDA=1)
endif()
```

`cuda/CMakeLists.txt`:

```cmake
enable_language(CUDA)
find_package(CUDAToolkit 12.8 REQUIRED)
if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12.8)
  message(FATAL_ERROR "CUDA ${CMAKE_CUDA_COMPILER_VERSION} cannot target sm_120; use the 'cuda' preset (CUDA 13.1)")
endif()
add_library(ntc_cuda STATIC ntc_cuda.cu)
target_include_directories(ntc_cuda PUBLIC ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(ntc_cuda PROPERTIES CUDA_ARCHITECTURES "120" CUDA_STANDARD 17 CUDA_SEPARABLE_COMPILATION OFF)
target_compile_options(ntc_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v --expt-relaxed-constexpr>)
# deliberately no --use_fast_math; --fmad stays at its default (on)
target_link_libraries(ntc_cuda PUBLIC CUDA::cudart_static)
add_executable(ntc_cuda_test ntc_cuda_test.cu)   # verification harness, section 10
target_link_libraries(ntc_cuda_test PRIVATE ntc_cuda)
set_target_properties(ntc_cuda_test PROPERTIES CUDA_ARCHITECTURES "120")
```

`CMakePresets.json` (the part that matters):

```json
{ "version": 6, "configurePresets": [
  { "name": "cpu",  "generator": "Visual Studio 17 2022", "architecture": "x64", "binaryDir": "${sourceDir}/build" },
  { "name": "cuda", "generator": "Visual Studio 17 2022", "architecture": "x64",
    "toolset": "cuda=13.1,host=x64", "binaryDir": "${sourceDir}/build_cuda",
    "cacheVariables": { "NTC_CUDA": "ON",
      "CUDAToolkit_ROOT": "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1" } } ] }
```

`cmake --preset cuda && cmake --build build_cuda --config Release`. The configure log must show `CMAKE_CUDA_COMPILER_VERSION 13.1.80`; if it shows 12.4 the toolset did not take (see risks). `CUDA_ARCHITECTURES "120"` emits sm_120 SASS plus compute_120 PTX; no `120a` needed (no architecture-specific features used).

`.gitignore` already ignores `build*/`, so `build_cuda/` needs no change.

## 2. Code structure

### 2.1 Backend seam in main.cpp

The training loop (`main.cpp` lines 1698-1773) calls exactly these operations per iteration: `mt.step`, `mt.step_fd`, `lt.step`, `qat_search`, `D.decode_full` (twice at print time: current `z` and the quantized `zq`), `mse_of`, `latent_stats`, `bitrate_stats`, `mt.adam.init` at the FD switch. The CUDA backend implements the first five and `decode_full`; everything else stays host code on the mirror.

```cpp
// cuda/ntc_cuda.h  (no CUDA types; safe to include from main.cpp)
struct CudaTrainer {
    // Refuses (returns false, fills why) anything outside the v1 envelope (section 9).
    bool init(const Decoder& D, const Image& target, const Options& o, std::string& why);
    void upload_model(const Decoder& D);            // z and mlp.p -> device (init, --load)
    void download_model(Decoder& D);                // device -> D.lat.z, D.mlp.p (print/save)
    double mlp_step(int it, const Options& oi, double& dstd, bool full);      // = MlpTrainer::step
    double mlp_step_fd(int it, const Options& oi, double& fd_rms, bool full); // = MlpTrainer::step_fd
    void reset_mlp_adam();                                                    // = mt.adam.init at fd_from
    void lat_step(int it, const Options& oi);                                 // = LatentTrainer::step
    void qat_search();                                                        // = qat_search()
    void decode_full(const float* z_override_host, Image& out);              // = Decoder::decode_full; null = device z
};
```

In `main`: `Options::cuda`; after `D` and `target` are built (and after `--load`), `if (o.cuda) { if (!cu.init(D, target, o, why)) { printf("--cuda: %s\n", why); return 1; } cu.upload_model(D); }`. In the loop, each of the five calls becomes `o.cuda ? cu.x(...) : existing`. At `do_print || do_save`: `cu.download_model(D)` then the existing host code, with `D.decode_full(...)` replaced by `cu.decode_full(nullptr, recon)` and `cu.decode_full(zq.data(), recon_q)`. `batch_loss` and `diff_std` keep their meaning (returned by the step functions exactly as today). The CPU path is not touched by any of this beyond the `? :` dispatch, so the 23.83 dB regression holds trivially for `ntc.exe` without `--cuda`.

### 2.2 Phase 1: the shared decoder core (`ntc_model.h`)

Recommended but severable. Move, unchanged in arithmetic, from `main.cpp` into `ntc_model.h`:

- `PI, MAXT, MAXOUT, MAXH, MAXL`, `Act`, `activate`, `BilinearTap`, `bilinear_tap`, `sample_latent` (lines 44, 273-316, 322-336).
- `BDCT_ZIGZAG`, `BC7_PARTITION2` (as `NTC_TABLE` arrays: `static const` on host, `__constant__` shadows on device selected by a macro).
- `QAT_GRID`, `qat_init_grid`, `qat_levels`, `qat_value`, `qat_index`, `qat_snap` (lines 1086-1097). On device `QAT_GRID` is a `__constant__` copy uploaded once.
- New PODs: `MlpShape { int nin, nout, nl; int w[MAXL]; Act act; float leak; bool clamp; }` and `PosSpec { int n; struct { int kind, n; } f[16]; int cellw, cellh; }`, `LevelDesc { int W, H, C, nearest; size_t off; }`, `ModelDesc { int W, H, nlev; LevelDesc lv[2]; PosSpec pos; MlpShape mlp; }`.
- `mlp_forward(const MlpShape&, const float* p, const float* in, float* out)` = lines 394-418 reading widths from the POD instead of `std::vector`; `pos_encode(const PosSpec&, u, v, t, t1, f)` = `PosEnc::encode` lines 536-615 iterating the POD array; `model_features(const ModelDesc&, const float* z, px, py, f)` = `Decoder::features` lines 694-709.
- `MLP` and `PosEnc` in `main.cpp` keep their vectors for parsing/printing and gain a `shape()` / `spec()` that fills the POD; `Decoder::features`/`pixel` become one-line forwarders.

All of it is annotated `NTC_HD` (`#ifdef __CUDACC__ #define NTC_HD __host__ __device__ #else #define NTC_HD #endif`) and uses `floorf/expf/sinf/cosf/tanhf` (the float overloads `main.cpp` already resolves to). The same float operations in the same order, so the CPU result is unchanged; verify with the 23.83 dB check and `fc /b` of `out\model.bin` against a pre-refactor build (the byte compare is the real test).

Why it is worth half a day: the GPU decode, selector search and per-pixel feature paths then call literally the same functions the CPU does; the verification harness tests only the hot templated forward (2.3) and the reductions, not a second copy of the positional encoder; every future positional kind, activation or level type works on both sides by construction; and the universal-decoder code will need these PODs anyway.

B-lite fallback: skip Phase 1, copy the ~200 lines into `ntc_kernels.cuh` as device functions with the same names, and let the harness catch drift.

### 2.3 The hot forward: `mlp_forward_t<MAXW>`

The reference `mlp_forward` keeps `bufA[MAXH], bufB[MAXH]` (128 floats each) and indexes them with runtime bounds: on the GPU those arrays go to local memory. The hot kernels use

```cpp
template <int MAXW>  // MAXW >= max(nin, every hidden width); nout <= MAXOUT always
__device__ __forceinline__ void mlp_forward_t(const MlpShape& s, const float* __restrict__ w /*shared*/, const float* in, float* out) {
    float a[MAXW], b[MAXW];
    #pragma unroll
    for (int i = 0; i < MAXW; i++) a[i] = (i < s.nin) ? in[i] : 0.f;
    int ncur = s.nin;
    for (int l = 0; l < s.nl; l++) {              // uniform trip count
        const int nh = s.w[l]; const float* b_ = w + nh * ncur;
        #pragma unroll
        for (int j = 0; j < MAXW; j++) if (j < nh) {
            float acc = b_[j]; const float* wr = w + j * ncur;
            #pragma unroll
            for (int i = 0; i < MAXW; i++) if (i < ncur) acc = fmaf(wr[i], a[i], acc);   // same order as CPU: s += w[i]*cur[i]
            b[j] = activate(s.act, acc, s.leak);
        }
        w = b_ + nh; ncur = nh; swap-by-copy a <- b (unrolled)
    }
    output layer as in mlp_forward (sigmoid via 1.f/(1.f+expf(-acc)) or clamp)
}
```

Instantiated for MAXW in {16, 32, 48, 64, 128}; the host picks the smallest that fits. Fully unrolled compile-time indices keep `a`, `b` in registers; MAXW = 128 will spill (documented, only for exotic runs). Weights are read from shared memory with every thread of the warp reading the same address at the same time (the loops are warp-uniform), which is a conflict-free broadcast. `ptxas -v` must be checked for MAXW = 48 (the typical 36-wide case): target < 128 registers so a 256-thread block still gets 2 blocks per SM.

Note on rounding: the CPU compiles `s += w[i] * cur[i]` under `/fp:fast`, which may or may not contract; nvcc contracts to FMA. Results differ at the 1e-7 relative level per multiply-add, which is the source of the "~1e-6" tolerance in section 10, and the reason exact tie-breaks in the selector search can occasionally differ.

## 3. Device-side representation

```cpp
struct DevLevel   { int W, H, C, nearest; size_t off;            // off: into the owning texture's z slice
                    const int *xb, *xe, *yb, *ye;                // nearest only: pixel range per texel column/row (qat_search step 1, uploaded once)
                    float sigma; int qat_bits[MAXC_QAT]; int qat; };   // qat > 0: level held on grid, excluded from ES
struct DevTexture { int W, H, T; int nlev; DevLevel lv[2];
                    size_t z_off, tgt_off, pix_off;              // slices into the set's flat arrays
                    float cw[MAXOUT]; float wsum; float inv_px; };  // inv_px = 1/(3 wsum W H) exactly as LatentTrainer
struct DevTexSet  { int ntex; const DevTexture* tex;              // descriptors (device array; also __constant__ copy for ntex <= 64)
                    float* z;  float* z_grad; float* z_m; float* z_v;   // concatenated per-texture latents + Adam state
                    const float* tgt;                            // interleaved targets, [pixel][3T], fp32 (uint8 later, section 7)
                    float* dpix; float* err; float* sse_t;       // per-pixel scratch, sized to the texture batch in flight
                    float* img;                                  // decoded image scratch (print/save)
                    MlpShape mlp; PosSpec pos;                   // one decoder format per set
                    float* w;  float* w_grad; float* w_m; float* w_v; int P; int nsets;   // weight set 0 = the model; others are scratch
                    float* feat; float* btgt; float* bnrm; int* bidx; int M;              // minibatch rows: features, targets, per-slot 1/(3 wsum_t), source pixel
                    double* part; double* dl; double* lsum; double* hh;                   // reduction partials and per-pair/per-weight results
                    uint64_t seed; };
```

- Latent levels: one flat `z` per set, texture-major, level-major, `(y, x, c)` with `c` fastest inside a level, i.e. the `LatentSet::z` layout per texture. Adam `m, v` mirror it. `z_grad` is zeroed with `cudaMemsetAsync` at the start of every latent step.
- Selectors: they ARE level 0 of a `--qat` run, stored as on-grid floats exactly like the CPU (no index array). The `__constant__ QAT_GRID[9][257]` copy is the single source of grid bit patterns on the device, mirroring the CPU rule.
- MLP weights: `w` holds `nsets` contiguous weight vectors of `P` floats; set 0 is the model. Kernels never read weights from global memory inside the pixel loop: each block copies its set into `__shared__ float ws[P]` first (P <= 12288 floats = 48 KB static shared; larger MLPs are refused in v1). Constant memory is not used for weights so that decode, ES, FD and search all share one code path.
- Minibatch: `bidx[M]` (kept for the harness), `feat[M * nin]`, `btgt[M * nout]`, `bnrm[M]`. Produced once per step by `k_batch_features`; the MLP kernels are texture-agnostic after that.
- Per-pixel buffers: `dpix` (loss difference of the current pair), `err` (weighted per-pixel error), `sse_t` (`[pixel][T]` unweighted SSE for the stats line), `img` (`[pixel][nout]`), all sized `W*H` times the number of textures in flight (1 in v1).
- Reduction scratch: `part[nsets * chunks]`, `dl[max(N, P)]`, `lsum[N]`, `hh[P]` doubles.

Memory for the typical config: z 2.4 MB, Adam 4.7 MB, target 3 MB, per-pixel scratch ~4 MB, minibatch 0.2 MB, weights 128 sets x 7 KB = 0.9 MB. Under 20 MB.

## 4. Kernels

Conventions: 256 threads per block unless stated; `blockIdx.y` = texture index within the set (v1: always 0); a block reads its `DevTexture` descriptor into registers first; all loops over MLP widths are warp-uniform. `bsum_d(double)` is the fixed-order block reduction of section 6.

### 4.1 `k_decode_loss` (replaces `Decoder::decode_full`, `Decoder::decode_err`, the pixel half of `mse_of`)

```
template<int MAXW> __global__ void k_decode_loss(DevTexSet S, const float* wset, int flags, float* img, float* err, float* sse_t)
  tex = S.tex[blockIdx.y]; shared ws[P] <- wset; __syncthreads()
  p = blockIdx.x*256 + tid; if p >= W*H return;  px = p % W; py = p / W
  float f[MAXW]; model_features(desc, S.z + tex.z_off, px, py, f)          // shared code, section 2.2
  float out[MAXOUT]; mlp_forward_t<MAXW>(S.mlp, ws, f, out)
  if flags & F_IMG:  img[p*nout + c] = out[c]
  if flags & F_ERR:  err[p] = sum_c cw[c] * ((out[c]-tgt[c]) * (out[c]-tgt[c]))       // float, same expression/order as decode_err
  if flags & F_SSE:  sse_t[p*T + t] = sum_{c in texture t} d*d                          // float per pixel; host sums in double = mse_of
```

Grid: `ceil(W*H/256)` = 1024 blocks for 512x512. Used at print/save time with set 0 (current `z`) and with a host-supplied `zq` (uploaded into a scratch slice) for the quantized PSNR; the images and `sse_t` are downloaded and the existing host `mse_of`-equivalent double sum runs on `sse_t`. Cost: 1 decode-equivalent.

### 4.2 `k_batch_features` (replaces `MlpTrainer::draw_batch` and the feature part of `Decoder::loss_subset`)

```
__global__ void k_batch_features(DevTexSet S, int step, int full)
  s = global thread id; if s >= M return
  if full: tex = 0, idx = s                                     // --mlp-full: every pixel in order, exactly like the CPU
  else:    tex = ntex > 1 ? uniform(noise_u32(seed, NS_BATCH_TEX, step, 0, s), ntex) : 0
           idx = uniform(noise_u32(seed, NS_BATCH_PIX, step, 0, s), W_tex*H_tex)
  bidx[s] = idx (+ tex<<? kept separately for N > 1); model_features(desc_tex, z_tex, idx%W, idx/W, feat + s*nin)
  btgt[s*nout + c] = tgt_tex[idx*nout + c];  bnrm[s] = 1 / (3 * wsum_tex)
```

16 blocks for M = 4096. The batch is drawn once per step and shared by all pairs and both signs, exactly the CPU's contract (METHOD.md section 4). `uniform(u, n)` is the 64-bit multiply-high map; it is not the CPU's `uniform_int_distribution`, which is fine (bit-identity is not required).

### 4.3 `k_mlp_es_eval`, `k_mlp_es_finish`, `k_mlp_es_grad` (replace `MlpTrainer::step`)

```
template<int MAXW> __global__ void k_mlp_es_eval(DevTexSet S, int step, float sigma, int N, int chunks)   // grid (chunks, 2N)
  set = blockIdx.y; pair = set >> 1; sign = (set & 1) ? -1.f : +1.f
  shared ws[P]: for k = tid; k < P; k += 256: ws[k] = w0[k] + sign * sigma * gauss(seed, NS_MLP, step, pair, k)   // == pp/pm on the CPU
  __syncthreads()
  float acc = 0
  for s = blockIdx.x*512 + tid, twice (2 pixels per thread): if s < M:
      out = mlp_forward_t<MAXW>(S.mlp, ws, feat + s*nin);  acc += bnrm[s] * sum_c cw[c]*(d*d)
  part[set*chunks + blockIdx.x] = bsum_d((double)acc)

__global__ void k_mlp_es_finish(DevTexSet S, int N, int chunks)          // one block, N threads
  i = tid: lp = sum_j part[(2i)*chunks + j]; lm = sum_j part[(2i+1)*chunks + j]  (double, fixed order)
  dl[i] = (lp - lm) / M;  lsum[i] = 0.5 * (lp + lm) / M                // == loss_subset's / (3 wsum |idx|) with bnrm folded in

__global__ void k_mlp_es_grad(DevTexSet S, int step, float scale, int N)   // P threads; scale = 1/(2 N sigma)
  k = tid: g = 0; for i in 0..N: g += (float)dl[i] * scale * gauss(seed, NS_MLP, step, i, k);  w_grad[k] = g
```

Then `k_adam(w, w_grad, w_m, w_v, lr, t)`. `mean`, `mean2`, `lmean` for the stats line are computed on the host from `dl[]`/`lsum[]` (2N doubles) when the line is printed; the values are kept on device until then so the loop never syncs. Grid for the typical config: 8 chunks x 128 sets = 1024 blocks; weight-set generation is ~7 Gaussians per thread, under 10% of the block's work. Cost: 2 decode-equivalents.

The block-per-weight-set structure is the same thing that later runs many independent decoders: `set -> (mlp_index, pair, sign)` is the only indexing rule; v1 has `mlp_index = 0`.

### 4.4 `k_fd_eval`, `k_fd_grad` (replace `MlpTrainer::step_fd`)

```
template<int MAXW> __global__ void k_fd_eval(DevTexSet S, float h)      // grid = P blocks, one weight per block
  j = blockIdx.x; shared ws[P] <- w0; __syncthreads()
  float orig = ws[j], wp = orig + h, wm = orig - h                        // float-realized steps, as on the CPU
  if tid == 0: ws[j] = wp;  __syncthreads()
  lp = bsum_d(sum over s = tid, tid+256, ... < M of bnrm[s] * weighted err with ws)   // 16 pixels per thread
  __syncthreads(); if tid == 0: ws[j] = wm; __syncthreads()
  lm = same
  if tid == 0: dl[j] = (lp - lm) / M;  hh[j] = (double)wp - (double)wm

__global__ void k_fd_grad(DevTexSet S)   // P threads
  w_grad[j] = (float)(dl[j] / hh[j])
```

The unperturbed loss `l0` (returned as `batch_loss`) is `k_mlp_es_eval` with N = 1, sigma = 0, reading only set 0 (one extra launch of 8 blocks). `fd_rms` = sqrt(sum dl^2 / P) on the host at print time. Grid: 1767 blocks of 256 threads, each 2 x 4096 pixel evaluations: 55 decode-equivalents in one launch. The per-thread float partial covers 16 terms; everything above that is double, so the tiny differences (`fd-rms` ~2e-8 on a loss of 2e-4 in the reference log) are resolved as accurately as on the CPU (section 12, fp32 reduction order).

### 4.5 `k_lat_pair` and `k_lat_gather` (replace `LatentTrainer::step`)

Per antithetic pair k of iteration `step`:

```
template<int MAXW> __global__ void k_lat_pair(DevTexSet S, int step, int pair, unsigned active_levels)
  tex, p, px, py as in 4.1; shared ws[P] <- w0
  float f[MAXW]; model_features(desc, z_tex, px, py, f)                    // unperturbed features, same code as decode
  float fp[MAXW] = f, fm[MAXW] = f
  for level l in active_levels (bit mask: qat level 0 excluded, --lat-alt picks one level):
      tap = bilinear_tap(L, u, v)                                           // nearest in v1: one texel, weight 1
      for each tap texel i with weight wt_i (1 for nearest), for c in 0..C:
          e = gauss(seed, NS_LAT, step, pair, tex_index_base + L.off + texel_i*C + c)
          fp[slot_l + c] += wt_i * L.sigma * e;  fm[slot_l + c] -= wt_i * L.sigma * e     // nearest: identical to z +- sigma*eps on the CPU
  ep = weighted err of mlp_forward_t(fp);  em = same for fm
  dpix[p] = (ep - em) * tex.inv_px                                         // the CPU's d, written at the pixel's own index

__global__ void k_lat_gather(DevTexSet S, int step, int pair, int level, float scale)   // one thread per texel; scale = 1/(2 K_l sigma_l)
  t = texel (tx, ty); rect = nearest ? [xb[tx], xe[tx]) x [yb[ty], ye[ty]) : closed-form 2r rectangle of METHOD 3.4
  float d = 0; for py in rect: for px in rect: d += dpix[py*W + px]        // fixed order = the CPU's dtex accumulation order per texel
  wgt = d * scale
  for c: z_grad[base + t*C + c] += wgt * gauss(seed, NS_LAT, step, pair, base + L.off + t*C + c)   // same noise as k_lat_pair
```

After K pairs: `k_adam(z, z_grad, z_m, z_v, lat_lr, t)` over the whole latent. A `--qat` level 0 never receives a gradient (its bit is clear in `active_levels` and it is never gathered), so Adam leaves it bit-exact, exactly as the CPU relies on (METHOD 3.5). `--lat-alt` is `active_levels` = one bit per pair, rotating with `step_no`, and `scale` uses `pairs_of[l]` as the CPU does. Cost per pair: 2 decode-equivalents fused in one launch (features computed once, forward twice), plus a gather that reads each pixel once per level. Fusing means `zp`, `zm` never exist; for bilinear levels (later) the sample of `z + sigma*eps` becomes `sample(z) + sigma*sample(eps)`, which is the same number up to rounding.

Occupancy note: a 128x128 level-1 gather is 16384 threads = 64 blocks, under one wave on 170 SMs; it reads 16 floats per thread and takes microseconds, so it does not matter, but the same structure is why coarse selector levels need the warp-per-texel variant later (4.6).

### 4.6 `k_qat_search` (replaces `qat_search`)

```
template<int MAXW> __global__ void k_qat_search(DevTexSet S)        // one thread per level-0 texel
  tex; L0 = tex.lv[0]; t = (tx, ty); if xb[tx] >= xe[tx] || yb[ty] >= ye[ty] return   // no pixel reads it
  shared ws[P] <- w0
  zt = z_tex + (ty*L0.W + tx) * C0                                    // private to this thread; other threads never read it
  for c in 0..C0:
      bits = L0.qat_bits[c]; levels = (1<<bits) - 1
      best_k = qat_index(zt[c], bits); best = cell_loss(c, qat_value(best_k, bits))    // current value first; ties keep it (strict <)
      for k in 0..levels: if k != best_k: sk = cell_loss(c, qat_value(k, bits)); if sk < best: best = sk, best_k = k
      zt[c] = qat_value(best_k, bits)                                 // visible to cell_loss for channel c+1 via model_features

  cell_loss(c, val): double sum = 0
      for py in [yb, ye): for px in [xb, xe):                          // row-major, the CPU's order
          float f[MAXW]; model_features(desc, z_tex, px, py, f); f[c] = val    // the feature-patching trick, recomputed (cheap) instead of cached
          out = mlp_forward_t<MAXW>(S.mlp, ws, f); sum += (double)(sum_q cw[q] * (d*d))   // float product, double add: the CPU's expression
      return sum
```

Grid: `ceil(L0.W*L0.H/256)`, i.e. 1024 blocks for a 512x512 selector level, 64 blocks for 128x128. The candidate loop is uniform across the block (same bits for every texel), so there is no divergence; only the cell-size loop varies for non-divisible sizes. Cost: `sum_c 2^B_c` decode-equivalents (10 for `--qat 3,1`), one launch. The materials case is the same kernel with `nout = 3T` and `cw` from the descriptor (shared selectors, weighted sum over all `3T` outputs).

Coarse selector levels (cells of 4x4 or more) leave most of the GPU idle with one thread per texel; the planned v1.5 variant assigns a warp per texel (lanes over cell pixels x candidates, fixed-order shuffle reduction in double). Not needed for the 512x512 configurations in the logs.

### 4.7 `k_adam` (replaces `Adam::step`, both instances)

```
__global__ void k_adam(float* theta, const float* g, float* m, float* v, int n, float lr, float c1, float c2)
  i: m[i] = b1*m[i] + (1-b1)*g[i]; v[i] = b2*v[i] + (1-b2)*g[i]*g[i]; theta[i] -= lr * (m[i]/c1) / (sqrtf(v[i]/c2) + eps)
```

`c1, c2` computed on the host from the step counter exactly as the CPU (`1 - pow(b1, t)`). Zero gradient with zero moments gives a zero update, as the `--qat` invariant needs. `reset_mlp_adam` = memset of `w_m, w_v` and `t = 0`.

### 4.8 Stats and eval

At `do_print || do_save`: `download_model(D)` (z 2.4 MB, w 7 KB), `k_decode_loss(F_IMG|F_SSE)` on set 0 -> download `img` and `sse_t`; the host computes `Mse` from `sse_t` in double (same formula as `mse_of`), runs the unchanged `latent_stats`, `bitrate_stats` (producing `zq`), uploads `zq` into a scratch latent slice, runs `k_decode_loss` again for `recon_q`. The `WARNING off-grid` check runs on the mirror unchanged. No reduction kernels are needed for stats in v1. This costs 2 decode-equivalents plus ~5 MB of transfers and ~5-10 ms of host time (the histogram in `bitrate_stats` over 590k values) per print; at `--print-every 10` and 2.3 ms iterations that is a visible ~20-30% overhead, so GPU runs should use `--print-every 50` or larger until the histogram moves to the device (v1.5).

## 5. RNG

`cuda/ntc_noise.h`, `__host__ __device__`, no state:

```cpp
enum NoiseStream : uint32_t { NS_LAT = 1, NS_MLP = 2, NS_BATCH_PIX = 3, NS_BATCH_TEX = 4 };
NTC_HD inline uint64_t mix64(uint64_t x);                         // splitmix64 finalizer
NTC_HD inline uint64_t noise_key(uint64_t seed, uint32_t stream, uint32_t step, uint32_t pair, uint64_t index)
    { return mix64(mix64(seed ^ ((uint64_t)stream << 56) ^ ((uint64_t)step << 32) ^ pair) + index * 0x9E3779B97F4A7C15ull); }
NTC_HD inline uint32_t noise_u32(...)    { return (uint32_t)(noise_key(...) >> 32); }
NTC_HD inline float gauss(...)           { uint64_t h = noise_key(...); u1 = ((h >> 40) + 1) * 2^-24 in (0,1]; u2 = (h & 0xFFFFFF) * 2^-24;
                                           return sqrtf(-2.f * logf(u1)) * cospif(2.f * u2); }   // Box-Muller, one sample per call
```

- Determinism: every perturbation is a pure function of `(seed, stream, iteration, pair, flat index)`. The same value is regenerated wherever it is needed (the +- kernel and the gather, the ES evaluation and the gradient), so nothing is stored and nothing can desynchronize. Same `--seed` -> same run, independent of block shapes, and the host harness can compute the identical numbers.
- Host `mt19937` keeps doing the latent init, the `--qat` snap and the MLP init (`main.cpp` lines 1482-1494), so the `iter 0` line of a `--cuda` run matches the CPU's to display precision, and `--load` continues from the same state on both backends.
- The device streams do not reproduce the CPU's `std::normal_distribution` sequence (which is already platform-specific, METHOD.md section 7), so CPU and GPU runs with the same seed are different ES samples of the same objective; that is the accepted "PSNR within ES noise" contract.
- Cost: ~40 instructions per Gaussian; the largest consumer is `k_lat_pair` regenerating a texel's C values per pixel (redundant across the 16 pixels of a cell but under 5% of a pixel's 3.4k FLOP).
- Annealing changes `sigma`, never the noise; `--seed` changes everything.

## 6. Reductions and determinism

- Latent attribution is a gather (thread per texel, section 4.5), so the scatter's border double-count guard is unnecessary: every pixel is inside exactly one nearest cell, and for bilinear (later) the closed-form rectangle credits each pixel once including the clamped border.
- Minibatch losses: float accumulation per thread over at most 16 terms, then `bsum_d`: `__shfl_xor_sync` butterfly over doubles within each warp, warp sums to `__shared__ double s[8]`, warp 0 adds `s[0..7]` in index order. Fixed tree, fixed order, same result every run. Cross-block partials are summed in index order by one thread in double.
- Image stats: per-pixel floats downloaded, summed in double on the host in pixel order (the CPU's `mse_of` order).
- No `atomicAdd`, no `cub::DeviceReduce` (its order is deterministic on a fixed config, but the explicit version is 30 lines and provably fixed).
- Result: same seed, same GPU, same driver, same toolkit -> byte-identical `model.bin`. Across GPUs/toolkits, `expf/logf/cospif` implementations may differ in the last ulp, so only PSNR-level agreement is claimed.

## 7. The texture-set abstraction and the universal decoder

v1 already runs on a `DevTexSet` with `ntex = 1`. What the future cases need, and how the layout above serves them:

**Universal decoder (train once over many textures, per-texture latents).**
- Residency: each texture contributes a `DevTexture` descriptor, a latent slice (z, m, v) and a target slice. A 512x512 texture with the reference format is 2.4 MB of latent + 4.7 MB Adam + 0.75 MB target as uint8 (`(float)v / 255.f` on read reproduces the CPU's float exactly); 1000 textures ~ 8 GB, well inside 32 GB. Only the per-pixel scratch (`dpix, err, img`) is sized to the batch of textures in flight, not to the set.
- Minibatch across textures: `k_batch_features` draws `tex ~ U(ntex)` then `pixel ~ U(W_tex H_tex)` and writes texture-agnostic rows (`feat, btgt, bnrm`). The MLP ES and FD kernels do not change at all. With `bnrm[s] = 1/(3 wsum_tex)` and the `/M` in the finish kernel, the batch loss is an unbiased estimate of `L_bar = (1/N) sum_t L_t` (draw textures uniformly; drawing pixels uniformly over the union would instead weight textures by pixel count, also a valid choice, just a different objective). Materials of different T in one set are not supported by "one decoder format per set"; a set is one format.
- What changes in the ES gradient: nothing in the estimator. `g = 1/(2 N sigma) sum_i [L_bar_M(w + sigma e_i) - L_bar_M(w - sigma e_i)] e_i` with the same shared batch per pair and across pairs. Per-texture latent steps are independent problems: `k_lat_pair`/`k_lat_gather` run with `blockIdx.y` over a batch of textures, each texture with its own `inv_px`, its own footprint tables and its own noise indices (the flat index includes the texture's `z_off`), so each texture's estimate is exactly the single-texture estimator of its own `L_t`; the `1/N` of the mean is a common scale Adam removes. Latent steps for all N textures cost `2K` decodes per texture per iteration; a schedule that updates a random subset of textures per iteration (round-robin over the set) keeps the iteration time bounded and is a pure host-side loop change.
- Phase two (fixed decoder, encode a new texture): `upload` the texture, run `k_qat_search` + `k_lat_pair/gather` + `k_adam` only (no MLP kernels). A block-latent *search* instead of ES (METHOD 8, "in-loop quantization of the block level") is a new kernel with `k_qat_search`'s structure (one thread or warp per level-1 texel, candidates over (channel, grid value), the cell's selectors re-searched inside the candidate loop), and the descriptors/tables already give it the pixel ranges.

**Batching many independent textures per run (throughput).** Same set, but the decoder weights become per texture: `w` holds `ntex` model sets, and the MLP kernels' `set -> (mlp_index, pair, sign)` map addresses them; `k_decode_loss`/`k_lat_pair` take the texture's weight set instead of set 0. Because v1 kernels already receive the weight set by index, this is a host-side change plus one index computation.

**Transcode-to-BC/ASTC in the loop.** The loss stage in `k_decode_loss` and `k_lat_pair` is the only thing that changes: instead of one thread per pixel computing its own error, a warp (or thread) per 4x4 output block decodes 16 pixels, runs the block encoder round trip, and writes 16 per-pixel errors. The footprint attribution is unchanged for nearest levels whose cells align with the codec block; a bilinear level's footprint grows by the block, which the closed-form rectangle absorbs. Keep the per-pixel error computation behind one device function (`pixel_err`) from day one so the block variant is a second function, not a second kernel family.

## 8. Thread/block shapes and back-of-envelope timings on a 5090

Model: a full 512x512 decode of the reference MLP (1767 weights, 8->36->36->3) is 262144 pixels x ~1.7k FMA = ~0.9 GFLOP. An RTX 5090 has 170 SMs and ~105 TFLOPS fp32 peak. This kernel family reads one shared-memory word per FMA (broadcast), which caps it at roughly a quarter of FMA peak, and the templated forward runs at modest occupancy; assume 10 TFLOPS effective for planning (100 us per decode-equivalent), with 25 TFLOPS as the optimistic case after float4 weight loads.

| Step (typical config) | Decode-eq | Blocks x threads | Estimated |
|---|---|---|---|
| `k_batch_features` | ~0 | 16 x 256 | 10 us |
| MLP ES: `k_mlp_es_eval` 64 pairs x 4096 px | 2 | 1024 x 256 | 0.2-0.3 ms |
| `k_mlp_es_finish/grad`, `k_adam` (P) | ~0 | 1 x 64; 7 x 256 | 20 us |
| Latent ES: 4 x `k_lat_pair` (level 1 only) | 8 | 4 x (1024 x 256) | 0.8 ms |
| 4 x `k_lat_gather` + `k_adam` (590k) | ~0 | 64 x 256; 2304 x 256 | 50 us |
| `k_qat_search` 3,1 bits | 10 | 1024 x 256 | 1.0 ms |
| ~18 launches (WDDM) | | | 0.1-0.2 ms |
| ES-phase iteration | 20 | | ~2.3 ms -> ~430 it/s (CPU 8.4) |
| FD phase adds `k_fd_eval` 1767 x 2 x 4096 | +55 | 1767 x 256 | +5.5 ms -> ~8 ms -> ~130 it/s (CPU 5.2) |
| Print (every 10): 2 decodes + 5 MB transfers + host histogram | 2 | | 8-12 ms per print |

Reference run (6000 iterations, FD from 4500): 4500 x 2.3 ms + 1500 x 8 ms = 22 s, plus prints/saves ~5 s at `--print-every 10` -> ~30 s versus 1150 s, about 40x. At 25 TFLOPS effective it is ~15 s. The 200-iteration regression config (64x64x4 bilinear, refused in v1) is not a GPU target; the nearest equivalents are.

Register budget: MAXW = 48 keeps `a[48], b[48]` plus loop state near 110-130 registers; at 128 registers a 256-thread block uses 32k of the SM's 64k registers, so 2 blocks (16 warps) per SM. Enough for this FMA-bound workload; verify with `-Xptxas=-v` at build time and with Nsight Compute once.

## 9. What `--cuda` refuses in v1, and how the rest fits later

`CudaTrainer::init` returns false with a message for:

| Refused in v1 | Why | Later |
|---|---|---|
| Any bilinear level (`--filter bilinear` on level 0 or 1, the default) | keeps the v1 test matrix to nearest taps | v1.5: `k_lat_pair` already sums weighted taps; `k_lat_gather` gets the closed-form rectangle of METHOD 3.4; 4.1 and 4.6 are unchanged (4.6 still needs nearest level 0, as on the CPU). Half a day. |
| `--deblock` | couples pixels; `loss_subset` decodes 3-5 pixels per ring pixel | v2: a `k_deblock_pass` after the full decode (the CPU's scratch-buffer form) for 4.1/4.5; ring-pixel crosses inside 4.3/4.4; gather over the union of taps (`add_tap`) in 4.5. One day. |
| `onehot`, `bc7part` | need the `BC7_PARTITION2` table and cell sizes on device | v1.5: `__constant__` table upload; trivial. |
| MLP with P > 12288 or a width > 128 | shared-memory weight set, MAXW instantiations | raise the cap with dynamic shared memory if ever needed |
| `--threads` | ignored with a note | |

Supported in v1: `--qat` with per-channel bits, `--qat-every`, 1 or 2 nearest levels, `--lat-alt`, `--lat2-sigma`, `--mlp-full`, `--mlp-fd`, `--mlp-freeze`, both anneals, materials T <= 4 with `--weights`, all activations, `--clamp`, positional kinds `uv fourier local lfourier lquad dct ldct ldct2 ldct4 lv1local lv1ldct bdct bdcte none` (`BDCT_ZIGZAG` is 32 ints in `__constant__`), `--load`, `--iters 0`.

## 10. Verification plan

Tolerances: `model_features` and the forward differ from the CPU only by FMA contraction and libm ulps, so outputs in [0,1] agree to ~1e-6 absolute; sums over thousands of terms are compared in double against a double oracle.

1. **Toolchain smoke** (step 0): `ntc_cuda_test --smoke` launches a trivial kernel, prints device name, `cudaRuntimeGetVersion` (must be 13010) and the SASS arch it ran on.
2. **Decode equality** (step 2): for each config in a matrix {T = 1, 2, 4} x {1 level, 2 levels} x {pos kinds above} x {leaky, relu, tanh, sine} x {sigmoid, clamp}, random-init a model with the host RNG, run `Decoder::decode_full` and `k_decode_loss(F_IMG|F_ERR)`: `max |img_gpu - img_cpu| < 2e-6`, `max |err_gpu - err_cpu| < 1e-6`. Also `--cuda --iters 0 --load out\model.bin` must print the same `iter 0` PSNR as the CPU to 0.01 dB on the checked-in models.
3. **Selector search** (step 3): same model, copy `z`, run `qat_search` (CPU) and `k_qat_search`; report the number of differing texels, and for each, both candidates' cell losses; require differing texels <= 0.05% and each a near-tie (relative gap < 1e-5); require the total weighted loss after both searches to agree within 1e-6 relative. Run for `--qat 1`, `3,1`, `4` and materials.
4. **Noise** (step 4): `k_dump_noise` writes `gauss(...)` for a range of keys; the host computes the same via `ntc_noise.h`: bit-identical. Histogram sanity: mean/variance of 1e7 samples within 1e-3.
5. **MLP ES step** (step 4): harness draws the batch on device, downloads `bidx`, forms `pp/pm` on the host with `gauss`, calls `D.loss_subset` for each pair: `|dl_gpu[i] - dl_cpu[i]| < 1e-8` (loss ~1e-3, `dl` ~1e-5). Then one Adam step both sides: `max |w_gpu - w_cpu| < 1e-7`.
6. **Latent step** (step 5): harness forms `zp/zm` on the host with `gauss`, calls `D.decode_err` twice, does the CPU scatter in double for the active level, compares `z_grad`: relative < 1e-5 per texel (max over the level), and after Adam `max |z_gpu - z_cpu| < 1e-6`; the `--qat` level-0 slice must be byte-identical before and after.
7. **FD step** (step 6): host loop over 32 sampled weights j calling `loss_subset` with `w[j] +- h`: `|dl_gpu[j] - dl_cpu[j]| < 1e-10` (dl ~1e-8), `hh` identical.
8. **End-to-end agreement**: the reference config for 3000 iterations, CPU vs GPU, seeds 1..3 each: GPU final PSNRs fall inside the CPU seed spread (expected +-0.1 dB); the material config `out_m1234_512c1q4_128c4` likewise. Also print-cadence independence: `--print-every 10` vs `100` give byte-identical `model.bin` on the GPU (no hidden host-device dependence).
9. **Determinism**: same command twice -> `fc /b` identical `model.bin` and identical log lines.
10. **CPU regression after every step that touches `main.cpp`**: `ntc.exe --iters 200 ...` prints 23.83 dB; after Phase 1, byte-compare `out\model.bin` with a build from a7fea48.

## 11. Implementation order, tests and effort

| Step | Work | Test | Effort |
|---|---|---|---|
| 0 | Presets, `cuda/CMakeLists.txt`, empty `ntc_cuda` lib, `--cuda` flag that refuses everything, `ntc_cuda_test --smoke` | 10.1; CPU build unchanged; confirm configure picks 13.1 | 0.5 d |
| 1 (optional) | `ntc_model.h` extraction with `NTC_HD`, PODs, forwarders in `main.cpp` | 10.10 (23.83 dB + byte-compare) | 0.5-1 d |
| 2 | `DevTexSet` upload/download, `k_decode_loss`, `mlp_forward_t<MAXW>`, stats path; `--cuda --iters 0` works | 10.2 | 1 d |
| 3 | `k_qat_search` | 10.3 | 0.5 d |
| 4 | `ntc_noise.h`, `k_batch_features`, MLP ES kernels, `k_adam`; single-level `--qat` runs train end to end (MLP + selectors) | 10.4, 10.5 | 1 d |
| 5 | `k_lat_pair`, `k_lat_gather`, `--lat-alt`; the reference two-level config trains | 10.6, 10.8 (ES phase) | 0.5-1 d |
| 6 | `k_fd_eval/grad`, `reset_mlp_adam`; full reference run | 10.7, 10.8, 10.9 | 0.5 d |
| 7 | Async loop (no per-iteration sync), print cadence, Nsight Systems pass, `ptxas -v` review, float4 weight loads if cheap | timings vs section 8 | 0.5 d |
| 8 (v1.5) | bilinear levels, `onehot/bc7part` tables, warp-per-texel search, histogram on device | 10.2/10.3 matrix extended | 1 d |

Critical path to a full typical run: steps 0, 2, 3, 4, 5, 6 without Phase 1 = about 3 days; with Phase 1, the harness and the perf pass, 5-6 days.

## 12. Risks

- **CUDA 13.1 + VS 2026.** `host_config.h` rejects `_MSC_VER >= 1950` and no CUDA props exist under VS 18. Use the VS 2022 generator. If the owner later installs a CUDA update, re-check that file before changing the preset. `-allow-unsupported-compiler` is not recommended: nvcc's own warning says wrong code is possible.
- **Wrong toolkit picked up.** PATH has 12.4 first. The preset's `toolset: cuda=13.1` plus `CUDAToolkit_ROOT` pins it; the FATAL_ERROR on `< 12.8` catches a silent fallback. Also `CUDA_PATH` in the environment should be checked (`echo %CUDA_PATH%`); if it points to 12.4, set it to 13.1 in the preset's `environment`.
- **MSVC flags leaking into nvcc.** The current `target_compile_options(ntc PRIVATE /W3 /O2 /fp:fast /arch:AVX2)` would be passed to any CUDA source in the same target; the plan keeps CUDA in its own static library and scopes the CXX flags with `$<COMPILE_LANGUAGE:CXX>` anyway. `OpenMP::OpenMP_CXX` already scopes `/openmp` to CXX in FindOpenMP.
- **Register pressure with MAXH = 128.** The reference `mlp_forward` on device spills to local memory; only the templated forward is used in hot kernels, MAXW picked per run, MAXW = 128 accepted as slow. Verify with `ptxas -v`; if MAXW = 48 exceeds ~128 registers, drop the second buffer by computing layer l+1 into a fresh array (the compiler renames) or cap threads per block at 128.
- **fp32 reduction order.** All loss sums cross the thread boundary in double with a fixed tree; FD differences of ~1e-8 on ~2e-4 losses are the sensitive case and are handled (4.4). The remaining CPU/GPU disagreement is per-term float rounding, identical in nature to the CPU's own.
- **Fast-math.** The CPU is `/fp:fast`; nvcc gets FMA contraction only. Do not add `--use_fast_math` (it swaps in `__expf/__sinf` with larger error and flushes denormals), and never compare grid values by recomputing the formula: `QAT_GRID` in `__constant__` is the single source, like on the CPU.
- **Occupancy and small grids.** Gathers over 128x128 levels and searches over coarse selector levels launch < 1 wave; harmless for the gather, a real slowdown for coarse `--qat` levels (warp-per-texel variant in v1.5).
- **WDDM launch latency and host stalls.** Windows display GPUs batch launches; keep the loop free of per-iteration `cudaMemcpy`/`cudaDeviceSynchronize` (only at print/save), use one stream, and enable Hardware-accelerated GPU scheduling in Windows. The `--print-every 10` host histogram is the largest remaining stall (4.8).
- **Shared-memory weight cap (P <= 12288).** Fine for every logged configuration (max 1983 weights); refused otherwise.
- **Phase 1 changing CPU codegen.** Moving the decoder into a header and iterating a POD instead of a vector does not change the float operation sequence, but `/fp:fast` gives the compiler latitude; the byte-compare of `model.bin` is the arbiter, and B-lite is the fallback.
- **Tie-breaking in the selector search.** FMA contraction makes a handful of near-tie texels choose differently from the CPU; the test tolerates a bounded count and checks each is a near-tie. This is not a bug and does not affect PSNR measurably.

## Appendix: reference configuration numbers used above

From `out_model_512c2q31_128c4_6k.log`: 512x512 model.png; level 0 512x512x2 nearest `--qat 3,1`; level 1 128x128x4 nearest; `--pos lv1local`; MLP 8->36->36->3 = 1767 params; MLP ES 64 pairs, batch 4096; latent ES 4 pairs (level 1 only); qat search every iteration = 10 decodes; FD from iteration 4500 (3534 evals x 4096 px = ~55 decode-equivalents); lr anneal from 3000. CPU: 8.4 it/s (ES phase), 5.2 it/s (FD phase), 1150.7 s total, 36.77 dB. Per-iteration decode-equivalents: ES phase 2 + 8 + 10 = 20 (+0.2 amortized stats), FD phase 73.

---

### Critical Files for Implementation

- `C:\dev\neural\main.cpp` — the reference implementation every kernel mirrors: `Decoder::features`/`pixel`/`decode_full`/`decode_err`/`loss_subset` (lines 694-803), `MlpTrainer::step`/`step_fd` (877-957), `LatentTrainer::step` (976-1076), `qat_search` and the grid helpers (1086-1169), `Adam::step` (837-847), the training loop and the backend seam (1698-1773), init/RNG order (1447-1498).
- `C:\dev\neural\CMakeLists.txt` — gains `option(NTC_CUDA)`, CXX-scoped MSVC flags, `add_subdirectory(cuda)`; plus the new `C:\dev\neural\CMakePresets.json` pinning "Visual Studio 17 2022" with `cuda=13.1`.
- `C:\dev\neural\cuda\ntc_kernels.cuh` (new) — `DevTexSet`/`DevTexture`/`DevLevel`, `mlp_forward_t<MAXW>`, and the eight kernels of section 4.
- `C:\dev\neural\cuda\ntc_cuda.cu` + `C:\dev\neural\cuda\ntc_cuda.h` (new) — `CudaTrainer`: buffers, uploads/downloads, launch sequencing, the v1 refusal list.
- `C:\dev\neural\ntc_model.h` (new, Phase 1) — the shared `NTC_HD` decoder core extracted from `main.cpp`; `C:\dev\neural\METHOD.md` sections 3.4 (gather form), 3.5 (search), 4 (batch contract) are the specifications the kernels must satisfy.
