# CUDA_PLAN_A.md — Plan A: minimal, lowest-risk CUDA backend for `ntc`

Target: RTX 5090 (sm_120), Windows 11, CUDA 13.1, VS 2022 toolset, CMake -> VS solution. CPU path stays the bit-identical reference. `--cuda` selects the backend; same command line, model file, outputs and stats.

Everything below quotes `C:\dev\neural\main.cpp` by function name and line as of commit `a7fea48`.

## 0. Summary of the design

- **Host loop unchanged.** `main()`, `Options`, annealing, phase switching, stats printing, PNG/model I/O, `bitrate_stats`, `mse_of`, both `Adam` structs for the MLP, and the `mt19937` stream for the MLP ES / minibatch / init all stay exactly as they are. `--cuda` only redirects the five hot functions:

  | CPU function (main.cpp) | GPU replacement | Where the hook goes |
  |---|---|---|
  | `Decoder::decode_full` (L719) | `k_decode<false>` | top of `decode_full` |
  | `Decoder::decode_err` (L730) | `k_decode<true>` | only called from `LatentTrainer::step`, which is replaced whole |
  | `Decoder::loss_subset` via `MlpTrainer::step` (L877) | `k_mlp_es_loss` | replaces the `#pragma omp parallel` block (L887-898) |
  | `Decoder::loss_subset` via `MlpTrainer::step_fd` (L927) | `k_mlp_fd_loss` | replaces L933-951 |
  | `LatentTrainer::step` (L976) | `k_perturb_latent`, `k_decode<true>` x2, `k_lat_attrib`, `k_lat_adam` | top of `step` |
  | `qat_search` (L1120) | `k_qat_search` | top of `qat_search` |

- **Device-resident state:** the flat latent `z` (all levels), its Adam moments, the target image, per-level texel->pixel range tables. The MLP weights (`D.mlp.p`, ~1.8k floats) stay **host-authoritative** and are uploaded (7 KB) before each kernel that reads them. The host `D.lat.z` is refreshed from the device only when the host needs it (stats/save/final), i.e. every `--print-every` iterations.
- **Kernels mirror the CPU functions one-to-one.** The pure math (`bilinear_tap`, `sample_latent`, `activate`, `mlp_forward`, `PosEnc::encode`, `Decoder::features`) is duplicated as `__device__` functions in `cuda_backend.cu` with a `// mirror of main.cpp:NNN` comment each, so `main.cpp` is not restructured. Unit checks (section 9) guard against drift.
- **RNG:** MLP perturbations and minibatch indices keep coming from the host `mt19937` (uploaded per step). The latent perturbation `eps` is generated on the device by a counter-based hash `(seed, step, pair, index) -> N(0,1)` (deterministic per seed, no state). A `--cuda-host-rng` debug mode instead draws `eps` on the host exactly as the CPU does and uploads it, which makes the GPU run consume the identical random stream as the CPU run (used for tight A/B checks).
- **Reductions:** per-thread float, per-block tree reduction in `double` (warp shuffle + shared), one `double` partial per block written to global, final fixed-order sum on the host in `double` (exactly the CPU's accumulation type). No float atomics anywhere; results are bitwise reproducible run to run.
- **fp32 throughout**, no `--use_fast_math`. FMA contraction left on (as MSVC `/fp:fast /arch:AVX2` does on the CPU).
- **v1 scope:** nearest-sampled levels only, `--qat` search, level-1 latent ES, MLP ES, FD, decode/eval/stats, materials, `--lat-alt`, `--mlp-full`, `--load`, `--iters 0`. Bilinear levels, `--deblock`, `onehot`, `bc7part` error out under `--cuda` (section 10 says how each fits later).

Expected speed on the owner's typical run (`model.png --latent 512 512 2 --qat 3,1 --latent2 128 128 4 --filter nearest,nearest --pos lv1local --mlp 36,36 --mlp-pairs 64`, 6000 iterations): CPU 1150 s (8.4 it/s ES phase, 5.2 it/s FD phase) -> GPU roughly 30-45 s (~200 it/s ES phase, ~100 it/s FD phase), i.e. 25-40x. Details in section 7.

## 1. Files and CMake

### 1.1 New files

- `cuda_backend.h` — plain C++ header, no CUDA types, included by `main.cpp` under `#ifdef NTC_CUDA`. Declares:
  ```cpp
  struct Decoder; struct Image; struct Options;   // forward
  bool cuda_init(const Decoder& D, const Image& target, const Options& o, std::string& err); // allocs, uploads, prints banner
  void cuda_shutdown();
  void cuda_upload_latent(const float* z_host);                       // host -> d_z (init, --load)
  void cuda_sync_latent_to_host(float* z_host);                       // d_z -> host (stats/save)
  void cuda_decode_full(const float* p_host, const float* z_host_or_null, Image& img); // null = device-resident z
  void cuda_mlp_es_losses(const float* p0, const float* eps, int N, float sigma,
                          const int* batch, int B, double* lp, double* lm);
  void cuda_mlp_fd_losses(const float* p0, float h, const int* batch, int B,
                          double* l0, double* lp, double* lm);
  struct CudaLatStep { int K; unsigned seed; int step_no; float sigma[2]; float lr, c1, c2;
                       unsigned char active[2][64] /* [pair][level] */; int pairs_of[2];
                       const float* host_eps /* null unless --cuda-host-rng; K*n floats */; };
  void cuda_latent_step(const CudaLatStep& s);
  void cuda_qat_search(const float* p_host, const int* ch_bits);
  ```
- `cuda_backend.cu` — everything CUDA: device config in `__constant__`, the `__device__` mirrors, the kernels, the host wrappers above, a `CUDA_CHECK` macro.

### 1.2 Changes to `main.cpp` (all under `#ifdef NTC_CUDA` or behind `o.cuda`)

1. `Options`: `bool cuda = false; bool cuda_host_rng = false; bool cuda_selftest = false;` and parsing of `--cuda`, `--cuda-host-rng`, `--cuda-selftest`. Without `NTC_CUDA`, `--cuda` prints `built without CUDA (configure with -DNTC_CUDA=ON)` and returns 1.
2. `usage()`: three lines.
3. After the model is initialised/loaded (after L1616) and before the banner: `if (o.cuda) { if (!cuda_init(D, target, o, err)) { printf("%s\n", err.c_str()); return 1; } }`. `cuda_init` prints `cuda     : NVIDIA GeForce RTX 5090 (sm_120, 170 SMs), CUDA 13.1`.
4. The v1 option guards (section 8) live in `cuda_init` and return an error string.
5. Hooks listed in section 0 (each 2-6 lines; the CPU bodies are untouched):
   - `decode_full`: `if (opt->cuda) { cuda_decode_full(p, z == lat.z.data() ? nullptr : z, img); return; }`. The `zq` decode passes the host `zq` pointer, which is uploaded to a scratch buffer.
   - `MlpTrainer::step`: after `draw_batch`, `if (o.cuda) cuda_mlp_es_losses(p0, eps.data(), N, o.mlp_sigma, batch.data(), (int)batch.size(), lp.data(), lm.data()); else { existing omp block }`, then `dl[i] = lp[i]-lm[i]; lsum[i] = 0.5*(lp[i]+lm[i])` as now. The rest (mean, `diff_std`, `grad`, `adam.step`) unchanged.
   - `MlpTrainer::step_fd`: same shape: `cuda_mlp_fd_losses(p0, h, batch..., &l0, lp, lm)`; `dl[j] = lp[j]-lm[j]`; `hh[j]` computed on the host exactly as now (`(double)(orig+h) - (double)(orig-h)` in float arithmetic, which is what the device also uses).
   - `LatentTrainer::step`: at the top, `if (o.cuda) { build CudaLatStep from K, o.lat_sigma/o.lat2_sigma, alt/level_of_pair/pairs_of (the existing lambdas), Adam constants from lt.adam (t++, c1, c2 as in Adam::step), lr; if (o.cuda_host_rng) draw eps for all K pairs with the existing draw loop into a K*n host vector; cuda_latent_step(s); step_no++; return; }`. Note the host `lt.adam.m/v` are then unused; `lt.adam.t` still counts steps so the bias-correction constants match the CPU.
   - `qat_search`: `if (D.opt->cuda) { cuda_qat_search(D.mlp.p.data(), ch_bits.data()); return; }`.
   - Stats/save block (L1733) and final block (L1774): `if (o.cuda) cuda_sync_latent_to_host(D.lat.z.data());` as the first statement. After that, `latent_stats`, `bitrate_stats`, the off-grid check, `save_model`, `save_latent_png` all run on the fresh host copy unchanged.
6. `--cuda-selftest` (section 9): a function `cuda_selftest(D, target, o, rng)` in `main.cpp` (host side, since it needs the CPU reference functions) that runs the per-kernel comparisons and exits.

Nothing else in `main.cpp` changes; the CPU build has no new code paths in its hot loop, so the `--iters 200` regression stays bit-identical (verify).

### 1.3 CMake

```cmake
cmake_minimum_required(VERSION 3.20)
project(ntc LANGUAGES CXX)
option(NTC_CUDA "Build the CUDA backend (--cuda)" OFF)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
find_package(OpenMP)

add_executable(ntc main.cpp)
if(OpenMP_CXX_FOUND)
  target_link_libraries(ntc PRIVATE OpenMP::OpenMP_CXX)
endif()
# Scope the host flags to C++ so nvcc never sees /fp:fast etc. Identical flags for the CPU build.
if(MSVC)
  target_compile_options(ntc PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W3 /O2 /fp:fast /arch:AVX2>)
else()
  target_compile_options(ntc PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall -O3 -ffast-math -march=native>)
endif()

if(NTC_CUDA)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 120)          # sm_120 SASS + compute_120 PTX; use "120-real" to drop the PTX
  endif()
  enable_language(CUDA)
  if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12.8)
    message(FATAL_ERROR "CUDA ${CMAKE_CUDA_COMPILER_VERSION} cannot target sm_120; configure with -T cuda=13.1 (VS generator) or -DCMAKE_CUDA_COMPILER=<13.1 nvcc>")
  endif()
  find_package(CUDAToolkit 12.8 REQUIRED)
  set(CMAKE_CUDA_STANDARD 17)
  target_sources(ntc PRIVATE cuda_backend.cu cuda_backend.h)
  target_compile_definitions(ntc PRIVATE NTC_CUDA=1)
  target_link_libraries(ntc PRIVATE CUDA::cudart_static)
  target_compile_options(ntc PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo -Xptxas=-v>)
  set_target_properties(ntc PROPERTIES CUDA_SEPARABLE_COMPILATION OFF CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()
```

Configure/build commands (the only supported way in v1):

```
cmake -S . -B build_cuda -G "Visual Studio 17 2022" -A x64 -T cuda=13.1 -DNTC_CUDA=ON
cmake --build build_cuda --config Release
```

Why `-T cuda=13.1`: with the Visual Studio generator CMake selects nvcc through the **toolset**, not through PATH or `CMAKE_CUDA_COMPILER`; the 12.4 on PATH would otherwise be picked and fail at sm_120 (the `VERSION_LESS 12.8` check catches that with a clear message). The CUDA 13.1 `BuildCustomizations` (`CUDA 13.1.props/targets`) are already installed in VS 2022 Community (`MSBuild/Microsoft/VC/v170/BuildCustomizations`), which is what the toolset string needs. `-T cuda="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1"` is the equivalent path form. `find_package(CUDAToolkit)` follows the selected nvcc, so no `CUDAToolkit_ROOT` is needed; set it explicitly if the search ever picks 12.4.

Host compiler: the VS 2022 v143 `cl.exe` of the generator. CUDA 13.1 officially supports VS 2022; if the owner generates with VS 2026 (v145 toolset) and nvcc rejects it, add `-allow-unsupported-compiler` to the CUDA options — but v1 should stay on `Visual Studio 17 2022`, which is the existing build. The CPU-only build (`-DNTC_CUDA=OFF`, default) does not enable the CUDA language at all, so it configures and builds exactly as today.

## 2. Device data layout

All device buffers are allocated once in `cuda_init` from the sizes in `Decoder`; nothing is allocated per iteration.

| Buffer | Type / size | Mirrors | Notes |
|---|---|---|---|
| `c_cfg` | `__constant__ DevCfg` | `Decoder` scalars, `MLP` widths, `PosEnc::feats`, `LatentSet::lv`, `cw[]`, `wsum` | see struct below |
| `d_target` | `float[W*H*nc]` | `Image target` (interleaved `r0 g0 b0 r1 ...`) | uploaded once |
| `d_z` | `float[n]` | `LatentSet::z` (level 0 first, `(y,x,c)`, c fastest) | resident; the truth during training |
| `d_zp`, `d_zm` | `float[n]` | `LatentTrainer::zp/zm` | perturbed copies (materialised, as on the CPU) |
| `d_eps` | `float[n]` | `LatentTrainer::eps` | one pair at a time |
| `d_grad`, `d_m`, `d_v` | `float[n]` each | `LatentTrainer::grad`, `Adam::m/v` | latent Adam lives on the device |
| `d_ep`, `d_em` | `float[W*H]` | `LatentTrainer::ep/em` | per-pixel weighted squared error |
| `d_zq` | `float[n]` | scratch for any host-supplied latent (`zq`, selftest) | |
| `d_img` | `float[W*H*nc]` | `Image recon` | decode output, downloaded on demand |
| `d_p`, `d_p_alt` | `float[P]` | `MLP::p` | uploaded before each kernel that uses it |
| `d_mlp_eps` | `float[Nmax*P]` | `MlpTrainer::eps` | uploaded per MLP ES step (64 x 1767 x 4 B = 450 KB) |
| `d_batch` | `int[max(mlp_batch, W*H)]` | `MlpTrainer::batch` | uploaded per MLP step |
| `d_partial` | `double[max(2*Nmax, 2*P) * nchunks]` | per-block loss partials | downloaded per MLP step |
| `d_range[l]` | `int xb[W_l], xe[W_l], yb[H_l], ye[H_l]` per level | the sweep at `qat_search` L1124-1126 | computed on the host at init for **every** nearest level, used by the gather and the search |
| `d_qat_grid` | `float[9*257]` | `QAT_GRID` | copied from the host table, so on-grid bit patterns are identical |

```cpp
static const int DEV_MAXH = 64;      // per-thread activation width; compile-time (raise to 128 if ever needed: doubles local memory)
static const int DEV_MAXPOS = 16;
struct DevLevel { int W, H, C, off, nearest; };
struct DevPos   { int kind, n; };
struct DevCfg {
  int W, H, nc, nin, nout, P, act, clamp_out, nhidden, hidden[MAXL];
  float leak;
  int nlev; DevLevel lv[2];
  int npos; DevPos pos[DEV_MAXPOS]; int cellw, cellh;
  float cw[MAXOUT]; float wsum;
};
```

Weights in shared memory: every kernel that evaluates the MLP starts with `extern __shared__ float sp[]; for (i = tid; i < P; i += blockDim) sp[i] = <its weight set>; __syncthreads();`. All threads of a warp read the same `sp[j*ncur+i]` at the same time (uniform loop indices), which is a shared-memory broadcast: one wavefront, no bank conflicts. This is why shared, not constant, memory is used everywhere: the ES kernel needs 2N *different* weight sets in flight (2 x 64 x 1767 floats = 900 KB, far over the 64 KB constant bank), and one code path for all kernels is simpler. Dynamic shared size = `P*4` bytes; for `P <= 12288` no opt-in is needed (48 KB); `cuda_init` rejects larger MLPs in v1 (or calls `cudaFuncSetAttribute(..., cudaFuncAttributeMaxDynamicSharedMemorySize, P*4)` up to 227 KB on sm_120 — trivial to add).

Per-thread activations: `float bufA[DEV_MAXH], bufB[DEV_MAXH]` exactly like `mlp_forward` (L395). Because the layer widths are runtime values, the compiler indexes these arrays dynamically and places them in **local memory** (L1-cached, per-thread private), not registers; the register file cannot be dynamically indexed. This is accepted in Plan A: local loads hit L1, the working set per thread is 2 x 36 x 4 = 288 B (the whole block's is 74 KB, within the 5090's 128 KB+ L1/shared per SM), and the kernel remains far faster than needed (section 7). Do not try to force registers in v1; templating the forward pass on `(nin, h1, h2)` for the 2-hidden-layer case is the documented follow-up if more speed is wanted. Check `-Xptxas -v` output: expect ~40-70 registers, a few hundred bytes of stack frame, and no spills beyond the activation arrays.

## 3. Device mirrors of the CPU math (`cuda_backend.cu`)

Each is a line-for-line copy of the CPU function with `__device__ __forceinline__`, `std::` replaced by the CUDA math functions of the same precision (`floorf`, `sinf`, `cosf`, `tanhf`, `expf`, `sqrtf`, `fminf/fmaxf`), `const Latent&` replaced by `const DevLevel&`:

- `dev_bilinear_tap` <- `bilinear_tap` (L279). v1 asserts `nearest` at init but keep the bilinear branch so the decode already works when the gather arrives.
- `dev_sample_latent` <- `sample_latent` (L302).
- `dev_activate` <- `activate` (L329).
- `dev_mlp_forward(const float* sp, const float* in, float* out)` <- `mlp_forward` (L394), reading widths from `c_cfg`.
- `dev_pos_encode(u, v, t, t1, f)` <- `PosEnc::encode` (L536), `switch` over `c_cfg.pos[]`. v1 implements `uv, fourier, local, lfourier, lquad, dct, ldct, ldct2, ldct4, lv1local, lv1ldct, bdct, bdcte` (needs a `__constant__ int BDCT_ZIGZAG[16][2]`); `onehot` and `bc7part` are rejected at init (they only need `cellw/cellh` and a 1 KB constant table; add when wanted).
- `dev_features(z, px, py, f)` <- `Decoder::features` (L694).
- `dev_pixel_err(out, target_ptr)` = the `sum_c cw[c]*(out[c]-t[c])^2` loop of `decode_err` (L770).

Use `PI` as the same float literal `3.14159265358979f`. Do not use the `__sinf`-style intrinsics; the plan wants 1e-6 agreement with the CPU.

## 4. Kernels

Thread/block conventions: 256 threads per block for pixel-parallel kernels (`__launch_bounds__(256)`), 1D grid over pixels or texels. `W*H` for a 512^2 image = 262144 threads = 1024 blocks: enough to fill 170 SMs (~1.5 blocks per SM per wave at 8 blocks/SM resident; occupancy is limited by local-memory bandwidth, not by block count).

### 4.1 `k_decode<bool ERR>` — replaces `decode_full` (L719) and `decode_err` (L730, non-deblock branch)

```cpp
template <bool ERR>
__global__ void __launch_bounds__(256)
k_decode(const float* __restrict__ p, const float* __restrict__ z,
         const float* __restrict__ target, float* __restrict__ out)
{
    extern __shared__ float sp[];
    for (int i = threadIdx.x; i < c_cfg.P; i += blockDim.x) sp[i] = p[i];
    __syncthreads();
    int pix = blockIdx.x * blockDim.x + threadIdx.x;
    if (pix >= c_cfg.W * c_cfg.H) return;
    int px = pix % c_cfg.W, py = pix / c_cfg.W;
    float f[DEV_MAXH]; dev_features(z, px, py, f);
    float o[MAXOUT];   dev_mlp_forward(sp, f, o);
    if (ERR) out[pix] = dev_pixel_err(o, target + (size_t)pix * c_cfg.nc);
    else     for (int c = 0; c < c_cfg.nc; c++) out[(size_t)pix * c_cfg.nc + c] = o[c];
}
```

`cuda_decode_full`: upload `p` to `d_p`; if a host `z` was given upload it to `d_zq`; launch `k_decode<false>`; `cudaMemcpy` `d_img` -> `img.rgb` (3 MB for one texture, 12 MB for four). Host then runs `mse_of`, PNG writes etc. unchanged. Cost: ~0.1-0.2 ms kernel + ~0.2-0.5 ms copy; only at print/save time.

### 4.2 Latent ES — replaces `LatentTrainer::step` (L976-1076)

Per pair `k`, per level `l`, all in one stream, no host sync until the end of the step:

**(a) `k_perturb_latent`** <- the draw loop L1004-1017. One thread per latent value `i`:
```cpp
__global__ void k_perturb_latent(unsigned seed, int step_no, int pair,
                                 const float* __restrict__ z, const float* __restrict__ host_eps /*or null*/,
                                 float* eps, float* zp, float* zm, float sigma0, float sigma1, int active0, int active1, int n)
{
    int i = ...; if (i >= n) return;
    int l = (c_cfg.nlev > 1 && i >= c_cfg.lv[1].off) ? 1 : 0;
    float sg = l ? sigma1 : sigma0; int active = l ? active1 : active0;
    float e = 0.f;
    if (active) e = host_eps ? host_eps[i] : hash_gauss(seed, (unsigned)step_no, (unsigned)pair, (unsigned)i);
    eps[i] = e; zp[i] = z[i] + sg * e; zm[i] = z[i] - sg * e;   // same expressions as L1011-1012
}
```
`active` encodes both the `--lat-alt` rule (host `level_of_pair(k) == l`) and the `--qat` rule (`level 0 inactive`), computed on the host from the existing lambdas and passed in `CudaLatStep::active[pair][level]`.

**(b) `k_decode<true>`** on `zp` -> `d_ep` and on `zm` -> `d_em` (two launches, identical to `decode_err`).

**(c) `k_lat_attrib`** <- the scatter L1029-1071 in **gather form**, one thread per texel of level `l`, fused with the `grad += w * eps` loop:
```cpp
__global__ void k_lat_attrib(int l, const float* __restrict__ ep, const float* __restrict__ em, const float* __restrict__ eps,
                             const int* xb, const int* xe, const int* yb, const int* ye, float inv_px, float scale, float* grad)
{
    const DevLevel& L = c_cfg.lv[l];
    int t = ...; if (t >= L.W * L.H) return;
    int tx = t % L.W, ty = t / L.W;
    float d = 0.f;                                       // == dtex[t] on the CPU
    for (int py = yb[ty]; py < ye[ty]; py++)             // nearest: the cell, row-major = the CPU's raster order
        for (int px = xb[tx]; px < xe[tx]; px++) {
            size_t q = (size_t)py * c_cfg.W + px;
            d += (ep[q] - em[q]) * inv_px;               // same expression as L1034
        }
    float w = d * scale;                                 // L1068
    size_t base = L.off + (size_t)t * L.C;
    for (int c = 0; c < L.C; c++) grad[base + c] += w * eps[base + c];   // L1070; grad zeroed once per step
}
```
For a nearest level the CPU visits a texel's cell pixels in raster order and adds `d` into `dtex[t]` in that same order, so `d` here is the identical float sequence: given identical `ep/em`, `dtex` is bit-identical. Texels with an empty range (level finer than the image) get `d = 0`, as on the CPU where no pixel scatters into them. `inv_px = 1/(3*wsum*W*H)` and `scale = 1/(2*pairs_of[l]*sigma_l)` come from the host exactly as computed at L986 and L1028. `grad` is zeroed by a `cudaMemsetAsync` at the start of the step (`std::fill` at L984).

**(d) `k_lat_adam`** <- `Adam::step` (L837), one thread per value, `c1, c2, lr` from the host (`t` incremented on the host so `lt.adam.t` keeps the CPU meaning):
```cpp
m[i] = b1*m[i] + (1-b1)*g[i]; v[i] = b2*v[i] + (1-b2)*g[i]*g[i];
z[i] -= lr * (m[i]/c1) / (sqrtf(v[i]/c2) + eps);
```
With `--qat`, level-0 `grad` stays exactly 0, so `m = v = 0` and the update is `-lr * 0 / (0 + 1e-8) = 0`: level 0 stays bit-exact on grid, as on the CPU (the off-grid WARNING check in `main` still runs on the synced copy).

Kernel count per latent step: `K * (1 + 2 + nlev_active) + 1` launches, ~13 for the typical run.

### 4.3 MLP ES — replaces the `loss_subset` calls in `MlpTrainer::step` (L887-898)

Grid `(2N, nchunks)` with `nchunks = ceil(B / 256)` (16 for the 4096 minibatch, 1024 for `--mlp-full`); block `(pair i, sign s)` = `blockIdx.x`, chunk = `blockIdx.y`:
```cpp
__global__ void __launch_bounds__(256)
k_mlp_es_loss(const float* __restrict__ p0, const float* __restrict__ eps, float sigma,
              const int* __restrict__ batch, int B, const float* __restrict__ z,
              const float* __restrict__ target, double* __restrict__ partial /*[2N][nchunks]*/)
{
    extern __shared__ float sp[];
    int i = blockIdx.x >> 1; float sgn = (blockIdx.x & 1) ? -sigma : sigma;
    const float* e = eps + (size_t)i * c_cfg.P;
    for (int k = threadIdx.x; k < c_cfg.P; k += blockDim.x) sp[k] = p0[k] + sgn * e[k];   // L893
    __syncthreads();
    int j = blockIdx.y * blockDim.x + threadIdx.x;
    float err = 0.f;
    if (j < B) { int pix = batch[j]; px = pix % W; py = pix / W;
                 float f[DEV_MAXH]; dev_features(z, px, py, f); float o[MAXOUT]; dev_mlp_forward(sp, f, o);
                 err = dev_pixel_err(o, target + (size_t)pix * c_cfg.nc); }
    double bs = block_reduce_double((double)err);       // warp shuffle + shared, fixed order
    if (threadIdx.x == 0) partial[blockIdx.x * gridDim.y + blockIdx.y] = bs;
}
```
Host wrapper: upload `p0`, `eps` (N*P), `batch`; launch; download `partial` (2N*nchunks doubles = 16 KB for the minibatch, 1 MB for `--mlp-full`); `lp[i] = sum_chunks partial[2i][*] / (3*wsum*B)`, `lm[i]` likewise, summed in `double` in chunk order (`loss_subset` L802 divides once at the end; do the same). The block tree in `double` reproduces the CPU's `double s +=` accumulation up to ordering; per-pixel `err` is a float on the CPU as well (it is accumulated into `double s` per pixel: the device keeps the per-pixel value in float and sums in double, same thing).

`block_reduce_double`: `__shfl_down_sync` works on `double`; 5 shuffle steps per warp, then 8 warp sums through shared memory added by thread 0 in warp order. sm_120's fp64 rate is 1/64 of fp32 but this is 256 adds per block against ~450k FMAs of MLP work: negligible. No atomics, deterministic.

Blocks per launch: 2 x 64 x 16 = 2048 blocks of 256 threads (524k threads, 2 decode-equivalents of work): a well-filled GPU.

### 4.4 FD — replaces `MlpTrainer::step_fd` L933-951

Grid `(2P, nchunks)`; block `(weight j, sign s)`:
```cpp
__global__ void __launch_bounds__(256)
k_mlp_fd_loss(const float* __restrict__ p0, float h, int j0 /*first weight of this launch*/,
              const int* __restrict__ batch, int B, const float* z, const float* target, double* partial)
{
    extern __shared__ float sp[];
    int j = j0 + (blockIdx.x >> 1); int minus = blockIdx.x & 1;
    for (int k = threadIdx.x; k < c_cfg.P; k += blockDim.x) sp[k] = p0[k];
    __syncthreads();
    if (threadIdx.x == 0) { float orig = sp[j]; sp[j] = minus ? orig - h : orig + h; }   // float-realized, L941-942
    __syncthreads();
    ... identical pixel evaluation and block reduction as 4.3 ...
}
```
The unperturbed `l0` is one extra launch of the same kernel with `h = 0` (or a one-block-row variant). Host: `lp[j], lm[j]` from the partials, then the existing lines `grad[j] = dl/hh`, `ss`, `diff_std`, `adam.step` unchanged. `hh[j]` computed on the host from `float wp = orig + h, wm = orig - h` as today: the device does the same float add (`orig + h` is a single `fadd`, no contraction possible), so the realized step matches.

Work: 2 x 1767 x 4096 = 14.5M pixel evaluations = 55 decode-equivalents per step; 56.5k blocks per launch. **Launch in slices of at most ~512 weights** (1024 x 16 blocks) so no single kernel runs longer than a few ms: with `--mlp-fd --mlp-full` (B = 262144) one weight is a full decode and the whole step is 3.5k decodes (~0.5 s); slicing keeps every launch far below the Windows WDDM 2-second TDR limit. Partials: 2P x 16 doubles = 450 KB per step download.

### 4.5 `k_qat_search` — replaces `qat_search` (L1120-1169)

One thread per level-0 texel; the thread owns its cell `[xb,xe) x [yb,ye)` and runs the CPU's coordinate descent verbatim:
```cpp
__global__ void __launch_bounds__(256)
k_qat_search(const float* __restrict__ p, float* z /*level 0 in place; level 1 read*/, const float* target,
             const int* xb, const int* xe, const int* yb, const int* ye, const int* ch_bits /*[C0]*/)
{
    extern __shared__ float sp[]; ... load p ...
    const DevLevel& L = c_cfg.lv[0];
    int t = ...; if (t >= L.W*L.H) return; int tx = t % L.W, ty = t / L.W;
    int x0 = xb[tx], x1 = xe[tx], y0 = yb[ty], y1 = ye[ty];
    if (x0 >= x1 || y0 >= y1) return;                    // texel no pixel reads (L1136)
    float* zt = z + (size_t)t * L.C;
    for (int c = 0; c < L.C; c++) {
        int bits = ch_bits[c], levels = (1 << bits) - 1;
        float loss[256]; for (k <= levels) loss[k] = 0.f;   // one accumulator per candidate
        for (py = y0..y1) for (px = x0..x1) {
            float f[DEV_MAXH]; dev_features(z, px, py, f);      // reads zt[0..C) incl. channels already updated (L1164)
            const float* tg = target + ((size_t)py*W + px) * nc;
            for (int k = 0; k <= levels; k++) { f[c] = c_qat_grid[bits][k];   // the patching trick, L1147
                float o[MAXOUT]; dev_mlp_forward(sp, f, o); loss[k] += dev_pixel_err(o, tg); }
        }
        int best_k = qat_index_dev(zt[c], bits); float best = loss[best_k];   // current value first (L1155-1156)
        for (int k = 0; k <= levels; k++) { if (k == best_k) continue; if (loss[k] < best) { best = loss[k]; best_k = k; } }  // strict <, same order (L1157-1161)
        zt[c] = c_qat_grid[bits][best_k];                 // L1163; visible to dev_features for channel c+1
    }
}
```
Loop order differs from the CPU (candidates inner, pixels outer) so that features are computed once per pixel per channel; per candidate the sequence of float ops per pixel is identical, only the cross-pixel sum order is the same (raster) but in float instead of the CPU's `double sum`. For 1x1 cells (the owner's 512x512 selector level) there is no sum at all and the per-candidate losses are the same float value as the CPU's up to the MLP evaluation itself (~1e-7 relative), so selector disagreements can only occur at near-ties; section 9 measures them. `loss[256]` lives in local memory (1 KB per thread worst case; 8-16 entries in practice). Threads: 262144 for the 512^2 level; for a 128x128 level (16-pixel cells) only 16k threads, each 16x the work: the GPU is under-filled but the search is then also only 10 decode-equivalents, ~1-2 ms; a warp-per-texel variant is the follow-up if 4x4-cell searches ever matter. Writes are to the thread's own texel only: no races. Level 1 is read-only during the search, as on the CPU. A `qat_index_dev` mirror of `qat_index` (L1092) uses `lrintf`/`__float2int_rn` for `std::lround` (values are exactly on grid, so the rounding mode is not exercised).

### 4.6 RNG: `hash_gauss`

```cpp
__host__ __device__ inline unsigned long long mix64(unsigned long long x) {   // splitmix64 finalizer
    x += 0x9E3779B97F4A7C15ull; x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull; return x ^ (x >> 31);
}
__host__ __device__ inline float hash_gauss(unsigned seed, unsigned step, unsigned pair, unsigned idx) {
    unsigned long long h = mix64(((unsigned long long)seed << 32 | step) ^ mix64(((unsigned long long)pair << 32) | idx));
    float u1 = ((unsigned)h >> 8) * 0x1p-24f + 0x1p-25f;           // (0,1), never 0
    float u2 = ((unsigned)(h >> 32) >> 8) * 0x1p-24f;
    return sqrtf(-2.f * logf(u1)) * cosf(6.28318530718f * u2);      // Box-Muller, one of the pair
}
```
Pure function of `(--seed, LatentTrainer::step_no, pair, flat index)`: no state, no ordering dependence, identical on host and device (the selftest compares a host evaluation against a device buffer bitwise). Alternative with the same properties: `curand_Philox4_32_10_t` with `curand_init(seed, idx, step*K + pair)` + `curand_normal` (header-only, no library); Philox's init is cheap, XORWOW's is not — do not use the default `curandState`. The hand-rolled one is preferred because the host reference is 6 lines.

Determinism statement for the plan: with `--cuda --seed S`, every random quantity is either drawn from the host `mt19937` (init, MLP eps, minibatch) in a fixed order, or from `hash_gauss` with fixed arguments; every reduction has a fixed order; every kernel writes each output from exactly one thread. Two runs with the same seed therefore produce byte-identical logs and model files. This is a test (section 9).

The host `mt19937` stream under `--cuda` differs from the CPU run only by the latent draws that are not made (same effect as the documented `--mlp-full` skipping the index draws); with `--cuda-host-rng` the draws are made and uploaded (`K*n` floats, 1 MB per step for the typical run, ~1-5 ms of host time), so the two runs consume the identical stream.

## 5. Host wrapper flow per iteration (typical `--qat` run)

```
MLP ES step   : host draws eps (N*P) + batch [mt19937, unchanged]
                upload p0 (7 KB), eps (450 KB), batch (16 KB); k_mlp_es_loss; download partials (16 KB)
                host: dl, dstd, grad, Adam            [unchanged]
Latent step   : upload p (7 KB); memset grad
                for k in 0..K-1: k_perturb_latent; k_decode<true>(zp); k_decode<true>(zm); k_lat_attrib(level 1)
                k_lat_adam
qat search    : upload p; k_qat_search           (every --qat-every iters and the last)
print/save    : k_decode<false> -> download recon (3 MB); cuda_sync_latent_to_host (2.4 MB)
                host: mse_of, latent_stats, bitrate_stats -> zq; upload zq; k_decode<false> -> download recon_q
                host: PNGs, save_model                [unchanged]
```
A single default stream; `cudaMemcpy` (synchronous) for the host-visible transfers is fine at this size. `CUDA_CHECK` after every launch (`cudaGetLastError`) and on every copy; on failure print and `exit(1)`.

## 6. What stays on the host, and why

- All of `main()`: option parsing, phase logic, annealing (`oi`), banners, stats line, `best_psnr`, PNG/model writes, `bitrate_stats`, `mse_of`, `latent_stats`, off-grid check. They read the synced host copies.
- MLP Adam, MLP `grad` accumulation (`grad[k] += w*e[k]`, 64 x 1767 MACs), `dstd`, `fd-rms`: unchanged CPU code, microseconds.
- MLP eps and minibatch draws (`mt19937` + `normal_distribution`): ~113k draws per step for 64 pairs, roughly 1.5-3 ms of single-thread host time. This is the largest host-side cost left and is comparable to the GPU time of the whole step; it is accepted in Plan A and listed as the first follow-up (draw them with `hash_gauss` in a kernel and keep them on the device; the host would then also need the grad accumulation moved, ~20 lines).
- Latent init (`N(0, lat_init)` on the host, then `cuda_upload_latent`), `--load`, `qat_snap_level0`: unchanged, then uploaded.
- Latent Adam moves to the device (section 4.2d) because otherwise `grad` and `z` (2.4 MB each) would round-trip every iteration; the `Adam` struct's `t` counter stays on the host.

## 7. Cost model and expected times (RTX 5090)

MLP `8 -> 36 -> 36 -> 3`: 288 + 1296 + 108 = 1692 MACs per pixel; a 512^2 decode is 0.44 G MACs, ~0.9 GFLOP. Inner loop per MAC: one shared load (broadcast), one local load (activation, L1), one FFMA, loop overhead: ~4 instructions per MAC, so ~1.8 G lane-instructions per decode. The 5090 issues ~1.6 T warp-instructions/s (170 SMs x 4 schedulers x 2.4 GHz) = ~52 T lane-instr/s: 35 us at perfect issue; with the LSU as the realistic bottleneck (2 loads per MAC per warp at 1 wavefront each: ~70 us) and imperfect occupancy, budget **~100 us per decode-equivalent, 200 us conservative**. Blackwell consumer parts have plenty of L1 for the 74 KB per block of activation local memory at 256 threads.

| Kernel | Work | Estimated time |
|---|---|---|
| `k_decode` (either) | 1 decode-eq | 0.1-0.2 ms |
| `k_perturb_latent` | 590k values, hash + log/cos | 20-30 us |
| `k_lat_attrib` (128x128 level) | 16k threads x 16 pixels | 10-20 us |
| `k_lat_adam` | 590k values | 10 us |
| Latent step (K = 4) | 8 decode-eq + small | ~1-1.8 ms |
| `k_mlp_es_loss` (64 pairs, 4096 px) | 2 decode-eq | 0.2-0.4 ms (+ ~0.1 ms uploads) |
| `k_mlp_fd_loss` (1767 weights, 4096 px) | 55 decode-eq | 6-11 ms |
| `k_qat_search` (512^2 x 2 ch, 3,1 bits) | 10 decode-eq | 1-2 ms |
| stats decode + downloads | 2 decode-eq + 8 MB copies | ~1 ms, every 10 iters |
| launch overhead | ~25 launches/iter x ~5-15 us (WDDM) | 0.15-0.4 ms |
| host: MLP eps draws | 113k `normal_distribution` | 1.5-3 ms |

Per iteration, ES phase: ~4-6 ms (170-250 it/s vs 8.4 on the CPU: 20-30x). FD phase: ~10-16 ms (60-100 it/s vs 5.2: 12-20x). The 6000-iteration run: ~35-45 s vs 1150 s (25-35x). The host-side eps draw and the local-memory MLP are the two known 2-3x levers left on the table, deliberately.

## 8. Options that error out with `--cuda` in v1 (checked in `cuda_init`)

- any level with bilinear sampling (`--filter bilinear`, the default): "`--cuda` v1 needs `--filter nearest[,nearest]`" (see 10.1; the decode side already handles bilinear, only `k_lat_attrib` is nearest-only)
- `--deblock`
- `--pos onehot`, `--pos bc7part:N`
- hidden width or `nin` > `DEV_MAXH` (64); `P*4` > 48 KB
- `--threads N`: printed as ignored (OpenMP still used by the remaining host code, harmless)
- `--cuda-host-rng` without `--cuda`: note and ignore

Everything else is supported: materials (`nc = 3T`, `cw`), `--weights`, `--latent2`, `--lat-alt`, `--qat B1,B2,...`, `--qat-every`, `--mlp-fd`, `--mlp-full`, `--mlp-freeze`, `--mlp-every`, annealing, `--clamp`, all activations, `--load`, `--iters 0`, `--seed`.

## 9. Verification plan

### 9.1 `--cuda-selftest` (per-kernel, same inputs, run after init, then exit)

All comparisons print max abs / max rel difference and PASS/FAIL against the stated tolerance; the CPU functions are called directly (they still exist unchanged).

1. **RNG**: for 1000 random `(step, pair, idx)` compare host `hash_gauss` with a device buffer filled by `k_perturb_latent(sigma=1, z=0)`: bitwise equal. Also check mean/variance of 1M draws (0 +- 0.003, 1 +- 0.005).
2. **Decode**: `decode_full` CPU vs `cuda_decode_full` on the initial model, and again on a model with random `p` scaled by 3 (exercise saturation): max abs diff over all `W*H*nc` outputs < 1e-5 (expected ~1e-6; the sigmoid compresses differences).
3. **decode_err**: `D.decode_err` vs `k_decode<true>`: max abs < 1e-6 absolute plus 1e-5 relative.
4. **MLP ES losses**: draw 8 pairs and a batch with the host RNG; CPU `loss_subset(pp)`, `loss_subset(pm)` vs `cuda_mlp_es_losses`: `|lp - lp_gpu| < 1e-6 * lp` and, the quantity that matters, `|dl - dl_gpu| < 1e-3 * dstd` where `dstd` is the std of the 8 `dl`.
5. **FD losses**: for 32 random weights `j`: CPU `lp, lm` (the loop of `step_fd`) vs `cuda_mlp_fd_losses`: relative error of `dl_j` < 1e-2 against the RMS of the 32 `dl_j` (FD differences are ~1e-8 on a 2e-4 loss; the double block reduction is what makes this pass).
6. **Latent step**: with `--cuda-host-rng` semantics, draw `K` eps on the host; run the CPU `LatentTrainer::step` on a copy of `D` and `cuda_latent_step` with the same eps on the device; compare `z` after Adam: max abs < 1e-5 (Adam normalises, so this is a direct test of `dtex` and `grad`). Also compare `grad` before Adam relative to its RMS (< 1e-4).
7. **qat search**: copy `z`; run CPU `qat_search` on the copy and `k_qat_search` on the device from the same start; report the number of level-0 values that differ. For each differing texel evaluate both choices' cell loss on the CPU: the GPU's choice must never be worse than the CPU's by more than 1e-6 absolute (near-tie flips only). Expected differing count: 0 to a few dozen out of 524k.
8. **Determinism**: run items 4-7 twice; device results bitwise identical.

### 9.2 End-to-end

1. **CPU regression unchanged**: rebuild the CPU-only configuration from the modified tree and confirm `ntc.exe --iters 200 ...` still prints exactly 23.83 dB (and identical stats lines). Then build with `-DNTC_CUDA=ON` and run the same command **without** `--cuda`: still 23.83 dB (the `NTC_CUDA` compile definition must not change the CPU path).
2. **Initial line**: with `--cuda` the `iter 0 ... (initial)` line must print the same mse/psnr as the CPU (same init, decode only).
3. **Tight A/B**: `--cuda --cuda-host-rng` vs CPU on a nearest `--qat` configuration for 200 iterations (e.g. the `model.png --latent 512 512 2 --qat 3,1 --latent2 128 128 4 --filter nearest,nearest --pos lv1local --mlp 36,36 --mlp-pairs 64` command at `--iters 200`; record the CPU number first). Same random stream, so only rounding differs: expect PSNR agreement within ~0.02 dB at 200 iterations and identical `qat` decode counts.
4. **Statistical agreement**: the full 3000/6000-iteration owner runs (`out_model_512c2q31_128c4_6k.log`: 36.77 dB; `out_2lv_512c1_128c4_qat3.log`; `out_m1234_512c1q2_128c4.log` for materials) with `--cuda` and 3 different seeds; expect final PSNR within the CPU seed-to-seed spread (measure it once on the CPU with two extra seeds, expected ~0.05-0.15 dB).
5. **Same-seed reproducibility**: two `--cuda` runs, `fc /b` on `model.bin` and diff of the logs (modulo timing fields): identical.
6. **`--load` round trip**: a GPU-trained `model.bin` loaded by the CPU path with `--iters 0` prints the same PSNR the GPU printed at the end.

## 10. Later extensions (not v1) and how they fit

### 10.1 Bilinear levels (~30-40 lines, unlocks the default configuration)
The decode already works (the `dev_bilinear_tap` bilinear branch). Attribution: keep the gather, one thread per texel, over the conservative footprint rectangle `px in [floor(r*(tx-1)) - 1, ceil(r*(tx+1)) + 1]` (METHOD §1.2/§3.4) intersected with the image, and credit a pixel iff `bilinear_tap` for that pixel yields `tx in {x0, x1}` and `ty in {y0, y1}` — a set-membership test, which reproduces the CPU's `if (t.x1 != t.x0)` distinct-texel guard exactly and needs no closed-form index arithmetic. Cost: ~4x the reads of the nearest gather, still microseconds. No atomics. The summation order over the footprint then differs from the CPU's raster scatter (bitwise `dtex` equality is lost; agreement ~1e-6 relative). The alternative, 4 `atomicAdd`s per pixel into `dtex`, is simpler but non-deterministic; rejected.

### 10.2 Deblocking
`k_decode<true>` becomes two kernels: raw decode into a scratch image, then a filter+error kernel mirroring `decode_err`'s deblock branch (L733-761) with `deblock_weights`/`deblock_pixel` mirrors. `k_decode<false>` adds a `deblock_image` kernel. `loss_subset`'s per-pixel cross (L783-795) is mirrored in the ES/FD kernels (3-5 MLP evaluations per ring pixel). Attribution: the gather membership test of 10.1 extended over the union of the pixel's own tap and its filtered neighbours' taps (the `add_tap` set, L1049-1061), over a rectangle dilated by one pixel. `--qat` + `--deblock` remains rejected as on the CPU.

### 10.3 `onehot`, `bc7part`
`cellw/cellh` are already in `DevCfg`; add `__constant__ unsigned char c_bc7[1024]`. Two `case`s in `dev_pos_encode`.

### 10.4 Performance follow-ups, in order of payoff
1. Draw the MLP eps and the minibatch on the device (`hash_gauss` with a different tag), accumulate the MLP `grad` on the device (one thread per weight, fixed order over pairs): removes 1.5-3 ms of host time per iteration.
2. Template `dev_mlp_forward` on `(nin, h1, h2)` for the 2-hidden-layer case via a small dispatch table (e.g. 8/36/36, 6/36/36, 10/24/24): activations become registers; expected 2-3x on all kernels.
3. Warp-per-texel `k_qat_search` for cells >= 2x2.
4. `cudaMemcpyAsync` with pinned host buffers and a second stream for the stats decode.

## 11. Implementation order, tests and effort

| Step | Work | Test before moving on | Effort |
|---|---|---|---|
| 0 | CMake option, `cuda_backend.{h,cu}` skeleton, `--cuda` flag, `cuda_init` (device query, allocations, `DevCfg`, uploads, v1 guards, banner), `CUDA_CHECK` | both configurations build; CPU regression 23.83 dB in both; `--cuda` prints the banner and runs the CPU path... i.e. no hooks yet | 0.5 d |
| 1 | Device mirrors (§3), `k_decode<false>`, `cuda_decode_full` hook, `--cuda-selftest` items 2 | selftest 2 passes; `--cuda --iters 0 --load <model>` prints the CPU's PSNR | 0.5-1 d |
| 2 | `k_decode<true>`, `k_mlp_es_loss`, `k_mlp_fd_loss`, block reduction, hooks in `MlpTrainer::step/step_fd` | selftest 3-5; a `--mlp-freeze 0` run is unaffected; a run with `--lat-lr 0` (latent frozen) trains the MLP to the same PSNR as the CPU within noise | 1 d |
| 3 | `hash_gauss`, `k_perturb_latent`, `k_lat_attrib`, `k_lat_adam`, `cuda_latent_step`, `--cuda-host-rng`, `cuda_sync_latent_to_host` at the two host sites | selftest 1, 6, 8; end-to-end 9.2.2 and 9.2.3 | 1 d |
| 4 | `k_qat_search`, `d_qat_grid`, hook | selftest 7; the `--qat 3,1` 200-iteration A/B | 0.5 d |
| 5 | Full runs (9.2.4-9.2.6), timing with `nsys`/`cudaEvent`s per kernel, `-Xptxas -v` review, block-size sanity (128 vs 256), FD slicing check against TDR with `--mlp-full` | numbers in a `CUDA_RESULTS.md` or the README table | 0.5-1 d |
| 6 (optional) | 10.1 bilinear gather | the default `kodim23` configuration at 200 iterations vs 23.83 dB within noise; selftest 6 extended to a bilinear level | 0.5 d |

Total for v1: about 4-5 engineer-days including verification.

## 12. Risks and mitigations

- **CUDA 13.1 + VS 2026 host compiler.** Not officially supported by nvcc 13.1; stay on the `Visual Studio 17 2022` generator (already the owner's build). If VS 2026's toolset gets picked up anyway (`-T v145`), `-allow-unsupported-compiler` usually works; not a v1 concern.
- **Wrong nvcc picked (12.4 on PATH).** Fixed by `-T cuda=13.1`; the `VERSION_LESS 12.8` check turns a cryptic `ptxas` error into a one-line message.
- **MSVC flags leaking into nvcc.** The existing `target_compile_options` must be wrapped in `$<$<COMPILE_LANGUAGE:CXX>:...>` (done above); otherwise nvcc fails on `/fp:fast`.
- **fp32 reductions.** Per-block reduction in `double` and host `double` final sums; no atomics. FD differences (1e-8 on 2e-4) are the sensitive case and selftest 5 measures them.
- **Register pressure / occupancy.** Runtime-width activation arrays go to local memory; `__launch_bounds__(256)`, check `-Xptxas -v` for spills beyond the arrays. Expected occupancy 25-50%, adequate for a compute-bound kernel with L1-resident locals. `DEV_MAXH = 64` keeps local frames at ~600 B/thread.
- **Fast-math.** Do not pass `--use_fast_math`; it swaps `sinf/cosf/expf/logf` for the approximate intrinsics and turns divisions approximate, breaking the 1e-6 agreement. Contraction stays on (as on the CPU).
- **qat near-tie flips** between CPU and GPU: expected, harmless (both are minimisers within rounding); the selftest quantifies them.
- **WDDM TDR (2 s)** on long kernels: slice the FD grid (§4.4); nothing else runs longer than a few ms.
- **Launch/sync overhead on WDDM**: ~25 launches and ~6 synchronous copies per iteration, 0.2-0.5 ms; acceptable. Do not add a `cudaDeviceSynchronize` per launch outside debug builds (make it a `NTC_CUDA_SYNC_DEBUG` define).
- **Host `mt19937` cost** for MLP eps (1.5-3 ms/iter) caps the ES-phase speedup near 30x; documented follow-up 10.4.1.
- **Code duplication** of the pure math in `cuda_backend.cu`: drift is caught by selftest 2-3; an optional later refactor moves those functions to a shared `ntc_core.h` with a `NTC_HD` (`__host__ __device__`) macro, after the tests exist.
- **Shared-memory cap** at 48 KB without opt-in (`P <= 12288`); larger MLPs are rejected in v1 with a message.

---

### Critical Files for Implementation
- `C:\dev\neural\main.cpp` — hooks at `Decoder::decode_full` (L719), `MlpTrainer::step` (L877) / `step_fd` (L927), `LatentTrainer::step` (L976), `qat_search` (L1120), the stats block (L1733) and the final block (L1774); `Options`/usage; `--cuda-selftest`
- `C:\dev\neural\cuda_backend.cu` (new) — device config, mirrors of the CPU math, the six kernels, host wrappers, `hash_gauss`
- `C:\dev\neural\cuda_backend.h` (new) — the plain-C++ API `main.cpp` calls
- `C:\dev\neural\CMakeLists.txt` — `NTC_CUDA` option, CXX-scoped flags, `enable_language(CUDA)`, sm_120, version guard
- `C:\dev\neural\METHOD.md` — §1.5, §3.4 (gather form), §3.5 (search invariants), §7 (RNG, QAT grid table) are the specification the kernels must honour
