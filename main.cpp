// ntc - toy neural texture compressor trained with Evolution Strategies.
//
//   I_hat(u,v) = MLP( bilinear(Z, u, v), phi(u,v) )
//
// Z   : low-res latent texture (LW x LH x LC floats), optionally plus a second,
//       coarser level (--latent2) whose bilinear sample is concatenated onto it.
//       Up to 4 same-size RGB textures (a material) share Z and the MLP, which
//       then has 3 outputs per texture; per-texture loss weights via --weights.
// MLP : tiny fully connected net, < ~2000 weights
// phi : (u,v) in [-1,1] plus a few Fourier features
//
// Both Z and the MLP are trained with antithetic ES (no backprop).
//   MLP    : global ES on a random pixel minibatch shared across all pairs.
//   Latent : all texels perturbed at once; each texel's loss change is measured
//            only over the pixels its bilinear footprint touches, so one
//            full-image decode pair yields a gradient estimate for every texel.

#define _CRT_SECURE_NO_WARNINGS
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#endif

#include "ntc_noise.h"   // counter-based noise: the GPU backend's RNG, also usable on the CPU (--rng hash)
#ifdef NTC_CUDA
#include "ntc_cuda.h"    // GPU backend (--cuda); see CUDA_PLAN_B.md
#endif

static const float PI = 3.14159265358979f;

// ---------------------------------------------------------------- options
struct Options {
    std::vector<std::string> inputs;    // positional PNGs, 1..MAXT; empty after parsing -> kodim23.png (checked in)
    std::vector<float> weights;         // per-texture loss weight; all 1 unless --weights was given
    bool weights_given = false;
    std::string outdir = "out";
    int crop = 512;            // center-crop input to crop x crop (0 = none)
    int LW = 64, LH = 64, LC = 4;
    int LW2 = 0, LH2 = 0, LC2 = 0;             // optional second (coarser) latent level; 0 0 0 = off
    std::string filter = "bilinear";           // per-level sampling: "bilinear" or "nearest", comma-separated per level
    bool deblock = false;                      // in-loop deblocking of the decoded image at level-0 block edges
    float deblock_falloff = 1.0f;              // edge-proximity falloff in pixels (1.0: only the ring adjacent to an edge)
    std::vector<int> hidden = { 24, 24 };  // MLP hidden layer widths
    std::string act = "leaky";             // leaky | relu | tanh | sine
    float leak = 0.01f;                    // negative-side slope of the leaky ReLU
    std::string pos = "uv";    // positional encoding spec, see PosEnc
    bool clamp_out = false;    // hard clamp instead of sigmoid
    int iters = 3000;
    int save_every = 100;
    int print_every = 10;
    unsigned seed = 1;
    // MLP ES
    int mlp_pairs = 32;
    int mlp_batch = 4096;
    float mlp_sigma = 0.02f;
    float mlp_lr = 0.005f;
    int mlp_every = 1;
    // Late-phase MLP options; the three START values are fractions of --iters (>= 1 = off).
    // Phases are gated by --mlp-every like the ES step. Precedence: freeze > fd > full.
    float mlp_fd_start = 1.0f;     // from here on, replace the MLP ES step with central finite differences
    float mlp_fd_h = 1e-3f;        // finite-difference step per weight
    bool mlp_fd_h_set = false;
    float mlp_freeze_start = 1.0f; // from here on, skip MLP updates entirely (latent-only phase; wins over fd/full)
    float mlp_full_start = 1.0f;   // from here on, the MLP ES step uses every pixel instead of the minibatch
    int mlp_full_pairs = -1;       // pairs per step in the full-image phase; < 0 = same as mlp_pairs
    bool lat_alt = false;          // two levels: perturb one level per pair, alternating
    // latent ES
    int lat_pairs = 4;
    float lat_sigma = 0.05f;
    float lat_lr = 0.02f;
    float lat_init = 0.1f;
    float lat2_sigma = -1.0f;  // ES sigma for the second level; <= 0 on the command line = same as lat_sigma (resolved after parsing)
    // Annealing: multiplier decays linearly from 1 at (start * iters) to final at the end.
    float lr_anneal_start = 1.0f, lr_anneal_final = 1.0f;       // applies to mlp_lr and lat_lr
    float sigma_anneal_start = 1.0f, sigma_anneal_final = 1.0f; // applies to mlp_sigma, lat_sigma and lat2_sigma
    int threads = 0;
    int qbits = 8;             // latent bit depth used for the reported quantized stats
    int qat = 0;               // --qat: largest per-channel bit depth of level 0 (0 = off); level 0 is held on fixed grids and updated by exhaustive search
    std::vector<int> qat_ch;   // --qat B or B1,B2,...: bits per level-0 channel (a single value applies to every channel)
    int qat_every = 1;         // run that search every N iterations
    std::string load;          // load model.bin instead of random init
    bool rng_hash = false;     // --rng hash: draw the ES perturbations and minibatches from the counter-based hash the GPU uses
    bool cuda = false;         // --cuda: train on the GPU (needs a build with NTC_CUDA)
    bool cuda_check = false;   // --cuda-check: compare the GPU kernels against the CPU code, then exit
};

static void usage() {
    printf(
        "ntc [options] [tex0.png [tex1.png ...]]   (default: kodim23.png; up to 4 same-size RGB textures of one material share the latent and MLP)\n"
        "  --weights w0,w1,...  per-texture loss weight, one entry per image (default 1 each). Relative:\n"
        "                       loss = sum_t w_t mse_t / sum_t w_t, so 2,2 == 1,1; a weight of 0 drops that\n"
        "                       texture from training (its PSNR is then meaningless). Not stored in model.bin:\n"
        "                       pass the same --weights with --load to get the same reported psnr.\n"
        "  --out DIR            output directory (default out)\n"
        "  --crop N             center-crop to NxN, 0 = none (512)\n"
        "  --latent W H C       latent texture size (64 64 4)\n"
        "  --latent2 W H C      optional second (typically coarser) latent level, e.g. 32 32 4 (off)\n"
        "  --deblock            in-loop deblocking: 5-tap cross filter on the decoded image at the level-0\n"
        "                       block edges (blocks = level-0 cells; meant for --filter nearest), applied\n"
        "                       inside every loss evaluation; boundary minibatch pixels cost 3-5 decodes\n"
        "  --deblock-falloff F  edge-proximity falloff in pixels (<= 1.5: only the boundary ring)\n"
        "  --filter M[,M]       latent sampling per level: bilinear | nearest (bilinear). With nearest,\n"
        "                       every pixel of a cell reads the same texel, so add a cell-position\n"
                       "feature (local, ldct:N, lv1local, ...) or the cell decodes to one color\n"
        "  --mlp W1,W2,...      MLP hidden layer widths (24,24); e.g. --mlp 32 or --mlp 16,16,16\n"
        "  --hidden N           shorthand for --mlp N,N\n"
        "  --act NAME           hidden activation: leaky | relu | tanh | sine (leaky)\n"
        "  --leak F             negative-side slope of the leaky ReLU (0.01)\n"
        "  --pos SPEC           positional features, comma list (uv). Kinds:\n"
        "                         uv | fourier:N | dct:N | local | lfourier:N | lquad | ldct:N | ldct2:N | ldct4:N |\n"
        "                         bdct:N (first N zig-zag ACs of the block DCT) | bdcte:I (AC by zig-zag index) | onehot |\n"
        "                         bc7part:N (first N BC7 two-subset partition patterns as +-1, 4x4 cells) |\n"
        "                         lv1local | lv1ldct:N | none\n"
        "                         e.g. --pos uv,local  or  --pos local,lfourier:2\n"
        "  --nfreq N            shorthand: --pos uv,fourier:N (N=0 -> uv only)\n"
        "  --clamp              hard-clamp output instead of sigmoid\n"
        "  --iters N            training iterations (3000)\n"
        "  --save-every N       write PNGs every N iters (100)\n"
        "  --print-every N      print stats every N iters (10)\n"
        "  --seed N\n"
        "  --mlp-pairs N --mlp-batch N --mlp-sigma F --mlp-lr F --mlp-every N\n"
        "  --mlp-fd START       from iteration START*iters on, train the MLP with central\n"
        "                       finite differences (2 evals per weight) instead of ES (off; beats --mlp-full)\n"
        "  --mlp-fd-h F         finite-difference step size (1e-3)\n"
        "  --mlp-full START     from iteration START*iters on, evaluate the MLP ES step on the full\n"
        "                       image instead of the minibatch (off)\n"
        "  --mlp-full-pairs N   ES pairs per step during the full-image phase (default: --mlp-pairs)\n"
        "  --mlp-freeze START   from iteration START*iters on, stop updating the MLP; latent-only\n"
        "                       phase (off; takes precedence over --mlp-fd and --mlp-full)\n"
        "  --lat-alt            with two latent levels, perturb only one level per antithetic pair,\n"
        "                       rotating across steps, so the levels add no crosstalk to each other (off)\n"
        "  --lat-pairs N --lat-sigma F --lat-lr F --lat-init F\n"
        "  --lat2-sigma F       ES sigma for the second latent level (<= 0 or absent: same as --lat-sigma)\n"
        "  --lr-anneal START FINAL     decay both learning rates linearly from 1x at\n"
        "                              iteration START*iters to FINAL x at the end (off)\n"
        "  --sigma-anneal START FINAL  same for all ES sigmas (off)\n"
        "  --threads N\n"
        "  --qbits N            latent bit depth for reported quantized bitrate/psnr (8)\n"
        "  --qat B[,B,...]      hold level 0 on a 2^B-value grid in [-1,1] (B = 1..8, one value or one per\n"
        "                       channel) and update it by an exact exhaustive per-texel search instead of\n"
        "                       ES (off). Needs --filter nearest on level 0 and no --deblock; level 1 (if\n"
        "                       any) and the MLP train as before\n"
        "  --qat-every N        run the level-0 search every N iterations (1); each run costs sum over channels of 2^B image decodes\n"
        "  --load model.bin     start from a saved model (use --iters 0 to just evaluate)\n"
        "  --rng MODE           ES noise source: mt (std::mt19937 + normal_distribution, platform-dependent) or\n"
        "                       hash (the GPU backend's counter-based generator; a CPU run then uses the same\n"
        "                       perturbations as a --cuda run with the same seed) (mt)\n"
        "  --cuda               train on the GPU (no deblocking; same outputs and model file)\n"
        "  --cuda-check         with --cuda: compare the GPU kernels against the CPU code and exit\n");
}

// ---------------------------------------------------------------- image
struct Image {
    int w = 0, h = 0, nc = 3;   // nc = 3 * textures; per pixel [r0 g0 b0 r1 g1 b1 ...]
    std::vector<float> rgb;     // w*h*nc, [0,1] (all textures interleaved, despite the name)
    float& at(int x, int y, int c) { return rgb[((size_t)y * w + x) * nc + c]; }
    float at(int x, int y, int c) const { return rgb[((size_t)y * w + x) * nc + c]; }
};

static bool load_png(const std::string& path, Image& img, int crop) {
    int w, h, n;
    unsigned char* data = stbi_load(path.c_str(), &w, &h, &n, 3);
    if (!data) return false;
    img.nc = 3;
    int cw = w, ch = h, ox = 0, oy = 0;
    if (crop > 0 && (w > crop || h > crop)) {
        cw = std::min(w, crop); ch = std::min(h, crop);
        ox = (w - cw) / 2; oy = (h - ch) / 2;
    }
    img.w = cw; img.h = ch; img.rgb.resize((size_t)cw * ch * 3);
    for (int y = 0; y < ch; y++)
        for (int x = 0; x < cw; x++)
            for (int c = 0; c < 3; c++)
                img.at(x, y, c) = data[((size_t)(y + oy) * w + (x + ox)) * 3 + c] / 255.0f;
    stbi_image_free(data);
    return true;
}

// Write texture t of img (any nc) as a 3-channel PNG. For nc == 3, t == 0 this
// is a plain RGB write.
static void save_png(const std::string& path, const Image& img, int t = 0) {
    std::vector<unsigned char> buf((size_t)img.w * img.h * 3);
    for (size_t p = 0; p < (size_t)img.w * img.h; p++)
        for (int c = 0; c < 3; c++)
            buf[p * 3 + c] = (unsigned char)std::lround(std::min(1.0f, std::max(0.0f, img.rgb[p * img.nc + 3 * t + c])) * 255.0f);
    stbi_write_png(path.c_str(), img.w, img.h, 3, buf.data(), img.w * 3);
}

// Interleave T same-size RGB images into one nc = 3T image.
static void pack_textures(const std::vector<Image>& tex, Image& out) {
    const int T = (int)tex.size();
    out.w = tex[0].w; out.h = tex[0].h; out.nc = 3 * T;
    out.rgb.resize((size_t)out.w * out.h * out.nc);
    for (size_t p = 0; p < (size_t)out.w * out.h; p++)
        for (int t = 0; t < T; t++)
            for (int c = 0; c < 3; c++) out.rgb[p * out.nc + 3 * t + c] = tex[t].rgb[p * 3 + c];
}

// Texture t of a packed image as a plain 3-channel image.
static void slice_texture(const Image& img, int t, Image& out) {
    out.w = img.w; out.h = img.h; out.nc = 3;
    out.rgb.resize((size_t)img.w * img.h * 3);
    for (size_t p = 0; p < (size_t)img.w * img.h; p++)
        for (int c = 0; c < 3; c++) out.rgb[p * 3 + c] = img.rgb[p * img.nc + 3 * t + c];
}

// Try the path as given, then relative to the executable's directory and its
// parents, so running from build/Release still finds the checked-in images.
// On success `name` is rewritten to the path that loaded.
static bool find_and_load(const char* argv0, std::string& name, Image& img, int crop) {
    std::vector<std::string> candidates = { name };
    std::string exe = argv0;
    size_t slash = exe.find_last_of("/\\");
    std::string dir = (slash == std::string::npos) ? "." : exe.substr(0, slash);
    for (int up = 0; up < 4; up++) { candidates.push_back(dir + "/" + name); dir += "/.."; }
    for (const std::string& c : candidates)
        if (load_png(c, img, crop)) { name = c; return true; }
    return false;
}

// Synthetic fallback so the program runs without an input image.
static void make_synthetic(Image& img, int n) {
    img.w = img.h = n; img.nc = 3; img.rgb.resize((size_t)n * n * 3);
    for (int y = 0; y < n; y++)
        for (int x = 0; x < n; x++) {
            float u = x / (float)n, v = y / (float)n;
            float r = 0.5f + 0.5f * std::sin(u * 12.0f + 3.0f * std::sin(v * 7.0f));
            float g = v;
            float b = ((x / 32 + y / 32) & 1) ? 0.8f : 0.2f;
            float dx = u - 0.5f, dy = v - 0.5f;
            if (dx * dx + dy * dy < 0.05f) { r = 1.0f; g = 0.9f; b = 0.1f; }
            img.at(x, y, 0) = r; img.at(x, y, 1) = g; img.at(x, y, 2) = b;
        }
}

// ---------------------------------------------------------------- latent
struct Latent {
    int W = 0, H = 0, C = 0;
    size_t off = 0;        // index of this level's first value in LatentSet::z
    bool nearest = false;  // nearest-texel sampling instead of bilinear (a cell = a block)
    size_t size() const { return (size_t)W * H * C; }
};

// All latent levels in one flat parameter vector, so one Adam state and one ES
// epsilon cover everything. Level 0 is --latent (off = 0, guaranteed by add());
// level 1, if present, is --latent2 (typically coarser, but any size works).
// Level l occupies z[off, off + size()) in (y, x, c) order with c fastest.
struct LatentSet {
    std::vector<Latent> lv;
    std::vector<float> z;
    void add(int W, int H, int C, bool nearest = false) {
        Latent L; L.W = W; L.H = H; L.C = C; L.off = z.size(); L.nearest = nearest;
        lv.push_back(L);
        z.resize(z.size() + L.size());
    }
    size_t size() const { return z.size(); }
    int channels() const { int c = 0; for (const Latent& L : lv) c += L.C; return c; }
    const float* level(int l) const { return z.data() + lv[l].off; }
};

// Map a pixel (px,py) of a W x H image to latent-space texel coordinates.
// Bilinear: UV = pixel center in [0,1]; texel centers sit at (i+0.5)/LW and the
// tap blends the 4 surrounding texels. Nearest: the cell is the texel's own
// footprint [i/LW, (i+1)/LW), x0 == x1, and fx,fy is the offset within that
// cell in [0,1), so a cell behaves like a block of r x r pixels.
struct BilinearTap {
    int x0, x1, y0, y1;   // clamped texel indices
    int ix, iy;           // unclamped floor(x), floor(y): needed for multi-texel periodic features
    float fx, fy;
};

static inline BilinearTap bilinear_tap(const Latent& L, float u, float v) {
    if (L.nearest) {
        float x = u * L.W, y = v * L.H;
        int ix = (int)std::floor(x), iy = (int)std::floor(y);
        BilinearTap t;
        t.ix = ix; t.iy = iy;
        t.fx = x - ix; t.fy = y - iy;
        t.x0 = t.x1 = std::max(0, std::min(L.W - 1, ix));
        t.y0 = t.y1 = std::max(0, std::min(L.H - 1, iy));
        return t;
    }
    float x = u * L.W - 0.5f, y = v * L.H - 0.5f;
    int x0 = (int)std::floor(x), y0 = (int)std::floor(y);
    BilinearTap t;
    t.ix = x0; t.iy = y0;
    t.fx = x - x0; t.fy = y - y0;
    t.x0 = std::max(0, std::min(L.W - 1, x0));
    t.x1 = std::max(0, std::min(L.W - 1, x0 + 1));
    t.y0 = std::max(0, std::min(L.H - 1, y0));
    t.y1 = std::max(0, std::min(L.H - 1, y0 + 1));
    return t;
}

static inline void sample_latent(const Latent& L, const float* z, const BilinearTap& t, float* out) {
    if (L.nearest) {
        const float* a = &z[((size_t)t.y0 * L.W + t.x0) * L.C];
        for (int k = 0; k < L.C; k++) out[k] = a[k];
        return;
    }
    const float* a = &z[((size_t)t.y0 * L.W + t.x0) * L.C];
    const float* b = &z[((size_t)t.y0 * L.W + t.x1) * L.C];
    const float* c = &z[((size_t)t.y1 * L.W + t.x0) * L.C];
    const float* d = &z[((size_t)t.y1 * L.W + t.x1) * L.C];
    float w00 = (1 - t.fx) * (1 - t.fy), w10 = t.fx * (1 - t.fy);
    float w01 = (1 - t.fx) * t.fy, w11 = t.fx * t.fy;
    for (int k = 0; k < L.C; k++)
        out[k] = w00 * a[k] + w10 * b[k] + w01 * c[k] + w11 * d[k];
}

// ---------------------------------------------------------------- MLP
// Fully connected net with an arbitrary list of hidden widths, e.g. {24,24}.
// Flat parameter layout, layer by layer: W[out*in] (row-major, one row per
// output unit) followed by b[out].
static const int MAXT = 4;        // max textures per material (all share the latent and the MLP)
static const int MAXOUT = 3 * MAXT; // max MLP outputs: 3 RGB channels per texture
static const int MAXH = 128;      // max units in any layer (incl. the input)
static const int MAXL = 8;        // max hidden layers

enum Act { ACT_LEAKY, ACT_RELU, ACT_TANH, ACT_SINE };

static inline float activate(Act a, float x, float leak) {
    switch (a) {
    case ACT_RELU:  return x > 0 ? x : 0.0f;
    case ACT_TANH:  return std::tanh(x);
    case ACT_SINE:  return std::sin(x);
    default:        return x > 0 ? x : leak * x;   // leaky ReLU
    }
}

static bool parse_act(const std::string& s, Act& a) {
    if (s == "leaky") a = ACT_LEAKY;
    else if (s == "relu") a = ACT_RELU;
    else if (s == "tanh") a = ACT_TANH;
    else if (s == "sine") a = ACT_SINE;
    else return false;
    return true;
}
static const char* act_name(Act a) {
    static const char* n[] = { "leaky", "relu", "tanh", "sine" };
    return n[a];
}

struct MLP {
    int nin = 0, nout = 3;
    std::vector<int> hidden;       // hidden layer widths
    Act act = ACT_LEAKY;
    float leak = 0.01f;            // leaky ReLU negative-side slope
    std::vector<float> p;
    size_t size() const { return p.size(); }

    // Widths of every layer, input first, output last.
    std::vector<int> widths() const {
        std::vector<int> w; w.push_back(nin);
        for (int h : hidden) w.push_back(h);
        w.push_back(nout);
        return w;
    }
    static size_t count(const std::vector<int>& w) {
        size_t n = 0;
        for (size_t l = 1; l < w.size(); l++) n += (size_t)w[l] * w[l - 1] + w[l];
        return n;
    }
    std::string describe() const {
        std::string s = std::to_string(nin);
        for (int h : hidden) s += " -> " + std::to_string(h);
        return s + " -> " + std::to_string(nout);
    }
    void init(int nin_, const std::vector<int>& hidden_, Act act_, std::mt19937& rng) {
        nin = nin_; hidden = hidden_; act = act_;
        std::vector<int> w = widths();
        p.assign(count(w), 0.0f);
        std::normal_distribution<float> N(0.0f, 1.0f);
        size_t o = 0;
        for (size_t l = 1; l < w.size(); l++) {
            bool last = (l + 1 == w.size());
            // He init for hidden layers, smaller for the output so it starts near mid-gray.
            float s = last ? std::sqrt(1.0f / w[l - 1]) : std::sqrt(2.0f / w[l - 1]);
            if (act == ACT_SINE && !last) s = (l == 1) ? 1.0f / w[l - 1] * 30.0f : std::sqrt(6.0f / w[l - 1]);
            for (int i = 0; i < w[l] * w[l - 1]; i++) p[o++] = N(rng) * s;
            o += w[l]; // biases stay zero
        }
    }
};

// Forward pass with an explicit parameter pointer so perturbed copies can be used.
static inline void mlp_forward(const MLP& m, const float* p, const float* in, float* out, bool clamp_out) {
    float bufA[MAXH], bufB[MAXH];
    const float* cur = in;
    int ncur = m.nin;
    float* nxt = bufA;
    for (size_t l = 0; l < m.hidden.size(); l++) {
        int nh = m.hidden[l];
        const float* W = p; const float* b = W + (size_t)nh * ncur;
        for (int j = 0; j < nh; j++) {
            float s = b[j]; const float* w = W + (size_t)j * ncur;
            for (int i = 0; i < ncur; i++) s += w[i] * cur[i];
            nxt[j] = activate(m.act, s, m.leak);
        }
        p = b + nh;
        cur = nxt; ncur = nh;
        nxt = (nxt == bufA) ? bufB : bufA;
    }
    const float* W = p; const float* b = W + (size_t)m.nout * ncur;
    for (int j = 0; j < m.nout; j++) {
        float s = b[j]; const float* w = W + (size_t)j * ncur;
        for (int i = 0; i < ncur; i++) s += w[i] * cur[i];
        if (clamp_out) out[j] = std::min(1.0f, std::max(0.0f, s + 0.5f));
        else out[j] = 1.0f / (1.0f + std::exp(-s));
    }
}

// ---------------------------------------------------------------- positional encoding
// phi(u,v): a list of named feature generators, composed from --pos "a,b:N,...".
// To add an experiment: add an enum value, a name in POS_NAMES, its output
// count in PosFeature::count(), and its evaluation in PosEnc::encode().
//
//   uv           global u,v mapped to [-1,1]                               (2)
//   fourier:N    sin/cos(2^k * 2*pi*u), same for v, k = 0..N-1              (4N)
//   local        fractional offset inside the bilinear cell, fx,fy -> [-1,1] (2)
//   lfourier:N   sin/cos(2^k * 2*pi*fx), same for fy: periodic per texel    (4N)
//   lquad        fx*fy, fx^2, fy^2 of the cell offset: cheap 2nd-order kernel (3)
//   dct:N        cos(pi*k*u), cos(pi*k*v), k = 1..N: the DCT (k,0),(0,k) bases   (2N)
//   ldct:N       same on the cell offset fx,fy                                    (2N)
//   ldct2:N      same on the offset within a 2x2-texel block (period 2 texels)   (2N)
//   ldct4:N      same with a 4-texel period                                      (2N)
//   bdct:N       the first N AC basis functions of the level-0 block's 2D DCT-II, in
//                JPEG zig-zag order over (u,v) = (horizontal, vertical frequency), each
//                cos(pi*u*fx)*cos(pi*v*fy) on the pixel-center cell offset; with nearest
//                4x4 cells these are exactly the 4x4 DCT basis samples          (N, N <= 15)
//   bdcte:I      a single AC basis function by zig-zag index I (1..15)          (1)
//   bc7part:N    the first N of the 64 BC7 two-subset partition patterns, one input per
//                pattern: +1 if the pixel's position in the 4x4 cell is in subset 1,
//                -1 if in subset 0 (requires 4x4 level-0 cells)               (N, N <= 64)
//   onehot       one indicator per pixel position within the level-0 cell (1 for the
//                pixel's own position, 0 elsewhere); with nearest 4x4 cells, 16 inputs.
//                The upper bound on any position basis: every other one is a linear
//                function of it                                            (cellw*cellh)
//   lv1local     cell offset of the SECOND latent level, fx,fy -> [-1,1]         (2)
//   lv1ldct:N    cos(pi*k*fx), cos(pi*k*fy) of the second level's cell offset    (2N)
enum PosKind { POS_UV, POS_FOURIER, POS_LOCAL, POS_LFOURIER, POS_LQUAD, POS_DCT, POS_LDCT, POS_LDCT2, POS_LDCT4, POS_LV1LOCAL, POS_LV1LDCT, POS_BDCT, POS_BDCTE, POS_ONEHOT, POS_BC7PART, POS_COUNT };
static const char* POS_NAMES[POS_COUNT] = { "uv", "fourier", "local", "lfourier", "lquad", "dct", "ldct", "ldct2", "ldct4", "lv1local", "lv1ldct", "bdct", "bdcte", "onehot", "bc7part" };

// JPEG zig-zag order of the 15 AC elements of a 4x4 DCT, as (u, v) = (horizontal,
// vertical frequency). Index 1 is (1,0), index 2 is (0,1), ...
static const int BDCT_ZIGZAG[16][2] = {
    {0,0}, {1,0}, {0,1}, {0,2}, {1,1}, {2,0}, {3,0}, {2,1}, {1,2}, {0,3}, {1,3}, {2,2}, {3,1}, {3,2}, {2,3}, {3,3} };
static const int BDCT_MAX = 15;

// BC7 two-subset partition table (64 patterns x 16 texels, row-major within the
// 4x4 block; value = subset index), from the BC7 specification.
static const unsigned char BC7_PARTITION2[64 * 16] = {
    0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1, 0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1, 0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1, 0,0,0,1,0,0,1,1,0,0,1,1,0,1,1,1, 0,0,0,0,0,0,0,1,0,0,0,1,0,0,1,1, 0,0,1,1,0,1,1,1,0,1,1,1,1,1,1,1, 0,0,0,1,0,0,1,1,0,1,1,1,1,1,1,1, 0,0,0,0,0,0,0,1,0,0,1,1,0,1,1,1,
    0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1, 0,0,1,1,0,1,1,1,1,1,1,1,1,1,1,1, 0,0,0,0,0,0,0,1,0,1,1,1,1,1,1,1, 0,0,0,0,0,0,0,0,0,0,0,1,0,1,1,1, 0,0,0,1,0,1,1,1,1,1,1,1,1,1,1,1, 0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1, 0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1, 0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,
    0,0,0,0,1,0,0,0,1,1,1,0,1,1,1,1, 0,1,1,1,0,0,0,1,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,1,0,0,0,1,1,1,0, 0,1,1,1,0,0,1,1,0,0,0,1,0,0,0,0, 0,0,1,1,0,0,0,1,0,0,0,0,0,0,0,0, 0,0,0,0,1,0,0,0,1,1,0,0,1,1,1,0, 0,0,0,0,0,0,0,0,1,0,0,0,1,1,0,0, 0,1,1,1,0,0,1,1,0,0,1,1,0,0,0,1,
    0,0,1,1,0,0,0,1,0,0,0,1,0,0,0,0, 0,0,0,0,1,0,0,0,1,0,0,0,1,1,0,0, 0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0, 0,0,1,1,0,1,1,0,0,1,1,0,1,1,0,0, 0,0,0,1,0,1,1,1,1,1,1,0,1,0,0,0, 0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0, 0,1,1,1,0,0,0,1,1,0,0,0,1,1,1,0, 0,0,1,1,1,0,0,1,1,0,0,1,1,1,0,0,
    0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1, 0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1, 0,1,0,1,1,0,1,0,0,1,0,1,1,0,1,0, 0,0,1,1,0,0,1,1,1,1,0,0,1,1,0,0, 0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0, 0,1,0,1,0,1,0,1,1,0,1,0,1,0,1,0, 0,1,1,0,1,0,0,1,0,1,1,0,1,0,0,1, 0,1,0,1,1,0,1,0,1,0,1,0,0,1,0,1,
    0,1,1,1,0,0,1,1,1,1,0,0,1,1,1,0, 0,0,0,1,0,0,1,1,1,1,0,0,1,0,0,0, 0,0,1,1,0,0,1,0,0,1,0,0,1,1,0,0, 0,0,1,1,1,0,1,1,1,1,0,1,1,1,0,0, 0,1,1,0,1,0,0,1,1,0,0,1,0,1,1,0, 0,0,1,1,1,1,0,0,1,1,0,0,0,0,1,1, 0,1,1,0,0,1,1,0,1,0,0,1,1,0,0,1, 0,0,0,0,0,1,1,0,0,1,1,0,0,0,0,0,
    0,1,0,0,1,1,1,0,0,1,0,0,0,0,0,0, 0,0,1,0,0,1,1,1,0,0,1,0,0,0,0,0, 0,0,0,0,0,0,1,0,0,1,1,1,0,0,1,0, 0,0,0,0,0,1,0,0,1,1,1,0,0,1,0,0, 0,1,1,0,1,1,0,0,1,0,0,1,0,0,1,1, 0,0,1,1,0,1,1,0,1,1,0,0,1,0,0,1, 0,1,1,0,0,0,1,1,1,0,0,1,1,1,0,0, 0,0,1,1,1,0,0,1,1,1,0,0,0,1,1,0,
    0,1,1,0,1,1,0,0,1,1,0,0,1,0,0,1, 0,1,1,0,0,0,1,1,0,0,1,1,1,0,0,1, 0,1,1,1,1,1,1,0,1,0,0,0,0,0,0,1, 0,0,0,1,1,0,0,0,1,1,1,0,0,1,1,1, 0,0,0,0,1,1,1,1,0,0,1,1,0,0,1,1, 0,0,1,1,0,0,1,1,1,1,1,1,0,0,0,0, 0,0,1,0,0,0,1,0,1,1,1,0,1,1,1,0, 0,1,0,0,0,1,0,0,0,1,1,1,0,1,1,1 };

struct PosFeature {
    PosKind kind;
    int n = 0;   // octave count for the fourier kinds
    int count() const {
        switch (kind) {
        case POS_UV:       return 2;
        case POS_FOURIER:  return 4 * n;
        case POS_LOCAL:    return 2;
        case POS_LFOURIER: return 4 * n;
        case POS_LQUAD:    return 3;
        case POS_DCT:      return 2 * n;
        case POS_LDCT:     return 2 * n;
        case POS_LDCT2:    return 2 * n;
        case POS_LDCT4:    return 2 * n;
        case POS_LV1LOCAL: return 2;
        case POS_LV1LDCT:  return 2 * n;
        case POS_BDCT:     return n;
        case POS_BDCTE:    return 1;
        case POS_ONEHOT:   return -1;   // depends on the cell size; PosEnc::count() resolves it
        case POS_BC7PART:  return n;
        default:           return 0;
        }
    }
};

struct PosEnc {
    std::vector<PosFeature> feats;
    std::string spec;   // canonical text form, stored in the model file
    int cellw = 1, cellh = 1;   // level-0 cell size in pixels, set by main before nin() is used (onehot needs it)

    int count() const { int c = 0; for (auto& f : feats) c += (f.kind == POS_ONEHOT) ? cellw * cellh : f.count(); return c; }
    bool uses_level1() const { for (auto& f : feats) if (f.kind == POS_LV1LOCAL || f.kind == POS_LV1LDCT) return true; return false; }
    bool uses_bc7part() const { for (auto& f : feats) if (f.kind == POS_BC7PART) return true; return false; }

    // Parse "uv,fourier:2,local". "none" or "" gives no positional input.
    bool parse(const std::string& s) {
        feats.clear(); spec.clear();
        if (s == "none" || s.empty()) { spec = "none"; return true; }
        for (size_t a = 0; a < s.size();) {
            size_t e = s.find(',', a);
            if (e == std::string::npos) e = s.size();
            std::string item = s.substr(a, e - a);
            a = e + 1;
            if (item.empty()) continue;
            PosFeature f;
            size_t colon = item.find(':');
            std::string name = item.substr(0, colon);
            f.n = (colon == std::string::npos) ? 1 : atoi(item.substr(colon + 1).c_str());
            int k = 0;
            while (k < POS_COUNT && name != POS_NAMES[k]) k++;
            if (k == POS_COUNT || f.n < 0) return false;
            f.kind = (PosKind)k;
            if (f.kind == POS_BDCT && f.n > BDCT_MAX) return false;
            if (f.kind == POS_BDCTE && (f.n < 1 || f.n > BDCT_MAX)) return false;
            if (f.kind == POS_BC7PART && (f.n < 1 || f.n > 64)) return false;
            if (f.count() == 0) continue;   // (onehot reports -1 here and is always kept)
            feats.push_back(f);
            if (!spec.empty()) spec += ",";
            spec += name;
            if (f.kind == POS_FOURIER || f.kind == POS_LFOURIER || f.kind == POS_DCT || f.kind == POS_LDCT || f.kind == POS_LDCT2 || f.kind == POS_LDCT4 || f.kind == POS_LV1LDCT || f.kind == POS_BDCT || f.kind == POS_BDCTE || f.kind == POS_BC7PART) spec += ":" + std::to_string(f.n);
        }
        if (spec.empty()) spec = "none";
        return true;
    }

    // u,v: global pixel-center UV in [0,1]. t: level 0's tap for that UV; t1: level 1's
    // tap (null without a second level; only the lv1* kinds read it).
    inline int encode(float u, float v, const BilinearTap& t, const BilinearTap* t1, float* f) const {
        int k = 0;
        for (const PosFeature& pf : feats) {
            switch (pf.kind) {
            case POS_UV:
                f[k++] = u * 2.0f - 1.0f;
                f[k++] = v * 2.0f - 1.0f;
                break;
            case POS_FOURIER: {
                float fr = 2.0f * PI;
                for (int o = 0; o < pf.n; o++, fr *= 2.0f) {
                    f[k++] = std::sin(fr * u); f[k++] = std::cos(fr * u);
                    f[k++] = std::sin(fr * v); f[k++] = std::cos(fr * v);
                }
                break;
            }
            case POS_LOCAL:
                f[k++] = t.fx * 2.0f - 1.0f;
                f[k++] = t.fy * 2.0f - 1.0f;
                break;
            case POS_LFOURIER: {
                float fr = 2.0f * PI;
                for (int o = 0; o < pf.n; o++, fr *= 2.0f) {
                    f[k++] = std::sin(fr * t.fx); f[k++] = std::cos(fr * t.fx);
                    f[k++] = std::sin(fr * t.fy); f[k++] = std::cos(fr * t.fy);
                }
                break;
            }
            case POS_LQUAD: {
                float x = t.fx * 2.0f - 1.0f, y = t.fy * 2.0f - 1.0f;
                f[k++] = x * y; f[k++] = x * x; f[k++] = y * y;
                break;
            }
            case POS_DCT:
                for (int o = 1; o <= pf.n; o++) { f[k++] = std::cos(PI * o * u); f[k++] = std::cos(PI * o * v); }
                break;
            case POS_LDCT:
                for (int o = 1; o <= pf.n; o++) { f[k++] = std::cos(PI * o * t.fx); f[k++] = std::cos(PI * o * t.fy); }
                break;
            case POS_LDCT2:
            case POS_LDCT4: {
                // Offset within a PxP-texel block, in [0,1): the cosines then repeat every P texels.
                int P = (pf.kind == POS_LDCT2) ? 2 : 4;
                float gx = ((((t.ix % P) + P) % P) + t.fx) / P, gy = ((((t.iy % P) + P) % P) + t.fy) / P;
                for (int o = 1; o <= pf.n; o++) { f[k++] = std::cos(PI * o * gx); f[k++] = std::cos(PI * o * gy); }
                break;
            }
            case POS_LV1LOCAL:
                f[k++] = t1->fx * 2.0f - 1.0f;
                f[k++] = t1->fy * 2.0f - 1.0f;
                break;
            case POS_LV1LDCT:
                for (int o = 1; o <= pf.n; o++) { f[k++] = std::cos(PI * o * t1->fx); f[k++] = std::cos(PI * o * t1->fy); }
                break;
            case POS_BDCT:
                for (int o = 1; o <= pf.n; o++)
                    f[k++] = std::cos(PI * BDCT_ZIGZAG[o][0] * t.fx) * std::cos(PI * BDCT_ZIGZAG[o][1] * t.fy);
                break;
            case POS_BDCTE:
                f[k++] = std::cos(PI * BDCT_ZIGZAG[pf.n][0] * t.fx) * std::cos(PI * BDCT_ZIGZAG[pf.n][1] * t.fy);
                break;
            case POS_BC7PART: {
                int ix = std::min(3, std::max(0, (int)(t.fx * 4)));
                int iy = std::min(3, std::max(0, (int)(t.fy * 4)));
                int pi = iy * 4 + ix;
                for (int o = 0; o < pf.n; o++) f[k++] = BC7_PARTITION2[o * 16 + pi] ? 1.0f : -1.0f;
                break;
            }
            case POS_ONEHOT: {
                // Pixel index within the cell from the cell offset (pixel centers at (i+0.5)/cellw).
                int ix = std::min(cellw - 1, std::max(0, (int)(t.fx * cellw)));
                int iy = std::min(cellh - 1, std::max(0, (int)(t.fy * cellh)));
                for (int j = 0; j < cellw * cellh; j++) f[k++] = (j == iy * cellw + ix) ? 1.0f : 0.0f;
                break;
            }
            default: break;
            }
        }
        return k;
    }
};

// ---------------------------------------------------------------- decoder
struct Decoder {
    const Options* opt;
    LatentSet lat;
    MLP mlp;
    PosEnc pos;
    int W = 0, H = 0; // output image size
    // Material loss: per-output-channel weight cw[c] = weights[c / 3]; the loss
    // divisor is 3 * wsum * pixels. Dividing by the weight sum makes --weights
    // purely relative (2,2 == 1,1) and leaves one texture of weight 1 unchanged.
    std::vector<float> cw;
    float wsum = 1.0f;

    // In-loop deblocking (Basis Universal's shader/in-loop filter): a 5-tap cross,
    // two 3-tap box filters per axis, each blended toward the original by a
    // geometric edge weight that is 1 on the ring of pixels adjacent to a
    // level-0 block edge and 0 elsewhere (falloff 1.0, K = 2). Content-blind:
    // which pixels change is fixed by geometry, so the dependency sets used by
    // footprint attribution are known in advance. Interior pixels pass through
    // bit-exact.
    bool deblock = false;
    float deblock_falloff = 1.0f;
    int bw = 1, bh = 1;   // block size in pixels = level-0 cell size
    int qat_bits = 0;     // --qat: largest level-0 channel bit depth (0 = off), stored in the model file
    std::vector<int> qat_ch;   // bits per level-0 channel (size C0 when on)
    mutable std::vector<float> scratch;   // unfiltered full decode for decode_err (serial caller only)

    // Edge weights for pixel (px,py): horizontal (left/right edge proximity) and vertical.
    inline void deblock_weights(int px, int py, float& wh, float& wv) const {
        float bx = (float)(px % bw) + 0.5f, by = (float)(py % bh) + 0.5f;   // position within the block
        float fo = deblock_falloff;
        float lp = 1.0f - std::min(1.0f, std::max(0.0f, bx / fo));
        float rp = 1.0f - std::min(1.0f, std::max(0.0f, (bw - bx) / fo));
        float tp = 1.0f - std::min(1.0f, std::max(0.0f, by / fo));
        float bp = 1.0f - std::min(1.0f, std::max(0.0f, (bh - by) / fo));
        const float K = 2.0f;
        wh = std::min(1.0f, K * std::max(lp, rp));
        wv = std::min(1.0f, K * std::max(tp, bp));
    }

    // Apply the filter to one pixel given its cross of samples (nc channels each).
    inline void deblock_pixel(const float* c0, const float* l1, const float* r1, const float* u1, const float* d1,
                              float wh, float wv, float* out) const {
        const int nc = mlp.nout;
        float tw = wh + wv;
        if (tw <= 0.0f) { for (int c = 0; c < nc; c++) out[c] = c0[c]; return; }
        for (int c = 0; c < nc; c++) {
            float fh = (l1[c] + c0[c] + r1[c]) * (1.0f / 3.0f);
            float fv = (u1[c] + c0[c] + d1[c]) * (1.0f / 3.0f);
            float hc = c0[c] + (fh - c0[c]) * wh;
            float vc = c0[c] + (fv - c0[c]) * wv;
            out[c] = (hc * wh + vc * wv) / tw;
        }
    }

    // Deblock a full decoded image in place (neighbors clamped at the image border).
    void deblock_image(Image& img) const {
        const int nc = img.nc;
        std::vector<float> src = img.rgb;
#pragma omp parallel for schedule(static)
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++) {
                float wh, wv; deblock_weights(x, y, wh, wv);
                if (wh <= 0.0f && wv <= 0.0f) continue;   // interior: untouched
                int xl = std::max(0, x - 1), xr = std::min(W - 1, x + 1), yu = std::max(0, y - 1), yd = std::min(H - 1, y + 1);
                deblock_pixel(&src[((size_t)y * W + x) * nc], &src[((size_t)y * W + xl) * nc], &src[((size_t)y * W + xr) * nc],
                              &src[((size_t)yu * W + x) * nc], &src[((size_t)yd * W + x) * nc], wh, wv, &img.rgb[((size_t)y * W + x) * nc]);
            }
    }

    int nin() const { return lat.channels() + pos.count(); }

    // Build the MLP input for pixel (px,py) from a given flat latent array:
    // level 0's sample, then each further level's sample at the same UV, then
    // the positional features (cell-relative kinds use level 0's tap, the lv1*
    // kinds level 1's).
    inline void features(const float* z, int px, int py, float* f) const {
        float u = (px + 0.5f) / W, v = (py + 0.5f) / H;
        const Latent& L0 = lat.lv[0];
        BilinearTap t = bilinear_tap(L0, u, v);
        sample_latent(L0, z, t, f);
        int k = L0.C;
        BilinearTap t1;
        for (size_t l = 1; l < lat.lv.size(); l++) {
            const Latent& L = lat.lv[l];
            BilinearTap tl = bilinear_tap(L, u, v);
            if (l == 1) t1 = tl;
            sample_latent(L, z + L.off, tl, f + k);
            k += L.C;
        }
        pos.encode(u, v, t, lat.lv.size() > 1 ? &t1 : nullptr, f + k);
    }

    // One pixel: all mlp.nout outputs (3 per texture) into out[].
    inline void pixel(const float* p, const float* z, int px, int py, float* out) const {
        float f[MAXH];
        features(z, px, py, f);
        mlp_forward(mlp, p, f, out, opt->clamp_out);
    }

    // Full-image decode into img (parallel over rows).
    void decode_full(const float* p, const float* z, Image& img) const {
        img.w = W; img.h = H; img.nc = mlp.nout; img.rgb.resize((size_t)W * H * img.nc);
#pragma omp parallel for schedule(static)
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++)
                pixel(p, z, x, y, &img.rgb[((size_t)y * W + x) * img.nc]);
        if (deblock) deblock_image(img);
    }

    // Per-pixel weighted squared error sum_c cw[c] (out_c - target_c)^2 over all
    // 3T channels (unnormalized; LatentTrainer divides by 3 * wsum * W * H).
    void decode_err(const float* p, const float* z, const Image& target, std::vector<float>& err) const {
        const int nc = mlp.nout;
        err.resize((size_t)W * H);
        if (deblock) {
            // The filter needs neighbors, so decode the whole (unfiltered) image into a
            // reusable scratch buffer, then measure each pixel through the filter on read.
            // Same values as decode_full + deblock_image, without the copy or the allocation.
            std::vector<float>& raw = scratch;
            raw.resize((size_t)W * H * nc);
#pragma omp parallel for schedule(static)
            for (int y = 0; y < H; y++)
                for (int x = 0; x < W; x++)
                    pixel(p, z, x, y, &raw[((size_t)y * W + x) * nc]);
#pragma omp parallel for schedule(static)
            for (int y = 0; y < H; y++)
                for (int x = 0; x < W; x++) {
                    const float* c0 = &raw[((size_t)y * W + x) * nc];
                    float out[MAXOUT];
                    const float* o = c0;
                    float wh, wv; deblock_weights(x, y, wh, wv);
                    if (wh > 0.0f || wv > 0.0f) {
                        int xl = std::max(0, x - 1), xr = std::min(W - 1, x + 1), yu = std::max(0, y - 1), yd = std::min(H - 1, y + 1);
                        deblock_pixel(c0, &raw[((size_t)y * W + xl) * nc], &raw[((size_t)y * W + xr) * nc],
                                      &raw[((size_t)yu * W + x) * nc], &raw[((size_t)yd * W + x) * nc], wh, wv, out);
                        o = out;
                    }
                    const float* t = &target.rgb[((size_t)y * W + x) * nc];
                    float e = 0;
                    for (int c = 0; c < nc; c++) { float d = o[c] - t[c]; e += cw[c] * (d * d); }
                    err[(size_t)y * W + x] = e;
                }
            return;
        }
#pragma omp parallel for schedule(static)
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++) {
                float out[MAXOUT];
                pixel(p, z, x, y, out);
                const float* t = &target.rgb[((size_t)y * W + x) * nc];
                float e = 0;
                for (int c = 0; c < nc; c++) { float d = out[c] - t[c]; e += cw[c] * (d * d); }
                err[(size_t)y * W + x] = e;
            }
    }

    // Weighted mean squared error on a pixel subset: sum_c cw[c] d^2 / (3 * wsum * |idx|).
    // Serial; called per-thread by the MLP trainers.
    double loss_subset(const float* p, const float* z, const Image& target, const std::vector<int>& idx) const {
        const int nc = mlp.nout;
        double s = 0;
        for (int i : idx) {
            int px = i % W, py = i / W;
            float out[MAXOUT];
            float wh = 0, wv = 0;
            if (deblock) deblock_weights(px, py, wh, wv);
            if (wh > 0.0f || wv > 0.0f) {
                // Boundary pixel: decode its cross and filter (interior pixels skip this).
                // deblock_pixel multiplies an axis's taps by its weight, so an axis with
                // weight 0 never reads its neighbors: skip those decodes (3 instead of 5).
                float c0[MAXOUT], l1[MAXOUT], r1[MAXOUT], u1[MAXOUT], d1[MAXOUT];
                pixel(p, z, px, py, c0);
                if (wh > 0.0f) { pixel(p, z, std::max(0, px - 1), py, l1); pixel(p, z, std::min(W - 1, px + 1), py, r1); }
                else { for (int c = 0; c < nc; c++) { l1[c] = c0[c]; r1[c] = c0[c]; } }
                if (wv > 0.0f) { pixel(p, z, px, std::max(0, py - 1), u1); pixel(p, z, px, std::min(H - 1, py + 1), d1); }
                else { for (int c = 0; c < nc; c++) { u1[c] = c0[c]; d1[c] = c0[c]; } }
                deblock_pixel(c0, l1, r1, u1, d1, wh, wv, out);
            } else {
                pixel(p, z, px, py, out);
            }
            const float* t = &target.rgb[(size_t)i * nc];
            for (int c = 0; c < nc; c++) { float d = out[c] - t[c]; s += cw[c] * (d * d); }
        }
        return s / (3.0 * wsum * idx.size());
    }
};

// Reporting-only error measures: the weighted training objective and the
// unweighted per-texture MSE (comparable to single-texture runs).
struct Mse {
    double weighted = 0;        // sum_t w_t SSE_t / (3 wsum W H)
    double tex[MAXT] = { 0 };   // SSE_t / (3 W H)
};
static Mse mse_of(const Image& a, const Image& b, const std::vector<float>& w, float wsum) {
    const int T = a.nc / 3;
    const size_t npix = (size_t)a.w * a.h;
    double sse[MAXT] = { 0 };
    for (size_t p = 0; p < npix; p++)
        for (int t = 0; t < T; t++) {
            const float* x = &a.rgb[p * a.nc + 3 * t];
            const float* y = &b.rgb[p * b.nc + 3 * t];
            for (int c = 0; c < 3; c++) { double d = x[c] - y[c]; sse[t] += d * d; }
        }
    Mse m;
    double ws = 0;
    for (int t = 0; t < T; t++) { m.tex[t] = sse[t] / (3.0 * npix); ws += (double)w[t] * sse[t]; }
    m.weighted = ws / (3.0 * wsum * npix);
    return m;
}
static double psnr_of(double mse) { return mse > 0 ? 10.0 * std::log10(1.0 / mse) : 99.0; }

// ---------------------------------------------------------------- Adam
struct Adam {
    std::vector<float> m, v;
    float b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;
    int t = 0;
    void init(size_t n) { m.assign(n, 0); v.assign(n, 0); t = 0; }
    // Gradient-descent step: theta -= lr * adam(g)
    void step(std::vector<float>& theta, const std::vector<float>& g, float lr) {
        t++;
        float c1 = 1.0f - std::pow(b1, (float)t), c2 = 1.0f - std::pow(b2, (float)t);
#pragma omp parallel for schedule(static)
        for (int i = 0; i < (int)theta.size(); i++) {
            m[i] = b1 * m[i] + (1 - b1) * g[i];
            v[i] = b2 * v[i] + (1 - b2) * g[i] * g[i];
            float mh = m[i] / c1, vh = v[i] / c2;
            theta[i] -= lr * mh / (std::sqrt(vh) + eps);
        }
    }
};

// ---------------------------------------------------------------- ES: MLP
// Antithetic ES on the MLP weights, evaluated on one shared random minibatch.
// g = 1/(2 N sigma) * sum_i [L(p + s e_i) - L(p - s e_i)] e_i
struct MlpTrainer {
    Adam adam;
    std::vector<float> grad;
    std::vector<float> eps;   // N x P
    std::vector<int> batch;
    int cur_it = 0;           // iteration of the current step (keys the --rng hash draws, as on the GPU)

    void init(size_t P) { adam.init(P); grad.assign(P, 0); }

    // The pixel set one MLP step is evaluated on: a random minibatch (with
    // replacement), or every pixel in order when `full` is set. The full-image
    // case makes no index draws, so relative to a minibatch run the RNG stream
    // differs only by those skipped draws.
    void draw_batch(const Decoder& D, std::mt19937& rng, const Options& o, bool full) {
        if (full) {
            batch.resize((size_t)D.W * D.H);
            for (size_t i = 0; i < batch.size(); i++) batch[i] = (int)i;
        } else {
            batch.resize(o.mlp_batch);
            if (o.rng_hash) {
                for (size_t s = 0; s < batch.size(); s++) batch[s] = (int)ntc_noise_uniform(ntc_noise_u32(o.seed, NS_BATCH_PIX, cur_it, 0, s), (uint32_t)(D.W * D.H));
            } else {
                std::uniform_int_distribution<int> U(0, D.W * D.H - 1);
                for (auto& b : batch) b = U(rng);
            }
        }
    }

    // Antithetic ES step. Returns the mean minibatch loss across all evaluations (for stats).
    double step(Decoder& D, const Image& target, std::mt19937& rng, const Options& o, double& diff_std, bool full) {
        const size_t P = D.mlp.size();
        const int N = full ? o.mlp_full_pairs : o.mlp_pairs;
        eps.resize((size_t)N * P);
        if (o.rng_hash) {
            for (int i = 0; i < N; i++) for (size_t k = 0; k < P; k++) eps[(size_t)i * P + k] = ntc_gauss(o.seed, NS_MLP, cur_it, i, k);
        } else {
            std::normal_distribution<float> Nd(0.0f, 1.0f);
            for (auto& e : eps) e = Nd(rng);
        }
        draw_batch(D, rng, o, full);

        std::vector<double> dl(N), lsum(N);
        const float* p0 = D.mlp.p.data();
#pragma omp parallel
        {
            std::vector<float> pp(P), pm(P);
#pragma omp for schedule(dynamic)
            for (int i = 0; i < N; i++) {
                const float* e = &eps[(size_t)i * P];
                for (size_t k = 0; k < P; k++) { pp[k] = p0[k] + o.mlp_sigma * e[k]; pm[k] = p0[k] - o.mlp_sigma * e[k]; }
                double lp = D.loss_subset(pp.data(), D.lat.z.data(), target, batch);
                double lm = D.loss_subset(pm.data(), D.lat.z.data(), target, batch);
                dl[i] = lp - lm; lsum[i] = 0.5 * (lp + lm);
            }
        }
        double mean = 0, mean2 = 0, lmean = 0;
        for (int i = 0; i < N; i++) { mean += dl[i]; mean2 += dl[i] * dl[i]; lmean += lsum[i]; }
        mean /= N; mean2 /= N; lmean /= N;
        diff_std = std::sqrt(std::max(0.0, mean2 - mean * mean));

        std::fill(grad.begin(), grad.end(), 0.0f);
        float scale = 1.0f / (2.0f * N * o.mlp_sigma);
        for (int i = 0; i < N; i++) {
            float w = (float)dl[i] * scale;
            const float* e = &eps[(size_t)i * P];
            for (size_t k = 0; k < P; k++) grad[k] += w * e[k];
        }
        adam.step(D.mlp.p, grad, o.mlp_lr);
        return lmean;
    }

    // Central finite differences, one weight at a time, on a shared minibatch:
    //   dL/dw_j ~= [ L(w + h e_j) - L(w - h e_j) ] / (2h)
    // Exact up to O(h^2) and float rounding, no random-direction noise, but it
    // costs 2P minibatch evaluations per step versus 2N for ES. Neither ES nor
    // backprop: numerical differentiation. Meant as a late-training polish
    // once ES noise, not step count, limits the decoder.
    // Returns the unperturbed minibatch loss; diff_std receives the RMS of the
    // per-weight loss differences (printed as "fd-rms"; not comparable to the
    // ES "dstd", which is the spread of O(sigma) random-direction differences).
    // Note that Adam's second-moment estimate carries ES noise power across the
    // switch, which would throttle the first few hundred FD steps; main resets
    // the MLP Adam state when the FD phase begins.
    double step_fd(Decoder& D, const Image& target, std::mt19937& rng, const Options& o, double& diff_std, bool full) {
        const size_t P = D.mlp.size();
        const float h = o.mlp_fd_h;
        draw_batch(D, rng, o, full);

        const float* p0 = D.mlp.p.data();
        double l0 = D.loss_subset(p0, D.lat.z.data(), target, batch);
        std::vector<double> dl(P), hh(P);
#pragma omp parallel
        {
            std::vector<float> pw(p0, p0 + P);   // per-thread working copy
#pragma omp for schedule(dynamic, 16)
            for (int j = 0; j < (int)P; j++) {
                float orig = pw[j];
                // Use the step sizes actually realized in float, not nominal h.
                float wp = orig + h, wm = orig - h;
                pw[j] = wp;
                double lp = D.loss_subset(pw.data(), D.lat.z.data(), target, batch);
                pw[j] = wm;
                double lm = D.loss_subset(pw.data(), D.lat.z.data(), target, batch);
                pw[j] = orig;
                dl[j] = lp - lm;
                hh[j] = (double)wp - (double)wm;
            }
        }
        double ss = 0;
        for (size_t j = 0; j < P; j++) { grad[j] = (float)(dl[j] / hh[j]); ss += dl[j] * dl[j]; }
        diff_std = std::sqrt(ss / P);
        adam.step(D.mlp.p, grad, o.mlp_lr);
        return l0;
    }
};

// ---------------------------------------------------------------- ES: latent
// All texels of all levels are perturbed simultaneously. For each antithetic
// pair we decode the full image twice and get per-pixel squared errors e+ and
// e-. Each pixel's (e+ - e-) is attributed, once per level, to the (up to) 4
// texels its bilinear tap reads on that level, so texel (x,y) accumulates the
// loss change over exactly its footprint.
//   g[x,y,c] = 1/(2 K sigma) * sum_pairs [ sum_{footprint} (e+ - e-) / (3 sum(w) W H) ] * eps[x,y,c]
struct LatentTrainer {
    Adam adam;
    std::vector<float> grad, eps, zp, zm;
    std::vector<float> ep, em;   // per-pixel errors
    std::vector<float> dtex;     // per-texel footprint loss difference (reused per level)
    int step_no = 0;             // counts steps, used to rotate --lat-alt across steps

    void init(size_t n) { adam.init(n); grad.assign(n, 0); step_no = 0; }

    void step(Decoder& D, const Image& target, std::mt19937& rng, const Options& o) {
        const LatentSet& LS = D.lat;
        const size_t n = LS.size();
        const int K = o.lat_pairs;
        eps.resize(n); zp.resize(n); zm.resize(n);
        size_t maxtex = 0;
        for (const Latent& L : LS.lv) maxtex = std::max(maxtex, (size_t)L.W * L.H);
        dtex.resize(maxtex);
        std::fill(grad.begin(), grad.end(), 0.0f);
        std::normal_distribution<float> Nd(0.0f, 1.0f);
        const float inv_px = 1.0f / (3.0f * D.wsum * D.W * D.H);
        // Per-level sigma: level 0 uses lat_sigma, further levels lat2_sigma (already resolved to lat_sigma if unset).
        auto sigma_of = [&](size_t l) { return l == 0 ? o.lat_sigma : o.lat2_sigma; };

        // --lat-alt: pair k of this step perturbs only level (k + step_no) mod nlevels;
        // the other levels are left unperturbed (eps = 0) so they add no crosstalk.
        // Rotating with step_no means any K alternates fairly over time (K < nlevels
        // included). Each level's gradient is then the standard 1/(2 K_l sigma_l)
        // antithetic estimator over only the K_l pairs that perturbed it, which
        // keeps it unbiased for that level's smoothed gradient.
        const size_t nlev = LS.lv.size();
        const bool alt = o.lat_alt && nlev > 1 && o.qat == 0;   // with --qat only level 1 is perturbed anyway
        auto level_of_pair = [&](int k) { return (size_t)((k + step_no) % (int)nlev); };
        std::vector<int> pairs_of(nlev, K);
        if (alt) for (size_t l = 0; l < nlev; l++) { pairs_of[l] = 0; for (int k = 0; k < K; k++) if (level_of_pair(k) == l) pairs_of[l]++; }

        for (int k = 0; k < K; k++) {
            // One epsilon over the whole flat vector, drawn level by level (level 0 first).
            for (size_t l = 0; l < nlev; l++) {
                const Latent& L = LS.lv[l];
                const float sg = sigma_of(l);
                const bool active = (!alt || level_of_pair(k) == l) && !(o.qat > 0 && l == 0);   // --qat: level 0 is discrete, eps = 0
                for (size_t i = L.off; i < L.off + L.size(); i++) {
                    if (active) {
                        eps[i] = o.rng_hash ? ntc_gauss(o.seed, NS_LAT, step_no, k, i) : Nd(rng);
                        zp[i] = LS.z[i] + sg * eps[i];
                        zm[i] = LS.z[i] - sg * eps[i];
                    } else {
                        eps[i] = 0.0f; zp[i] = LS.z[i]; zm[i] = LS.z[i];
                    }
                }
            }
            D.decode_err(D.mlp.p.data(), zp.data(), target, ep);
            D.decode_err(D.mlp.p.data(), zm.data(), target, em);

            // Scatter each pixel's loss difference into the distinct texels its tap
            // reads, once per level with that level's own tap and border guard.
            for (size_t l = 0; l < nlev; l++) {
                if (alt && level_of_pair(k) != l) continue;   // this level was not perturbed in this pair
                if (o.qat > 0 && l == 0) continue;            // discrete level 0: no scatter, no ES gradient
                const Latent& L = LS.lv[l];
                const size_t ntex = (size_t)L.W * L.H;
                const float scale = 1.0f / (2.0f * pairs_of[l] * sigma_of(l));
                std::fill(dtex.begin(), dtex.begin() + ntex, 0.0f);
                for (int py = 0; py < D.H; py++) {
                    float v = (py + 0.5f) / D.H;
                    for (int px = 0; px < D.W; px++) {
                        float u = (px + 0.5f) / D.W;
                        float d = (ep[(size_t)py * D.W + px] - em[(size_t)py * D.W + px]) * inv_px;
                        float wh = 0, wv = 0;
                        if (D.deblock) D.deblock_weights(px, py, wh, wv);
                        if (wh <= 0.0f && wv <= 0.0f) {
                            BilinearTap t = bilinear_tap(L, u, v);
                            dtex[(size_t)t.y0 * L.W + t.x0] += d;
                            if (t.x1 != t.x0) dtex[(size_t)t.y0 * L.W + t.x1] += d;
                            if (t.y1 != t.y0) {
                                dtex[(size_t)t.y1 * L.W + t.x0] += d;
                                if (t.x1 != t.x0) dtex[(size_t)t.y1 * L.W + t.x1] += d;
                            }
                        } else {
                            // Deblocked pixel: it also depends on the texels read by the neighbors
                            // the filter mixes in, so credit the union of those taps' texels once each.
                            size_t tex[20]; int nt = 0;
                            auto add_tap = [&](int qx, int qy) {
                                BilinearTap t = bilinear_tap(L, (qx + 0.5f) / D.W, (qy + 0.5f) / D.H);
                                size_t cand[4] = { (size_t)t.y0 * L.W + t.x0, (size_t)t.y0 * L.W + t.x1,
                                                   (size_t)t.y1 * L.W + t.x0, (size_t)t.y1 * L.W + t.x1 };
                                for (size_t c : cand) {
                                    bool dup = false;
                                    for (int j = 0; j < nt; j++) if (tex[j] == c) { dup = true; break; }
                                    if (!dup) tex[nt++] = c;
                                }
                            };
                            add_tap(px, py);
                            if (wh > 0.0f) { add_tap(std::max(0, px - 1), py); add_tap(std::min(D.W - 1, px + 1), py); }
                            if (wv > 0.0f) { add_tap(px, std::max(0, py - 1)); add_tap(px, std::min(D.H - 1, py + 1)); }
                            for (int j = 0; j < nt; j++) dtex[tex[j]] += d;
                        }
                    }
                }
                for (int ty = 0; ty < L.H; ty++)
                    for (int tx = 0; tx < L.W; tx++) {
                        float w = dtex[(size_t)ty * L.W + tx] * scale;
                        size_t base = L.off + ((size_t)ty * L.W + tx) * L.C;
                        for (int c = 0; c < L.C; c++) grad[base + c] += w * eps[base + c];
                    }
            }
        }
        adam.step(D.lat.z, grad, o.lat_lr);
        step_no++;   // rotate the --lat-alt assignment for the next step
    }
};

// ---------------------------------------------------------------- QAT: discrete level 0
// With --qat B every level-0 value is one of 2^B grid points -1 + 2k/(2^B-1), k = 0..2^B-1.
// The grid is fixed (no per-channel min/max), so these three functions are the single
// source of truth for the search, the initial snap, the loader and the bitrate histogram.
// Each channel has its own bit depth B (1..8). The grid values come from tables filled
// once (qat_init_grid), so every on-grid comparison sees the same bit pattern regardless
// of how the compiler evaluates the expression at different call sites.
static float QAT_GRID[9][257];   // [bits][index]
static void qat_init_grid() {
    for (int b = 1; b <= 8; b++) { int levels = (1 << b) - 1; for (int k = 0; k <= levels; k++) QAT_GRID[b][k] = -1.0f + 2.0f * k / (float)levels; }
}
static inline int qat_levels(int bits) { return (1 << bits) - 1; }   // largest grid index
static inline float qat_value(int k, int bits) { return QAT_GRID[bits][k]; }
static inline int qat_index(float v, int bits) {
    const int levels = qat_levels(bits);
    int k = (int)std::lround((v + 1.0f) * 0.5f * levels);
    return std::max(0, std::min(levels, k));
}
static inline float qat_snap(float v, int bits) { return qat_value(qat_index(v, bits), bits); }
// Snap every level-0 value to its channel's grid; returns how many values moved.
static size_t qat_snap_level0(LatentSet& LS, const std::vector<int>& ch_bits) {
    const Latent& L = LS.lv[0]; size_t moved = 0;
    for (size_t i = 0; i < L.size(); i++) { float sv = qat_snap(LS.z[i], ch_bits[i % L.C]); if (sv != LS.z[i]) { LS.z[i] = sv; moved++; } }
    return moved;
}
static int qat_decodes(const std::vector<int>& ch_bits) { int d = 0; for (int b : ch_bits) d += 1 << b; return d; }
static std::string qat_spec(const std::vector<int>& ch_bits) {
    std::string r; for (size_t c = 0; c < ch_bits.size(); c++) r += (c ? "," : "") + std::to_string(ch_bits[c]); return r;
}

// Exact coordinate descent on the discrete level 0. With nearest sampling and no deblocking
// a pixel's output depends on exactly one level-0 texel (Decoder::features copies that
// texel's C values into f[0..C) and nothing after depends on level-0 values), so the
// per-texel objective, the summed cw-weighted squared error over the texel's cell, is
// separable across texels: texels are searched independently and in parallel, and within
// a texel the channels are searched one after another, each over all 2^B grid values with
// the others held fixed. Ties keep the current value, so the loss never increases, and
// texels without pixels (latent finer than the image) are left alone. The MLP inputs of
// the cell's pixels are computed once per texel and only f[c] is patched per candidate,
// which is bit-identical to calling Decoder::pixel with the candidate written into z.
// The loss normalization (1 / (3 wsum W H)) does not affect the argmin and is omitted.
static void qat_search(Decoder& D, const Image& target, const std::vector<int>& ch_bits) {
    const Latent& L = D.lat.lv[0];
    const int nc = D.mlp.nout, nin = D.nin();
    // Pixel range [xb, xe) x [yb, ye) of each texel, from the same nearest tap features() uses.
    std::vector<int> xb(L.W, D.W), xe(L.W, 0), yb(L.H, D.H), ye(L.H, 0);
    for (int px = 0; px < D.W; px++) { int t = bilinear_tap(L, (px + 0.5f) / D.W, 0.5f).x0; xb[t] = std::min(xb[t], px); xe[t] = std::max(xe[t], px + 1); }
    for (int py = 0; py < D.H; py++) { int t = bilinear_tap(L, 0.5f, (py + 0.5f) / D.H).y0; yb[t] = std::min(yb[t], py); ye[t] = std::max(ye[t], py + 1); }
    const float* p = D.mlp.p.data();
    float* z = D.lat.z.data();   // level 0 starts at offset 0; each thread writes only its own texels
#pragma omp parallel
    {
        std::vector<float> feat;   // per-thread: nin inputs for each pixel of the current cell
#pragma omp for schedule(dynamic)
        for (int ty = 0; ty < L.H; ty++) {
            for (int tx = 0; tx < L.W; tx++) {
                const int x0 = xb[tx], x1 = xe[tx], y0 = yb[ty], y1 = ye[ty];
                if (x0 >= x1 || y0 >= y1) continue;
                const int npx = (x1 - x0) * (y1 - y0);
                feat.resize((size_t)npx * nin);
                { int j = 0; for (int py = y0; py < y1; py++) for (int px = x0; px < x1; px++, j++) D.features(z, px, py, &feat[(size_t)j * nin]); }
                float* zt = z + ((size_t)ty * L.W + tx) * L.C;
                for (int c = 0; c < L.C; c++) {
                    const int bits = ch_bits[c], levels = qat_levels(bits);
                    auto cell_loss = [&](float val) {
                        double sum = 0; int j = 0;
                        for (int py = y0; py < y1; py++)
                            for (int px = x0; px < x1; px++, j++) {
                                float* f = &feat[(size_t)j * nin]; f[c] = val;
                                float out[MAXOUT];
                                mlp_forward(D.mlp, p, f, out, D.opt->clamp_out);
                                const float* t = &target.rgb[((size_t)py * D.W + px) * nc];
                                for (int q = 0; q < nc; q++) { float d = out[q] - t[q]; sum += D.cw[q] * (d * d); }
                            }
                        return sum;
                    };
                    int best_k = qat_index(zt[c], bits);            // zt[c] is on-grid: this is its index
                    double best = cell_loss(qat_value(best_k, bits));   // the current value first; ties keep it
                    for (int k = 0; k <= levels; k++) {
                        if (k == best_k) continue;
                        double sk = cell_loss(qat_value(k, bits));
                        if (sk < best) { best = sk; best_k = k; }
                    }
                    const float v = qat_value(best_k, bits);
                    zt[c] = v;
                    for (int j = 0; j < npx; j++) feat[(size_t)j * nin + c] = v;   // fixed for the next channel
                }
            }
        }
    }
}

// ---------------------------------------------------------------- I/O helpers
static void save_latent_png(const std::string& path, const Latent& L, const float* z) {
    // Channels laid out side by side, each mapped from [-1.5,1.5] to [0,255].
    int w = L.W * L.C, h = L.H;
    std::vector<unsigned char> buf((size_t)w * h);
    for (int y = 0; y < h; y++)
        for (int c = 0; c < L.C; c++)
            for (int x = 0; x < L.W; x++) {
                float v = z[((size_t)y * L.W + x) * L.C + c];
                float m = std::min(1.0f, std::max(0.0f, v / 3.0f + 0.5f));
                buf[(size_t)y * w + c * L.W + x] = (unsigned char)std::lround(m * 255.0f);
            }
    stbi_write_png(path.c_str(), w, h, 1, buf.data(), w);
}

static void save_model(const std::string& path, const Decoder& D) {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) return;
    // v6 header: magic, LW0, LH0, LC0, nin, poslen, act, nlayers, nlevels, textures,
    // then one int per level (0 bilinear, 1 nearest), then (W,H,C) for each level
    // after the first, then the hidden widths, then
    // poslen bytes of positional spec text, then the flat latent (level 0
    // first, each level (y,x,c) with c fastest), then the MLP parameters
    // (3 * textures outputs).
    int hdr[10] = { 0x4E54433A, D.lat.lv[0].W, D.lat.lv[0].H, D.lat.lv[0].C, D.mlp.nin,
                    (int)D.pos.spec.size(), (int)D.mlp.act, (int)D.mlp.hidden.size(), (int)D.lat.lv.size(), D.mlp.nout / 3 };
    fwrite(hdr, sizeof(hdr), 1, f);
    for (size_t l = 0; l < D.lat.lv.size(); l++) { int nr = D.lat.lv[l].nearest ? 1 : 0; fwrite(&nr, sizeof(int), 1, f); }
    { int db = D.deblock ? 1 : 0; fwrite(&db, sizeof(int), 1, f); fwrite(&D.deblock_falloff, sizeof(float), 1, f); }   // v7
    fwrite(&D.mlp.leak, sizeof(float), 1, f);   // v8
    fwrite(&D.qat_bits, sizeof(int), 1, f);     // v9: largest --qat bit depth (0 = off)
    for (int c = 0; c < D.lat.lv[0].C; c++) { int b = D.qat_bits > 0 ? D.qat_ch[c] : 0; fwrite(&b, sizeof(int), 1, f); }   // v10: bits per level-0 channel
    for (size_t l = 1; l < D.lat.lv.size(); l++) {
        int d[3] = { D.lat.lv[l].W, D.lat.lv[l].H, D.lat.lv[l].C };
        fwrite(d, sizeof(d), 1, f);
    }
    fwrite(D.mlp.hidden.data(), sizeof(int), D.mlp.hidden.size(), f);
    fwrite(D.pos.spec.data(), 1, D.pos.spec.size(), f);
    fwrite(D.lat.z.data(), sizeof(float), D.lat.z.size(), f);
    fwrite(D.mlp.p.data(), sizeof(float), D.mlp.p.size(), f);
    fclose(f);
}

static void latent_stats(const Latent& L, const float* z, float& mean, float& sd, float& mx) {
    double s = 0, s2 = 0; mx = 0;
    for (size_t i = 0; i < L.size(); i++) { float v = z[i]; s += v; s2 += (double)v * v; mx = std::max(mx, std::fabs(v)); }
    mean = (float)(s / L.size());
    sd = (float)std::sqrt(std::max(0.0, s2 / L.size() - (s / L.size()) * (s / L.size())));
}


// ---------------------------------------------------------------- bitrate
// Effective compressed size estimate. The latent is quantized to 8 bits per
// channel with a per-channel min/max scale (2 floats per channel overhead), and
// the MLP weights are counted as fp16. Two latent numbers are reported: raw
// 8 bits/texel/channel, and the zeroth-order entropy of the quantized symbols.
// The dequantized latent is also returned so the caller can measure the PSNR
// the codec would actually achieve at that bitrate.
struct BitrateStats {
    double bpp_fp32;      // everything stored as fp32
    double bpp_q8;        // 8-bit latent (raw) + fp16 MLP
    double bpp_q8_ent;    // entropy-coded 8-bit latent + fp16 MLP
    double bits_mlp;      // fp16 MLP bits
};

// With qat_bits > 0, level 0 is already discrete: it is charged qat_bits per value with no
// min/max header, its entropy is that of the 2^qat_bits grid indices, and zq keeps its values
// as they are (re-quantizing them with a min/max scale would be wrong). Other levels use qbits.
static BitrateStats bitrate_stats(const LatentSet& LS, const MLP& m, int W, int H, int qbits, const std::vector<int>& qat_ch, std::vector<float>& zq) {
    BitrateStats s;
    const double npix = (double)W * H;
    s.bits_mlp = m.size() * 16.0;
    s.bpp_fp32 = (LS.size() * 32.0 + m.size() * 32.0) / npix;
    zq.resize(LS.size());
    double ent_bits = 0, header_bits = 0, raw_bits = 0;
    for (size_t l = 0; l < LS.lv.size(); l++) {   // each level quantized independently, per channel
        const Latent& L = LS.lv[l];
        const bool fixed = (l == 0 && !qat_ch.empty());   // discrete level: fixed per-channel grids, no scale header
        const float* z = LS.z.data() + L.off;
        float* q = zq.data() + L.off;
        const size_t nl = L.size();
        const double ntex = (double)L.W * L.H;
        if (!fixed) raw_bits += nl * (double)qbits;
        for (int c = 0; c < L.C; c++) {
            const int bits = fixed ? qat_ch[c] : qbits;
            const int levels = (1 << bits) - 1;
            if (fixed) raw_bits += ntex * (double)bits;
            float lo = 1e30f, hi = -1e30f;
            if (!fixed) for (size_t i = c; i < nl; i += L.C) { lo = std::min(lo, z[i]); hi = std::max(hi, z[i]); }
            float range = std::max(hi - lo, 1e-6f);
            std::vector<int> hist(levels + 1, 0);
            for (size_t i = c; i < nl; i += L.C) {
                int qi;
                if (fixed) { qi = qat_index(z[i], bits); q[i] = z[i]; }
                else {
                    qi = (int)std::lround((z[i] - lo) / range * levels);
                    qi = std::max(0, std::min(levels, qi));
                    q[i] = lo + qi / (float)levels * range;
                }
                hist[qi]++;
            }
            for (int qv = 0; qv <= levels; qv++)
                if (hist[qv]) ent_bits -= hist[qv] * std::log2(hist[qv] / ntex);
        }
        if (!fixed) header_bits += L.C * 2 * 32.0; // per-channel min/max, per level
    }
    s.bpp_q8 = (raw_bits + header_bits + s.bits_mlp) / npix;
    s.bpp_q8_ent = (ent_bits + header_bits + s.bits_mlp) / npix;
    return s;
}

// ---------------------------------------------------------------- main
#ifdef NTC_CUDA
// --cuda-check: run the deterministic GPU kernels and the CPU code on identical inputs
// and report the differences. The decode and the selector search must agree to float
// rounding; the ES / FD loss differences are compared per pair / per weight against the
// CPU's loss_subset evaluated with the same (hash-generated) perturbations and batch.
static int cuda_check(Decoder& D, const Image& target, const Options& o, ntc_cuda::Trainer& cu) {
    printf("cuda-check: %s\n", cu.banner().c_str());
    const size_t npix = (size_t)D.W * D.H;
    int fails = 0;
    // 1. full decode
    {
        Image a, b;
        D.decode_full(D.mlp.p.data(), D.lat.z.data(), a);
        b.w = D.W; b.h = D.H; b.nc = D.mlp.nout; b.rgb.resize(npix * b.nc);
        cu.decode_full(nullptr, b.rgb.data());
        double mx = 0; for (size_t i = 0; i < a.rgb.size(); i++) mx = std::max(mx, (double)std::fabs(a.rgb[i] - b.rgb[i]));
        bool ok = mx < 1e-5;
        printf("  decode          : max |gpu - cpu| = %.3e  %s\n", mx, ok ? "PASS" : "FAIL"); fails += !ok;
    }
    // 2. MLP ES losses: 8 pairs, lr 0 (weights unchanged), same batch and perturbations on the CPU
    {
        const int N = 8, it = 1;
        cu.mlp_step(it, o.mlp_sigma, 0.0f, N, o.mlp_batch, false);
        std::vector<int> bidx; cu.debug_last_batch(bidx);
        std::vector<double> dl; cu.debug_last_dl(dl);
        const size_t P = D.mlp.size();
        std::vector<float> pp(P), pm(P);
        double mxrel = 0, dstd = 0;
        for (int i = 0; i < N; i++) dstd += dl[i] * dl[i];
        dstd = std::sqrt(dstd / N);
        for (int i = 0; i < N; i++) {
            for (size_t k = 0; k < P; k++) { float e = ntc_gauss(o.seed, NS_MLP, it, i, k); pp[k] = D.mlp.p[k] + o.mlp_sigma * e; pm[k] = D.mlp.p[k] - o.mlp_sigma * e; }
            double lp = D.loss_subset(pp.data(), D.lat.z.data(), target, bidx), lm = D.loss_subset(pm.data(), D.lat.z.data(), target, bidx);
            mxrel = std::max(mxrel, std::fabs((lp - lm) - dl[i]) / std::max(dstd, 1e-300));
        }
        bool ok = mxrel < 1e-3;
        printf("  mlp ES dl       : max |gpu - cpu| / rms(dl) = %.3e over %d pairs  %s\n", mxrel, N, ok ? "PASS" : "FAIL"); fails += !ok;
    }
    // 3. FD losses: 24 sampled weights against the CPU's central differences on the same batch
    {
        const int it = 2;
        cu.mlp_step_fd(it, o.mlp_fd_h, 0.0f, o.mlp_batch, false);
        std::vector<int> bidx; cu.debug_last_batch(bidx);
        std::vector<double> dl; cu.debug_last_dl(dl);
        const size_t P = D.mlp.size();
        std::vector<float> pw(D.mlp.p);
        double rms = 0; for (size_t j = 0; j < P; j++) rms += dl[j] * dl[j]; rms = std::sqrt(rms / P);
        double mxrel = 0;
        for (int q = 0; q < 24; q++) {
            size_t j = (size_t)((q * 7919) % P);
            float orig = pw[j], wp = orig + o.mlp_fd_h, wm = orig - o.mlp_fd_h;
            pw[j] = wp; double lp = D.loss_subset(pw.data(), D.lat.z.data(), target, bidx);
            pw[j] = wm; double lm = D.loss_subset(pw.data(), D.lat.z.data(), target, bidx);
            pw[j] = orig;
            mxrel = std::max(mxrel, std::fabs((lp - lm) - dl[j]) / std::max(rms, 1e-300));
        }
        bool ok = mxrel < 1e-2;
        printf("  mlp FD dl       : max |gpu - cpu| / rms(dl) = %.3e over 24 weights  %s\n", mxrel, ok ? "PASS" : "FAIL"); fails += !ok;
    }
    // 4. latent ES gradient: one step with lr 0 on the GPU, the same estimator on the CPU
    //    with the same hash-generated perturbations (nearest levels, decode_err, raster
    //    gather), compared before Adam relative to the gradient's RMS.
    if (!(o.qat > 0 && D.lat.lv.size() == 1)) {
        const int K = o.lat_pairs;
        cu.lat_step(1, o.lat_sigma, o.lat2_sigma, 0.0f, K, o.lat_alt);   // first call: noise step 0
        std::vector<float> gg; cu.debug_last_zgrad(gg);
        const LatentSet& LS = D.lat;
        const size_t n = LS.size(), nlev = LS.lv.size();
        std::vector<float> gc(n, 0.0f), zp(n), zm(n), ep, em;
        const float inv_px = 1.0f / (3.0f * D.wsum * D.W * D.H);
        const bool alt = o.lat_alt && nlev > 1 && o.qat == 0;
        auto level_of_pair = [&](int k) { return (size_t)(k % (int)nlev); };   // step_no = 0
        std::vector<int> pairs_of(nlev, K);
        if (alt) for (size_t l = 0; l < nlev; l++) { pairs_of[l] = 0; for (int k = 0; k < K; k++) if (level_of_pair(k) == l) pairs_of[l]++; }
        for (int k = 0; k < K; k++) {
            std::vector<bool> act(nlev);
            for (size_t l = 0; l < nlev; l++) act[l] = (!alt || level_of_pair(k) == l) && !(o.qat > 0 && l == 0);
            for (size_t l = 0; l < nlev; l++) {
                const Latent& L = LS.lv[l]; const float sg = l == 0 ? o.lat_sigma : o.lat2_sigma;
                for (size_t i = L.off; i < L.off + L.size(); i++) {
                    float e = act[l] ? ntc_gauss(o.seed, NS_LAT, 0, k, i) : 0.0f;
                    zp[i] = LS.z[i] + sg * e; zm[i] = LS.z[i] - sg * e;
                }
            }
            D.decode_err(D.mlp.p.data(), zp.data(), target, ep);
            D.decode_err(D.mlp.p.data(), zm.data(), target, em);
            for (size_t l = 0; l < nlev; l++) {
                if (!act[l]) continue;
                const Latent& L = LS.lv[l]; const float sg = l == 0 ? o.lat_sigma : o.lat2_sigma;
                const float scale = 1.0f / (2.0f * pairs_of[l] * sg);
                std::vector<float> dtex((size_t)L.W * L.H, 0.0f);
                for (int py = 0; py < D.H; py++)
                    for (int px = 0; px < D.W; px++) {
                        BilinearTap t = bilinear_tap(L, (px + 0.5f) / D.W, (py + 0.5f) / D.H);
                        float dd = (ep[(size_t)py * D.W + px] - em[(size_t)py * D.W + px]) * inv_px;
                        dtex[(size_t)t.y0 * L.W + t.x0] += dd;
                        if (t.x1 != t.x0) dtex[(size_t)t.y0 * L.W + t.x1] += dd;
                        if (t.y1 != t.y0) { dtex[(size_t)t.y1 * L.W + t.x0] += dd; if (t.x1 != t.x0) dtex[(size_t)t.y1 * L.W + t.x1] += dd; }
                    }
                for (size_t ti = 0; ti < dtex.size(); ti++) {
                    float w = dtex[ti] * scale; size_t base = L.off + ti * L.C;
                    for (int c = 0; c < L.C; c++) gc[base + c] += w * ntc_gauss(o.seed, NS_LAT, 0, k, base + c);
                }
            }
        }
        double rms = 0, mx = 0; size_t cnt = 0;
        for (size_t i = 0; i < n; i++) { rms += (double)gc[i] * gc[i]; cnt++; }
        rms = std::sqrt(rms / std::max<size_t>(cnt, 1));
        for (size_t i = 0; i < n; i++) mx = std::max(mx, (double)std::fabs(gg[i] - gc[i]));
        bool ok = mx < 1e-3 * rms;
        printf("  latent ES grad  : max |gpu - cpu| / rms(grad) = %.3e over %zu values  %s\n", mx / std::max(rms, 1e-300), n, ok ? "PASS" : "FAIL"); fails += !ok;
    }
    // 5. selector search from the same state
    if (o.qat > 0) {
        std::vector<float> z0 = D.lat.z;
        qat_search(D, target, o.qat_ch);                 // CPU, in place
        std::vector<float> zc = D.lat.z;
        D.lat.z = z0; cu.upload_model(D.lat.z.data(), D.mlp.p.data());
        cu.qat_search();
        std::vector<float> zg(D.lat.z.size()), ptmp(D.mlp.size());
        cu.download_model(zg.data(), ptmp.data());
        const size_t n0 = D.lat.lv[0].size();
        size_t diff = 0; for (size_t i = 0; i < n0; i++) if (zc[i] != zg[i]) diff++;
        Image a, b; D.lat.z = zc; D.decode_full(D.mlp.p.data(), D.lat.z.data(), a);
        D.lat.z = zg; D.decode_full(D.mlp.p.data(), D.lat.z.data(), b);
        double mc = mse_of(target, a, o.weights, D.wsum).weighted, mg = mse_of(target, b, o.weights, D.wsum).weighted;
        // Per-texel cell losses of both results, on the CPU: a differing texel is a tie
        // (or a rounding-level near tie) unless the GPU's choice is measurably worse.
        const Latent& L0 = D.lat.lv[0];
        std::vector<double> cc((size_t)L0.W * L0.H, 0.0), cg((size_t)L0.W * L0.H, 0.0);
        for (int py = 0; py < D.H; py++)
            for (int px = 0; px < D.W; px++) {
                BilinearTap t = bilinear_tap(L0, (px + 0.5f) / D.W, (py + 0.5f) / D.H);
                size_t ti = (size_t)t.y0 * L0.W + t.x0, pi = (size_t)py * D.W + px;
                const float* tg = &target.rgb[pi * a.nc];
                for (int c = 0; c < a.nc; c++) {
                    float dc = a.rgb[pi * a.nc + c] - tg[c], dg = b.rgb[pi * b.nc + c] - tg[c];
                    cc[ti] += D.cw[c] * (dc * dc); cg[ti] += D.cw[c] * (dg * dg);
                }
            }
        size_t worse = 0, better = 0, texdiff = 0;
        for (size_t ti = 0; ti < cc.size(); ti++) {
            bool d = false; for (int c = 0; c < L0.C; c++) if (zc[ti * L0.C + c] != zg[ti * L0.C + c]) d = true;
            if (!d) continue;
            texdiff++;
            if (cg[ti] > cc[ti] * (1.0 + 1e-4) + 1e-12) worse++;
            if (cc[ti] > cg[ti] * (1.0 + 1e-4) + 1e-12) better++;
        }
        // Rounding flips near-ties both ways; a real bug would be one-sided and change the loss.
        bool ok = worse <= better + better / 2 + 16 && std::fabs(mg - mc) <= mc * 1e-6;
        printf("  qat search      : %zu of %zu values differ in %zu texels: gpu worse %zu, better %zu, ties %zu; loss after search cpu %.6e gpu %.6e  %s\n",
            diff, n0, texdiff, worse, better, texdiff - worse - better, mc, mg, ok ? "PASS" : "FAIL"); fails += !ok;
    }
    printf("cuda-check: %s\n", fails ? "FAILED" : "all passed");
    return fails ? 1 : 0;
}
#endif

int main(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&](int k = 1) { if (i + k >= argc) { usage(); exit(1); } return argv[i + k]; };
        if (a == "--out") { o.outdir = next(); i++; }
        else if (a == "--crop") { o.crop = atoi(next()); i++; }
        else if (a == "--latent") { o.LW = atoi(next(1)); o.LH = atoi(next(2)); o.LC = atoi(next(3)); i += 3; }
        else if (a == "--latent2") { o.LW2 = atoi(next(1)); o.LH2 = atoi(next(2)); o.LC2 = atoi(next(3)); i += 3; }
        else if (a == "--filter") { o.filter = next(); i++; }
        else if (a == "--deblock") { o.deblock = true; }
        else if (a == "--deblock-falloff") { o.deblock_falloff = (float)atof(next()); i++; }
        else if (a == "--hidden") { int n = atoi(next()); o.hidden = { n, n }; i++; }
        else if (a == "--mlp") {
            o.hidden.clear();
            std::string list = next(); i++;
            for (size_t s = 0; s < list.size();) {
                size_t e = list.find(',', s);
                if (e == std::string::npos) e = list.size();
                if (e > s) o.hidden.push_back(atoi(list.substr(s, e - s).c_str()));
                s = e + 1;
            }
        }
        else if (a == "--act") { o.act = next(); i++; }
        else if (a == "--leak") { o.leak = (float)atof(next()); i++; }
        else if (a == "--nfreq") { int n = atoi(next()); i++; o.pos = n > 0 ? "uv,fourier:" + std::to_string(n) : "uv"; }
        else if (a == "--pos") { o.pos = next(); i++; }
        else if (a == "--clamp") { o.clamp_out = true; }
        else if (a == "--iters") { o.iters = atoi(next()); i++; }
        else if (a == "--save-every") { o.save_every = atoi(next()); i++; }
        else if (a == "--print-every") { o.print_every = atoi(next()); i++; }
        else if (a == "--seed") { o.seed = (unsigned)atoi(next()); i++; }
        else if (a == "--mlp-pairs") { o.mlp_pairs = atoi(next()); i++; }
        else if (a == "--mlp-batch") { o.mlp_batch = atoi(next()); i++; }
        else if (a == "--mlp-sigma") { o.mlp_sigma = (float)atof(next()); i++; }
        else if (a == "--mlp-lr") { o.mlp_lr = (float)atof(next()); i++; }
        else if (a == "--mlp-every") { o.mlp_every = atoi(next()); i++; }
        else if (a == "--mlp-fd") { o.mlp_fd_start = (float)atof(next()); i++; }
        else if (a == "--mlp-fd-h") { o.mlp_fd_h = (float)atof(next()); o.mlp_fd_h_set = true; i++; }
        else if (a == "--mlp-freeze") { o.mlp_freeze_start = (float)atof(next()); i++; }
        else if (a == "--mlp-full") { o.mlp_full_start = (float)atof(next()); i++; }
        else if (a == "--mlp-full-pairs") { o.mlp_full_pairs = atoi(next()); i++; }
        else if (a == "--lat-alt") { o.lat_alt = true; }
        else if (a == "--lat-pairs") { o.lat_pairs = atoi(next()); i++; }
        else if (a == "--lat-sigma") { o.lat_sigma = (float)atof(next()); i++; }
        else if (a == "--lat-lr") { o.lat_lr = (float)atof(next()); i++; }
        else if (a == "--lat-init") { o.lat_init = (float)atof(next()); i++; }
        else if (a == "--lat2-sigma") { o.lat2_sigma = (float)atof(next()); i++; }
        else if (a == "--lr-anneal") { o.lr_anneal_start = (float)atof(next(1)); o.lr_anneal_final = (float)atof(next(2)); i += 2; }
        else if (a == "--sigma-anneal") { o.sigma_anneal_start = (float)atof(next(1)); o.sigma_anneal_final = (float)atof(next(2)); i += 2; }
        else if (a == "--threads") { o.threads = atoi(next()); i++; }
        else if (a == "--qbits") { o.qbits = atoi(next()); i++; }
        else if (a == "--qat") {
            std::string v = next(); i++;
            o.qat_ch.clear();
            size_t pos = 0;
            while (pos <= v.size()) {
                size_t e = v.find(',', pos); if (e == std::string::npos) e = v.size();
                std::string tok = v.substr(pos, e - pos);
                if (tok.empty() || tok.find_first_not_of("0123456789") != std::string::npos) { o.qat_ch.assign(1, -1); break; }
                o.qat_ch.push_back(atoi(tok.c_str()));
                pos = e + 1;
            }
            o.qat = 0; for (int b : o.qat_ch) o.qat = std::max(o.qat, b);
        }
        else if (a == "--qat-every") { o.qat_every = atoi(next()); i++; }
        else if (a == "--rng") { std::string m = next(); i++; if (m == "hash") o.rng_hash = true; else if (m == "mt") o.rng_hash = false; else { printf("--rng: mt | hash\n"); return 1; } }
        else if (a == "--cuda") { o.cuda = true; }
        else if (a == "--cuda-check") { o.cuda_check = true; }
        else if (a == "--load") { o.load = next(); i++; }
        else if (a == "--weights") {
            o.weights.clear(); o.weights_given = true;
            std::string list = next(); i++;
            for (size_t s = 0; s <= list.size();) {
                size_t e = list.find(',', s);
                if (e == std::string::npos) e = list.size();
                std::string tok = list.substr(s, e - s);
                char* end = nullptr;
                double w = tok.empty() ? 0.0 : strtod(tok.c_str(), &end);
                if (tok.empty() || *end != 0 || !std::isfinite(w)) { printf("--weights: bad entry '%s' in '%s'\n", tok.c_str(), list.c_str()); return 1; }
                o.weights.push_back((float)w);
                s = e + 1;
            }
        }
        else if (a == "-h" || a == "--help") { usage(); return 0; }
        else if (a[0] == '-') { printf("unknown option %s\n", a.c_str()); usage(); return 1; }
        else o.inputs.push_back(a);
    }
    const bool default_input = o.inputs.empty();
    if (default_input) o.inputs.push_back("kodim23.png");
    const int T = (int)o.inputs.size();
    if (T > MAXT) { printf("at most %d textures per material\n", MAXT); return 1; }
    if (!o.weights_given) o.weights.assign(T, 1.0f);
    if ((int)o.weights.size() != T) { printf("--weights needs %d entries (one per input image), got %zu\n", T, o.weights.size()); return 1; }
    float wsum = 0.0f;
    for (float w : o.weights) { if (w < 0.0f) { printf("--weights must be >= 0\n"); return 1; } wsum += w; }
    if (wsum <= 0.0f) { printf("--weights must not all be zero\n"); return 1; }
    Act act;
    if (!parse_act(o.act, act)) { printf("unknown activation %s\n", o.act.c_str()); return 1; }
    if (o.hidden.empty() || (int)o.hidden.size() > MAXL) { printf("need 1..%d hidden layers\n", MAXL); return 1; }
    if ((o.LW2 > 0 || o.LH2 > 0 || o.LC2 > 0) && (o.LW2 < 1 || o.LH2 < 1 || o.LC2 < 1)) { printf("--latent2 needs W H C all > 0 (or 0 0 0 = off)\n"); return 1; }
    if (o.lat2_sigma > 0.0f && o.LW2 == 0) printf("note: --lat2-sigma is ignored without --latent2\n");
    if (o.lat2_sigma <= 0.0f) o.lat2_sigma = o.lat_sigma;   // resolve the sentinel once; everything below uses it unconditionally
    if (o.mlp_fd_h <= 0.0f) { printf("--mlp-fd-h must be > 0\n"); return 1; }
    if (o.LW < 1 || o.LH < 1 || o.LC < 1 || o.LC > MAXH) { printf("--latent needs W, H >= 1 and 1..%d channels\n", MAXH); return 1; }
    if (o.LW2 > 0 && (o.LH2 < 1 || o.LC2 < 1 || o.LC2 > MAXH)) { printf("--latent2 needs W, H >= 1 and 1..%d channels\n", MAXH); return 1; }
    if (o.qat_ch.size() == 1 && o.qat_ch[0] < 0) { printf("--qat: malformed bit list (use B or B1,B2,... with B = 1..8)\n"); return 1; }
    for (int b : o.qat_ch) if (b < 0 || b > 8) { printf("--qat needs 1..8 bits per channel (0 = off)\n"); return 1; }
    if (o.qat_ch.size() > 1 && (int)o.qat_ch.size() != o.LC) { printf("--qat lists %zu bit depths but level 0 has %d channels\n", o.qat_ch.size(), o.LC); return 1; }
    if (o.qat > 0) {
        for (int b : o.qat_ch) if (b == 0) { printf("--qat: every channel needs 1..8 bits when quantization is on\n"); return 1; }
        if (o.qat_ch.size() == 1) o.qat_ch.assign(o.LC, o.qat_ch[0]);   // one value applies to every channel
    } else o.qat_ch.clear();
    if (o.qat_every < 1) { printf("--qat-every must be >= 1\n"); return 1; }
    if (o.qat > 0 && o.deblock) { printf("--qat cannot be combined with --deblock: the filter couples neighboring cells, so the per-texel search would not be exact\n"); return 1; }
    if (o.qat == 0 && o.qat_every != 1) printf("note: --qat-every is ignored without --qat\n");
    if (o.qat > 0 && o.lat_alt && o.LW2 > 0) printf("note: --lat-alt has no effect with --qat (level 0 is never ES-perturbed; every pair perturbs level 1)\n");
    if (o.mlp_full_pairs < 0) o.mlp_full_pairs = o.mlp_pairs;
    if (o.mlp_pairs < 1 || o.mlp_full_pairs < 1 || o.lat_pairs < 1) { printf("pair counts must be >= 1\n"); return 1; }
    if (o.lat_alt && o.LW2 == 0) printf("note: --lat-alt has no effect with a single latent level\n");
    if (o.mlp_full_pairs != o.mlp_pairs && o.mlp_full_start >= 1.0f) printf("note: --mlp-full-pairs is ignored without --mlp-full\n");
    if (o.mlp_fd_h_set && o.mlp_fd_start >= 1.0f) printf("note: --mlp-fd-h is ignored without --mlp-fd\n");
    if (o.mlp_fd_start < 1.0f && o.mlp_freeze_start <= o.mlp_fd_start) printf("note: --mlp-fd never runs, --mlp-freeze starts first\n");
    if (o.mlp_full_start < 1.0f && (o.mlp_freeze_start <= o.mlp_full_start || o.mlp_fd_start <= o.mlp_full_start)) printf("note: --mlp-full is shadowed by --mlp-freeze or --mlp-fd\n");
    for (int h : o.hidden) if (h < 1 || h > MAXH) { printf("hidden width must be 1..%d\n", MAXH); return 1; }
#ifdef _OPENMP
    if (o.threads > 0) omp_set_num_threads(o.threads);
    printf("OpenMP threads: %d\n", omp_get_max_threads());
#endif

#ifdef _WIN32
    _mkdir(o.outdir.c_str());
#else
    mkdir(o.outdir.c_str(), 0755);
#endif

    // Output file naming: unchanged for one texture, "_tK" suffix per texture otherwise.
    auto tex_name = [&](const char* stem, int t, int it = -1) {
        char name[96];
        if (it >= 0) snprintf(name, sizeof(name), "%s_%06d", stem, it); else snprintf(name, sizeof(name), "%s", stem);
        std::string sname = o.outdir + "/" + name;
        if (T > 1) sname += "_t" + std::to_string(t);
        return sname + ".png";
    };

    // Load every texture (same crop), require identical sizes, interleave into one target.
    Image target;
    {
        std::vector<Image> tex(T);
        for (int t = 0; t < T; t++) {
            if (!find_and_load(argv[0], o.inputs[t], tex[t], o.crop)) {
                if (default_input) {   // nothing named on the command line: keep the synthetic fallback
                    int nsyn = o.crop ? o.crop : 512;
                    printf("could not load %s, using synthetic %dx%d image\n", o.inputs[t].c_str(), nsyn, nsyn);
                    make_synthetic(tex[t], nsyn);
                } else { printf("could not load %s\n", o.inputs[t].c_str()); return 1; }
            }
            if (t > 0 && (tex[t].w != tex[0].w || tex[t].h != tex[0].h)) {
                printf("%s is %dx%d after cropping but %s is %dx%d; all textures of a material must have the same size\n",
                    o.inputs[t].c_str(), tex[t].w, tex[t].h, o.inputs[0].c_str(), tex[0].w, tex[0].h);
                return 1;
            }
        }
        pack_textures(tex, target);
    }
    for (int t = 0; t < T; t++) save_png(tex_name("target", t), target, t);

    std::mt19937 rng(o.seed);
    Decoder D;
    D.opt = &o;
    if (!D.pos.parse(o.pos)) { printf("bad --pos spec: %s\n", o.pos.c_str()); return 1; }
    D.W = target.w; D.H = target.h;
    // Per-level sampling filter: "bilinear" or "nearest", comma-separated; a single entry applies to level 0 only.
    bool near0 = false, near1 = false;
    {
        std::vector<std::string> fl;
        for (size_t a = 0; a <= o.filter.size();) {
            size_t e = o.filter.find(',', a); if (e == std::string::npos) e = o.filter.size();
            fl.push_back(o.filter.substr(a, e - a)); a = e + 1;
        }
        if (fl.size() > 2) { printf("--filter takes at most two entries (one per latent level)\n"); return 1; }
        for (size_t l = 0; l < fl.size(); l++) {
            bool nr;
            if (fl[l] == "bilinear") nr = false; else if (fl[l] == "nearest") nr = true;
            else { printf("--filter: unknown mode '%s' (bilinear | nearest)\n", fl[l].c_str()); return 1; }
            if (l == 0) near0 = nr; else near1 = nr;
        }
        if (fl.size() > 1 && o.LW2 == 0) printf("note: second --filter entry ignored without --latent2\n");
    }
    if (o.qat > 0 && !near0) { printf("--qat needs nearest sampling on level 0 (--filter nearest): with bilinear a texel is read by pixels of 4 cells, so the per-texel search would not be exact\n"); return 1; }
    D.lat.add(o.LW, o.LH, o.LC, near0);
    D.qat_bits = o.qat;
    D.qat_ch = o.qat_ch;
    if (o.qat > 0) qat_init_grid();
    if (o.LW2 > 0) D.lat.add(o.LW2, o.LH2, o.LC2, near1);
    if (D.pos.uses_level1() && o.LW2 == 0) { printf("--pos uses lv1* features but there is no --latent2\n"); return 1; }
    D.deblock = o.deblock;
    D.deblock_falloff = o.deblock_falloff;
    D.bw = std::max(1, D.W / o.LW); D.bh = std::max(1, D.H / o.LH);
    if (o.deblock && (D.W % o.LW != 0 || D.H % o.LH != 0)) { printf("--deblock needs the image size to be a multiple of the level-0 latent size\n"); return 1; }
    if (o.deblock && !(o.deblock_falloff > 0.0f)) { printf("--deblock-falloff must be > 0\n"); return 1; }
    if (o.deblock && (D.bw < 2 || D.bh < 2)) { printf("--deblock needs level-0 cells of at least 2x2 pixels\n"); return 1; }
    {
        std::normal_distribution<float> N(0.0f, o.lat_init);
        for (auto& z : D.lat.z) z = N(rng);
        // --qat: snap level 0 onto its grid. The Gaussian draw is kept (same RNG stream as
        // without --qat, so the MLP init is identical) and rounded to the nearest grid value.
        if (o.qat > 0) qat_snap_level0(D.lat, o.qat_ch);
    }
    D.pos.cellw = std::max(1, D.W / o.LW); D.pos.cellh = std::max(1, D.H / o.LH);   // for the onehot kind
    if (D.pos.uses_bc7part() && (D.pos.cellw != 4 || D.pos.cellh != 4)) { printf("--pos bc7part needs 4x4 level-0 cells (got %dx%d)\n", D.pos.cellw, D.pos.cellh); return 1; }
    D.mlp.nout = 3 * T;   // init() sizes the output layer from nout; must precede it
    if (D.nin() > MAXH) { printf("too many MLP inputs\n"); return 1; }
    if (D.mlp.nout > MAXOUT) { printf("too many MLP outputs\n"); return 1; }   // implied by T <= MAXT; belt and braces for the out[MAXOUT] buffers
    D.mlp.init(D.nin(), o.hidden, act, rng);
    D.mlp.leak = o.leak;
    D.wsum = wsum;
    D.cw.resize(D.mlp.nout);
    for (int c = 0; c < D.mlp.nout; c++) D.cw[c] = o.weights[c / 3];
    if (!o.load.empty()) {
        FILE* f = fopen(o.load.c_str(), "rb");
        // Accepted layouts: v10 (magic 0x4E54433A, v9 plus one int of --qat bits per level-0 channel),
        // v9 (magic 0x4E544339, v8 plus an int of --qat bits, 0 = off),
        // v8 (magic 0x4E544338, v7 plus the leaky-ReLU slope float),
        // v7 (magic 0x4E544337, v6 plus a deblock int and a falloff float after the filters),
        // v6 (magic 0x4E544336, v5 plus one filter int per level after textures),
        // v5 (magic 0x4E544335, v4 plus a textures int after nlevels; MLP has 3*textures outputs),
        // v4 (magic 0x4E544334, 9 ints incl. nlevels, extra-level dims, widths, spec, latent, mlp),
        // v3 (magic 0x4E544333, 8 ints, widths, spec, latent, mlp), v2 (magic 0x4E544332, nfreq instead of spec),
        // legacy (six ints LW LH LC nin nh nfreq, no magic; two leaky layers of width nh).
        int hdr[8];
        if (!f || fread(hdr, sizeof(int), 6, f) != 6) { printf("cannot read %s\n", o.load.c_str()); return 1; }
        std::vector<int> saved_hidden;
        std::string saved_pos;
        std::vector<Latent> saved_lv;   // W,H,C of each saved level
        bool hdr_ok;
        auto nfreq_spec = [](int n) { return n > 0 ? "uv,fourier:" + std::to_string(n) : std::string("uv"); };
        int saved_T = 1;
        std::vector<int> saved_filter;   // per level, 0 bilinear / 1 nearest (v6+; older files are bilinear)
        int saved_deblock = 0; float saved_falloff = 1.0f, saved_leak = 0.01f;
        int saved_qat = 0;   // v9+; older files are continuous (0)
        std::vector<int> saved_qat_ch;   // v10+: per channel; v9: saved_qat for every channel
        if (hdr[0] == 0x4E54433A || hdr[0] == 0x4E544339 || hdr[0] == 0x4E544338 || hdr[0] == 0x4E544337 || hdr[0] == 0x4E544336 || hdr[0] == 0x4E544335 || hdr[0] == 0x4E544334 || hdr[0] == 0x4E544333 || hdr[0] == 0x4E544332) {
            bool v10 = hdr[0] == 0x4E54433A;
            bool v9 = v10 || hdr[0] == 0x4E544339;
            bool v8 = v9 || hdr[0] == 0x4E544338;
            bool v7 = v8 || hdr[0] == 0x4E544337;
            bool v6 = v7 || hdr[0] == 0x4E544336;
            bool v5 = v6 || hdr[0] == 0x4E544335;
            bool v4 = v5 || hdr[0] == 0x4E544334;
            bool v3 = v4 || hdr[0] == 0x4E544333;
            hdr_ok = fread(hdr + 6, sizeof(int), 2, f) == 2 && hdr[7] >= 1 && hdr[7] <= MAXL;
            int nlevels = 1;
            if (hdr_ok && v4) hdr_ok = fread(&nlevels, sizeof(int), 1, f) == 1 && nlevels >= 1 && nlevels <= 2;
            if (hdr_ok && v5) hdr_ok = fread(&saved_T, sizeof(int), 1, f) == 1 && saved_T >= 1 && saved_T <= MAXT;
            saved_filter.assign(nlevels, 0);
            if (hdr_ok && v6) for (int l = 0; l < nlevels && hdr_ok; l++) hdr_ok = fread(&saved_filter[l], sizeof(int), 1, f) == 1 && (saved_filter[l] == 0 || saved_filter[l] == 1);
            if (hdr_ok && v7) hdr_ok = fread(&saved_deblock, sizeof(int), 1, f) == 1 && fread(&saved_falloff, sizeof(float), 1, f) == 1;
            if (hdr_ok && v8) hdr_ok = fread(&saved_leak, sizeof(float), 1, f) == 1;
            if (hdr_ok && v9) hdr_ok = fread(&saved_qat, sizeof(int), 1, f) == 1 && saved_qat >= 0 && saved_qat <= 8;
            if (hdr_ok) hdr_ok = hdr[3] >= 1 && hdr[3] <= MAXH;   // level-0 channel count bounds the v10 per-channel list
            saved_qat_ch.assign(hdr_ok ? hdr[3] : 0, saved_qat);
            if (hdr_ok && v10) {
                int mx = 0;
                for (size_t c = 0; c < saved_qat_ch.size() && hdr_ok; c++) { hdr_ok = fread(&saved_qat_ch[c], sizeof(int), 1, f) == 1 && saved_qat_ch[c] >= 0 && saved_qat_ch[c] <= 8; mx = std::max(mx, saved_qat_ch[c]); }
                if (hdr_ok && mx != saved_qat) hdr_ok = false;   // the max field and the per-channel list must agree
            }
            if (hdr_ok) {
                saved_lv.resize(nlevels);
                saved_lv[0].W = hdr[1]; saved_lv[0].H = hdr[2]; saved_lv[0].C = hdr[3];
                for (int l = 1; l < nlevels && hdr_ok; l++) {
                    int d[3] = { 0, 0, 0 };
                    hdr_ok = fread(d, sizeof(int), 3, f) == 3;
                    if (hdr_ok) { saved_lv[l].W = d[0]; saved_lv[l].H = d[1]; saved_lv[l].C = d[2]; }
                }
            }
            if (hdr_ok) {
                saved_hidden.resize(hdr[7]);
                hdr_ok = fread(saved_hidden.data(), sizeof(int), saved_hidden.size(), f) == saved_hidden.size();
            }
            if (hdr_ok) {
                if (v3) {
                    // hdr[5] is the positional spec length, followed by the spec text.
                    hdr_ok = hdr[5] >= 0 && hdr[5] < 4096;
                    if (hdr_ok) {
                        saved_pos.resize(hdr[5]);
                        hdr_ok = fread(&saved_pos[0], 1, saved_pos.size(), f) == saved_pos.size();
                    }
                } else {
                    saved_pos = nfreq_spec(hdr[5]);   // v2 stored nfreq
                }
            }
        } else {
            // Legacy header: LW, LH, LC, nin, nh, nfreq; always two leaky layers of width nh.
            saved_hidden = { hdr[4], hdr[4] };
            saved_pos = nfreq_spec(hdr[5]);
            int lw = hdr[0], lh = hdr[1], lc = hdr[2], nin = hdr[3];
            hdr[1] = lw; hdr[2] = lh; hdr[3] = lc; hdr[4] = nin; hdr[6] = ACT_LEAKY; hdr[7] = 2;
            saved_lv.resize(1);
            saved_lv[0].W = lw; saved_lv[0].H = lh; saved_lv[0].C = lc;
            saved_filter.assign(1, 0);
            hdr_ok = true;
        }
        bool dims_ok = hdr_ok && saved_lv.size() == D.lat.lv.size();
        for (size_t l = 0; dims_ok && l < saved_lv.size(); l++)
            dims_ok = saved_lv[l].W == D.lat.lv[l].W && saved_lv[l].H == D.lat.lv[l].H && saved_lv[l].C == D.lat.lv[l].C
                   && (saved_filter[l] == 1) == D.lat.lv[l].nearest;
        bool deblock_ok = (saved_deblock == 1) == D.deblock && (!D.deblock || saved_falloff == D.deblock_falloff);
        bool leak_ok = (act != ACT_LEAKY) || saved_leak == o.leak;
        bool qat_ok = (saved_qat == 0 && o.qat == 0) || (saved_qat == 0 && o.qat > 0)   // continuous -> --qat warm start is allowed (snapped below)
                   || (saved_qat > 0 && o.qat > 0 && saved_qat_ch == o.qat_ch);
        if (!hdr_ok || !dims_ok || !deblock_ok || !leak_ok || !qat_ok || hdr[4] != D.mlp.nin || saved_T != T
            || saved_pos != D.pos.spec || hdr[6] != (int)act || saved_hidden != o.hidden) {
            std::string mlp, l2;
            for (size_t k = 0; k < saved_hidden.size(); k++) mlp += (k ? "," : "") + std::to_string(saved_hidden[k]);
            if (saved_lv.size() > 1) l2 = " --latent2 " + std::to_string(saved_lv[1].W) + " " + std::to_string(saved_lv[1].H) + " " + std::to_string(saved_lv[1].C);
            else if (D.lat.lv.size() > 1) l2 = " (no --latent2)";
            std::string flt;
            for (size_t l = 0; l < saved_filter.size(); l++) flt += (l ? "," : "") + std::string(saved_filter[l] ? "nearest" : "bilinear");
            std::string dbs = saved_deblock ? " --deblock --deblock-falloff " + std::to_string(saved_falloff) : std::string(" (no --deblock)");
            if (hdr[6] == ACT_LEAKY && saved_leak != 0.01f) dbs += " --leak " + std::to_string(saved_leak);
            if (saved_qat > 0) dbs += " --qat " + qat_spec(saved_qat_ch); else if (o.qat > 0) dbs += " (no --qat)";
            printf("%s was saved with --latent %d %d %d%s --filter %s%s --mlp %s --act %s --pos %s and %d texture(s); pass the same options\n",
                o.load.c_str(), hdr[1], hdr[2], hdr[3], l2.c_str(), flt.c_str(), dbs.c_str(), mlp.c_str(), act_name((Act)std::max(0, std::min(3, hdr[6]))), saved_pos.c_str(), saved_T);
            fclose(f);
            return 1;
        }
        bool ok = fread(D.lat.z.data(), sizeof(float), D.lat.z.size(), f) == D.lat.z.size()
               && fread(D.mlp.p.data(), sizeof(float), D.mlp.p.size(), f) == D.mlp.p.size();
        fclose(f);
        if (!ok) { printf("truncated %s\n", o.load.c_str()); return 1; }
        if (o.qat > 0) {   // a --qat file is already on-grid; a continuous file (warm start) is snapped here
            size_t moved = qat_snap_level0(D.lat, o.qat_ch);
            if (moved) printf("note: %zu level-0 values were off the --qat %s grid and were snapped (warm start from a continuous model)\n", moved, qat_spec(o.qat_ch).c_str());
        }
        printf("loaded   : %s\n", o.load.c_str());
    }

#ifndef NTC_CUDA
    if (o.cuda) { printf("this build has no CUDA backend (configure with -DNTC_CUDA=ON, or the 'cuda' CMake preset)\n"); return 1; }
#else
    ntc_cuda::Trainer cu;
    if (o.cuda) {
        ntc_cuda::ModelDesc md;
        md.W = D.W; md.H = D.H; md.T = T;
        md.nlev = (int)D.lat.lv.size();
        for (int l = 0; l < md.nlev; l++) { md.lv[l].W = D.lat.lv[l].W; md.lv[l].H = D.lat.lv[l].H; md.lv[l].C = D.lat.lv[l].C; md.lv[l].off = D.lat.lv[l].off; md.lv[l].nearest = D.lat.lv[l].nearest; }
        if (D.pos.feats.size() > (size_t)ntc_cuda::MAX_POS) { printf("--cuda: too many positional features\n"); return 1; }
        md.npos = (int)D.pos.feats.size();
        for (int q = 0; q < md.npos; q++) { md.pos[q].kind = (int)D.pos.feats[q].kind; md.pos[q].n = D.pos.feats[q].n; }
        md.cellw = D.pos.cellw; md.cellh = D.pos.cellh;
        md.nin = D.mlp.nin; md.nout = D.mlp.nout;
        if (o.hidden.size() > (size_t)ntc_cuda::MAX_HIDDEN) { printf("--cuda: too many hidden layers\n"); return 1; }
        md.nhidden = (int)o.hidden.size(); for (int l = 0; l < md.nhidden; l++) md.hidden[l] = o.hidden[l];
        md.act = (int)D.mlp.act; md.leak = D.mlp.leak; md.clamp_out = o.clamp_out;
        for (int c = 0; c < ntc_cuda::MAX_OUT; c++) md.cw[c] = c < D.mlp.nout ? D.cw[c] : 0.0f;
        md.wsum = D.wsum;
        for (int c = 0; c < ntc_cuda::MAX_CH; c++) md.qat_bits[c] = (o.qat > 0 && c < (int)o.qat_ch.size()) ? o.qat_ch[c] : 0;
        md.qat = o.qat;
        md.qat_grid = &QAT_GRID[0][0];
        md.max_pairs = std::max(o.mlp_pairs, o.mlp_full_pairs);
        md.max_batch = o.mlp_batch;
        md.seed = o.seed;
        if (o.deblock) { printf("--cuda v1 does not support --deblock\n"); return 1; }
        std::string why;
        if (!cu.init(md, D.lat.z.data(), D.mlp.p.data(), target.rgb.data(), why)) { printf("--cuda: %s\n", why.c_str()); return 1; }
        if (o.threads) printf("note: --threads is ignored with --cuda\n");
    }
    if (o.cuda && o.cuda_check) return cuda_check(D, target, o, cu);
#endif
    const double bits_latent = D.lat.size() * 32.0;
    const double bits_mlp = D.mlp.size() * 32.0;
    if (T == 1) printf("image    : %dx%d (%s)\n", D.W, D.H, o.inputs[0].c_str());
    else {
        printf("images   : %dx%d x%d textures (", D.W, D.H, T);
        for (int t = 0; t < T; t++) printf("%s%s", t ? ", " : "", o.inputs[t].c_str());
        printf(")\nweights  :");
        for (int t = 0; t < T; t++) printf(" %g", o.weights[t]);
        printf("   (loss = sum_t w_t mse_t / sum_t w_t; mse/psnr/best are on that loss; tex lists are unweighted per-texture psnr)\n");
    }
    printf("latent   : %dx%dx%d = %zu floats, %s\n", o.LW, o.LH, o.LC, D.lat.lv[0].size(), near0 ? "nearest (cells are blocks)" : "bilinear");
    if (D.lat.lv.size() > 1)
        printf("latent2  : %dx%dx%d = %zu floats (both levels %zu), %s\n", o.LW2, o.LH2, o.LC2, D.lat.lv[1].size(), D.lat.size(), near1 ? "nearest" : "bilinear");
    printf("mlp      : %s = %zu params, %s hidden%s, %s output\n", D.mlp.describe().c_str(), D.mlp.size(), act_name(D.mlp.act),
        (D.mlp.act == ACT_LEAKY && o.leak != 0.01f) ? (" (leak " + std::to_string(o.leak) + ")").c_str() : "", o.clamp_out ? "clamp" : "sigmoid");
    printf("inputs   : %d latent + %d positional (%s)\n", D.lat.channels(), D.pos.count(), D.pos.spec.c_str());
#ifdef NTC_CUDA
    if (o.cuda) printf("cuda     : %s\n", cu.banner().c_str());
#endif
    // Raw quantized size: every level at --qbits with a 2-float min/max per channel, except a
    // --qat level 0, which is charged its own bit count and needs no scale.
    double bits_q = D.mlp.size() * 16.0;
    for (size_t l = 0; l < D.lat.lv.size(); l++) {
        const bool fixed = (l == 0 && o.qat > 0);
        if (fixed) { for (int c = 0; c < D.lat.lv[l].C; c++) bits_q += (double)D.lat.lv[l].W * D.lat.lv[l].H * o.qat_ch[c]; }
        else bits_q += D.lat.lv[l].size() * (double)o.qbits + D.lat.lv[l].C * 64.0;
    }
    printf("bitrate  : fp32 %.3f bpp (latent %.3f + mlp %.3f); %d-bit latent + fp16 mlp %.3f bpp raw%s\n",
        (bits_latent + bits_mlp) / (D.W * D.H), bits_latent / (D.W * D.H), bits_mlp / (D.W * D.H),
        o.qbits, bits_q / (D.W * D.H), o.qat > 0 ? " (level 0 at its --qat depth)" : "");
    if (T > 1)
        printf("bpp/tex  : fp32 %.3f bpp, %d-bit latent + fp16 mlp %.3f bpp raw (shared bits / %d textures)\n",
            (bits_latent + bits_mlp) / ((double)D.W * D.H * T), o.qbits, bits_q / ((double)D.W * D.H * T), T);
    printf("mlp ES   : %d pairs, batch %d, sigma %g, lr %g, every %d%s\n", o.mlp_pairs, o.mlp_batch, o.mlp_sigma, o.mlp_lr, o.mlp_every, o.rng_hash ? " (hash noise)" : "");
    // Phase start iterations: first it with it/iters >= start (start >= 1 -> never).
    auto phase_iter = [&](float start) { return start >= 1.0f ? o.iters + 1 : (int)std::ceil((double)start * o.iters); };
    const int fd_from = phase_iter(o.mlp_fd_start), full_from = phase_iter(o.mlp_full_start), freeze_from = phase_iter(o.mlp_freeze_start);
    if (fd_from <= o.iters)
        printf("mlp FD   : from iteration %d, h = %g (%zu weights -> %zu evals per step, ~%.0f decode-equivalents on the minibatch)\n",
            fd_from, o.mlp_fd_h, D.mlp.size(), 2 * D.mlp.size(), 2.0 * D.mlp.size() * o.mlp_batch / (D.W * D.H));
    if (full_from <= o.iters)
        printf("mlp full : from iteration %d, ES on all %d pixels with %d pairs per step (%d decodes per step)\n",
            full_from, D.W * D.H, o.mlp_full_pairs, 2 * o.mlp_full_pairs);
    if (freeze_from <= o.iters)
        printf("mlp frozen from iteration %d (latent-only phase)\n", freeze_from);
    if (o.lat_alt && D.lat.lv.size() > 1 && o.qat == 0)
        printf("latent   : --lat-alt, one level perturbed per pair, rotating across steps\n");
    if (o.deblock)
        printf("deblock  : in-loop 5-tap cross at %dx%d block edges, falloff %g px (boundary ring only for falloff <= 1.5)\n", D.bw, D.bh, o.deblock_falloff);
    if (o.qat > 0)
        printf("qat      : level 0 quantized in-loop to %s bits per channel, exhaustive per-texel search every %d iters (%d image decodes)\n",
            qat_spec(o.qat_ch).c_str(), o.qat_every, qat_decodes(o.qat_ch));
    if (!(o.qat > 0 && D.lat.lv.size() == 1))
        printf("latent ES: %d pairs (full image), sigma %g, lr %g%s\n", o.lat_pairs, o.lat_sigma, o.lat_lr, o.qat > 0 ? " (level 1 only)" : "");
    if (D.lat.lv.size() > 1)
        printf("latent2 ES: sigma %g (lr shared with level 0)\n", o.lat2_sigma);
    if (o.lr_anneal_start < 1.0f)
        printf("anneal   : learning rates x1 -> x%g from iteration %d to the end\n", o.lr_anneal_final, (int)std::ceil((double)o.lr_anneal_start * o.iters));
    if (o.sigma_anneal_start < 1.0f)
        printf("anneal   : sigmas x1 -> x%g from iteration %d to the end\n", o.sigma_anneal_final, (int)std::ceil((double)o.sigma_anneal_start * o.iters));
    fflush(stdout);

    MlpTrainer mt; mt.init(D.mlp.size());
    LatentTrainer lt; lt.init(D.lat.size());

    Image recon, recon_q;
    std::vector<float> zq;
    // Full decode with the current weights and a latent (the model's own or e.g. the
    // post-hoc quantized copy); on the GPU the model's latent is the device-resident one.
    auto decode_with = [&](const float* z, Image& img) {
#ifdef NTC_CUDA
        if (o.cuda) {
            img.w = D.W; img.h = D.H; img.nc = D.mlp.nout; img.rgb.resize((size_t)D.W * D.H * img.nc);
            cu.decode_full(z == D.lat.z.data() ? nullptr : z, img.rgb.data());
            return;
        }
#endif
        D.decode_full(D.mlp.p.data(), z, img);
    };
    auto t0 = std::chrono::steady_clock::now();
    auto elapsed = [&]() { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); };

    decode_with(D.lat.z.data(), recon);
    Mse m = mse_of(target, recon, o.weights, wsum), mq;
    double mse = m.weighted;
    printf("iter %6d  mse %.6f  psnr %6.2f dB  (initial)", 0, mse, psnr_of(mse));
    if (T > 1) { printf(" | tex"); for (int t = 0; t < T; t++) printf(" %.2f", psnr_of(m.tex[t])); }
    printf("\n");
    for (int t = 0; t < T; t++) save_png(tex_name("recon", t, 0), recon, t);
    save_latent_png(o.outdir + "/latent_000000.png", D.lat.lv[0], D.lat.level(0));
    if (D.lat.lv.size() > 1) save_latent_png(o.outdir + "/latent2_000000.png", D.lat.lv[1], D.lat.level(1));
    fflush(stdout);

    double best_psnr = -1;
    double batch_loss = 0, diff_std = 0;
    for (int it = 1; it <= o.iters; it++) {
        // Per-iteration copy of the options with annealed learning rates / sigmas.
        auto anneal = [&](float start, float fin) {
            float t = (float)it / (float)o.iters;
            if (t <= start || start >= 1.0f) return 1.0f;
            float u = (t - start) / (1.0f - start);
            return 1.0f + (fin - 1.0f) * std::min(1.0f, u);
        };
        Options oi = o;
        float lrm = anneal(o.lr_anneal_start, o.lr_anneal_final);
        float sm = anneal(o.sigma_anneal_start, o.sigma_anneal_final);
        oi.mlp_lr *= lrm; oi.lat_lr *= lrm;
        oi.mlp_sigma *= sm; oi.lat_sigma *= sm;
        oi.lat2_sigma *= sm;
        // Late-phase switches (precedence: freeze > fd > full). Start iterations computed once above.
        bool mlp_frozen = it >= freeze_from;
        bool mlp_fd = !mlp_frozen && it >= fd_from;
        bool mlp_full = it >= full_from;
        if (it == freeze_from && it <= o.iters) { printf("iter %6d  MLP frozen; latent-only from here\n", it); fflush(stdout); }
        if (it == fd_from && !mlp_frozen) {
            // Fresh optimizer state for a different gradient estimator: the ES phase's
            // second-moment estimate would otherwise throttle FD steps for ~1000 iterations.
            mt.adam.init(D.mlp.size());
#ifdef NTC_CUDA
            if (o.cuda) cu.reset_mlp_adam();
#endif
            printf("iter %6d  MLP switched to finite differences%s; Adam state reset\n", it, mlp_full ? " on the full image" : ""); fflush(stdout);
        }
        if (it == full_from && !mlp_frozen && !mlp_fd) { printf("iter %6d  MLP ES switched to the full image\n", it); fflush(stdout); }
        if (!mlp_frozen && it % o.mlp_every == 0) {
#ifdef NTC_CUDA
            if (o.cuda) {
                const int M = mlp_full ? D.W * D.H : o.mlp_batch;
                if (mlp_fd) cu.mlp_step_fd(it, oi.mlp_fd_h, oi.mlp_lr, M, mlp_full);
                else cu.mlp_step(it, oi.mlp_sigma, oi.mlp_lr, mlp_full ? o.mlp_full_pairs : o.mlp_pairs, M, mlp_full);
            } else
#endif
            { mt.cur_it = it; batch_loss = mlp_fd ? mt.step_fd(D, target, rng, oi, diff_std, mlp_full) : mt.step(D, target, rng, oi, diff_std, mlp_full); }
        }
        const char* mlp_stat = mlp_frozen ? "frozen" : (mlp_fd ? "fd-rms" : "dstd");
        // --qat with a single level: nothing is ES-perturbed, so skip the (2K decodes) ES step entirely.
#ifdef NTC_CUDA
        if (o.cuda) {
            if (!(o.qat > 0 && D.lat.lv.size() == 1)) cu.lat_step(it, oi.lat_sigma, oi.lat2_sigma, oi.lat_lr, o.lat_pairs, o.lat_alt);
            if (o.qat > 0 && (it % o.qat_every == 0 || it == o.iters)) cu.qat_search();
        } else
#endif
        {
            if (!(o.qat > 0 && D.lat.lv.size() == 1)) lt.step(D, target, rng, oi);
            if (o.qat > 0 && (it % o.qat_every == 0 || it == o.iters)) qat_search(D, target, o.qat_ch);
        }

        bool do_print = (it % o.print_every == 0) || it == o.iters;
        bool do_save = (it % o.save_every == 0) || it == o.iters;
        if (do_print || do_save) {
#ifdef NTC_CUDA
            if (o.cuda) { cu.download_model(D.lat.z.data(), D.mlp.p.data()); if (!mlp_frozen) cu.mlp_stats(batch_loss, diff_std); }
#endif
            decode_with(D.lat.z.data(), recon);
            m = mse_of(target, recon, o.weights, wsum);
            mse = m.weighted;
            double ps = psnr_of(mse);
            best_psnr = std::max(best_psnr, ps);
            float lm, lsd, lmx; latent_stats(D.lat.lv[0], D.lat.level(0), lm, lsd, lmx);
            float l2m = 0, l2sd = 0, l2mx = 0;
            if (D.lat.lv.size() > 1) latent_stats(D.lat.lv[1], D.lat.level(1), l2m, l2sd, l2mx);
            // Bitrate and quality when the latent is actually quantized to 8 bits.
            BitrateStats bs = bitrate_stats(D.lat, D.mlp, D.W, D.H, o.qbits, o.qat_ch, zq);
            decode_with(zq.data(), recon_q);
            mq = mse_of(target, recon_q, o.weights, wsum);
            double ps_q = psnr_of(mq.weighted);
            double sec = elapsed();
            printf("iter %6d  mse %.6f  psnr %6.2f dB  best %6.2f", it, mse, ps, best_psnr);
            if (T > 1) { printf(" | tex"); for (int t = 0; t < T; t++) printf(" %.2f", psnr_of(m.tex[t])); }
            printf(" | q%d psnr %6.2f @ %.3f bpp (ent %.3f)", o.qbits, ps_q, bs.bpp_q8, bs.bpp_q8_ent);
            if (T > 1) { printf(" (%.3f/tex) qtex", bs.bpp_q8 / T); for (int t = 0; t < T; t++) printf(" %.2f", psnr_of(mq.tex[t])); }
            printf(" | mlp batch %.5f %s %.2e | lat mean %+.3f sd %.3f max %.2f", batch_loss, mlp_stat, diff_std, lm, lsd, lmx);
            if (D.lat.lv.size() > 1) printf(" | lat2 mean %+.3f sd %.3f max %.2f", l2m, l2sd, l2mx);
            if (o.qat > 0) {   // invariant: level 0 is always on its grid
                size_t off_grid = 0;
                for (size_t i = 0; i < D.lat.lv[0].size(); i++) if (qat_snap(D.lat.z[i], o.qat_ch[i % D.lat.lv[0].C]) != D.lat.z[i]) off_grid++;
                if (off_grid) printf(" | WARNING %zu level-0 values off-grid", off_grid);
            }
            printf(" | %.1fs (%.2f it/s)\n", sec, it / sec);
            fflush(stdout);
        }
        if (do_save) {
            char name[64];
            for (int t = 0; t < T; t++) save_png(tex_name("recon", t, it), recon, t);
            snprintf(name, sizeof(name), "/latent_%06d.png", it);
            save_latent_png(o.outdir + name, D.lat.lv[0], D.lat.level(0));
            if (D.lat.lv.size() > 1) {
                snprintf(name, sizeof(name), "/latent2_%06d.png", it);
                save_latent_png(o.outdir + name, D.lat.lv[1], D.lat.level(1));
            }
            save_model(o.outdir + "/model.bin", D);
        }
    }
    {
        BitrateStats bs = bitrate_stats(D.lat, D.mlp, D.W, D.H, o.qbits, o.qat_ch, zq);
        decode_with(zq.data(), recon_q);
        mq = mse_of(target, recon_q, o.weights, wsum);
        double ps_q = psnr_of(mq.weighted);
        for (int t = 0; t < T; t++) {
            save_png(tex_name("recon_q_final", t), recon_q, t);
            // Target | quantized-latent reconstruction, so the picture matches the quoted bitrate.
            Image a, b, sbs;
            slice_texture(target, t, a); slice_texture(recon_q, t, b);
            sbs.w = a.w * 2; sbs.h = a.h; sbs.nc = 3;
            sbs.rgb.resize((size_t)sbs.w * sbs.h * 3);
            for (int y = 0; y < a.h; y++) {
                memcpy(&sbs.rgb[((size_t)y * sbs.w) * 3], &a.rgb[((size_t)y * a.w) * 3], (size_t)a.w * 3 * sizeof(float));
                memcpy(&sbs.rgb[((size_t)y * sbs.w + a.w) * 3], &b.rgb[((size_t)y * a.w) * 3], (size_t)a.w * 3 * sizeof(float));
            }
            save_png(tex_name("side_by_side", t), sbs);
        }
        printf("done: final psnr %.2f dB (best %.2f)", psnr_of(mse), best_psnr);
        if (T > 1) { printf(" tex"); for (int t = 0; t < T; t++) printf(" %.2f", psnr_of(m.tex[t])); }
        printf(" at fp32 %.3f bpp", bs.bpp_fp32);
        if (T > 1) printf(" (%.3f/tex)", bs.bpp_fp32 / T);
        printf(" | %d-bit latent + fp16 mlp: psnr %.2f dB", o.qbits, ps_q);
        if (T > 1) { printf(" qtex"); for (int t = 0; t < T; t++) printf(" %.2f", psnr_of(mq.tex[t])); }
        printf(" at %.3f bpp raw, %.3f bpp entropy-coded", bs.bpp_q8, bs.bpp_q8_ent);
        if (T > 1) printf(" (%.3f, %.3f /tex)", bs.bpp_q8 / T, bs.bpp_q8_ent / T);
        printf(" | %.1fs\n", elapsed());
    }
    return 0;
}
