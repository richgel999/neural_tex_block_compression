// Host-facing API of the CUDA backend. Plain C++: no CUDA types, safe to include from
// main.cpp. The backend owns the latent, the MLP weights and both Adam states on the
// device for the whole run; the host Decoder is a mirror refreshed by download_model()
// at print/save time so the unchanged host code (stats, bitrate, PNG and model I/O)
// keeps working. Every kernel is written against a small set of textures sharing one
// decoder format (v1 always runs one texture), see CUDA_PLAN_B.md.
#pragma once
#include <stdint.h>
#include <string>
#include <vector>

namespace ntc_cuda {

static const int MAX_LEVELS = 2;
static const int MAX_POS = 16;
static const int MAX_OUT = 12;      // = MAXOUT in main.cpp (3 channels x 4 textures)
static const int MAX_HIDDEN = 8;    // = MAXL
static const int MAX_CH = 16;       // max level-0 channels with per-channel --qat bits

// Positional feature kinds, in the same order as PosKind in main.cpp.
enum PosKindDev { PK_UV, PK_FOURIER, PK_LOCAL, PK_LFOURIER, PK_LQUAD, PK_DCT, PK_LDCT, PK_LDCT2, PK_LDCT4,
                  PK_LV1LOCAL, PK_LV1LDCT, PK_BDCT, PK_BDCTE, PK_ONEHOT, PK_BC7PART };

struct LevelDesc { int W = 0, H = 0, C = 0; size_t off = 0; bool nearest = false; };
struct PosDesc { int kind = 0, n = 0; };

// Everything the device needs to know about one model; filled by main.cpp.
struct ModelDesc {
    int W = 0, H = 0;            // image size
    int T = 1;                   // textures (nout = 3T)
    int nlev = 0; LevelDesc lv[MAX_LEVELS];
    int npos = 0; PosDesc pos[MAX_POS];
    int cellw = 1, cellh = 1;
    int nin = 0, nout = 3;
    int nhidden = 0; int hidden[MAX_HIDDEN];
    int act = 0;                 // 0 leaky, 1 relu, 2 tanh, 3 sine (Act in main.cpp)
    float leak = 0.01f;
    bool clamp_out = false;
    float cw[MAX_OUT];           // per-output-channel loss weight
    float wsum = 1.0f;
    int qat_bits[MAX_CH];        // per level-0 channel; all 0 when --qat is off
    int qat = 0;                 // largest qat bit depth (0 = off)
    const float* qat_grid = nullptr;   // the host's QAT_GRID[9][257]: copied verbatim so on-grid bit patterns match the CPU exactly
    int max_pairs = 64;          // largest N the MLP ES step will be called with (sizes the per-pair buffers)
    int max_batch = 4096;        // largest minibatch M (--mlp-batch); --mlp-full uses W*H, which is always covered
    uint64_t seed = 1;
};

class Trainer {
public:
    Trainer();
    ~Trainer();
    // Allocates device state and uploads the initial latent, weights and target
    // (interleaved W*H*nout floats). Returns false with a reason for anything the
    // v1 backend does not support (bilinear levels, deblocking, table-based positional
    // kinds, oversized MLPs).
    bool init(const ModelDesc& d, const float* z, const float* p, const float* target, std::string& why);
    std::string banner() const;              // device name / arch / toolkit line for the run header

    void upload_model(const float* z, const float* p);
    void download_model(float* z, float* p);

    // One antithetic ES step on the MLP (= MlpTrainer::step). N pairs, minibatch of M
    // pixels drawn on the device (or every pixel when full is set).
    void mlp_step(int it, float sigma, float lr, int N, int M, bool full);
    // One central-differences step (= MlpTrainer::step_fd).
    void mlp_step_fd(int it, float h, float lr, int M, bool full);
    void reset_mlp_adam();
    // Stats of the most recent MLP step: mean minibatch loss and dstd (ES) or fd-rms (FD).
    void mlp_stats(double& batch_loss, double& diff_std);

    // One latent ES step (= LatentTrainer::step): K pairs, per-level sigma, --lat-alt.
    void lat_step(int it, float sigma0, float sigma1, float lr, int K, bool lat_alt);
    // Exact per-texel search on the discrete level 0 (= qat_search).
    void qat_search();
    // Decode the whole image with the device latent (z_host == null) or a host-supplied
    // latent (e.g. the post-hoc quantized one) into img (W*H*nout floats).
    void decode_full(const float* z_host, float* img);

    // Verification helpers (--cuda-check): the minibatch indices and per-pair loss
    // differences of the last MLP ES step, and the per-weight differences of the last FD step.
    void debug_last_batch(std::vector<int>& bidx) const;
    void debug_last_dl(std::vector<double>& dl) const;
    void debug_last_zgrad(std::vector<float>& g) const;   // the latent ES gradient of the last lat_step (before Adam)

    struct Impl;   // device state (public so the file-local launch helpers can reach it)
    Impl* im;
};

} // namespace ntc_cuda
