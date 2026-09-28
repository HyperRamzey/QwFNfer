#pragma once
// Per-sequence state. Three different kinds, because the layers differ:
//
//   12 sparse-attention layers : K/V cache, grows with context
//                                @128K, q8_0: 1.71 GB   (f16: 3.22 GB)
//   12 indexer caches          : one 128-dim key per token, @128K f16: 0.40 GB
//   36 gated-DeltaNet layers   : a [128,128,48] recurrent state per layer,
//                                CONSTANT in context -- 113 MB f32 for all 36
//    + short conv histories for DeltaNet (k=4) and PLE (k=4, dilation=3)
//
// The constant-size recurrent state for 3/4 of the layers is why long context is
// cheap on this architecture: only 12 layers pay per-token KV.
//
// v1 scope: one sequence, no unified KV, no recurrent rollback. That matches the
// -np 1 configuration this engine targets anyway (unified KV + parallel requests
// is known to corrupt QSA block pooling upstream).

#include "qwfn_hparams.h"

#include <string>
#include <vector>

#include "ggml.h"
#include "ggml-backend.h"

namespace qwfn {

struct state_config {
    uint32_t  n_ctx   = 131072;
    ggml_type type_k  = GGML_TYPE_Q8_0;
    ggml_type type_v  = GGML_TYPE_Q8_0;
    ggml_type type_idx = GGML_TYPE_F16;   // indexer keys; pooled+normed before use
    // Keep the indexer cache and/or the KV cache in pinned host memory (read by
    // the device over PCIe) so their VRAM goes to the expert tier.
    bool      idx_host = false;
    bool      kv_host  = false;
};

// The K and V caches are separate tensors with separate types all the way down, and
// the graph builds each view from its own tensor's type, so a split is a spelling
// question rather than a new cache. "q8_0", "q4_0" and "f16" set both; "K/V" of those
// sets them apart, e.g. "q8_0/q4_0" for a K that wants the extra bit and a V that
// does not -- at 131K that is 1.71 GB against 1.71/0.91, and V is the larger share of
// the gather. Whether the attention kernel actually runs two types at once is a
// measured question, not an assumed one: see the note where it is used.
bool parse_kv_spec(const std::string & spec, ggml_type & type_k, ggml_type & type_v,
                   std::string & err);

class state {
public:
    ~state();
    state() = default;
    state(const state &) = delete;
    state & operator=(const state &) = delete;

    bool init(const hparams * hp, const state_config & cfg,
              ggml_backend_buffer_type_t buft, std::string & err);

    // Attention layers: il is the absolute layer index; these return null for
    // recurrent layers.
    ggml_tensor * k_cache(uint32_t il) const;
    ggml_tensor * v_cache(uint32_t il) const;
    ggml_tensor * idx_cache(uint32_t il) const;

    // Recurrent layers.
    ggml_tensor * rs_state(uint32_t il) const;   // [128,128,48]
    ggml_tensor * rs_conv(uint32_t il) const;    // [d_conv-1, conv_dim]
    ggml_tensor * ple_conv() const;              // [(k-1)*dilation, hc_dim]

    // Zero the recurrent and conv state. With a fixed input this makes every
    // forward pass identical, which is how the engine is checked for
    // tier-independence: the same token must give the same logits whether its
    // experts are served from CPU RAM or from VRAM.
    void reset();

    size_t bytes() const { return bytes_; }
    std::string summary() const;

    uint32_t n_ctx() const { return cfg_.n_ctx; }

private:
    const hparams * hp_ = nullptr;
    state_config    cfg_;

    ggml_context *        ctx_ = nullptr;
    ggml_backend_buffer_t buf_ = nullptr;
    // The KV cache and/or the indexer key cache in pinned host memory instead
    // (state_config::kv_host / idx_host); the device kernels read them over PCIe
    // and the VRAM they occupied goes to the expert tier.
    ggml_context *        ctx_h_ = nullptr;
    ggml_backend_buffer_t buf_h_ = nullptr;
    size_t                bytes_h_ = 0;
    bool                  kv_host_ = false, idx_host_ = false;

    std::vector<ggml_tensor *> k_, v_, idx_, rs_, conv_;
    ggml_tensor * ple_conv_ = nullptr;
    size_t bytes_ = 0;
};

} // namespace qwfn
