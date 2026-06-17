#include "models.h"

void llama_model_nllb::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);

    uint32_t dec_start_token_id;
    if (ml.get_key(LLM_KV_DECODER_START_TOKEN_ID, dec_start_token_id, false)) {
        hparams.dec_start_token_id = dec_start_token_id;
    }

    hparams.dec_n_layer = hparams.n_layer();
    ml.get_key(LLM_KV_DECODER_BLOCK_COUNT, hparams.dec_n_layer, false);

    switch (hparams.n_layer()) {
        case 12:
            switch (hparams.n_ff()) {
                case 2048: type = LLM_TYPE_700M; break; // nllb-200-distilled-600M (closest match)
                case 4096: type = LLM_TYPE_1_3B; break; // nllb-200-distilled-1.3B
                default:   type = LLM_TYPE_UNKNOWN;
            } break;
        case 24: type = LLM_TYPE_3B; break; // nllb-200-3.3B
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_nllb::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);
    // NLLB positional embeddings include M2M100 offset of +2
    pos_embd = create_tensor(tn(LLM_TENSOR_POS_EMBD, "weight"), {n_embd, hparams.n_ctx_train + 2}, 0);

    output_norm_enc   = create_tensor(tn(LLM_TENSOR_ENC_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output_norm_enc_b = create_tensor(tn(LLM_TENSOR_ENC_OUTPUT_NORM, "bias"),   {n_embd}, 0);
    output_norm       = create_tensor(tn(LLM_TENSOR_DEC_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output_norm_b     = create_tensor(tn(LLM_TENSOR_DEC_OUTPUT_NORM, "bias"),   {n_embd}, 0);
    output            = create_tensor(tn(LLM_TENSOR_OUTPUT,          "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);

    if (output == NULL) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
    }

    const int     dec_n_layer  = hparams.dec_n_layer;

    if (dec_n_layer > n_layer) {
        layers.resize(dec_n_layer);
    }

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm_enc   = create_tensor(tn(LLM_TENSOR_ENC_ATTN_NORM, "weight", i), {n_embd}, 0);
        layer.attn_norm_enc_b = create_tensor(tn(LLM_TENSOR_ENC_ATTN_NORM, "bias",   i), {n_embd}, 0);

        layer.wq_enc = create_tensor(tn(LLM_TENSOR_ENC_ATTN_Q,   "weight", i), {n_embd, n_embd_k_gqa}, 0);
        layer.bq_enc = create_tensor(tn(LLM_TENSOR_ENC_ATTN_Q,   "bias",   i), {n_embd_k_gqa}, 0);
        layer.wk_enc = create_tensor(tn(LLM_TENSOR_ENC_ATTN_K,   "weight", i), {n_embd, n_embd_k_gqa}, 0);
        layer.bk_enc = create_tensor(tn(LLM_TENSOR_ENC_ATTN_K,   "bias",   i), {n_embd_k_gqa}, 0);
        layer.wv_enc = create_tensor(tn(LLM_TENSOR_ENC_ATTN_V,   "weight", i), {n_embd, n_embd_v_gqa}, 0);
        layer.bv_enc = create_tensor(tn(LLM_TENSOR_ENC_ATTN_V,   "bias",   i), {n_embd_v_gqa}, 0);
        layer.wo_enc = create_tensor(tn(LLM_TENSOR_ENC_ATTN_OUT, "weight", i), {n_embd_v_gqa, n_embd}, 0);
        layer.bo_enc = create_tensor(tn(LLM_TENSOR_ENC_ATTN_OUT, "bias",   i), {n_embd}, 0);

        layer.ffn_norm_enc   = create_tensor(tn(LLM_TENSOR_ENC_FFN_NORM, "weight", i), {n_embd}, 0);
        layer.ffn_norm_enc_b = create_tensor(tn(LLM_TENSOR_ENC_FFN_NORM, "bias",   i), {n_embd}, 0);
        layer.ffn_down_enc   = create_tensor(tn(LLM_TENSOR_ENC_FFN_DOWN, "weight", i), {n_ff, n_embd}, 0);
        layer.ffn_down_enc_b = create_tensor(tn(LLM_TENSOR_ENC_FFN_DOWN, "bias",   i), {n_embd}, 0);
        layer.ffn_up_enc     = create_tensor(tn(LLM_TENSOR_ENC_FFN_UP,   "weight", i), {n_embd, n_ff}, 0);
        layer.ffn_up_enc_b   = create_tensor(tn(LLM_TENSOR_ENC_FFN_UP,   "bias",   i), {n_ff}, 0);
    }

    for (int i = 0; i < dec_n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm   = create_tensor(tn(LLM_TENSOR_DEC_ATTN_NORM, "weight", i), {n_embd}, 0);
        layer.attn_norm_b = create_tensor(tn(LLM_TENSOR_DEC_ATTN_NORM, "bias",   i), {n_embd}, 0);

        layer.wq = create_tensor(tn(LLM_TENSOR_DEC_ATTN_Q,   "weight", i), {n_embd, n_embd_k_gqa}, 0);
        layer.bq = create_tensor(tn(LLM_TENSOR_DEC_ATTN_Q,   "bias",   i), {n_embd_k_gqa}, 0);
        layer.wk = create_tensor(tn(LLM_TENSOR_DEC_ATTN_K,   "weight", i), {n_embd, n_embd_k_gqa}, 0);
        layer.bk = create_tensor(tn(LLM_TENSOR_DEC_ATTN_K,   "bias",   i), {n_embd_k_gqa}, 0);
        layer.wv = create_tensor(tn(LLM_TENSOR_DEC_ATTN_V,   "weight", i), {n_embd, n_embd_v_gqa}, 0);
        layer.bv = create_tensor(tn(LLM_TENSOR_DEC_ATTN_V,   "bias",   i), {n_embd_v_gqa}, 0);
        layer.wo = create_tensor(tn(LLM_TENSOR_DEC_ATTN_OUT, "weight", i), {n_embd_v_gqa, n_embd}, 0);
        layer.bo = create_tensor(tn(LLM_TENSOR_DEC_ATTN_OUT, "bias",   i), {n_embd}, 0);

        layer.attn_norm_cross   = create_tensor(tn(LLM_TENSOR_DEC_CROSS_ATTN_NORM, "weight", i), {n_embd}, 0);
        layer.attn_norm_cross_b = create_tensor(tn(LLM_TENSOR_DEC_CROSS_ATTN_NORM, "bias",   i), {n_embd}, 0);

        layer.wq_cross = create_tensor(tn(LLM_TENSOR_DEC_CROSS_ATTN_Q,   "weight", i), {n_embd, n_embd_k_gqa}, 0);
        layer.bq_cross = create_tensor(tn(LLM_TENSOR_DEC_CROSS_ATTN_Q,   "bias",   i), {n_embd_k_gqa}, 0);
        layer.wk_cross = create_tensor(tn(LLM_TENSOR_DEC_CROSS_ATTN_K,   "weight", i), {n_embd, n_embd_k_gqa}, 0);
        layer.bk_cross = create_tensor(tn(LLM_TENSOR_DEC_CROSS_ATTN_K,   "bias",   i), {n_embd_k_gqa}, 0);
        layer.wv_cross = create_tensor(tn(LLM_TENSOR_DEC_CROSS_ATTN_V,   "weight", i), {n_embd, n_embd_v_gqa}, 0);
        layer.bv_cross = create_tensor(tn(LLM_TENSOR_DEC_CROSS_ATTN_V,   "bias",   i), {n_embd_v_gqa}, 0);
        layer.wo_cross = create_tensor(tn(LLM_TENSOR_DEC_CROSS_ATTN_OUT, "weight", i), {n_embd_v_gqa, n_embd}, 0);
        layer.bo_cross = create_tensor(tn(LLM_TENSOR_DEC_CROSS_ATTN_OUT, "bias",   i), {n_embd}, 0);

        layer.ffn_norm   = create_tensor(tn(LLM_TENSOR_DEC_FFN_NORM, "weight", i), {n_embd}, 0);
        layer.ffn_norm_b = create_tensor(tn(LLM_TENSOR_DEC_FFN_NORM, "bias",   i), {n_embd}, 0);

        layer.ffn_down   = create_tensor(tn(LLM_TENSOR_DEC_FFN_DOWN, "weight", i), {n_ff, n_embd}, 0);
        layer.ffn_down_b = create_tensor(tn(LLM_TENSOR_DEC_FFN_DOWN, "bias",   i), {n_embd}, 0);
        layer.ffn_up     = create_tensor(tn(LLM_TENSOR_DEC_FFN_UP,   "weight", i), {n_embd, n_ff}, 0);
        layer.ffn_up_b   = create_tensor(tn(LLM_TENSOR_DEC_FFN_UP,   "bias",   i), {n_ff}, 0);
    }
}

std::unique_ptr<llm_graph_context> llama_model_nllb::build_arch_graph(const llm_graph_params & params) const {
    switch (params.gtype) {
        case LLM_GRAPH_TYPE_ENCODER:
            return std::make_unique<graph<true>>(*this, params);
        case LLM_GRAPH_TYPE_DEFAULT:
        case LLM_GRAPH_TYPE_DECODER:
            return std::make_unique<graph<false>>(*this, params);
        default:
            GGML_ABORT("invalid graph type");
    };
}
