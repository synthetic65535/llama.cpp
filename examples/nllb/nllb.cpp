#include "../common/arg.h"
#include "../common/common.h"
#include "beam-search/beam-search.h"
#include "llama.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    common_params params;

    params.n_predict = 128;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    auto            llama_init = common_init_from_params(params);
    llama_model *   model      = llama_init->model();
    llama_context * ctx        = llama_init->context();

    if (model == NULL) {
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);

    // Find lang tokens
    llama_token src_lang_token = -1;
    llama_token tgt_lang_token = -1;
    for (int i = 0; i < llama_vocab_n_tokens(vocab); i++) {
        const char * text = llama_vocab_get_text(vocab, i);
        if (params.nllb_src_lang == text) {
            src_lang_token = i;
        }
        if (params.nllb_tgt_lang == text) {
            tgt_lang_token = i;
        }
        if (src_lang_token != -1 && tgt_lang_token != -1) {
            break;
        }
    }

    if (src_lang_token == -1) {
        fprintf(stderr, "error: source language %s not found\n", params.nllb_src_lang.c_str());
        return 1;
    }
    if (tgt_lang_token == -1) {
        fprintf(stderr, "error: target language %s not found\n", params.nllb_tgt_lang.c_str());
        return 1;
    }

    // Tokenize prompt
    std::vector<llama_token> tokens;
    tokens.push_back(src_lang_token);

    std::vector<llama_token> text_tokens = common_tokenize(vocab, params.prompt, false, true);
    tokens.insert(tokens.end(), text_tokens.begin(), text_tokens.end());
    tokens.push_back(llama_vocab_eos(vocab));

    // Encode
    llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());
    if (llama_encode(ctx, batch) != 0) {
        fprintf(stderr, "error: llama_encode failed\n");
        return 1;
    }

    // Decode initialization
    std::vector<llama_token> initial_tokens;
    initial_tokens.push_back(llama_vocab_eos(vocab));
    initial_tokens.push_back(tgt_lang_token);

    if (params.nllb_beam_size > 1) {
        // Beam search
        llama_beam::beam_search_params bparams;
        bparams.beam_size  = params.nllb_beam_size;
        bparams.max_length = params.n_predict;

        llama_beam::beam_search_engine engine(ctx, bparams);
        auto result = engine.search(initial_tokens, [vocab](llama_token t) { return llama_vocab_is_eog(vocab, t); });

        if (!result.hypotheses.empty()) {
            const auto & best = result.best();
            for (size_t i = 0; i < best.tokens.size(); i++) {
                char buf[128];
                int  n = llama_token_to_piece(vocab, best.tokens[i], buf, sizeof(buf), (i == 0 ? 1 : 0), true);
                if (n > 0) {
                    buf[n] = '\0';
                    printf("%s", buf);
                }
            }
            printf("\n");
        }
    } else {
        // Greedy search
        batch = llama_batch_get_one(initial_tokens.data(), initial_tokens.size());
        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "error: llama_decode failed at initial step\n");
            return 1;
        }

        llama_token new_token;
        for (int i = 0; i < params.n_predict; i++) {
            auto * logits  = llama_get_logits(ctx);
            int    n_vocab = llama_vocab_n_tokens(vocab);

            new_token       = 0;
            float max_logit = -1e10;
            for (int j = 0; j < n_vocab; j++) {
                if (logits[j] > max_logit) {
                    max_logit = logits[j];
                    new_token = j;
                }
            }

            if (llama_vocab_is_eog(vocab, new_token)) {
                break;
            }

            char buf[128];
            int  n = llama_token_to_piece(vocab, new_token, buf, sizeof(buf), (i == 0 ? 1 : 0), true);
            if (n > 0) {
                buf[n] = '\0';
                printf("%s", buf);
                fflush(stdout);
            }

            batch = llama_batch_get_one(&new_token, 1);
            if (llama_decode(ctx, batch) != 0) {
                fprintf(stderr, "error: llama_decode failed at step %d\n", i);
                break;
            }
        }
        printf("\n");
    }

    return 0;
}
