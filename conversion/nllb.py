from __future__ import annotations

import json
import os

from typing import Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, SentencePieceTokenTypes, TextModel, gguf, logger


@ModelBase.register("M2M100ForConditionalGeneration")
class NLLBModel(TextModel):
    """
    NLLB (No Language Left Behind) model converter.
    NLLB models use the M2M-100 encoder-decoder architecture.
    Supports: nllb-200-distilled-600M, nllb-200-distilled-1.3B, nllb-200-3.3B
    """
    model_arch = gguf.MODEL_ARCH.NLLB

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shared_token_embeddings_found = False

    def set_vocab(self):
        # NLLB uses SentencePiece tokenizer like T5
        os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
        from sentencepiece import SentencePieceProcessor
        from sentencepiece import sentencepiece_model_pb2 as model

        tokenizer_path = self.dir_model / 'tokenizer.model'
        if not tokenizer_path.is_file():
            tokenizer_path = self.dir_model / 'spiece.model'

        if not tokenizer_path.is_file():
            raise FileNotFoundError(f"File not found: {tokenizer_path}")

        sentencepiece_model = model.ModelProto()  # pyright: ignore[reportAttributeAccessIssue]
        sentencepiece_model.ParseFromString(open(tokenizer_path, "rb").read())

        add_prefix = sentencepiece_model.normalizer_spec.add_dummy_prefix
        remove_whitespaces = sentencepiece_model.normalizer_spec.remove_extra_whitespaces
        precompiled_charsmap = sentencepiece_model.normalizer_spec.precompiled_charsmap

        tokenizer = SentencePieceProcessor()
        tokenizer.LoadFromFile(str(tokenizer_path))

        vocab_size = self.hparams.get('vocab_size', tokenizer.vocab_size())

        tokens: list[bytes] = [f"[PAD{i}]".encode("utf-8") for i in range(vocab_size)]
        scores: list[float] = [-10000.0] * vocab_size
        toktypes: list[int] = [SentencePieceTokenTypes.UNUSED] * vocab_size

        for token_id in range(tokenizer.vocab_size()):
            piece = tokenizer.IdToPiece(token_id)
            text = piece.encode("utf-8")
            score = tokenizer.GetScore(token_id)

            toktype = SentencePieceTokenTypes.NORMAL
            if tokenizer.IsUnknown(token_id):
                toktype = SentencePieceTokenTypes.UNKNOWN
            elif tokenizer.IsControl(token_id):
                toktype = SentencePieceTokenTypes.CONTROL
            elif tokenizer.IsUnused(token_id):
                toktype = SentencePieceTokenTypes.UNUSED
            elif tokenizer.IsByte(token_id):
                toktype = SentencePieceTokenTypes.BYTE

            tokens[token_id] = text
            scores[token_id] = score
            toktypes[token_id] = toktype

        added_tokens_file = self.dir_model / 'added_tokens.json'
        if added_tokens_file.is_file():
            with open(added_tokens_file, "r", encoding="utf-8") as f:
                added_tokens_json = json.load(f)
                for key in added_tokens_json:
                    token_id = added_tokens_json[key]
                    if token_id >= vocab_size:
                        logger.warning(f'ignore token {token_id}: id is out of range, max={vocab_size - 1}')
                        continue
                    tokens[token_id] = key.encode("utf-8")
                    scores[token_id] = -1000.0
                    toktypes[token_id] = SentencePieceTokenTypes.USER_DEFINED

        self.gguf_writer.add_tokenizer_model("llama")  # Use llama tokenizer type
        self.gguf_writer.add_tokenizer_pre("default")
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_scores(scores)
        self.gguf_writer.add_token_types(toktypes)
        self.gguf_writer.add_add_space_prefix(add_prefix)
        self.gguf_writer.add_remove_extra_whitespaces(remove_whitespaces)
        if precompiled_charsmap:
            self.gguf_writer.add_precompiled_charsmap(precompiled_charsmap)

        special_vocab = gguf.SpecialVocab(self.dir_model, n_vocab=len(tokens))
        special_vocab.add_to_gguf(self.gguf_writer)

    def set_gguf_parameters(self):
        # NLLB uses max_position_embeddings for context length
        n_ctx = self.find_hparam(["max_position_embeddings"], optional=True)
        if n_ctx is None:
            logger.warning("Couldn't find max_position_embeddings in config.json, assuming 1024")
            n_ctx = 1024

        self.gguf_writer.add_context_length(n_ctx)
        self.gguf_writer.add_embedding_length(self.hparams["d_model"])
        self.gguf_writer.add_feed_forward_length(self.hparams["encoder_ffn_dim"])
        self.gguf_writer.add_block_count(self.block_count)

        # Add decoder block count if different from encoder
        if (dec_n_layer := self.hparams.get("decoder_layers")) is not None:
            self.gguf_writer.add_decoder_block_count(dec_n_layer)

        self.gguf_writer.add_head_count(self.hparams["encoder_attention_heads"])

        # NLLB uses standard attention (no separate d_kv like T5)
        # head_dim = d_model / num_heads
        head_dim = self.hparams["d_model"] // self.hparams["encoder_attention_heads"]
        self.gguf_writer.add_key_length(head_dim)
        self.gguf_writer.add_value_length(head_dim)

        # NLLB uses 1e-5 for layer norm epsilon
        layer_norm_eps = self.hparams.get("layer_norm_eps", 1e-5)
        self.gguf_writer.add_layer_norm_eps(layer_norm_eps)
        self.gguf_writer.add_layer_norm_rms_eps(layer_norm_eps)

        # Decoder start token
        self.gguf_writer.add_decoder_start_token_id(self.hparams.get("decoder_start_token_id", 2))
        self.gguf_writer.add_eos_token_id(self.hparams.get("eos_token_id", 2))
        self.gguf_writer.add_bos_token_id(self.hparams.get("bos_token_id", 0))
        self.gguf_writer.add_pad_token_id(self.hparams.get("pad_token_id", 1))

        self.gguf_writer.add_file_type(self.ftype)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        del bid  # unused

        # NLLB models share token embeddings between encoder and decoder
        # Handle "shared.weight", "encoder.embed_tokens.weight", or "decoder.embed_tokens.weight"
        if name in ["decoder.embed_tokens.weight", "encoder.embed_tokens.weight", "shared.weight"]:
            if not self.shared_token_embeddings_found:
                name = "shared.weight"
                self.shared_token_embeddings_found = True
            else:
                logger.debug(f"Skipping shared tensor {name!r} in safetensors so that convert can end normally.")
                return []

        return [(self.map_tensor_name(name), data_torch)]
