# NLLB Support in llama.cpp

The [NLLB (No Language Left Behind)](https://github.com/facebookresearch/fairseq/tree/main/examples/nllb) model family is a collection of encoder-decoder models designed for high-quality machine translation across 200+ languages.

`llama.cpp` supports NLLB-200 models (distilled-600M, distilled-1.3B, and 3.3B) with both greedy and beam search decoding.

## Quick Start

1. **Obtain the model**: You can convert an NLLB model from Hugging Face to GGUF format.

   ```bash
   # Clone the model from Hugging Face
   git clone https://huggingface.co/facebook/nllb-200-distilled-600M

   # Convert to GGUF
   python convert_hf_to_gguf.py nllb-200-distilled-600M
   ```

2. **Build the example**:

   ```bash
   cmake -B build
   cmake --build build --target llama-nllb
   ```

3. **Run translation**:

   ```bash
   # English to French translation
   ./build/bin/llama-nllb -m nllb-200-distilled-600M/ggml-model-f16.gguf -p "Hello, how are you?" -sl eng_Latn -tl fra_Latn
   ```

## Command Line Arguments

The `llama-nllb` example supports the following NLLB-specific arguments:

- `-sl`, `--src-lang LANG`: Source language code (e.g., `eng_Latn`, `fra_Latn`, `spa_Latn`).
- `-tl`, `--tgt-lang LANG`: Target language code (e.g., `als_Latn`, `zho_Hans`, `deu_Latn`).
- `--beam-size N`: Beam size for beam search (default: 1, which means greedy decoding).

Standard `llama.cpp` arguments like `-m` (model path), `-n` (max tokens), and `-t` (threads) are also supported.

## Architecture

NLLB models use an encoder-decoder architecture:
1. **Encoder**: Processes the source text in the source language.
2. **Decoder**: Generates the translation in the target language, attending to the encoder's output.

The implementation in `llama.cpp` handles the complex tokenization requirements of NLLB, including the prefixing of language tokens and the specific EOS/BOS handling required for correct translation.

## Verification

Numerical verification scripts and a testing framework can be found in `tests/nllb/`. These scripts compare the `llama.cpp` implementation against the Hugging Face reference implementation to ensure functional equivalence.

```bash
# Run the integration test
python tests/nllb/test_nllb.py
```

## Supported Models

The following NLLB-200 variants are verified to work:
- [nllb-200-distilled-600M](https://huggingface.co/facebook/nllb-200-distilled-600M)
- [nllb-200-distilled-1.3B](https://huggingface.co/facebook/nllb-200-distilled-1.3B)
- [nllb-200-3.3B](https://huggingface.co/facebook/nllb-200-3.3B)
