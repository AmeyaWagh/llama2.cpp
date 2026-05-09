# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## About

C++ rewrite of [@karpathy](https://github.com/karpathy/llama2.c)'s llama2.c. Implements the Llama 2 transformer architecture as a pure C++20 header-only library with a custom tensor class, targeting CPU inference (with hooks for future CUDA/AVX512).

## Build

**CMake (primary):**
```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make
```

**Makefile targets (legacy, for the original C `run.c`):**
```bash
make run       # basic
make runfast   # -Ofast
make runomp    # OpenMP
```

Build outputs:
- `build/run_llama` — inference binary (compiled with `-fsanitize=address`)
- `build/libllama2cpp_core.so` — shared library
- `build/test_tensor`, `build/test_tokenizer` — test binaries

**Run inference:**
```bash
./build/run_llama stories15M.bin
./build/run_llama stories15M.bin -t 0.8 -p 0.9 -n 256 -i "Once upon a time"
./build/run_llama stories15M.bin -m chat
```

## Tests

**C++ unit tests (gtest):**
```bash
cd build && ctest
# Or run individually:
./build/test_tensor
./build/test_tokenizer
```

**Python forward-pass comparison:**
```bash
pytest test_all.py
```

## Code Formatting

clang-format with Google style:
```bash
clang-format -i <file>
```

## Architecture

All C++ headers live under `include/llama2cpp/`. The library is split into a low-level tensor layer and a high-level inference layer.

### Tensor layer (`include/llama2cpp/transformer/`)

- **`types.hpp`** — `float32_t`, `float64_t` aliases
- **`memory.hpp`** — `CpuMemory<T>` handles raw allocation/deallocation; abstraction layer for future CUDA/AVX512 backends
- **`tensor.hpp`** — `Tensor<T, MemoryType>` — N-dimensional tensor with automatic stride calculation, zero-copy views/slices, and reshape. All model weights and activations are this type.
- **`ops.hpp`** — Standalone math ops: `matmul`, `rmsnorm`, `softmax`, `silu`, `hadamard`, `dot`. Most ops are OpenMP-parallelized.

### Inference layer (`include/llama2cpp/`)

- **`tokenizer.hpp`** — BPE tokenizer. Loads `.bin` vocab files. Encodes strings → token IDs; decodes token IDs → string pieces.
- **`sampler.hpp`** — Token sampling: greedy (temperature=0), multinomial, top-p (nucleus). XORshift PRNG for reproducibility.
- **`transformer/transformer.hpp`** — Core model:
  - `TransformerConfig` — hyperparameters (dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len)
  - `TransformerWeights` — all model parameters as tensors, loaded from binary checkpoint
  - `Attention` — multi-head attention with KV-cache and RoPE positional encoding; supports MQA (n_heads ≥ n_kv_heads)
  - `FeedForward` — SwiGLU: `w2(silu(w1(x)) * w3(x))`
  - `TransformerBlock` — attention + FFN with RMSNorm pre-norm and residuals
  - `Transformer` — stacks N blocks; handles weight loading from `.bin` checkpoints
- **`llama2.hpp`** — `Llama2` top-level class; `generate` mode (free text) and `chat` mode (dialogue); manages the full token generation loop
- **`utils.hpp`** — Miscellaneous helpers

### Entry point

`src/main.cpp` — CLI argument parsing; dispatches to `Llama2::generate` or `Llama2::chat`.

### Inference flow

1. Load checkpoint → `TransformerWeights` tensors
2. Load tokenizer `.bin` → vocab + merge rules
3. Encode prompt → token IDs
4. Per-step loop: embed → N × (Attention + FFN + residuals) → RMSNorm → linear classifier → sample next token → decode → print
5. Stop at EOS token or `--steps` limit

### Model checkpoint format

Binary `.bin` files: `TransformerConfig` header followed by raw float32 weight matrices in a fixed order (embeddings, per-layer norms, QKV projections, output projections, FFN weights, final norm + classifier).

## Open TODOs (from README)

- Quantized model support
- CUDA tensor backend
- AVX512 CPU backend
- Modularize RoPE
- Package tokenizer/sampler as standalone library
