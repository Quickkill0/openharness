# GGUF header fixtures

Header-only copies of real GGUF files (metadata + tensor-info table, no tensor
data), used by the planner's accounting and context-memory tests. Tokenizer arrays and oversized
strings were stripped to keep them small; every tensor record and all
architecture keys are intact.

| File | Source |
| --- | --- |
| `glm53/GLM-5.3-Flash-UD-Q2_K_XL-0000N-of-00004.gguf` | `unsloth/GLM-5.3-Flash-GGUF`, `UD-Q2_K_XL/` (4-way split) |
| `Qwen3.6-35B-A3B-UD-Q4_K_M.header.gguf` | `unsloth/Qwen3.6-35B-A3B-GGUF` |
| `Qwen3-8B-Q4_K_M.header.gguf` | `unsloth/Qwen3-8B-GGUF` |
| `Qwen3-0.6B-Q8_0.header.gguf` | `Qwen/Qwen3-0.6B-GGUF` (dense GQA calibration) |
| `gemma-3-1b-it-Q4_K_M.header.gguf` | `unsloth/gemma-3-1b-it-GGUF` (sliding-window calibration) |
| `GLM-4.7-Flash-Q4_K_M.header.gguf` | `unsloth/GLM-4.7-Flash-GGUF` (MLA calibration) |

Real file sizes, for checking the tensor-byte sums: GLM shards 9,429,984 +
49,294,975,936 + 49,949,266,048 + 9,466,399,584 bytes; Qwen3.6 22,134,528,992
bytes; Qwen3-8B 5,027,784,512 bytes.

The context-memory calibration records in `src/planner/context_tests.rs` were
captured with llama.cpp b12182 `llama-fit-params` dry runs on sparse stubs
built from these headers (the tensor data was never downloaded).
