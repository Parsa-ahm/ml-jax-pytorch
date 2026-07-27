# 02 — Tiny LLM that speaks Python

_(Rung 2 — locked after rung 1 ships.)_

The language rite of passage. Train a small GPT-style transformer on Python source
so it autocompletes code, then whittle inference.

- **baseline:** a nanoGPT-style decoder transformer trained on a Python corpus.
- **optimized axis:** KV-cache + int8 quantization → higher tokens/sec.
- **deliverable:** tokens/sec before vs after, plus generated-Python samples.

Full spec written when we get here.
