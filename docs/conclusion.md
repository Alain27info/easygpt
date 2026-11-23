# Conclusion

You have built, from `torch.Tensor` up, every component of a modern language model:

- a **tokenizer**: characters first, then a byte-level BPE trained on your data;
- the **model**: embeddings, causal multi-head attention with rotary position embeddings, SwiGLU feed-forward networks, RMSNorm in a pre-norm residual architecture;
- the **training**: cross-entropy, AdamW with a warmup + cosine schedule, gradient clipping, checkpointing;
- the **decoding**: temperature and nucleus sampling.

You have also trained two small models on a CPU: a 0.9M-parameter character-level Shakespeare model and a 2.6M-parameter TinyStories storyteller.

## From here to a "real" LLM

Architecturally, your `TransformerLM` is very close to [Llama 3](https://arxiv.org/abs/2407.21783) or [Qwen 2.5](https://arxiv.org/abs/2412.15115). What separates it from them is mostly **scale** and **engineering**:

| | easygpt (TinyStories preset) | Llama 3 8B |
|---|---|---|
| parameters | 2.6M | 8B |
| layers × width | 4 × 192 | 32 × 4096 |
| vocabulary | 2,048 | 128,256 |
| context | 128 | 8,192 (then 128K) |
| training tokens | 16M | 15T |
| compute | ~25 min on a laptop CPU | ~1.3M H100-hours |

The main extra ingredients:

- **Data**: trillions of tokens, carefully filtered and deduplicated, mixing web pages, code, books and maths.
- **Efficiency**: [mixed-precision](https://arxiv.org/abs/1710.03740) (bfloat16) training, fused attention kernels ([FlashAttention](https://arxiv.org/abs/2205.14135)), `torch.compile`, data/tensor/pipeline parallelism over thousands of GPUs.
- **Inference**: KV cache, [grouped-query attention](https://arxiv.org/abs/2305.13245), quantization, [speculative decoding](https://arxiv.org/abs/2211.17192).
- **Post-training**: supervised fine-tuning on instructions and dialogues, then preference optimization ([RLHF](https://arxiv.org/abs/2203.02155), [DPO](https://arxiv.org/abs/2305.18290)) and reinforcement learning on verifiable tasks: this is what turns a text predictor into an assistant.

## Where to go next

- The rest of [Stanford CS336](https://stanford-cs336.github.io/): systems (kernels, parallelism), scaling laws, data, alignment.
- [nanoGPT](https://github.com/karpathy/nanoGPT) and [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) to reproduce GPT-2 (124M) and see how far careful engineering goes.
- [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/) for the original encoder-decoder architecture.
- [easygrad](https://svaiter.github.io/easygrad/) to open the last black box we used: automatic differentiation.
