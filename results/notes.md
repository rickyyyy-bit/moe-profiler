# Results notes

## Stage 6: sparsity versus batch size

MoE sparsity is strongest at small batches because each request routes only a small
top-k subset of experts per layer. As batch size grows, tokens from more prompts are
combined and their expert sets overlap imperfectly, so the union of activated experts
increases. The activated ratio should therefore rise from a low batch-1 baseline but
typically sub-linearly: later requests increasingly revisit experts already present in
the batch. This is why large serving batches can erode the parameter-loading advantage
of sparse models even though each individual token still uses the same top-k experts.

## Small real-model trace

On `tiny-random/qwen3-moe` (revision
`10a349dcb488b10c27aa4a3c1dbefb74c41565c3`) I found an activated-expert ratio of
1.00 at batch sizes 1, 2, 4, and 8 across four PyTorch CPU passes. This eight-expert
random test model saturates within the tokens of a batch-one prompt, so the result
validates real Qwen3-MoE router-hook compatibility but is not evidence for the
low-base production-model sparsity curve.
