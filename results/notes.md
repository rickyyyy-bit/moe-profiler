# Results notes

## Stage 6: sparsity versus batch size

MoE sparsity is strongest at small batches because each request routes only a small
top-k subset of experts per layer. As batch size grows, tokens from more prompts are
combined and their expert sets overlap imperfectly, so the union of activated experts
increases. The activated ratio should therefore rise from a low batch-1 baseline but
typically sub-linearly: later requests increasingly revisit experts already present in
the batch. This is why large serving batches can erode the parameter-loading advantage
of sparse models even though each individual token still uses the same top-k experts.

Do not treat a tiny/random model as evidence for this curve. If batch 1 already
activates nearly every expert, first distinguish a trace bug from toy-model
saturation; either outcome invalidates the run as a production sparsity result.
