# Parallelisms Reference

Generated from [llm-training-inference-parallelisms-whitepaper.md](/Users/huba/llm-training-inference-parallelisms-whitepaper.md). The original section structure is preserved below; each section contains local source links when present, otherwise a one-sentence summary.

## Abstract

Sources:

- [KV-cache](https://arxiv.org/abs/2309.06180)
- [ZeRO++](https://www.deepspeed.ai/tutorials/zeropp/)

## How To Read This Paper

Sources:

- [DP](https://pytorch.org/tutorials/beginner/dist_overview.html)
- [FSDP](https://docs.pytorch.org/docs/stable/fsdp.html)
- [ZeRO](https://www.deepspeed.ai/tutorials/zero/)
- [HSDP](https://github.com/pytorch/torchtitan/blob/main/docs/fsdp.md)
- [TP](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html)
- [CP](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/context_parallel.html)
- [PP](https://docs.pytorch.org/docs/stable/distributed.pipelining.html)
- [EP](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/features/parallel-strategy.md)
- [MoE](https://en.wikipedia.org/wiki/Mixture_of_experts)
- [P/D](https://arxiv.org/abs/2401.09670)

## 1. Shared System Model, Notation, Placement, and Evidence Policy

Summary: This section covers Shared System Model, Notation, Placement, and Evidence Policy.

### 1.1 Evidence Grades and Source Interpretation

Summary: This subsection covers Evidence Grades and Source Interpretation.

### 1.2 Shared Symbols and Transformer Shapes

Summary: This subsection covers Shared Symbols and Transformer Shapes.

### 1.3 Device Meshes, Rank Coordinates, and Process Groups

Sources:

- [NVIDIA Megatron-Core parallelism guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html)
- [TorchTitan](https://github.com/pytorch/torchtitan)
- [PyTorch DTensor](https://docs.pytorch.org/docs/stable/distributed.tensor.html)
- [DeepSpeed](https://github.com/deepspeedai/DeepSpeed)
- [MaxText](https://github.com/AI-Hypercomputer/maxtext)
- [GSPMD](https://arxiv.org/abs/2105.04663)
- [Pathways](https://research.google/pubs/pathways-asynchronous-distributed-dataflow-for-ml/)

### 1.4 Placement States and Tensor Layout Transitions

Sources:

- [NVIDIA Megatron-Core parallelism guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html)
- [Megatron-Core context parallelism](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/context_parallel.html)
- [TorchTitan](https://github.com/pytorch/torchtitan)
- [MaxText](https://github.com/AI-Hypercomputer/maxtext)
- [JAX distributed arrays and automatic parallelization](https://jax.readthedocs.io/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html)

### 1.5 Communication Primitives and Cost Models

Sources:

- [NVIDIA NCCL collective operations](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2297/user-guide/docs/usage/collectives.html)
- [NVIDIA NCCL point-to-point operations](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/p2p.html)
- [NVIDIA Megatron-Core parallelism guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html)
- [PyTorch distributed overview](https://pytorch.org/tutorials/beginner/dist_overview.html)
- [Ultra-Scale Playbook](https://nanotron-ultrascale-playbook.static.hf.space/index.html)

### 1.6 Memory, Bandwidth, and Overlap Accounting

Sources:

- [DeepSpeed ZeRO tutorial](https://www.deepspeed.ai/tutorials/zero/)
- [ZeRO-Infinity paper page](https://www.microsoft.com/en-us/research/publication/zero-infinity-breaking-the-gpu-memory-wall-for-extreme-scale-deep-learning/)
- [DeepSpeed ZeRO++ tutorial](https://www.deepspeed.ai/tutorials/zeropp/)
- [TorchTitan FSDP/HSDP notes](https://github.com/pytorch/torchtitan/blob/main/docs/fsdp.md)
- [PyTorch FSDP2 tutorial](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
- [Megatron-Core distributed optimizer](https://docs.nvidia.com/megatron-core/developer-guide/0.15.0/api-guide/dist_optimizer.html)
- [Ultra-Scale Playbook](https://nanotron-ultrascale-playbook.static.hf.space/index.html)

### 1.7 How Later Sections Use This Shared Layer

Summary: This subsection covers How Later Sections Use This Shared Layer.

## 2. Data Parallelism, FSDP, ZeRO, HSDP, and State Sharding

Summary: This section covers Data Parallelism, FSDP, ZeRO, HSDP, and State Sharding.

### 2.1 DDP: Replicated State, Batch Shards, and Gradient All-Reduce

Sources:

- [PyTorch DistributedDataParallel](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)
- [PyTorch DDP design note](https://docs.pytorch.org/docs/stable/notes/ddp.html)
- [PyTorch DDP tutorial](https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [PyTorch DDP communication hooks](https://docs.pytorch.org/docs/2.9/ddp_comm_hooks.html)
- [NCCL collective operations](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2297/user-guide/docs/usage/collectives.html)

### 2.2 The Primitive Mechanisms Behind Sharded Data Parallelism

Sources:

- [NCCL collective operations](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2297/user-guide/docs/usage/collectives.html)
- [NCCL point-to-point operations](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/p2p.html)
- [PyTorch FSDP2 tutorial](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
- [Megatron-Core distributed optimizer](https://docs.nvidia.com/megatron-core/developer-guide/0.15.0/api-guide/dist_optimizer.html)

### 2.3 ZeRO, ZeRO++, and Distributed Optimizers

Sources:

- [DeepSpeed ZeRO++ tutorial](https://www.deepspeed.ai/tutorials/zeropp/)
- [Microsoft Research ZeRO++ ICLR 2024 publication](https://www.microsoft.com/en-us/research/publication/zero-extremely-efficient-collective-communication-for-giant-model-training/)
- [Megatron-Core distributed optimizer docs](https://docs.nvidia.com/megatron-core/developer-guide/0.15.0/api-guide/dist_optimizer.html)
- [Microsoft Research ZeRO paper page](https://www.microsoft.com/en-us/research/publication/zero-memory-optimizations-toward-training-trillion-parameter-models/)
- [DeepSpeed ZeRO tutorial](https://www.deepspeed.ai/tutorials/zero/)
- [NCCL all-reduce, all-gather, reduce-scatter, all-to-all definitions](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2297/user-guide/docs/usage/collectives.html)

### 2.4 FSDP and FSDP2 Execution

Sources:

- [PyTorch FSDP2 tutorial](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
- [PyTorch `fully_shard` API](https://docs.pytorch.org/docs/2.8/distributed.fsdp.fully_shard.html)
- [TorchTitan FSDP/HSDP notes](https://github.com/pytorch/torchtitan/blob/main/docs/fsdp.md)
- [PyTorch Distributed Checkpoint](https://docs.pytorch.org/docs/stable/distributed.checkpoint.html)
- [Megatron-Core `fsdp_dtensor` optimizer checkpoint format](https://docs.nvidia.com/megatron-core/developer-guide/nightly/apidocs/core/core.optimizer.distrib_optimizer.html)

### 2.5 HSDP: Hybrid Sharded Data Parallelism

Sources:

- [TorchTitan FSDP/HSDP notes](https://github.com/pytorch/torchtitan/blob/main/docs/fsdp.md)
- [PyTorch `fully_shard` `reshard_after_forward` docs](https://docs.pytorch.org/docs/2.8/distributed.fsdp.fully_shard.html)
- [DeepSpeed ZeRO++ tutorial](https://www.deepspeed.ai/tutorials/zeropp/)
- [NCCL collective definitions](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2297/user-guide/docs/usage/collectives.html)

### 2.6 Offload: CPU and NVMe as Additional State Tiers

Sources:

- [PyTorch `CPUOffloadPolicy` in `fully_shard`](https://docs.pytorch.org/docs/2.8/distributed.fsdp.fully_shard.html)
- [Microsoft Research ZeRO-Offload](https://www.microsoft.com/en-us/research/publication/zero-offload-democratizing-billion-scale-model-training/)
- [Microsoft Research ZeRO-Infinity](https://www.microsoft.com/en-us/research/publication/zero-infinity-breaking-the-gpu-memory-wall-for-extreme-scale-deep-learning/)
- [SC21 ZeRO-Infinity proceedings page](https://sc21.supercomputing.org/proceedings/tech_paper/tech_paper_pages/pap464.html)

### 2.7 Low Precision, Quantized Communication, and Platform Support

Sources:

- [DeepSpeed ZeRO++ tutorial](https://www.deepspeed.ai/tutorials/zeropp/)
- [Microsoft Research ZeRO++ publication](https://www.microsoft.com/en-us/research/publication/zero-extremely-efficient-collective-communication-for-giant-model-training/)
- [PyTorch/TorchTitan float8 all-gather discussion](https://discuss.pytorch.org/t/distributed-w-torchtitan-enabling-float8-all-gather-in-fsdp2/209323)
- [NVIDIA Transformer Engine FP8 delayed scaling](https://nvidia.github.io/TransformerEngine/features/low_precision_training/fp8_delayed_scaling/fp8_delayed_scaling.html)
- [NVIDIA Transformer Engine FP8 primer](https://docs.nvidia.com/deeplearning/transformer-engine-releases/release-2.6/user-guide/examples/fp8_primer.html)
- [COAT project page](https://research.nvidia.com/labs/eai/publication/coat/)
- [Megatron-Core distributed optimizer docs](https://docs.nvidia.com/megatron-core/developer-guide/0.15.0/api-guide/dist_optimizer.html)

### 2.8 Choosing Among DDP, FSDP2, ZeRO, ZeRO++, HSDP, and Offload

Sources:

- [PyTorch FSDP2 tutorial](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
- [TorchTitan FSDP/HSDP notes](https://github.com/pytorch/torchtitan/blob/main/docs/fsdp.md)
- [DeepSpeed ZeRO++ tutorial](https://www.deepspeed.ai/tutorials/zeropp/)
- [Megatron-Core distributed optimizer](https://docs.nvidia.com/megatron-core/developer-guide/0.15.0/api-guide/dist_optimizer.html)
- [COAT project page](https://research.nvidia.com/labs/eai/publication/coat/)
- [ZeRO-Offload paper page](https://www.microsoft.com/en-us/research/publication/zero-offload-democratizing-billion-scale-model-training/)
- [ZeRO-Infinity paper page](https://www.microsoft.com/en-us/research/publication/zero-infinity-breaking-the-gpu-memory-wall-for-extreme-scale-deep-learning/)

### 2.9 Sources and Lineage

Sources:

- [FSDP2 tutorial](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
- [`fully_shard`](https://docs.pytorch.org/docs/2.8/distributed.fsdp.fully_shard.html)
- [FSDP/HSDP notes](https://github.com/pytorch/torchtitan/blob/main/docs/fsdp.md)
- [tutorial](https://www.deepspeed.ai/tutorials/zeropp/)
- [ICLR 2024 publication page](https://www.microsoft.com/en-us/research/publication/zero-extremely-efficient-collective-communication-for-giant-model-training/)
- [API and memory-saving documentation](https://docs.nvidia.com/megatron-core/developer-guide/0.15.0/api-guide/dist_optimizer.html)
- [project page](https://research.nvidia.com/labs/eai/publication/coat/)
- [FP8 delayed scaling](https://nvidia.github.io/TransformerEngine/features/low_precision_training/fp8_delayed_scaling/fp8_delayed_scaling.html)
- [float8 all-gather discussion](https://discuss.pytorch.org/t/distributed-w-torchtitan-enabling-float8-all-gather-in-fsdp2/209323)
- [collective operations](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2297/user-guide/docs/usage/collectives.html)
- [point-to-point API](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/p2p.html)
- [`DistributedDataParallel`](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)
- [DDP design note](https://docs.pytorch.org/docs/stable/notes/ddp.html)
- [DDP tutorial](https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [ZeRO paper page](https://www.microsoft.com/en-us/research/publication/zero-memory-optimizations-toward-training-trillion-parameter-models/)
- [ZeRO-Offload paper page](https://www.microsoft.com/en-us/research/publication/zero-offload-democratizing-billion-scale-model-training/)
- [ZeRO-Infinity paper page](https://www.microsoft.com/en-us/research/publication/zero-infinity-breaking-the-gpu-memory-wall-for-extreme-scale-deep-learning/)
- [proceedings page](https://sc21.supercomputing.org/proceedings/tech_paper/tech_paper_pages/pap464.html)

### 2.10 Caveats

Summary: This subsection covers Caveats.

## 3. Tensor Parallelism, Sequence-Parallel Activations, and Vocabulary Parallelism

Summary: This section covers Tensor Parallelism, Sequence-Parallel Activations, and Vocabulary Parallelism.

### 3.1 TP-Specific Notation and Layout Contract

Sources:

- [PyTorch tensor parallel docs](https://docs.pytorch.org/docs/stable/distributed.tensor.parallel)
- [PyTorch DTensor docs](https://docs.pytorch.org/docs/stable/distributed.tensor.html)
- [Megatron-Core parallelism guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html)
- [Megatron-Core tensor-parallel layer APIs](https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.tensor_parallel.layers.html)

### 3.2 Column-Parallel Linear Layers

Sources:

- [PyTorch `ColwiseParallel`](https://docs.pytorch.org/docs/stable/distributed.tensor.parallel)
- [Megatron-Core `ColumnParallelLinear`](https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.tensor_parallel.layers.html)
- [Megatron-LM model-parallel paper](https://arxiv.org/abs/1909.08053)
- [NVIDIA MegatronLM explanation](https://research.nvidia.com/labs/adlr/MegatronLM/)

### 3.3 Row-Parallel Linear Layers

Sources:

- [PyTorch `RowwiseParallel`](https://docs.pytorch.org/docs/stable/distributed.tensor.parallel)
- [Megatron-Core `RowParallelLinear`](https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.tensor_parallel.layers.html)
- [Megatron-LM paper](https://arxiv.org/abs/1909.08053)
- [TorchTitan async TP discussion](https://discuss.pytorch.org/t/distributed-w-torchtitan-introducing-async-tensor-parallelism-in-pytorch/209487)

### 3.4 Transformer Block Recipes: MLP and Attention

Summary: This subsection covers Transformer Block Recipes: MLP and Attention.

#### MLP / SwiGLU

Summary: This subsection covers MLP / SwiGLU.

#### Attention

Sources:

- [Megatron-LM model-parallel paper](https://arxiv.org/abs/1909.08053)
- [NVIDIA MegatronLM transformer-block explanation](https://research.nvidia.com/labs/adlr/MegatronLM/)
- [Megatron-Core tensor-parallel APIs](https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.tensor_parallel.layers.html)
- [PyTorch tensor parallel docs](https://docs.pytorch.org/docs/stable/distributed.tensor.parallel)

### 3.5 Sequence-Parallel Activations as a TP Companion

Sources:

- [PyTorch `SequenceParallel`](https://docs.pytorch.org/docs/stable/distributed.tensor.parallel)
- [Megatron-Core parallelism guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html)
- [Reducing Activation Recomputation in Large Transformer Models, MLSys 2023](https://proceedings.mlsys.org/paper_files/paper/2023/hash/80083951326cf5b35e5100260d64ed81-Abstract-mlsys2023.html)
- [Megatron Bridge parallelisms docs](https://docs.nvidia.com/nemo/megatron-bridge/latest/parallelisms.html)

### 3.6 Vocabulary Parallelism

Summary: This subsection covers Vocabulary Parallelism.

#### Input Embedding

Summary: This subsection covers Input Embedding.

#### LM Head and Cross-Entropy

Sources:

- [Megatron-Core `VocabParallelEmbedding`](https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.tensor_parallel.layers.html)
- [Balancing Pipeline Parallelism with Vocabulary Parallelism, MLSys 2025](https://proceedings.mlsys.org/paper_files/paper/2025/hash/10e400a587ff6925e4e26333b419ff55-Abstract-Conference.html)
- [VocabularyParallelism repository](https://github.com/sail-sg/VocabularyParallelism)
- [Liger Kernel paper](https://arxiv.org/abs/2410.10989)
- [Megatron-LM repository](https://github.com/NVIDIA/Megatron-LM)

### 3.7 Async Tensor Parallelism

Sources:

- [TorchTitan async TP post](https://discuss.pytorch.org/t/distributed-w-torchtitan-introducing-async-tensor-parallelism-in-pytorch/209487)
- [TorchTitan repository](https://github.com/pytorch/torchtitan)
- [PyTorch tensor parallel docs](https://docs.pytorch.org/docs/stable/distributed.tensor.parallel)
- [CoCoNeT / Breaking the Computation and Communication Abstraction Barrier, ASPLOS 2022](https://www.microsoft.com/en-us/research/publication/breaking-the-computation-and-communication-abstraction-barrier-in-distributed-machine-learning-workloads/)

### 3.8 TP-Aware CUDA and Kernel Techniques

Summary: This subsection covers TP-Aware CUDA and Kernel Techniques.

#### 3.8.1 Local GEMM Shape Quality

Summary: This subsection covers Local GEMM Shape Quality.

#### 3.8.2 Fused Epilogues and Layout-Stable Collectives

Summary: This subsection covers Fused Epilogues and Layout-Stable Collectives.

#### 3.8.3 Low-Precision Communication and Scale Layout

Summary: This subsection covers Low-Precision Communication and Scale Layout.

#### 3.8.4 TP-Aware Megakernels

Sources:

- [Transformer Engine MXFP8 docs](https://nvidia.github.io/TransformerEngine/features/low_precision_training/mxfp8/mxfp8.html)
- [Transformer Engine FP8 block scaling docs](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/fp8_blockwise_scaling/fp8_blockwise_scaling.html)
- [Hazy Research TP megakernel post](https://hazyresearch.stanford.edu/blog/2025-09-28-tp-llama-main)
- [Megatron-Core tensor-parallel APIs](https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.tensor_parallel.layers.html)
- [CUTLASS documentation](https://docs.nvidia.com/cutlass/)

### 3.9 Choosing a TP Degree

Sources:

- [Megatron-Core parallelism guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html)
- [TorchTitan repository](https://github.com/pytorch/torchtitan)
- [PyTorch DTensor docs](https://docs.pytorch.org/docs/stable/distributed.tensor.html)
- [Hugging Face Accelerate Megatron-LM guide](https://huggingface.co/docs/accelerate/main/usage_guides/megatron_lm)

### 3.10 Sources and Lineage

Sources:

- [`torch.distributed.tensor.parallel`](https://docs.pytorch.org/docs/stable/distributed.tensor.parallel)
- [`DTensor`](https://docs.pytorch.org/docs/stable/distributed.tensor.html)
- [async tensor parallelism](https://discuss.pytorch.org/t/distributed-w-torchtitan-introducing-async-tensor-parallelism-in-pytorch/209487)
- [TorchTitan repository](https://github.com/pytorch/torchtitan)
- [Megatron-Core parallelism guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html)
- [`ColumnParallelLinear`, `RowParallelLinear`, and `VocabParallelEmbedding`](https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.tensor_parallel.layers.html)
- [Vocabulary Parallelism](https://proceedings.mlsys.org/paper_files/paper/2025/hash/10e400a587ff6925e4e26333b419ff55-Abstract-Conference.html)
- [MXFP8](https://nvidia.github.io/TransformerEngine/features/low_precision_training/mxfp8/mxfp8.html)
- [TP megakernel analysis](https://hazyresearch.stanford.edu/blog/2025-09-28-tp-llama-main)
- [Megatron-LM tensor model parallelism](https://arxiv.org/abs/1909.08053)
- [Reducing Activation Recomputation in Large Transformer Models](https://proceedings.mlsys.org/paper_files/paper/2023/hash/80083951326cf5b35e5100260d64ed81-Abstract-mlsys2023.html)

### 3.11 Caveats

Summary: This subsection covers Caveats.

## 4. Pipeline Parallelism: Depth-Axis Parallelism and Modern Schedules

Sources:

- [Zero Bubble Pipeline Parallelism](https://proceedings.iclr.cc/paper_files/paper/2024/hash/d5a8e37f38a08c68162452dcba89ae9c-Abstract-Conference.html)
- [Pipeline Parallelism with Controllable Memory](https://proceedings.neurips.cc/paper_files/paper/2024/hash/527dad0b9159805289906d5740a0bdd3-Abstract-Conference.html)
- [PipeOffload](https://icml.cc/virtual/2025/poster/45468)
- [Balancing Pipeline Parallelism with Vocabulary Parallelism](https://proceedings.mlsys.org/paper_files/paper/2025/hash/10e400a587ff6925e4e26333b419ff55-Abstract-Conference.html)
- [DeepSeek-V3 DualPipe](https://arxiv.org/abs/2412.19437)
- [DeepSeek DualPipe repository](https://github.com/deepseek-ai/DualPipe)
- [DualPipeV](https://sail.sea.com/blog/articles/63)
- [PyTorch `torch.distributed.pipelining`](https://docs.pytorch.org/docs/2.9/distributed.pipelining.html)
- [TorchTitan zero-bubble post](https://discuss.pytorch.org/t/distributed-w-torchtitan-training-with-zero-bubble-pipeline-parallelism/214420)
- [GPipe](https://arxiv.org/abs/1811.06965)
- [PipeDream](https://arxiv.org/abs/1806.03377)
- [Megatron-LM interleaved PP](https://arxiv.org/abs/2104.04473)

### 4.1 System Model, Variables, and Shapes

Summary: This subsection covers System Model, Variables, and Shapes.

### 4.2 Primitive: Stage Partitioning and Boundary Contracts

Sources:

- [PyTorch pipeline docs](https://docs.pytorch.org/docs/2.9/distributed.pipelining.html)
- [PyTorch pipeline tutorial](https://docs.pytorch.org/tutorials/intermediate/pipelining_tutorial.html)
- [Megatron-LM interleaved PP lineage](https://arxiv.org/abs/2104.04473)

### 4.3 Primitive: Microbatching, Warmup, Steady State, and Drain

Sources:

- [GPipe lineage](https://arxiv.org/abs/1811.06965)
- [PipeDream/1F1B lineage](https://arxiv.org/abs/1806.03377)
- [Zero Bubble PP](https://proceedings.iclr.cc/paper_files/paper/2024/hash/d5a8e37f38a08c68162452dcba89ae9c-Abstract-Conference.html)

### 4.4 Primitive: Point-to-Point Send/Recv and Collective Composition

Sources:

- [PyTorch `torch.distributed.pipelining`](https://docs.pytorch.org/docs/2.9/distributed.pipelining.html)
- [TorchTitan zero-bubble discussion](https://discuss.pytorch.org/t/distributed-w-torchtitan-training-with-zero-bubble-pipeline-parallelism/214420)
- [DeepSeek DualPipe repository](https://github.com/deepseek-ai/DualPipe)

### 4.5 Primitive: The Schedule DAG and the F/B/W Split

Sources:

- [Zero Bubble PP](https://proceedings.iclr.cc/paper_files/paper/2024/hash/d5a8e37f38a08c68162452dcba89ae9c-Abstract-Conference.html)
- [TorchTitan zero-bubble post](https://discuss.pytorch.org/t/distributed-w-torchtitan-training-with-zero-bubble-pipeline-parallelism/214420)
- [PyTorch schedule docs](https://docs.pytorch.org/docs/2.9/distributed.pipelining.html)

### 4.6 Zero Bubble Pipeline Parallelism

Sources:

- [ICLR 2024 Zero Bubble paper](https://proceedings.iclr.cc/paper_files/paper/2024/hash/d5a8e37f38a08c68162452dcba89ae9c-Abstract-Conference.html)
- [arXiv summary](https://huggingface.co/papers/2401.10241)
- [Sail zero-bubble repository](https://github.com/sail-sg/zero-bubble-pipeline-parallelism)
- [PyTorch `ScheduleInterleavedZeroBubble` and `ScheduleZBVZeroBubble`](https://docs.pytorch.org/docs/2.9/distributed.pipelining.html)

### 4.7 Controllable-Memory Pipeline Schedules

Sources:

- [NeurIPS 2024 Controllable Memory paper](https://proceedings.neurips.cc/paper_files/paper/2024/hash/527dad0b9159805289906d5740a0bdd3-Abstract-Conference.html)
- [arXiv summary](https://huggingface.co/papers/2405.15362)
- [Sail zero-bubble repository with schedule implementations](https://github.com/sail-sg/zero-bubble-pipeline-parallelism)
- [Sea AI Lab V-shaped schedule discussion](https://sail.sea.com/blog/articles/63)

### 4.8 PipeOffload: Activation Offload and Prefetch as Schedule Primitives

Sources:

- [ICML 2025 PipeOffload poster](https://icml.cc/virtual/2025/poster/45468)
- [Sea AI Lab publication page](https://sail.sea.com/research/publications/34)
- [Controllable Memory PP](https://proceedings.neurips.cc/paper_files/paper/2024/hash/527dad0b9159805289906d5740a0bdd3-Abstract-Conference.html)

### 4.9 Vocabulary-Balanced Pipeline Parallelism

Sources:

- [MLSys 2025 Vocabulary Parallelism paper](https://proceedings.mlsys.org/paper_files/paper/2025/hash/10e400a587ff6925e4e26333b419ff55-Abstract-Conference.html)
- [OpenReview page](https://openreview.net/forum?id=nlRGXpglaO)
- [arXiv summary](https://huggingface.co/papers/2411.05288)
- [Controllable Memory PP](https://proceedings.neurips.cc/paper_files/paper/2024/hash/527dad0b9159805289906d5740a0bdd3-Abstract-Conference.html)

### 4.10 DualPipe and DualPipeV: Pipeline Scheduling for MoE Communication

Sources:

- [DeepSeek-V3 technical report](https://arxiv.org/abs/2412.19437)
- [DeepSeek DualPipe repository](https://github.com/deepseek-ai/DualPipe)
- [DualPipeV analysis](https://sail.sea.com/blog/articles/63)
- [PyTorch pipeline schedule docs](https://docs.pytorch.org/docs/2.9/distributed.pipelining.html)

### 4.11 PyTorch Pipelining and TorchTitan Runtime Support

Sources:

- [PyTorch pipeline docs](https://docs.pytorch.org/docs/2.9/distributed.pipelining.html)
- [PyTorch pipeline tutorial](https://docs.pytorch.org/tutorials/intermediate/pipelining_tutorial.html)
- [TorchTitan zero-bubble post](https://discuss.pytorch.org/t/distributed-w-torchtitan-training-with-zero-bubble-pipeline-parallelism/214420)
- [TorchTitan repository](https://github.com/pytorch/torchtitan)

### 4.12 Optimizer Barriers, Gradient Synchronization, and Correctness

Sources:

- [Zero Bubble PP](https://proceedings.iclr.cc/paper_files/paper/2024/hash/d5a8e37f38a08c68162452dcba89ae9c-Abstract-Conference.html)
- [PipeDream lineage](https://arxiv.org/abs/1806.03377)
- [PyTorch pipeline docs](https://docs.pytorch.org/docs/2.9/distributed.pipelining.html)

### 4.13 Composing PP With Tensor, Context, Expert, and Data Parallelism

Sources:

- [DeepSeek-V3 technical report](https://arxiv.org/abs/2412.19437)
- [DeepSeek DualPipe repository](https://github.com/deepseek-ai/DualPipe)
- [PyTorch TorchTitan zero-bubble post](https://discuss.pytorch.org/t/distributed-w-torchtitan-training-with-zero-bubble-pipeline-parallelism/214420)
- [Megatron-LM large-scale training](https://arxiv.org/abs/2104.04473)

### 4.14 Choosing a Modern PP Schedule

Summary: This subsection covers Choosing a Modern PP Schedule.

### 4.15 Lineage: Older PP Work as Context

Sources:

- [GPipe](https://arxiv.org/abs/1811.06965)
- [PipeDream](https://arxiv.org/abs/1806.03377)
- [Megatron-LM interleaved PP](https://arxiv.org/abs/2104.04473)

### 4.16 Practical Caveats

Summary: This subsection covers Practical Caveats.

## 5. Sequence Parallelism and Context Parallelism

Summary: This section covers Sequence Parallelism and Context Parallelism.

### 5.1 Notation, Tensor Shapes, and the Attention Operator

Sources:

- [\left\lfloor \frac{rS}{c} \right\rfloor, \left\lfloor \frac{(r+1)S}{c} \right\rfloor \right), $$ with local length $$ S_r = |\mathcal{S}_r|. $$ For balanced padded sequences, $S_r \approx S/c$. The local sequence-sharded tensors are $$ X^{(r)} \in \mathbb{R}^{B \times S_r \times H}, $$ $$ Q^{(r)} \in \mathbb{R}^{B \times S_r \times h_q \times d}, \qquad K^{(r)},V^{(r)} \in \mathbb{R}^{B \times S_r \times h_{kv} \times d}, $$ and the local output shard is $$ O^{(r)} \in \mathbb{R}^{B \times S_r \times h_q \times d}. $$ The core challenge is visible in these shapes: local $Q^{(r)}$ is enough to compute outputs for local query tokens, but dense attention needs $K,V$ from every token position, not only $\mathcal{S}_r$. **Sources.** Recent docs: [Megatron-Core context parallelism](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/context_parallel.html)
- [NVIDIA Dynamic-CP blog](https://developer.nvidia.com/blog/speeding-up-variable-length-training-with-dynamic-context-parallelism-and-nvidia-megatron-core/)
- [PyTorch FlexAttention + FlashAttention-4](https://pytorch.org/blog/flexattention-flashattention-4-fast-and-flexible/)
- [cuDNN NSA](https://docs.nvidia.com/deeplearning/cudnn/frontend/latest/fe-oss-apis/nsa.html)
- [DeepSpeed-Ulysses](https://arxiv.org/abs/2309.14509)
- [Ring Attention](https://arxiv.org/abs/2310.01889)
- [Megatron sequence parallelism](https://proceedings.mlsys.org/paper_files/paper/2023/hash/80083951326cf5b35e5100260d64ed81-Abstract-mlsys2023.html)

### 5.2 The Primitive Operations and Their Collectives

Sources:

- [Megatron-Core parallelism guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html)
- [Megatron-Core CP docs](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/context_parallel.html)
- [NVIDIA Dynamic-CP blog](https://developer.nvidia.com/blog/speeding-up-variable-length-training-with-dynamic-context-parallelism-and-nvidia-megatron-core/)
- [Hugging Face Ulysses documentation](https://huggingface.co/docs/transformers/main/deepspeed_alst)
- [DeepSpeed-Ulysses paper](https://arxiv.org/abs/2309.14509)
- [Ring Attention paper](https://arxiv.org/abs/2310.01889)
- [USP repo](https://github.com/feifeibear/long-context-attention)

### 5.3 Megatron Sequence Parallelism: Activation Sharding, Not Full Long-Context Attention

Sources:

- [Megatron-Core parallelism guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html)
- [Accelerate Megatron-LM guide](https://huggingface.co/docs/accelerate/v1.10.0/usage_guides/megatron_lm)
- [Reducing Activation Recomputation in Large Transformer Models](https://proceedings.mlsys.org/paper_files/paper/2023/hash/80083951326cf5b35e5100260d64ed81-Abstract-mlsys2023.html)

### 5.4 Ulysses: Sequence Shards Become Head Shards Through All-to-All

Sources:

- [Hugging Face Ulysses sequence parallelism docs](https://huggingface.co/docs/transformers/main/deepspeed_alst)
- [Ultra-long Ulysses + Ring implementation note](https://huggingface.co/blog/exploding-gradients/ulysses-ring-attention)
- [DeepSpeed-Ulysses](https://arxiv.org/abs/2309.14509)
- [USP paper](https://arxiv.org/abs/2405.07719)
- [USP/YunChang repo](https://github.com/feifeibear/long-context-attention)

### 5.5 Ring Attention: KV Streaming With Point-to-Point Send/Recv

Sources:

- [Ring Attention with Blockwise Transformers](https://arxiv.org/abs/2310.01889)
- [USP paper](https://arxiv.org/abs/2405.07719)
- [USP/YunChang repo](https://github.com/feifeibear/long-context-attention)
- [ring-flash-attention repo](https://github.com/zhuzilin/ring-flash-attention)
- [Megatron-Core CP docs](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/context_parallel.html)

### 5.6 The Numerical Primitive: Blockwise Online Softmax

Sources:

- [FlashAttention](https://arxiv.org/abs/2205.14135)
- [FlashAttention-3](https://arxiv.org/abs/2407.08608)
- [PyTorch FlexAttention + FlashAttention-4](https://pytorch.org/blog/flexattention-flashattention-4-fast-and-flexible/)
- [Ring Attention](https://arxiv.org/abs/2310.01889)
- [Megatron-Core CP docs](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/context_parallel.html)

### 5.7 Megatron Context Parallelism: Sequence Sharding for the Whole Network

Sources:

- [Megatron-Core context parallelism](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/context_parallel.html)
- [Megatron-Core parallelism guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html)
- [NVIDIA Dynamic-CP blog](https://developer.nvidia.com/blog/speeding-up-variable-length-training-with-dynamic-context-parallelism-and-nvidia-megatron-core/)
- [USP/YunChang repo](https://github.com/feifeibear/long-context-attention)

### 5.8 Dynamic Context Parallelism: Choose CP Size Per Microbatch

Sources:

- [NVIDIA Dynamic Context Parallelism with Megatron Core](https://developer.nvidia.com/blog/speeding-up-variable-length-training-with-dynamic-context-parallelism-and-nvidia-megatron-core/)
- [Megatron-Core CP](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/context_parallel.html)
- [Megatron-Core GitHub](https://github.com/NVIDIA/Megatron-LM)

### 5.9 THD and Ragged Packing: Token-Head-Dimension Layout

Sources:

- [NVIDIA Dynamic-CP blog](https://developer.nvidia.com/blog/speeding-up-variable-length-training-with-dynamic-context-parallelism-and-nvidia-megatron-core/)
- [cuDNN NSA](https://docs.nvidia.com/deeplearning/cudnn/frontend/latest/fe-oss-apis/nsa.html)
- [PyTorch FlexAttention + FlashAttention-4](https://pytorch.org/blog/flexattention-flashattention-4-fast-and-flexible/)

### 5.10 Sparse and Hybrid Attention: Communicate Selected KV, Not Necessarily All KV

Sources:

- [cuDNN NSA documentation](https://docs.nvidia.com/deeplearning/cudnn/frontend/latest/fe-oss-apis/nsa.html)
- [PyTorch FlexAttention + FlashAttention-4](https://pytorch.org/blog/flexattention-flashattention-4-fast-and-flexible/)
- [DeepSeek-V4 official model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro)
- [SGLang/Miles DeepSeek-V4 Day-0 report](https://www.lmsys.org/blog/2026-04-25-deepseek-v4/)
- [SGLang DeepSeek-V4 cookbook](https://docs.sglang.io/cookbook/autoregressive/DeepSeek/DeepSeek-V4)
- [Native Sparse Attention paper](https://arxiv.org/abs/2502.11089)
- [Mistral sliding-window attention](https://arxiv.org/abs/2310.06825)

### 5.11 Inference KV and Context Placement

Sources:

- [SGLang/Miles DeepSeek-V4 Day-0 report](https://www.lmsys.org/blog/2026-04-25-deepseek-v4/)
- [SGLang DeepSeek-V4 cookbook](https://docs.sglang.io/cookbook/autoregressive/DeepSeek/DeepSeek-V4)
- [TensorRT-LLM disaggregated serving](https://nvidia.github.io/TensorRT-LLM/1.2.0rc6/features/disagg-serving.html)
- [TensorRT-LLM disaggregated serving blog](https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog5_Disaggregated_Serving_in_TensorRT-LLM.html)
- [vLLM PagedAttention design](https://docs.vllm.ai/en/v0.18.0/design/paged_attention/)
- [PagedAttention paper](https://arxiv.org/abs/2309.06180)
- [SGLang paper](https://arxiv.org/abs/2312.07104)
- [SGLang RadixCache implementation notes](https://deepwiki.com/sgl-project/sglang/5.2-kernel-implementations)

### 5.12 How to Choose Among SP, Ulysses, Ring, CP, Dynamic-CP, and Sparse Context Placement

Sources:

- [NVIDIA Dynamic-CP blog](https://developer.nvidia.com/blog/speeding-up-variable-length-training-with-dynamic-context-parallelism-and-nvidia-megatron-core/)
- [Megatron-Core CP docs](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/context_parallel.html)
- [SGLang/Miles DeepSeek-V4](https://www.lmsys.org/blog/2026-04-25-deepseek-v4/)
- [PyTorch FlexAttention + FlashAttention-4](https://pytorch.org/blog/flexattention-flashattention-4-fast-and-flexible/)
- [USP paper](https://arxiv.org/abs/2405.07719)
- [DeepSpeed-Ulysses](https://arxiv.org/abs/2309.14509)
- [Ring Attention](https://arxiv.org/abs/2310.01889)

### 5.13 Evidence Boundaries and Terminology Caveats

Sources:

- [DeepSeek-V4 official model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro)
- [SGLang/Miles DeepSeek-V4 runtime report](https://www.lmsys.org/blog/2026-04-25-deepseek-v4/)
- [SGLang DeepSeek-V4 cookbook](https://docs.sglang.io/cookbook/autoregressive/DeepSeek/DeepSeek-V4)
- [Megatron-Core CP docs](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/context_parallel.html)
- [cuDNN NSA docs](https://docs.nvidia.com/deeplearning/cudnn/frontend/latest/fe-oss-apis/nsa.html)

## 6. Expert and MoE Parallelism

Sources:

- [DeepSeek-V4](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro)
- [SGLang/Miles DeepSeek-V4 report](https://www.lmsys.org/blog/2026-04-25-deepseek-v4/)
- [DeepEP](https://github.com/deepseek-ai/DeepEP)
- [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM)
- [DeepSeek-V3](https://arxiv.org/abs/2412.19437)
- [TensorRT-LLM Wide-EP](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/features/parallel-strategy.md)
- [Megatron-Core](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html)
- [GShard](https://research.google/pubs/gshard-scaling-giant-models-with-conditional-computation-and-automatic-sharding/)
- [Switch Transformer](https://arxiv.org/abs/2101.03961)
- [Tutel](https://proceedings.mlsys.org/paper_files/paper/2023/hash/5616d34cf8ff73942cfd5aa922842556-Abstract-mlsys2023.html)
- [MegaBlocks](https://proceedings.mlsys.org/paper_files/paper/2023/hash/5a54f79333768effe7e8927bcccffe40-Abstract-mlsys2023.html)

### 6.1 Symbols, Shapes, and the MoE Layer

Sources:

- [DeepSeek-V4 model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro)
- [SGLang/Miles DeepSeek-V4](https://www.lmsys.org/blog/2026-04-25-deepseek-v4/)
- [DeepSeek-V3](https://arxiv.org/abs/2412.19437)
- [Megatron-Core parallelism guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html)
- [GShard](https://research.google/pubs/gshard-scaling-giant-models-with-conditional-computation-and-automatic-sharding/)
- [Switch Transformer](https://arxiv.org/abs/2101.03961)

### 6.2 Primitive Technique: Routing and Top-k Assignment

Sources:

- [DeepSeek-V3](https://arxiv.org/abs/2412.19437)
- [DeepSeek-V4 model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro)
- [TensorRT-LLM Wide-EP docs](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/features/parallel-strategy.md)
- [Qwen3 technical report](https://huggingface.co/papers/2505.09388)
- [Switch Transformer](https://arxiv.org/abs/2101.03961)
- [GShard](https://research.google/pubs/gshard-scaling-giant-models-with-conditional-computation-and-automatic-sharding/)

### 6.3 Primitive Technique: Expert Ownership and Local Shard Shapes

Sources:

- [TensorRT-LLM expert and Wide-EP docs](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/features/parallel-strategy.md)
- [Megatron-Core parallelism guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html)
- [DeepEP](https://github.com/deepseek-ai/DeepEP)
- [SGLang/Miles DeepSeek-V4](https://www.lmsys.org/blog/2026-04-25-deepseek-v4/)
- [Tutel](https://proceedings.mlsys.org/paper_files/paper/2023/hash/5616d34cf8ff73942cfd5aa922842556-Abstract-mlsys2023.html)

### 6.4 Primitive Technique: Permutation, Packing, and Inverse Permutation

Sources:

- [DeepEP dispatch/combine examples](https://github.com/deepseek-ai/DeepEP)
- [DeepGEMM grouped GEMM layouts](https://github.com/deepseek-ai/DeepGEMM)
- [SGLang/Miles DeepSeek-V4](https://www.lmsys.org/blog/2026-04-25-deepseek-v4/)
- [MegaBlocks](https://proceedings.mlsys.org/paper_files/paper/2023/hash/5a54f79333768effe7e8927bcccffe40-Abstract-mlsys2023.html)
- [Tutel](https://proceedings.mlsys.org/paper_files/paper/2023/hash/5616d34cf8ff73942cfd5aa922842556-Abstract-mlsys2023.html)

### 6.5 Primitive Technique: All-to-All Dispatch, Notify, and Combine

Sources:

- [DeepEP](https://github.com/deepseek-ai/DeepEP)
- [SGLang/Miles DeepSeek-V4](https://www.lmsys.org/blog/2026-04-25-deepseek-v4/)
- [DeepSeek-V3](https://arxiv.org/abs/2412.19437)
- [TensorRT-LLM Wide-EP docs](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/features/parallel-strategy.md)
- [Tutel](https://proceedings.mlsys.org/paper_files/paper/2023/hash/5616d34cf8ff73942cfd5aa922842556-Abstract-mlsys2023.html)

### 6.6 Primitive Technique: Capacity, Dropless Execution, and Padding

Sources:

- [MegaBlocks](https://proceedings.mlsys.org/paper_files/paper/2023/hash/5a54f79333768effe7e8927bcccffe40-Abstract-mlsys2023.html)
- [DeepSeek-V3](https://arxiv.org/abs/2412.19437)
- [Megatron-Core MoE support](https://developer.nvidia.com/megatron-core)
- [Switch Transformer](https://arxiv.org/abs/2101.03961)
- [GShard](https://research.google/pubs/gshard-scaling-giant-models-with-conditional-computation-and-automatic-sharding/)

### 6.7 Primitive Technique: Grouped GEMM, MegaBlocks, DeepGEMM, and Mega MoE

Sources:

- [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM)
- [SGLang/Miles DeepSeek-V4](https://www.lmsys.org/blog/2026-04-25-deepseek-v4/)
- [DeepEP](https://github.com/deepseek-ai/DeepEP)
- [MegaBlocks](https://proceedings.mlsys.org/paper_files/paper/2023/hash/5a54f79333768effe7e8927bcccffe40-Abstract-mlsys2023.html)
- [CUTLASS/CuTe background](https://github.com/NVIDIA/cutlass)
- [Tutel](https://proceedings.mlsys.org/paper_files/paper/2023/hash/5616d34cf8ff73942cfd5aa922842556-Abstract-mlsys2023.html)

### 6.8 Primitive Technique: Backward Pass, Weight Gradients, and Overlap

Sources:

- [DeepEP](https://github.com/deepseek-ai/DeepEP)
- [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM)
- [DeepSeek-V3](https://arxiv.org/abs/2412.19437)
- [SGLang/Miles DeepSeek-V4 RL stack](https://www.lmsys.org/blog/2026-04-25-deepseek-v4/)
- [Megatron-Core](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html)

### 6.9 Primitive Technique: DeepEP Modes and Network-Aware EP

Sources:

- [DeepEP](https://github.com/deepseek-ai/DeepEP)
- [SGLang/Miles DeepSeek-V4](https://www.lmsys.org/blog/2026-04-25-deepseek-v4/)
- [DeepGEMM Mega MoE](https://github.com/deepseek-ai/DeepGEMM)
- [TensorRT-LLM Wide-EP docs](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/features/parallel-strategy.md)
- [NVIDIA NVSHMEM documentation](https://docs.nvidia.com/nvshmem/)

### 6.10 Primitive Technique: Wide-EP Serving and Hot-Expert Replication

Sources:

- [TensorRT-LLM Wide-EP docs](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/features/parallel-strategy.md)
- [SGLang/Miles DeepSeek-V4](https://www.lmsys.org/blog/2026-04-25-deepseek-v4/)
- [DeepSeek-V4 model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro)
- [DeepEP](https://github.com/deepseek-ai/DeepEP)
- [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM)

### 6.11 Composition With TP, SP, PP, CP, DP, and Precision

Sources:

- [Megatron-Core parallelism guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html)
- [SGLang/Miles DeepSeek-V4](https://www.lmsys.org/blog/2026-04-25-deepseek-v4/)
- [DeepSeek-V3](https://arxiv.org/abs/2412.19437)
- [TensorRT-LLM parallel strategy](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/features/parallel-strategy.md)
- [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM)
- [Transformer Engine FP8 docs](https://docs.nvidia.com/deeplearning/transformer-engine-releases/release-2.8/user-guide/index.html)

### 6.12 End-to-End MoE Execution Recipe

Sources:

- [DeepEP](https://github.com/deepseek-ai/DeepEP)
- [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM)
- [TensorRT-LLM Wide-EP docs](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/features/parallel-strategy.md)
- [Megatron-Core](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html)
- [SGLang/Miles DeepSeek-V4](https://www.lmsys.org/blog/2026-04-25-deepseek-v4/)
- [MegaBlocks](https://proceedings.mlsys.org/paper_files/paper/2023/hash/5a54f79333768effe7e8927bcccffe40-Abstract-mlsys2023.html)
- [Tutel](https://proceedings.mlsys.org/paper_files/paper/2023/hash/5616d34cf8ff73942cfd5aa922842556-Abstract-mlsys2023.html)

### 6.13 Evidence Boundaries and Caveats

Sources:

- [DeepSeek-V4 model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro)
- [SGLang/Miles DeepSeek-V4](https://www.lmsys.org/blog/2026-04-25-deepseek-v4/)
- [DeepSeek-V3](https://arxiv.org/abs/2412.19437)
- [DeepEP](https://github.com/deepseek-ai/DeepEP)
- [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM)
- [TensorRT-LLM Wide-EP](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/features/parallel-strategy.md)
- [Megatron-Core](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html)
- [Qwen3](https://huggingface.co/papers/2505.09388)
- [MegaBlocks](https://proceedings.mlsys.org/paper_files/paper/2023/hash/5a54f79333768effe7e8927bcccffe40-Abstract-mlsys2023.html)
- [Tutel](https://proceedings.mlsys.org/paper_files/paper/2023/hash/5616d34cf8ff73942cfd5aa922842556-Abstract-mlsys2023.html)
- [Switch Transformer](https://arxiv.org/abs/2101.03961)
- [GShard](https://research.google/pubs/gshard-scaling-giant-models-with-conditional-computation-and-automatic-sharding/)

## 7. Hybrid Mesh Parallelism

Sources:

- [NVIDIA Megatron-Core parallelism guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html)
- [TorchTitan](https://github.com/pytorch/torchtitan)
- [MaxText](https://github.com/AI-Hypercomputer/maxtext)
- [GSPMD](https://arxiv.org/abs/2105.04663)
- [Pathways](https://research.google/pubs/pathways-asynchronous-distributed-dataflow-for-ml/)
- [DeepSpeed](https://github.com/deepspeedai/DeepSpeed)
- [Ultra-Scale Playbook](https://nanotron-ultrascale-playbook.static.hf.space/index.html)

### 7.1 Rank Mapping and Topology-Aware Placement

Sources:

- [NVIDIA Megatron-Core](https://developer.nvidia.com/megatron-core)
- [NVIDIA Transformer Engine](https://docs.nvidia.com/deeplearning/transformer-engine-releases/release-2.8/user-guide/index.html)
- [TorchTitan](https://github.com/pytorch/torchtitan)
- [Ultra-Scale Playbook](https://nanotron-ultrascale-playbook.static.hf.space/index.html)
- [MegaScale](https://arxiv.org/abs/2402.15627)
- [DeepSpeed](https://www.deepspeed.ai/)

### 7.2 Checkpoint Resharding and Restart Semantics

Sources:

- [TorchTitan distributed checkpointing context](https://github.com/pytorch/torchtitan)
- [PyTorch Distributed Checkpoint](https://docs.pytorch.org/docs/stable/distributed.checkpoint.html)
- [Megatron-Core](https://docs.nvidia.com/megatron-core/developer-guide/latest/index.html)
- [MaxText](https://maxtext.readthedocs.io/)
- [DeepSpeed ZeRO checkpointing documentation](https://deepspeed.readthedocs.io/en/stable/zero3.html)

### 7.3 Composition Rules

Sources:

- [Ultra-Scale Playbook](https://nanotron-ultrascale-playbook.static.hf.space/index.html)
- [NVIDIA Megatron-Core parallelism guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html)
- [TorchTitan](https://github.com/pytorch/torchtitan)
- [DeepSpeed](https://www.deepspeed.ai/)
- [MaxText](https://github.com/AI-Hypercomputer/maxtext)
- [GSPMD](https://arxiv.org/abs/2105.04663)
- [Pathways](https://research.google/pubs/pathways-asynchronous-distributed-dataflow-for-ml/)

### 7.4 System Perspectives

Sources:

- [Megatron-Core](https://developer.nvidia.com/megatron-core)
- [TorchTitan](https://github.com/pytorch/torchtitan)
- [MaxText](https://github.com/AI-Hypercomputer/maxtext)
- [GSPMD](https://arxiv.org/abs/2105.04663)
- [Pathways](https://research.google/pubs/pathways-asynchronous-distributed-dataflow-for-ml/)
- [DeepSpeed](https://www.deepspeed.ai/)
- [Ultra-Scale Playbook](https://nanotron-ultrascale-playbook.static.hf.space/index.html)

### 7.5 Caveats and Evidence Boundaries

Summary: This subsection covers Caveats and Evidence Boundaries.

## 8. Inference Parallelism and Serving Systems

Summary: This section covers Inference Parallelism and Serving Systems.

### 8.1 Primitive Technique: Replicas and Request Routing

Sources:

- [SLO](https://en.wikipedia.org/wiki/Service-level_objective)
- [RAG](https://en.wikipedia.org/wiki/Retrieval-augmented_generation)

### 8.2 Primitive Technique: Continuous Batching

Summary: This subsection covers Continuous Batching.

### 8.3 Primitive Technique: Paged KV Cache

Summary: This subsection covers Paged KV Cache.

### 8.4 Primitive Technique: Prefix and Radix Caching

Summary: This subsection covers Prefix and Radix Caching.

### 8.5 Primitive Technique: Prefill/Decode Disaggregation

Summary: This subsection covers Prefill/Decode Disaggregation.

### 8.6 Primitive Technique: Chunked Prefill and SplitFuse

Summary: This subsection covers Chunked Prefill and SplitFuse.

### 8.7 Primitive Technique: Speculative and [MTP](https://arxiv.org/abs/2412.19437) (multi-token prediction) Decoding

Sources:

- [MTP](https://arxiv.org/abs/2412.19437)

### 8.8 Primitive Technique: TP, PP, EP, and CP for Serving

Summary: This subsection covers TP, PP, EP, and CP for Serving.

### 8.9 End-to-End Serving Map

Summary: This subsection covers End-to-End Serving Map.

### 8.10 Sources and Caveats

Summary: This subsection covers Sources and Caveats.

## 9. Cross-Cutting CUDA and Platform Techniques

Summary: This section covers Cross-Cutting CUDA and Platform Techniques.

### 9.1 Overlap: Streams, Events, Buckets, and Progress

Summary: This subsection covers Overlap: Streams, Events, Buckets, and Progress.

### 9.2 CUDA Graphs and Static Metadata

Summary: This subsection covers CUDA Graphs and Static Metadata.

### 9.3 Low Precision as Payload Plus Scale Layout

Summary: This subsection covers Low Precision as Payload Plus Scale Layout.

### 9.4 Attention Kernels: FlashAttention, cuDNN SDPA, NSA, and FlashMLA

Sources:

- [SDPA](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)

### 9.5 CUTLASS, CuTe, GEMM, and Grouped Expert Compute

Sources:

- [CUTLASS](https://docs.nvidia.com/cutlass/)
- [CuTe](https://docs.nvidia.com/cutlass/media/docs/cpp/cute/01_layout.html)
- [WGMMA](https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-instructions-wgmma)
- [TMA](https://docs.nvidia.com/cutlass/media/docs/cpp/cute/0z_tma_tensors.html)
- [UMMA](https://docs.nvidia.com/cutlass/media/docs/cpp/blackwell_functionality.html)

### 9.6 NVSHMEM, GPU-Initiated Communication, and Persistent Kernels

Sources:

- [NVSHMEM](https://docs.nvidia.com/nvshmem/)
- [MPK](https://github.com/mirage-project/mirage)

### 9.7 Kernel Authoring DSLs: Triton, Pallas, TileLang, and ThunderKittens

Summary: This subsection covers Kernel Authoring DSLs: Triton, Pallas, TileLang, and ThunderKittens.

## 10. Frontier Model and Runtime Case Studies

Summary: This section covers Frontier Model and Runtime Case Studies.

### 10.1 DeepSeek-V3: MoE, FP8, DualPipe, and Expert Overlap

Summary: This subsection covers DeepSeek-V3: MoE, FP8, DualPipe, and Expert Overlap.

### 10.2 DeepSeek-V4: Hybrid Attention, mHC, FP4 Experts, and Serving Layout

Sources:

- [mHC](https://arxiv.org/abs/2512.24880)
- [TRTLLM-Gen](https://nvidia.github.io/TensorRT-LLM/)

### 10.3 Meta Llama 3 and TorchTitan: Dense Hybrid Parallelism

Summary: This subsection covers Meta Llama 3 and TorchTitan: Dense Hybrid Parallelism.

### 10.4 Qwen, Kimi, and Open MoE Directions

Summary: This subsection covers Qwen, Kimi, and Open MoE Directions.

### 10.5 Google Pathways, GSPMD, and MaxText: Compiler-Mesh Parallelism

Summary: This subsection covers Google Pathways, GSPMD, and MaxText: Compiler-Mesh Parallelism.

## 11. Source Carryover From The GPU-Kernel Paper

Summary: This section covers Source Carryover From The GPU-Kernel Paper.

## 12. Open Problems and Evidence Boundaries

Summary: This section covers Open Problems and Evidence Boundaries.

### 12.1 Unresolved Technical Questions

Summary: This subsection covers Unresolved Technical Questions.

### 12.2 The Main Boundary

Summary: This subsection covers The Main Boundary.

## 13. Conclusion

Summary: This section covers Conclusion.

## Bibliography and Source Notes

Summary: This section covers Bibliography and Source Notes.

### Recent Training, Parallelism, and Runtime Systems

Sources:

- [DeepSeek-AI, "DeepSeek-V3 Technical Report," arXiv:2412.19437, 2024/2025](https://arxiv.org/abs/2412.19437)
- [DeepSeek-AI, DeepSeek-V3 repository](https://github.com/deepseek-ai/DeepSeek-V3)
- [DeepSeek-AI, DualPipe repository](https://github.com/deepseek-ai/DualPipe)
- [DeepSeek-AI, DeepEP: expert-parallel communication library](https://github.com/deepseek-ai/DeepEP)
- [DeepSeek-AI, DeepGEMM: FP8/FP4 GEMM and Mega MoE kernels](https://github.com/deepseek-ai/DeepGEMM)
- [DeepSeek-AI, FlashMLA optimized MLA attention repository](https://github.com/deepseek-ai/FlashMLA)
- [DeepSeek-AI, "DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence," official model card/report](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro)
- [NVIDIA Technical Blog, "Build with DeepSeek V4 Using NVIDIA Blackwell and GPU-Accelerated Endpoints," 2026](https://developer.nvidia.com/blog/build-with-deepseek-v4-using-nvidia-blackwell-and-gpu-accelerated-endpoints/)
- [LMSYS/SGLang/Miles, "DeepSeek-V4 on Day 0: From Fast Inference to Verified RL with SGLang and Miles," 2026](https://www.lmsys.org/blog/2026-04-25-deepseek-v4/)
- [NVIDIA Megatron-Core developer information and parallelism guide (developer.nvidia.com)](https://developer.nvidia.com/megatron-core)
- [NVIDIA Megatron-Core developer information and parallelism guide (docs.nvidia.com)](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html)
- [NVIDIA/Megatron-LM repository](https://github.com/NVIDIA/Megatron-LM)
- [Deepak Narayanan et al., "Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM," SC 2021 / arXiv:2104.04473](https://arxiv.org/abs/2104.04473)
- [PyTorch/TorchTitan repository and paper (github.com)](https://github.com/pytorch/torchtitan)
- [PyTorch/TorchTitan repository and paper (huggingface.co)](https://huggingface.co/papers/2410.06511)
- [PyTorch FSDP/FSDP2 documentation](https://docs.pytorch.org/docs/stable/fsdp.html)
- [PyTorch FSDP2 tutorial and fully_shard API documentation (docs.pytorch.org)](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
- [PyTorch FSDP2 tutorial and fully_shard API documentation (docs.pytorch.org)](https://docs.pytorch.org/docs/2.8/distributed.fsdp.fully_shard.html)
- [PyTorch torch.distributed.pipelining documentation](https://docs.pytorch.org/docs/2.9/distributed.pipelining.html)
- [PyTorch Forum, "Introducing Async Tensor Parallelism in PyTorch," 2024](https://discuss.pytorch.org/t/distributed-w-torchtitan-introducing-async-tensor-parallelism-in-pytorch/209487)
- [PyTorch Forum, "Training with Zero-Bubble Pipeline Parallelism," 2024](https://discuss.pytorch.org/t/distributed-w-torchtitan-training-with-zero-bubble-pipeline-parallelism/214420)
- [PyTorch Forum, "Enabling Float8 All-Gather in FSDP2," 2024](https://discuss.pytorch.org/t/distributed-w-torchtitan-enabling-float8-all-gather-in-fsdp2/209323)
- [DeepSpeed repository and documentation (github.com)](https://github.com/deepspeedai/DeepSpeed)
- [DeepSpeed repository and documentation (deepspeed.readthedocs.io)](https://deepspeed.readthedocs.io/)
- [DeepSpeed ZeRO documentation](https://deepspeed.readthedocs.io/en/stable/zero3.html)
- [DeepSpeed ZeRO++ tutorial](https://www.deepspeed.ai/tutorials/zeropp/)
- [Guanhua Wang et al., "ZeRO++: Extremely Efficient Collective Communication for Large Model Training," ICLR 2024](https://www.microsoft.com/en-us/research/publication/zero-extremely-efficient-collective-communication-for-large-model-training/)
- [Samyam Rajbhandari et al., "ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning," SC 2021](https://www.microsoft.com/en-us/research/publication/zero-infinity-breaking-the-gpu-memory-wall-for-extreme-scale-deep-learning/)
- [Ziheng Jiang et al., "MegaScale: Scaling Large Language Model Training to More Than 10,000 GPUs," arXiv:2402.15627, 2024](https://arxiv.org/abs/2402.15627)
- [NVIDIA Technical Blog, "Speeding Up Variable-Length Training with Dynamic Context Parallelism and NVIDIA Megatron Core," 2026](https://developer.nvidia.com/blog/speeding-up-variable-length-training-with-dynamic-context-parallelism-and-nvidia-megatron-core/)
- [Hugging Face/Nanotron, "The Ultra-Scale Playbook: Training LLMs on GPU Clusters," 2025](https://nanotron-ultrascale-playbook.static.hf.space/index.html)

### Pipeline Parallelism

Sources:

- [Penghui Qi et al., "Zero Bubble Pipeline Parallelism," ICLR 2024 / arXiv:2401.10241](https://arxiv.org/abs/2401.10241)
- [Sea AI Lab, "Zero Bubble Pipeline Parallelism" blog and repository (sail.sea.com)](https://sail.sea.com/blog/articles/56)
- [Sea AI Lab, "Zero Bubble Pipeline Parallelism" blog and repository (github.com)](https://github.com/sail-sg/zero-bubble-pipeline-parallelism)
- [Penghui Qi et al., "Pipeline Parallelism with Controllable Memory," NeurIPS 2024](https://sail.sea.com/research/publications/57)
- [Sea AI Lab, "Pipeline Parallelism with Controllable Memory," 2025](https://sail.sea.com/blog/articles/57)
- [Xinyi Wan et al., "PipeOffload: Improving Scalability of Pipeline Parallelism with Memory Optimization," 2025](https://sail.sea.com/research/publications/70)
- [Man Tsung Yeung et al., "Balancing Pipeline Parallelism with Vocabulary Parallelism," MLSys 2025](https://sail.sea.com/research/publications/69)
- [Sea AI Lab, "DualPipe could be better without the Dual," 2025](https://sail.sea.com/blog/articles/63)
- [Yanping Huang et al., "GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism," NeurIPS 2019](https://arxiv.org/abs/1811.06965)
- [Deepak Narayanan et al., "PipeDream: Generalized Pipeline Parallelism for DNN Training," SOSP 2019 / arXiv:1806.03377](https://arxiv.org/abs/1806.03377)

### Sequence, Context, and Attention Parallelism

Sources:

- [NVIDIA Megatron-Core context parallelism documentation](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/context_parallel.html)
- [Sam Ade Jacobs et al., "DeepSpeed Ulysses: System Optimizations for Enabling Training of Extreme Long Sequence Transformer Models," 2023](https://www.microsoft.com/en-us/research/publication/deepspeed-ulysses-system-optimizations-for-enabling-training-of-extreme-long-sequence-transformer-models/)
- [Hao Liu, Matei Zaharia, Pieter Abbeel, "Ring Attention with Blockwise Transformers for Near-Infinite Context," ICLR 2024 / arXiv:2310.01889](https://arxiv.org/abs/2310.01889)
- [Jiarui Fang and Shangchun Zhao, "A Unified Sequence Parallelism Approach for Long Context Generative AI," arXiv:2405.07719, 2024](https://arxiv.org/abs/2405.07719)
- [DeepSeek-AI, "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model," arXiv:2405.04434, 2024](https://arxiv.org/abs/2405.04434)
- [Tri Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness," arXiv:2205.14135, 2022](https://arxiv.org/abs/2205.14135)
- [Tri Dao, "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning," arXiv:2307.08691, 2023](https://arxiv.org/abs/2307.08691)
- [Jay Shah et al., "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision," arXiv:2407.08608, 2024](https://arxiv.org/abs/2407.08608)
- [Ted Zadouri et al., "FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling," arXiv:2603.05451, 2026](https://arxiv.org/abs/2603.05451)
- [Dao-AILab, flash-attention repository and FA4 blog (github.com)](https://github.com/Dao-AILab/flash-attention)
- [Dao-AILab, flash-attention repository and FA4 blog (tridao.me)](https://tridao.me/blog/2026/flash4/)
- [PyTorch, "FlexAttention + FlashAttention-4: Fast and Flexible Attention," 2026](https://pytorch.org/blog/flexattention-flashattention-4-fast-and-flexible/)
- [NVIDIA cuDNN Frontend attention and Native Sparse Attention APIs (docs.nvidia.com)](https://docs.nvidia.com/deeplearning/cudnn/frontend/latest/operations/Attention.html)
- [NVIDIA cuDNN Frontend attention and Native Sparse Attention APIs (docs.nvidia.com)](https://docs.nvidia.com/deeplearning/cudnn/frontend/latest/fe-oss-apis/nsa.html)

### MoE and Expert Parallelism

Sources:

- [Damai Dai et al., "DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models," arXiv:2401.06066, 2024](https://arxiv.org/abs/2401.06066)
- [Dmitry Lepikhin et al., "GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding," ICLR 2021](https://research.google/pubs/gshard-scaling-giant-models-with-conditional-computation-and-automatic-sharding/)
- [William Fedus et al., "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity," JMLR 2022](https://www.jmlr.org/papers/v23/21-0998.html)
- [Trevor Gale et al., "MegaBlocks: Efficient Sparse Training with Mixture-of-Experts," MLSys 2023 / arXiv:2211.15841](https://arxiv.org/abs/2211.15841)
- [Databricks, megablocks repository](https://github.com/databricks/megablocks)
- [Changho Hwang et al., "Tutel: Adaptive Mixture-of-Experts at Scale," MLSys 2023 / arXiv:2206.03382](https://arxiv.org/abs/2206.03382)
- [TensorRT-LLM parallelism documentation, including EP and Wide-EP](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/features/parallel-strategy.md)
- [Qwen2 Technical Report, arXiv:2407.10671, 2024](https://arxiv.org/abs/2407.10671)
- [Qwen3 Technical Report, arXiv:2505.09388, 2025](https://arxiv.org/abs/2505.09388)
- [NVIDIA NeMo Qwen3 MoE model coverage and recipes](https://docs.nvidia.com/nemo/automodel/latest/model-coverage/llm/qwen/qwen3-moe.html)

### Inference and Serving

Sources:

- [Woosuk Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention," SOSP 2023 / arXiv:2309.06180](https://arxiv.org/abs/2309.06180)
- [vLLM documentation](https://docs.vllm.ai/)
- [SGLang documentation](https://docs.sglang.io/)
- [SGLang DeepSeek-V4 cookbook and RadixAttention/RadixCache documentation (docs.sglang.io)](https://docs.sglang.io/cookbook/autoregressive/DeepSeek/DeepSeek-V4)
- [TensorRT-LLM documentation](https://nvidia.github.io/TensorRT-LLM/)
- [Connor Holmes et al., "DeepSpeed-FastGen: High-throughput Text Generation for LLMs via MII and DeepSpeed-Inference," arXiv:2401.08671, 2024](https://arxiv.org/abs/2401.08671)
- [Yinmin Zhong et al., "DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving," OSDI 2024 / arXiv:2401.09670](https://arxiv.org/abs/2401.09670)
- [Pratyush Patel et al., "Splitwise: Efficient generative LLM inference using phase splitting," ISCA 2024 / arXiv:2311.18677](https://arxiv.org/abs/2311.18677)
- [Amey Agrawal et al., "SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills," arXiv:2308.16369, 2023](https://arxiv.org/abs/2308.16369)
- [Xupeng Miao et al., "SpecInfer: Accelerating Generative LLM Serving with Speculative Inference and Token Tree Verification," arXiv:2305.09781, 2023](https://arxiv.org/abs/2305.09781)
- [Tianle Cai et al., "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads," ICML 2024 / arXiv:2401.10774](https://arxiv.org/abs/2401.10774)
- [FasterDecoding, Medusa repository](https://github.com/FasterDecoding/Medusa)

### Low Precision, Kernels, and Platform

Sources:

- [NVIDIA Transformer Engine documentation](https://docs.nvidia.com/deeplearning/transformer-engine-releases/release-2.8/user-guide/index.html)
- [NVIDIA Transformer Engine FP8 delayed scaling documentation](https://nvidia.github.io/TransformerEngine/features/low_precision_training/fp8_delayed_scaling/fp8_delayed_scaling.html)
- [NVIDIA Transformer Engine MXFP8 documentation](https://nvidia.github.io/TransformerEngine/features/low_precision_training/mxfp8/mxfp8.html)
- [NVIDIA Transformer Engine NVFP4 documentation](https://nvidia.github.io/TransformerEngine/features/low_precision_training/nvfp4/nvfp4.html)
- [NVIDIA CUTLASS/CuTe documentation and Blackwell kernel abstractions](https://docs.nvidia.com/cutlass/)
- [NVIDIA NCCL documentation](https://docs.nvidia.com/deeplearning/nccl/)
- [NVIDIA NVSHMEM documentation/developer page (developer.nvidia.com)](https://developer.nvidia.com/nvshmem)
- [NVIDIA NVSHMEM documentation/developer page (docs.nvidia.com)](https://docs.nvidia.com/nvshmem/)
- [NVIDIA CUDA platform documentation](https://docs.nvidia.com/cuda/)
- [NVIDIA Apex optimizer and fused layer norm documentation (nvidia.github.io)](https://nvidia.github.io/apex/optimizers.html)
- [NVIDIA Apex optimizer and fused layer norm documentation (nvidia.github.io)](https://nvidia.github.io/apex/layernorm.html)
- [NVIDIA NeMo and Megatron Bridge documentation for activation recomputation, communication overlap, and optimizer integration (docs.nvidia.com)](https://docs.nvidia.com/nemo-framework/)
- [NVIDIA NeMo and Megatron Bridge documentation for activation recomputation, communication overlap, and optimizer integration (docs.nvidia.com)](https://docs.nvidia.com/nemo/megatron-bridge/)
- [PyTorch activation checkpointing documentation](https://docs.pytorch.org/docs/stable/checkpoint.html)
- [Haocheng Xi et al., "COAT: Compressing Optimizer states and Activation for Memory-Efficient FP8 Training," arXiv:2410.19313, 2024](https://arxiv.org/abs/2410.19313)
- [NVIDIA Technical Blog, "Advancing Emerging Optimizers for Accelerated LLM Training with NVIDIA Megatron," 2026](https://developer.nvidia.com/blog/advancing-emerging-optimizers-for-accelerated-llm-training-with-nvidia-megatron/)
- ["Scaling Muon: A 2x More Efficient Optimizer for Large Language Models," arXiv:2502.16982, 2025](https://arxiv.org/abs/2502.16982)
- [Zhenda Xie et al., "mHC: Manifold-Constrained Hyper-Connections," arXiv:2512.24880, 2025](https://arxiv.org/abs/2512.24880)
- [Pin-Lun Hsu et al., "Liger-Kernel: Efficient Triton Kernels for LLM Training," arXiv:2410.10989, 2024](https://arxiv.org/abs/2410.10989)
- [LinkedIn, Liger-Kernel repository](https://github.com/linkedin/Liger-Kernel)
- [Tile-AI, tilelang repository and "TileLang: A Composable Tiled Programming Model for AI Systems," arXiv:2504.17577 (github.com)](https://github.com/tile-ai/tilelang)
- [Tile-AI, tilelang repository and "TileLang: A Composable Tiled Programming Model for AI Systems," arXiv:2504.17577 (arxiv.org)](https://arxiv.org/abs/2504.17577)
- [Native Sparse Attention public material: arXiv:2502.11089 and fla-org/native-sparse-attention (arxiv.org)](https://arxiv.org/abs/2502.11089)
- [Native Sparse Attention public material: arXiv:2502.11089 and fla-org/native-sparse-attention (github.com)](https://github.com/fla-org/native-sparse-attention)
- [HazyResearch, ThunderKittens repository and paper (github.com)](https://github.com/HazyResearch/ThunderKittens)
- [HazyResearch, ThunderKittens repository and paper (arxiv.org)](https://arxiv.org/abs/2410.20399)
- [HazyResearch, "Look Ma, No Bubbles! Designing a Low-Latency Megakernel for Llama-1B," 2025](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles)
- [HazyResearch Megakernels and Mirage MPK repositories (github.com)](https://github.com/HazyResearch/Megakernels)
- [HazyResearch Megakernels and Mirage MPK repositories (github.com)](https://github.com/mirage-project/mirage)

### Compiler and Mesh Systems

Sources:

- [Yuanzhong Xu et al., "GSPMD: General and Scalable Parallelization for ML Computation Graphs," arXiv:2105.04663, 2021](https://arxiv.org/abs/2105.04663)
- [Paul Barham et al., "Pathways: Asynchronous Distributed Dataflow for ML," MLSys 2022](https://research.google/pubs/pathways-asynchronous-distributed-dataflow-for-ml/)
- [Aakanksha Chowdhery et al., "PaLM: Scaling Language Modeling with Pathways," arXiv:2204.02311, 2022](https://arxiv.org/abs/2204.02311)
- [AI-Hypercomputer, MaxText repository and docs (github.com)](https://github.com/AI-Hypercomputer/maxtext)
- [AI-Hypercomputer, MaxText repository and docs (maxtext.readthedocs.io)](https://maxtext.readthedocs.io/)
- [JAX Pallas documentation](https://docs.jax.dev/en/latest/pallas/index.html)
- [OpenAI Triton (openai.com)](https://openai.com/research/triton)
- [OpenAI Triton (triton-lang.org)](https://triton-lang.org/main/)

### Model Reports Used As Parallelism Context

Sources:

- [Meta AI, "The Llama 3 Herd of Models," arXiv:2407.21783, 2024](https://arxiv.org/abs/2407.21783)
- [NVIDIA Blog, "New Open Source Qwen3-Next Models Preview Hybrid MoE Architecture..." 2025](https://developer.nvidia.com/blog/new-open-source-qwen3-next-models-preview-hybrid-moe-architecture-delivering-improved-accuracy-and-accelerated-parallel-processing-across-nvidia-platform/)
- [Moonshot AI, "Kimi K2: Open Agentic Intelligence," arXiv:2507.20534, 2025, and MoonshotAI/Kimi-K2 repository (huggingface.co)](https://huggingface.co/papers/2507.20534)
- [Moonshot AI, "Kimi K2: Open Agentic Intelligence," arXiv:2507.20534, 2025, and MoonshotAI/Kimi-K2 repository (github.com)](https://github.com/MoonshotAI/Kimi-K2)

## Verification Notes

Summary: This section covers Verification Notes.
