# GPU Kernel Authoring Techniques for Frontier-Scale LLM Training

**A technique-centered academic white paper**

**Scope:** public sources through 2026-04-27

**Primary platform:** NVIDIA [CUDA](https://docs.nvidia.com/cuda/) (NVIDIA's GPU programming platform) GPUs, with selective comparison to [TPU](https://cloud.google.com/tpu/docs/intro-to-tpu) (Google Tensor Processing Unit accelerators), [ROCm](https://rocm.docs.amd.com/) (AMD GPU compute software stack), and compiler [DSLs](https://en.wikipedia.org/wiki/Domain-specific_language) (domain-specific languages)
**Goal:** explain the kernel-authoring techniques first, then point to papers, docs, code, and LLM-assisted workflows for deeper study and implementation inspection.

## Abstract

Frontier-scale LLM training is increasingly limited by the shape of GPU kernels rather than by the Transformer equations alone. A modern training step is a schedule of tiled [GEMMs](https://docs.nvidia.com/cuda/cublas/#cublas-t-gemm) (general matrix-multiply kernels), attention reductions, softmax statistics, quantization scales, all-to-all token exchange, activation recomputation, fused losses, and optimizer updates. The best public systems treat these operations as algorithm-hardware co-design problems: reduce [HBM](https://www.jedec.org/standards-documents/technology-focus-areas/high-bandwidth-memory-hbm) (high-bandwidth memory) traffic, overlap tensor cores with data movement and communication, preserve numerical stability under [FP8/FP4](https://nvidia.github.io/TransformerEngine/examples/fp8_primer.html) (8-bit and 4-bit floating-point formats), and move scheduling decisions closer to the GPU.

This paper rewrites the LLM training kernel landscape around techniques rather than frameworks. Each section explains one technique, decomposes it into primitive techniques, links to further reading, and gives implementation examples where the technique appears in public papers, documentation, repositories, or agent workflows. Frameworks such as Triton, Pallas, TileLang, [CUTLASS](https://docs.nvidia.com/cutlass/) (CUDA Templates for Linear Algebra Subroutines)/[CuTe](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/01_layout.html) (CUTLASS tensor-layout algebra), ThunderKittens, Transformer Engine, Megatron-Core, DeepSpeed, [NCCL](https://docs.nvidia.com/deeplearning/nccl/) (NVIDIA Collective Communications Library), and [NVSHMEM](https://docs.nvidia.com/nvshmem/) (GPU-side OpenSHMEM-style communication library) are mentioned only as implementation vehicles for a technique.

The through-line is simple: the frontier is moving from isolated operator kernels toward resource orchestration. FlashAttention-4 shows that Blackwell attention is no longer only about reducing [IO](https://en.wikipedia.org/wiki/Input/output) (input/output traffic); it also needs software exponentials, conditional rescaling, [TMEM](https://research.colfax-intl.com/cutlass-tutorial-writing-gemm-kernels-using-tensor-memory-for-nvidia-blackwell-gpus/) (Blackwell Tensor Memory for tensor-core accumulator staging), and [2-CTA](https://tridao.me/blog/2026/flash4/) (two cooperating CUDA thread blocks) backward scheduling. DeepSeek-V3/V4 show that sparse [MoE](https://arxiv.org/abs/2211.15841) (Mixture-of-Experts routing) training makes communication, grouped GEMM, FP8/FP4 scaling, and optimizer design part of the kernel problem. Megakernel work shows a likely next direction: persistent, internally scheduled GPU programs that fuse larger subgraphs where launch overhead, HBM round-trips, or communication bubbles dominate. LLM-assisted kernel generation adds another layer: code agents can explore schedules, generate candidate kernels, and run correctness/performance loops, but public benchmarks still show that expert-designed or expert-supervised workflows are required for trustworthy training kernels.

## How to Read This Paper

Each technique section follows the same pattern:

| Part | Purpose |
|---|---|
| **Technique** | The core algorithmic and CUDA/platform idea. |
| **Primitive Techniques** | Smaller mechanisms that compose the top-level technique, with how each mechanism is used in training kernels. |
| **Why It Matters For Training** | The bottleneck or stability problem it addresses. |
| **Explore Further** | Primary papers, official docs, and canonical repositories. |
| **Implementation Examples** | Public implementations or integration points to inspect. |
| **Caveats** | Where the technique is limited, inferred, or hardware-specific. |

Evidence labels:

| Grade | Meaning |
|---|---|
| `paper` | Peer-reviewed paper, arXiv preprint, or official technical report. |
| `official repo` | Repository owned by the project or organization. |
| `official docs/blog/report` | Vendor docs, model cards, official engineering blogs, official reports, or technical pages. |
| `third-party integration` | Runtime or downstream integration report. |
| `inferred` | Kernel consequence inferred from public architecture, not directly disclosed. |

Primitive-technique subsections include inline evidence or implementation links. The source grade follows the source class above: arXiv and independent technical reports are `paper`, project-owned GitHub repositories are `official repo`, vendor/model documentation is `official docs/blog/report`, downstream runtime posts are `third-party integration`, and claims that go beyond disclosed implementation details are explicitly described as inferred or architecture-implied. Composite table labels such as `paper + official repo` mean the row is supported by more than one evidence class; shorter labels such as `official blog` are sublabels of the broader official-docs/blog/report class.

## Executive Technique Map

| Technique | Main Training Pressure | Core Sources | Implementation Examples | Evidence Grade |
|---|---|---|---|---|
| IO-aware attention and online softmax | Avoid $O(n^2)$ score/probability materialization in HBM | [FlashAttention](https://arxiv.org/abs/2205.14135), [FlashAttention-2](https://arxiv.org/abs/2307.08691) | [FlashAttention repository](https://github.com/Dao-AILab/flash-attention), [cuDNN SDPA](https://docs.nvidia.com/deeplearning/cudnn/frontend/latest/operations/Attention.html) ([scaled dot-product attention](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)) | `paper` + `official repo/docs` |
| Hopper/Blackwell asynchronous attention pipelines | Overlap tensor cores, softmax, and memory movement | [FlashAttention-3](https://arxiv.org/abs/2407.08608), [FA3](https://arxiv.org/abs/2407.08608) (FlashAttention-3) PyTorch blog, [FlashAttention-4](https://arxiv.org/abs/2603.05451) | FlashAttention-3/4 code in [`flash-attention`](https://github.com/Dao-AILab/flash-attention), [FA4](https://arxiv.org/abs/2603.05451) (FlashAttention-4) blog | `paper` + `official blog/repo` |
| Softmax bottleneck reduction | Exponential and rescaling become limiting on modern GPUs | [FA3 blog](https://docs.pytorch.org/blog/flashattention-3/), [FA4 blog](https://tridao.me/blog/2026/flash4/) | FA4 conditional rescaling/software exp; cuDNN attention | `paper` + `official blog` |
| Low-precision scaling as layout | $\mathrm{FP8}/\mathrm{FP4}$ require explicit scales, $a_{\max}$, and tensor layouts | [Transformer Engine docs](https://docs.nvidia.com/deeplearning/transformer-engine-releases/release-2.8/user-guide/index.html), [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM), [COAT](https://arxiv.org/abs/2410.19313) ([Compressing Optimizer states and Activation for memory-efficient FP8 training](https://arxiv.org/abs/2410.19313)) | Transformer Engine, DeepGEMM $\mathrm{FP8}/\mathrm{FP4}$, COAT | `official docs/repo` + `paper` |
| Fused memory-bound training operators | Avoid temporary tensors for norms, RoPE, SwiGLU, and losses | [Liger paper](https://arxiv.org/abs/2410.10989), [`Liger-Kernel`](https://github.com/linkedin/Liger-Kernel) | Liger RMSNorm/RoPE/SwiGLU/cross-entropy/post-training losses | `paper` + `official repo` |
| Recomputation and deterministic backward | Save activation memory; control nondeterministic atomics | [FA2](https://arxiv.org/abs/2307.08691) (FlashAttention-2)/[FA3](https://arxiv.org/abs/2407.08608) (FlashAttention-3)/[FA4](https://arxiv.org/abs/2603.05451) (FlashAttention-4) papers and blogs | FlashAttention backward, FA4 deterministic mode | `paper` + `official blog/repo` |
| MoE routing as data layout and communication | Sparse experts create all-to-all plus grouped GEMM bottlenecks | [DeepSeek-V3](https://arxiv.org/abs/2412.19437), [DeepEP](https://github.com/deepseek-ai/DeepEP) (DeepSeek expert-parallel communication library), [MegaBlocks](https://arxiv.org/abs/2211.15841), [Tutel](https://arxiv.org/abs/2206.03382) | DeepEP dispatch/combine, MegaBlocks block-sparse MoE, Tutel adaptive MoE | `paper` + `official repo` |
| Expert grouped GEMM and MoE mega-kernels | Variable expert token counts and communication bubbles | [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM), [SGLang/Miles V4 post](https://www.lmsys.org/blog/2026-04-25-deepseek-v4/) | DeepGEMM grouped GEMM and Mega MoE | `official repo` + `third-party integration` |
| Distributed overlap | Hide collectives under useful work | [MegaScale](https://arxiv.org/abs/2402.15627), [DeepSeek-V3](https://arxiv.org/abs/2412.19437), [NCCL docs](https://docs.nvidia.com/deeplearning/nccl/), [NVSHMEM](https://developer.nvidia.com/nvshmem) | DeepSeek DualPipe, DeepEP hooks, Megatron-Core, NCCL/NVSHMEM | `paper` + `official docs/repo` |
| Long-context sparse/hybrid attention | $10^6$-token context requires non-quadratic attention paths | [DeepSeek-V2](https://arxiv.org/abs/2405.04434), [DeepSeek-V4 model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro), [TileLang](https://github.com/tile-ai/tilelang) | [MLA](https://arxiv.org/abs/2405.04434) (Multi-head Latent Attention)/[CSA](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro) (Compressed Sparse Attention)/[HCA](https://developer.nvidia.com/blog/build-with-deepseek-v4-using-nvidia-blackwell-and-gpu-accelerated-endpoints/) (Heavily Compressed Attention), TileLang attention, cuDNN [NSA](https://docs.nvidia.com/deeplearning/cudnn/frontend/latest/fe-oss-apis/nsa.html) (Native Sparse Attention) | `paper` + `official docs/blog/report` |
| Optimizer and state-compression kernels | Optimizer states and matrix optimizers become bandwidth/compute kernels | [Apex FusedAdam](https://nvidia.github.io/apex/optimizers.html), [COAT](https://arxiv.org/abs/2410.19313), [Muon/NVIDIA Megatron post](https://developer.nvidia.com/blog/advancing-emerging-optimizers-for-accelerated-llm-training-with-nvidia-megatron/) | Apex multi-tensor Adam, COAT $\mathrm{FP8}$ states, Megatron Muon | `official docs/blog/report` + `paper` |
| Persistent and megakernel scheduling | Remove launch boundaries and inter-kernel bubbles | [Hazy Megakernels](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles), [`Megakernels`](https://github.com/HazyResearch/Megakernels), Mirage [MPK](https://github.com/mirage-project/mirage) (Mirage Persistent Kernel) | Hazy Llama megakernel, Mirage persistent kernel, DeepGEMM Mega MoE | `official blog/repo` |
| Authoring abstraction and automated search | Make kernel schedules editable and testable | [Triton](https://openai.com/research/triton), [Pallas](https://docs.jax.dev/en/latest/pallas/index.html), [TileLang](https://github.com/tile-ai/tilelang), [KernelBench](https://arxiv.org/abs/2502.10517) | Liger Triton kernels, FA4 CuTe-DSL, TileLang kernels, KernelBench | `official docs/repo` + `paper` |
| LLM-assisted kernel generation and optimization | Turn kernel authoring into a supervised generate-test-profile loop | [KernelBench](https://arxiv.org/abs/2502.10517), [GPU Kernel Scientist](https://arxiv.org/abs/2506.20807), [CudaForge](https://arxiv.org/abs/2511.01884) | Claude Code subagents/hooks, Codex cloud/app/[CLI](https://developers.openai.com/codex/cli) (command-line interface) workflows, Nsight-guided agent loops | `paper` + `official docs/blog/report` |

## Evidence and Benchmark Normalization

This paper treats benchmark claims as evidence about a specific context, not as universal rankings. When a source reports a speedup, the source should be read with its hardware, precision, shape family, and baseline attached.

| Source Claim | Context To Preserve | What It Supports | What It Does Not Prove | Evidence Grade |
|---|---|---|---|---|
| [FA4](https://arxiv.org/abs/2603.05451) (FlashAttention-4) reports Blackwell speedups and high utilization | B200, [BF16](https://nvidia.github.io/TransformerEngine/examples/fp8_primer.html) (bfloat16) attention, paper/blog figure-specific sequence and head shapes, baselines including cuDNN $9.13$, Triton, and FlashAttention-2; the FA4 blog also notes newer cuDNN versions adopted many optimizations and can be similar | Blackwell attention bottlenecks moved toward exponentials, shared-memory traffic, TMEM, $2$-CTA backward, and scheduling | End-to-end pretraining speedup for arbitrary models or non-Blackwell GPUs | `paper` + `official blog` |
| FA4 deterministic backward reaches near nondeterministic throughput in reported benchmarks | FA4 blog reports deterministic backward at about $85\text{--}90\%$ of nondeterministic throughput; context is FA4 backward on Blackwell attention | Semaphore-style deterministic reduction can be practical when paired with [CTA](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#thread-hierarchy) (cooperative thread array/thread block) swizzling and [SPT](https://en.wikipedia.org/wiki/Shortest_job_next) (shortest-processing-time-first scheduling) ordering | Deterministic reductions are always cheap in other kernels or hardware generations | `official blog` |
| DeepSeek-V4 model card reports $10^6$-token context, [CSA](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro) (Compressed Sparse Attention)/[HCA](https://developer.nvidia.com/blog/build-with-deepseek-v4-using-nvidia-blackwell-and-gpu-accelerated-endpoints/) (Heavily Compressed Attention), mHC, Muon, and $\mathrm{FP4}+\mathrm{FP8}$ mixed precision | Official model-level report: V4-Pro has $1.6\mathrm{T}$ total and $49\mathrm{B}$ active parameters; V4-Flash has $284\mathrm{B}$ total and $13\mathrm{B}$ active; instruct weights use $\mathrm{FP4}$ expert parameters plus mostly $\mathrm{FP8}$ elsewhere | Architecture and precision choices imply kernel pressure around sparse/hybrid attention, residual mixing, optimizer kernels, and low-precision layout | Exact pretraining CUDA kernels, schedules, or measured kernel speedups | `official docs/blog/report` |
| mHC reports efficient infrastructure and $6.7\%$ additional time overhead | [mHC](https://arxiv.org/abs/2512.24880) (Manifold-Constrained Hyper-Connections) paper reports Sinkhorn-Knopp projection, TileLang/mixed-precision kernels, selective recomputation, DualPipe overlap, and in-house large-scale training with expansion rate $4$ | Residual-path mixing can be made trainable with fused small-matrix kernels and recomputation | DeepSeek-V4 production mHC kernels are identical to the paper or to third-party runtime integrations | `paper` |
| Dynamic Context Parallelism reports THD layout and microbatch-specific CP | NVIDIA blog describes variable-length packing, [THD](https://developer.nvidia.com/blog/speeding-up-variable-length-training-with-dynamic-context-parallelism-and-nvidia-megatron-core/) (token-head-dimension packed layout), `PackedSeqParams` (packed-sequence metadata) carrying `cp_size` and `cp_group`, broadcast of `num_micro_batches`, `max_seqlen`, and `cu_seqlens`, plus cost-model/solver/simulator scheduling | Variable-length long-context training needs kernel metadata and distributed scheduler support together | A single attention kernel can solve all variable-length imbalance without scheduler changes | `official blog` |
| NVIDIA Megatron Muon blog reports throughput under Muon and AdamW | [GB300 NVL72](https://www.nvidia.com/en-us/data-center/gb300-nvl72/) (NVIDIA rack-scale Blackwell system), [MXFP8](https://nvidia.github.io/TransformerEngine/features/low_precision_training/mxfp8/mxfp8.html) (microscaling FP8), Kimi K2 on $256$ GPUs with [PP](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html) (pipeline parallelism) $=4$ / [DP](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html) (data parallelism) $=64$ / [EP](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/moe.html) (expert parallelism) $=64$ and Qwen3 $30\mathrm{B}$ on $8$ GPUs with [DP](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html) $=8$ / [EP](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/moe.html) $=8$; baseline is AdamW in the same stack | Matrix optimizers can be integrated into large-scale training with duplicated, distributed, and blockwise Newton-Schulz modes | DeepSeek-V4's undisclosed optimizer implementation or convergence claims beyond the cited systems | `official blog` |
| KernelBench/CudaForge measure LLM-generated kernel loops | KernelBench: $250$ PyTorch ML workloads and PyTorch baselines; CudaForge: KernelBench-style evaluation across A100, RTX 6000, RTX 4090, and RTX 3090 with Coder/Judge plus Nsight feedback | Agentic generate-test-profile loops are measurable and useful for bounded kernels | Unsupervised production adoption for distributed training kernels, attention backward, or $\mathrm{FP8}/\mathrm{FP4}$ scale-management code | `paper` |

## Operational Triage Guide

When prioritizing which evidence or implementation to inspect first, start from the observed bottleneck rather than from the framework name.

| Observed Symptom | First Technique To Inspect | Why This Is The Likely Root | Initial Sources | Evidence Grade |
|---|---|---|---|---|
| Activation memory grows as $O(n^2)$ with context length | IO-aware attention, probability-free backward, sparse/hybrid attention | Scores or probabilities are being materialized or recomputed with the wrong contract | [FlashAttention](https://arxiv.org/abs/2205.14135), [DeepSeek-V4](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro), [cuDNN NSA](https://docs.nvidia.com/deeplearning/cudnn/frontend/latest/fe-oss-apis/nsa.html) | `paper` + `official docs/blog/report` |
| Tensor cores are underused while attention dominates | Asynchronous tensor-core/data-movement pipelining and softmax bottleneck reduction | Non-MMA work such as copies, exponentials, rescaling, or shared-memory traffic is on the critical path | [FA3](https://arxiv.org/abs/2407.08608), [FA4](https://arxiv.org/abs/2603.05451), [FA4 blog](https://tridao.me/blog/2026/flash4/) | `paper` + `official blog` |
| Low-precision training is fast but unstable | Scale policy, amax synchronization, stochastic rounding, and quantized epilogues | The failure is often scale metadata, clipping, or stale dynamic range rather than matmul throughput | [Transformer Engine](https://docs.nvidia.com/deeplearning/transformer-engine-releases/release-2.8/user-guide/index.html), [COAT](https://arxiv.org/abs/2410.19313), [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) | `official docs/repo` + `paper` |
| MoE layers have long tail latency | Routing histogram, all-to-all overlap, grouped GEMM scheduling, and SM budgeting | Expert imbalance couples token permutation, communication volume, and variable-size expert matmuls | [DeepEP](https://github.com/deepseek-ai/DeepEP), [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM), [MegaBlocks](https://arxiv.org/abs/2211.15841) | `official repo` + `paper` |
| Variable-length batches cause data-parallel or pipeline stragglers | THD packing, cost-model scheduling, and microbatch-specific [CP](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/context_parallel.html) (context parallelism) | Token balance does not equal attention-[FLOP](https://en.wikipedia.org/wiki/FLOPS) (floating-point-operation) or activation-memory balance | [Dynamic Context Parallelism](https://developer.nvidia.com/blog/speeding-up-variable-length-training-with-dynamic-context-parallelism-and-nvidia-megatron-core/), [MegaScale](https://arxiv.org/abs/2402.15627) | `official blog` + `paper` |
| Optimizer time is no longer negligible | Multi-tensor fused updates, sharded optimizer state, $\mathrm{FP8}$ state compression, or Muon matrix kernels | Once attention/GEMM are optimized, optimizer memory traffic and matrix preconditioning become visible | [Apex FusedAdam](https://nvidia.github.io/apex/optimizers.html), [COAT](https://arxiv.org/abs/2410.19313), [NVIDIA Muon blog](https://developer.nvidia.com/blog/advancing-emerging-optimizers-for-accelerated-llm-training-with-nvidia-megatron/) | `official docs/blog/report` + `paper` |
| A new custom kernel is plausible but risky | Correctness oracle, benchmark harness, generated-code inspection, and agentic generate-test-profile loop | Candidate code needs evidence before speed claims are meaningful | [KernelBench](https://arxiv.org/abs/2502.10517), [CudaForge](https://arxiv.org/abs/2511.01884), [PyTorch custom ops](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html) | `paper` + `official docs/blog/report` |

## 1. IO-Aware Attention and Online Softmax

### Technique

The IO-aware attention technique computes exact attention without materializing the full $QK^\top$ (query-key transpose product) score matrix or the full softmax probability matrix in HBM. The kernel streams blocks of Q, K, and V (query, key, and value tensors) through on-chip memory, maintains each row's running maximum and normalization sum, and updates the output tile incrementally. This is the core FlashAttention idea: optimize reads and writes between HBM and [SRAM](https://en.wikipedia.org/wiki/Static_random-access_memory) (static random-access memory used here as a shorthand for on-chip storage) rather than only counting FLOPs.

Mathematically, the dense operator being preserved is:

$$
S = \frac{QK^\top}{\sqrt{d_k}} + M,\qquad
P = \operatorname{softmax}(S),\qquad
O = PV,
$$

where $d_k$ is the key dimension and $M$ is an additive mask whose invalid entries are usually $-\infty$ or a numerically safe sentinel. IO-aware attention changes the schedule and storage of $S$ and $P$; it does not approximate this formula.

Symbols: $Q$, $K$, and $V$ are query, key, and value matrices for one attention head or head group; $K^\top$ is the transposed key matrix; $S$ is the scaled score matrix; $P$ is the row-wise softmax probability matrix; $O$ is the attention output; $\operatorname{softmax}$ normalizes each score row; and $\sqrt{d_k}$ is the standard attention scaling factor.

### Primitive Techniques

#### [Q/K/V](https://arxiv.org/abs/1706.03762) (Query/Key/Value) Block Streaming

Q/K/V block streaming partitions the query rows and key/value columns into tiles that fit in the GPU's on-chip storage budget instead of constructing sequence-length by sequence-length tensors. A forward kernel typically assigns a block of query rows to a [CTA](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#thread-hierarchy) (cooperative thread array, CUDA's thread-block abstraction), streams one or more K/V tiles across that query tile, and writes only the final attention output for those rows. The full score matrix $S=QK^\top$ and probability matrix $P=\operatorname{softmax}(S)$ are logical objects, not HBM-resident activations.

The tile-level dataflow is:

$$
\begin{array}{c}
Q_{I,:}\in\mathbb{R}^{B_q\times d}
\quad
K_{J,:},V_{J,:}\in\mathbb{R}^{B_k\times d}
\\[2mm]
\Downarrow
\\[-1mm]
S_{I,J}=Q_{I,:}K_{J,:}^{\top}\in\mathbb{R}^{B_q\times B_k}
\\[2mm]
\Downarrow
\\[-1mm]
\left(m_I,\ell_I,A_I\right)
\leftarrow
\operatorname{OnlineSoftmaxUpdate}
\left(S_{I,J},V_{J,:}\right)
\\[2mm]
\Downarrow
\\[-1mm]
O_{I,:}=A_I/\ell_I
\end{array}
$$

Diagram symbols: $I$ is a query-row tile, $J$ is a streamed key/value tile, $B_q$ and $B_k$ are the query and key/value tile row counts, $d$ is the head dimension, $S_{I,J}$ is the temporary score tile, $m_I$ is the row maximum, $\ell_I$ is the row denominator, $A_I$ is the unnormalized output accumulator, and only $O_{I,:}$ plus compact row statistics survive beyond the kernel.

In training, this is the primitive that converts attention activation memory from quadratic storage into a compact output plus softmax-statistics contract. It also sets the shape of the backward pass: because the probabilities were not saved, backward revisits the same Q/K/V tile schedule and reconstructs local score/probability tiles as needed.

CUDA details that matter are tile size, head dimension, data layout, and occupancy. Tiles must leave room for Q/K/V staging, output accumulators, per-row softmax state, masks, and synchronization structures; larger tiles reduce HBM traffic but can reduce resident CTAs through register and shared-memory pressure. The [FlashAttention repository](https://github.com/Dao-AILab/flash-attention) README is a useful implementation entry point because it documents CUDA support, head-dimension limits, packed-QKV interfaces, variable-length paths, and the separate Hopper/CuTe-DSL implementations. Inspect first: [FlashAttention](https://arxiv.org/abs/2205.14135) for the IO model and the [FlashAttention repository](https://github.com/Dao-AILab/flash-attention) for the public CUDA-facing implementation surface.

#### On-Chip Score Tile

An on-chip score tile is the temporary $QK^\top$ submatrix produced for one Q tile and one K tile. The tile is scaled, masked, reduced through softmax, and immediately consumed by the $PV$ update. It should be thought of as an ephemeral register/shared-memory artifact rather than an activation tensor.

In LLM training kernels, this tile is where the dense attention formula is fused into a single schedule: tensor-core or [SIMT](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#simt-architecture) (single-instruction, multiple-thread) matrix multiply produces scores, mask logic invalidates disallowed positions, softmax converts the tile into probabilities relative to the running row statistics, and the result contributes to the output accumulator. Keeping this whole chain inside the kernel avoids separate score, mask, softmax, dropout, and value-matmul launches.

CUDA details center on fragment layout and numerical format. Implementations commonly stage Q/K/V through shared memory, hold score fragments and accumulators in registers, and use [FP32](https://en.wikipedia.org/wiki/Single-precision_floating-point_format) (32-bit floating point) or equivalent wider intermediates for reductions even when input/output tensors are [FP16](https://en.wikipedia.org/wiki/Half-precision_floating-point_format) (16-bit floating point), BF16, or lower precision. Bank conflicts, shared-memory swizzles, vectorized loads, and tensor-core tile shapes determine whether the score tile is cheap enough to recompute in backward. Inspect first: [FlashAttention](https://arxiv.org/abs/2205.14135) for why scores are consumed without HBM writes and [cuDNN SDPA](https://docs.nvidia.com/deeplearning/cudnn/frontend/latest/operations/Attention.html) for NVIDIA's graph-level exposure of fused scaled dot-product attention.

#### Online Max/Sum Softmax Recurrence

The online softmax recurrence maintains two row statistics while K/V blocks stream by: a running maximum $m$ and a running denominator $\ell$. For a new score block, the kernel computes $m_{\mathrm{new}}=\max(m_{\mathrm{old}}, \max(S_{\mathrm{block}}))$ and rescales the old denominator into the new numerical frame before adding the new block's exponentials. This produces the same softmax as dense attention, but it requires only per-row state rather than all scores.

For score block $S_j$ and value block $V_j$, a row-wise recurrence can be written as:

$$
m_j = \max\left(m_{j-1}, \max(S_j)\right),
$$

$$
\ell_j =
e^{m_{j-1}-m_j}\ell_{j-1}
 + \sum_{x \in S_j} e^{x-m_j},
$$

$$
A_j =
e^{m_{j-1}-m_j}A_{j-1}
 + e^{S_j-m_j}V_j,\qquad
O = \frac{A_T}{\ell_T}.
$$

Symbols: $j$ indexes the current streamed K/V block; $S_j$ and $V_j$ are that block's score and value tiles; $m_j$ is the running row maximum after block $j$; $m_{j-1}$ is the previous maximum; $\ell_j$ is the running softmax denominator; $A_j$ is the unnormalized output accumulator; $x$ ranges over scores in $S_j$; $T$ is the final streamed block; and $O=A_T/\ell_T$ is the normalized attention output. CUDA implementations often use base-2 exponentials, but the recurrence is the same after multiplying scores by $\log_2 e$, where $e$ is Euler's number.

Training kernels use this recurrence in forward to generate exact outputs and in backward to reconstruct local probabilities from Q, K, V, and the saved log-sum-exp/statistics. The recurrence is also the reason IO-aware attention remains exact: each tile is normalized against the final row-wise softmax denominator even though the denominator was discovered incrementally.

CUDA details that matter are the precision and placement of the row state. The row maximum and denominator are normally kept in FP32-like accumulators, reduced across lanes or warps, and updated in a way that avoids overflow for long contexts. Exponential throughput can become a bottleneck on newer GPUs, which is why later FlashAttention variants pay close attention to rescaling and softmax scheduling. Inspect first: [FlashAttention](https://arxiv.org/abs/2205.14135) for the recurrence and [FlashAttention-2](https://arxiv.org/abs/2307.08691) for the work-partitioning refinements around non-matmul work.

#### Output Accumulator Rescaling

Output accumulator rescaling is the companion to online softmax. The kernel maintains an unnormalized output accumulator for each query row; when a later score tile raises the running maximum, the old accumulator must be multiplied by $e^{m_{\mathrm{old}}-m_{\mathrm{new}}}$ before the new tile contribution is added. After all K/V blocks have been visited, the accumulator is divided by the final denominator.

This primitive preserves equivalence to dense $\operatorname{softmax}(QK^\top)V$ while allowing $PV$ to be accumulated tile by tile. It is especially important for training because the same algebra appears again when backward recomputes local probabilities and gradient terms from saved statistics rather than stored probabilities.

CUDA details are surprisingly concrete: accumulator rescaling consumes registers, row-wise broadcasts, and exponential or `exp2` work that does not run on tensor cores. On architectures where tensor-core throughput grows faster than special-function or scalar throughput, rescaling can become a visible cost; FlashAttention-4's conditional rescaling work is a later example of optimizing this primitive rather than changing the mathematical attention result. Inspect first: [FlashAttention](https://arxiv.org/abs/2205.14135) for the accumulator update and [FlashAttention-4](https://arxiv.org/abs/2603.05451) for the Blackwell-era rescaling pressure.

#### Softmax Statistics [ABI](https://en.wikipedia.org/wiki/Application_binary_interface) (Application Binary Interface)

The softmax statistics ABI is the forward/backward contract that replaces saved probability matrices. Instead of storing $P$, the forward pass stores compact row statistics such as log-sum-exp. Backward receives Q, K, V, the forward output, $dO$ (gradient of output), and these statistics, then reconstructs the local probabilities needed for $dQ$, $dK$, and $dV$.

This ABI is visible in production APIs. cuDNN SDPA can generate a training `stats` tensor in forward and requires that tensor for SDPA backward; the latest documentation exposes the stats tensor as FP32 with per-batch, per-head, per-query-row shape. The FlashAttention Python interface similarly surfaces `softmax_lse` as a returned tensor from the CUDA operator, while full probabilities are optional and primarily useful for diagnostics or dropout-related paths.

CUDA/platform details include shape conventions, mask alignment, dropout/[RNG](https://en.wikipedia.org/wiki/Random_number_generation) (random-number-generator) state, and determinism. The saved statistics are only valid for the exact scale, mask, sequence lengths, and layout used by forward, so backward APIs need those attributes to match. Deterministic backward modes may use different schedules or extra memory to avoid nondeterministic accumulation. Inspect first: [cuDNN SDPA](https://docs.nvidia.com/deeplearning/cudnn/frontend/latest/operations/Attention.html) for the `stats` training interface and [`flash-attention`](https://github.com/Dao-AILab/flash-attention) for the `softmax_lse` interface and deterministic backward option.

#### Probability-Free Backward

Probability-free backward recomputes the local score and probability tiles during backpropagation instead of reading a saved $P$ tensor. For each tile, the kernel reconstructs $S_{\mathrm{block}}=Q_{\mathrm{block}}K_{\mathrm{block}}^\top$, applies the same scale and mask, subtracts the saved row log-sum-exp, and regenerates the probabilities needed for gradients such as $dV=P^\top dO$, $dP=dO\,V^\top$, and the softmax derivative feeding $dQ$ and $dK$.

For one recomputed tile, the usual dense-gradient identities are:

$$
dV = P^\top dO,\qquad
dP = dO\,V^\top,
$$

$$
D_i = \sum_j P_{ij}dP_{ij},\qquad
dS_{ij} = P_{ij}\left(dP_{ij}-D_i\right),
$$

$$
dQ = \frac{1}{\sqrt{d_k}}dS\,K,\qquad
dK = \frac{1}{\sqrt{d_k}}dS^\top Q.
$$

Symbols: $dO$ is the upstream gradient of the attention output; $dV$, $dP$, $dS$, $dQ$, and $dK$ are gradients with respect to values, probabilities, scores, queries, and keys; $P_{ij}$ and $dP_{ij}$ are row $i$, column $j$ probability and probability-gradient entries; $D_i$ is the row-wise dot product $\sum_j P_{ij}dP_{ij}$ used by the softmax derivative; and $d_k$ is the key dimension used in the forward scaling.

FlashAttention-style backward evaluates these identities tile by tile and reduces partial $dQ$, $dK$, and $dV$ contributions according to the chosen work partition.

In training, this trades additional matrix multiply and softmax work for a large activation-memory reduction. The trade is attractive because Q, K, V, output, and compact row statistics are far smaller than the full probability matrix at long sequence lengths. It also makes recomputation a first-class kernel-authoring concern: the backward pass is not merely the derivative formula, but a second IO-aware schedule with its own reductions and write conflicts.

CUDA details include split reductions, accumulation order, and atomic avoidance. Multiple CTAs may contribute to the same $dK$ or $dV$ when work is split over query blocks, so high-performance backward kernels need deterministic or nondeterministic reduction strategies, careful staging of $dO$, and a policy for dropout masks or RNG replay when dropout is enabled. Inspect first: [FlashAttention-2](https://arxiv.org/abs/2307.08691) for backward work partitioning and [cuDNN SDPA backward](https://docs.nvidia.com/deeplearning/cudnn/frontend/latest/operations/Attention.html) for the API contract that passes forward tensors plus stats into backward.

#### Tile-Local Masking

Tile-local masking applies causal, padding, sliding-window, [ALiBi](https://arxiv.org/abs/2108.12409) (Attention with Linear Biases), ragged-sequence, or block-sparse constraints inside the score tile before the online softmax update. Masked positions are removed from the local maximum and denominator, usually by replacing their score with a numerically safe negative sentinel before exponentiation.

Training kernels use tile-local masking to keep attention variants fused. A causal decoder kernel does not need a separate triangular-mask tensor, a variable-length batch does not need to materialize padded scores, and a local-window attention path can reuse the same online recurrence while skipping out-of-window columns. The key requirement is that forward and backward apply identical mask semantics so the saved statistics describe the same logical scores that backward recomputes.

CUDA/platform details include diagonal alignment, fully masked rows, and block-mask granularity. cuDNN exposes causal and padding options, variable sequence lengths, diagonal alignment controls, and a Flex Attention block-mask path whose implementation assumes a 128 by 128 block size. FlashAttention's public interface documents causal, windowed, ALiBi, variable-length, [MQA](https://arxiv.org/abs/1911.02150) (multi-query attention)/[GQA](https://arxiv.org/abs/2305.13245) (grouped-query attention), and paged-cache variants, which are useful places to inspect how mask metadata becomes kernel parameters. Inspect first: [cuDNN attention](https://docs.nvidia.com/deeplearning/cudnn/frontend/latest/operations/Attention.html), [cuDNN NSA](https://docs.nvidia.com/deeplearning/cudnn/frontend/latest/fe-oss-apis/nsa.html), and [`flash-attention`](https://github.com/Dao-AILab/flash-attention).

#### Row/Head/Block Work Partitioning

Row/head/block work partitioning decides which CTA, warp, or warpgroup owns each slice of the attention computation. The simple mapping is one CTA per batch-head-query tile, but long sequences, small batch sizes, causal masks, and backward reductions often need additional parallelism over query blocks, key blocks, heads, or warps within a block.

In LLM training kernels, this primitive is the difference between an IO-efficient algorithm and a fast GPU program. FlashAttention-2 identifies poor occupancy and unnecessary shared-memory traffic as major remaining costs after FA1, then improves parallelism by splitting work across thread blocks even for a single head and by distributing work across warps inside a block to reduce communication. Backward adds further constraints because partial $dQ$, $dK$, and $dV$ results must be reduced without excessive atomics or scratch memory.

CUDA details include CTA residency, warp-level reductions, shared-memory traffic, and load balance under masks. Causal attention creates triangular work where early and late query blocks see different numbers of valid keys; grouped-query attention changes the relationship between Q heads and K/V heads; variable-length batches create uneven row counts. Hopper and Blackwell implementations also expose warpgroup scheduling choices that are not present in older Ampere-style kernels. Inspect first: [FlashAttention-2](https://arxiv.org/abs/2307.08691) for the work-partitioning analysis and [FlashAttention-3](https://arxiv.org/abs/2407.08608) for Hopper-era scheduling context.

### Why It Matters For Training

Training needs both forward activations and backward gradients. Standard attention can require quadratic activation storage in sequence length. IO-aware attention changes the memory contract: save compact softmax statistics and recompute score/probability tiles in backward. This makes long context and larger batch sizes practical without approximating attention.

### Explore Further

- `paper`: [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
- `paper`: [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691)
- `official repo`: [FlashAttention repository](https://github.com/Dao-AILab/flash-attention)
- `official docs`: [cuDNN Frontend Attention / SDPA](https://docs.nvidia.com/deeplearning/cudnn/frontend/latest/operations/Attention.html)

### Implementation Examples

- [FlashAttention repository](https://github.com/Dao-AILab/flash-attention): inspect the main repository for the CUDA and Hopper implementations. The README documents FA1/FA2 usage, FA3 beta, and FA4 CuTe-DSL usage.
- [cuDNN Frontend SDPA](https://docs.nvidia.com/deeplearning/cudnn/frontend/latest/operations/Attention.html): inspect how NVIDIA exposes SDPA as a graph operation with a `stats` tensor for training backward.
- FlashAttention-2 paper: inspect the work-partitioning changes that reduce non-matmul FLOPs and improve single-head parallelism.

### Caveats

The IO-aware algorithm is portable in principle, but the best implementation is not portable line-for-line. A100, H100, and B200 require different scheduling choices because tensor-core throughput, shared-memory bandwidth, and special-function throughput scale differently.

## 2. Asynchronous Tensor-Core and Data-Movement Pipelining

### Technique

Modern attention and GEMM kernels overlap data movement with matrix multiply. On Hopper, the key primitives are [WGMMA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html) (warpgroup matrix multiply-accumulate, an asynchronous tensor-core instruction family) and [TMA](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/0z_tma_tensors.html) (Tensor Memory Accelerator, a multidimensional asynchronous copy engine) for global-to-shared transfers. Kernels split work across producer warps that move data and consumer warps that issue tensor-core work. On Blackwell, [FA4](https://arxiv.org/abs/2603.05451) (FlashAttention-4) uses fully asynchronous [UMMA](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html) (Blackwell/CUTLASS term for the `tcgen05` tensor-core matrix-multiply path) operations, tensor memory, and larger tile shapes.

### Primitive Techniques

#### Warpgroup Tensor-Core Issue

Warpgroup tensor-core issue is the act of feeding matrix-multiply work to tensor cores at the granularity expected by the newest NVIDIA architectures. On Hopper, the primitive is `wgmma.mma_async`: a warpgroup-level asynchronous [MMA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html) (matrix multiply-accumulate) instruction, normally involving four warps that cooperate on a larger tile than older warp-level `mma.sync` instructions. On Blackwell, the analogous path is the 5th-generation tensor-core family exposed in [PTX](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html) (Parallel Thread Execution, NVIDIA's virtual instruction set architecture) and CUTLASS/CuTe as `tcgen05.mma` or UMMA-style operations. The key idea is not just a larger MMA tile; it is that the matrix operation can be launched and then synchronized later, giving the kernel scheduler space to issue loads, stores, reductions, or softmax work while the tensor-core pipeline is busy.

In LLM training kernels, this primitive is the compute heart of attention and dense or expert GEMMs. Attention forward uses it for $QK^\top$ and $PV$; attention backward chains MMAs for recomputing scores and for $dQ$, $dK$, $dV$, and related gradient products. MLP and MoE kernels use the same idea in dense, grouped, or block-scaled GEMM loops. The authoring question is therefore how to keep WGMMA/UMMA issue slots full while avoiding register spills and synchronization bubbles around the non-MMA work.

For ordinary tiled matrix multiplication, one CTA or warpgroup usually owns an output tile $C_{I,J}$ and streams the reduction dimension in panels:

$$
\begin{array}{c}
C_{I,J}
\;=\;
\displaystyle\sum_{r=0}^{R-1}
A_{I,K_r}B_{K_r,J}
\\[3mm]
\begin{array}{c|cccc}
 & K_0 & K_1 & \cdots & K_{R-1}\\ \hline
A_{I,:} & A_{I,K_0} & A_{I,K_1} & \cdots & A_{I,K_{R-1}}\\
B_{:,J} & B_{K_0,J} & B_{K_1,J} & \cdots & B_{K_{R-1},J}
\end{array}
\end{array}
$$

The GPU-kernel question is how many $K_r$ panels can be prefetched, staged, and multiplied while the output accumulator for $C_{I,J}$ stays resident.

Diagram symbols: $C_{I,J}$ is the output tile with row tile $I$ and column tile $J$; $A_{I,K_r}$ and $B_{K_r,J}$ are the $r$-th reduction panels of input matrices $A$ and $B$; $K_r$ names a slice of the reduction dimension; and $R$ is the number of streamed reduction panels.

Platform details matter. Hopper WGMMA generally consumes descriptors for shared-memory operands, imposes layout and swizzle constraints, and needs explicit ordering around asynchronous groups. Blackwell `tcgen05.mma` changes the tradeoff: accumulators live in tensor memory, operand $A$ can be sourced from tensor memory or shared memory, operand $B$ is sourced from shared memory, and `cta_group::1` or `cta_group::2` selects whether one CTA or a CTA pair participates. Blackwell also adds native support for narrow and block-scaled formats, so tile shape, scale-factor layout, and tensor-memory placement become part of the MMA contract rather than epilogue bookkeeping.

Inspect first: [FlashAttention-3](https://arxiv.org/abs/2407.08608) for Hopper WGMMA/TMA attention scheduling, [FlashAttention-4](https://arxiv.org/abs/2603.05451) and the [FA4 blog](https://tridao.me/blog/2026/flash4/) for Blackwell UMMA/TMEM changes, the NVIDIA PTX documentation for [`wgmma.mma_async` and `tcgen05.mma`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html), the [CUTLASS Blackwell SM100 GEMM notes](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html) ([SM100](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html) is the Blackwell-generation streaming-multiprocessor target used by CUTLASS), FA3's [forward mainloop implementation](https://github.com/Dao-AILab/flash-attention/blob/main/hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp) ([SM90](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/hopper_gemm.html) is the Hopper-generation streaming-multiprocessor target), and FA4's [Blackwell forward kernel](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/cute/flash_fwd_sm100.py).

#### [TMA](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/0z_tma_tensors.html) (Tensor Memory Accelerator) / Bulk Async Copies

TMA, the Tensor Memory Accelerator, is Hopper's multidimensional asynchronous copy path for moving tensor tiles between global memory and shared memory. At the PTX level this is exposed through `cp.async.bulk.tensor`-style instructions and tensor-map descriptors; at the CUTLASS/CuTe level it appears as TMA tensors and copy atoms. Compared with ordinary per-thread global loads, TMA can offload address calculation, out-of-bounds predication, and bulk movement to a hardware copy path, which frees registers and instruction slots for the warps that are doing math.

Training kernels use TMA to prefetch the next tiles of $Q$, $K$, $V$, MLP weights, activation tiles, or expert GEMM operands while tensor cores consume the current tile. In attention, a producer warpgroup can issue TMA loads for future $K$ and $V$ blocks while consumer warpgroups compute the current $QK^\top$, run softmax, and issue $PV$. In GEMM and MoE kernels, TMA carries the role that hand-vectorized global-to-shared load loops used to fill, but with better support for large tiles and irregular tensor layouts.

The hard CUDA details are synchronization and layout. TMA completion is usually tracked with memory barriers, and data produced through the asynchronous proxy must be fenced before ordinary shared-memory consumers rely on it. Tensor maps describe rank, stride, swizzle, bounds, and element type, so a kernel author must align the TMA tile layout with the WGMMA/UMMA operand layout. TMA can also prefetch to [L2](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory-l2-access-management) (level-2 cache) before the shared-memory transfer, but prefetch distance is shape- and occupancy-dependent; too much lookahead consumes shared memory and barriers that could otherwise hold more CTAs or larger accumulator fragments.

Inspect first: the [PyTorch FA3 blog](https://docs.pytorch.org/blog/flashattention-3/) for the attention-level explanation of TMA, WGMMA, and warp specialization; the NVIDIA PTX documentation for [`cp.async.bulk.tensor`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html); CUTLASS/CuTe's [TMA tensor tutorial](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/0z_tma_tensors.html); FA3's [forward mainloop implementation](https://github.com/Dao-AILab/flash-attention/blob/main/hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp); and [ThunderKittens](https://github.com/HazyResearch/ThunderKittens) for compact tile abstractions over WGMMA/TMA-style copies.

#### Producer-Consumer Warp Specialization

Producer-consumer warp specialization assigns different warps or warpgroups inside one CTA to different jobs. A producer group is responsible for feeding the pipeline, typically by issuing TMA or bulk copies and advancing shared-memory buffers. Consumer groups issue WGMMA/UMMA, run softmax or elementwise work, and store results. This is a software scheduling strategy that exposes the hardware's independent copy, tensor-core, and CUDA-core pipelines instead of relying on one uniform group of warps to interleave every instruction stream.

LLM training kernels use this pattern when the operation mixes fast matrix multiply with slower data movement or scalar work. FlashAttention is the canonical example: one set of warps can prepare $K$/$V$ tiles while another set computes $QK^\top$, and other warps or later phases can reduce row maxima, evaluate exponentials, or rescale outputs. GEMM kernels use the same split more simply: producers keep $A$ and $B$ tiles arriving, consumers run the K-loop MMAs and epilogue. MoE kernels can combine the split with routing or grouped scheduling so that expert tiles are loaded without starving tensor cores.

The important platform detail is that specialization increases scheduling freedom but also increases the synchronization surface. Hopper WGMMA operates at warpgroup granularity, so consumer groups must be sized and synchronized around 128-thread warpgroup semantics. Producers and consumers communicate through shared memory and named barriers or memory barriers; any mismatch between stage count, barrier arrival count, or buffer reuse can turn a performance optimization into a deadlock. On Blackwell, UMMA can be launched differently and accumulates in tensor memory, but the same architectural idea remains: choose which warps launch copies, which issue MMA, and which run non-MMA work so that bottleneck units overlap rather than serialize.

Inspect first: [FlashAttention-3](https://arxiv.org/abs/2407.08608) for producer-consumer asynchrony in attention, the [PyTorch warp-specialization blog](https://pytorch.org/blog/warp-specialization/) for a compiler and Triton view of the same technique, FA3's [backward mainloop implementation](https://github.com/Dao-AILab/flash-attention/blob/main/hopper/mainloop_bwd_sm90_tma_gmma_ws.hpp), FA4's [pipeline helpers](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/cute/pipeline.py), and [ThunderKittens](https://github.com/HazyResearch/ThunderKittens) for a minimal DSL-style surface.

#### Multi-Stage Shared-Memory Pipeline

A multi-stage shared-memory pipeline is a circular buffer over tiles. Stage 0 might be ready for MMA, stage 1 might be receiving a TMA load, and stage 2 might be waiting to be overwritten after consumers finish. A two-stage version is often called ping-pong buffering; deeper versions add more lookahead to hide longer memory or synchronization latency. The primitive is not the buffer alone but the protocol tying together load issue, barrier arrival, wait, MMA consumption, and buffer release.

One way to visualize the schedule is as a time-skewed circular buffer:

$$
\begin{array}{c|ccc}
\text{time} & \text{stage 0} & \text{stage 1} & \text{stage 2}\\ \hline
t       & \operatorname{MMA}(K_r)      & \operatorname{TMA}(K_{r+1}) & \operatorname{free}\\
t+1     & \operatorname{free}          & \operatorname{MMA}(K_{r+1}) & \operatorname{TMA}(K_{r+2})\\
t+2     & \operatorname{TMA}(K_{r+3})  & \operatorname{free}         & \operatorname{MMA}(K_{r+2})
\end{array}
$$

The labels are schematic: real kernels also carry barriers, swizzled shared-memory layouts, and separate producer/consumer warpgroup roles.

Diagram symbols: $t$, $t+1$, and $t+2$ are successive pipeline time steps; each stage is one slot in a circular shared-memory buffer; $\operatorname{TMA}(K_r)$ means asynchronously loading the $K_r$ operand panel; $\operatorname{MMA}(K_r)$ means issuing tensor-core multiply-accumulate work for that panel; and `free` means the stage can be reused once its producer/consumer dependencies are satisfied.

In training kernels, these stages are what let one tile arrive while another tile is multiplied and a previous tile is reduced, normalized, or stored. In attention forward, $K$ and $V$ tiles can be staged ahead while the current score tile is being processed. In backward, the same concept applies to recomputed scores, probability tiles, $dO$, and gradient operands. In dense and grouped GEMM, a multi-stage K-loop hides global-memory latency and keeps operand tiles close to the tensor cores.

CUDA details determine whether the pipeline is useful. More stages increase latency tolerance, but each stage costs shared memory and often barriers, which can reduce occupancy or force smaller MMA tiles. Stage buffers must use layouts compatible with WGMMA/UMMA descriptors, and they often need swizzles to avoid shared-memory bank conflicts. Barriers must distinguish producer completion from consumer release; reusing a buffer before all consumers have finished is a correctness bug, while waiting too conservatively leaves tensor cores idle.

Inspect first: the [PyTorch FA3 blog](https://docs.pytorch.org/blog/flashattention-3/) for ping-pong and intra-warpgroup overlap, CUTLASS/CuTe's [pipeline and synchronization documentation](https://docs.nvidia.com/cutlass/), FA3's [SM90 pipeline helpers](https://github.com/Dao-AILab/flash-attention/blob/main/hopper/sm90_pipeline_no_cluster.hpp) and [named-barrier helpers](https://github.com/Dao-AILab/flash-attention/blob/main/hopper/named_barrier.hpp), and FA4's [pipeline helpers](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/cute/pipeline.py).

#### Ping-Pong And Double-Buffered Schedules

Ping-pong scheduling is the two-buffer special case of a staged pipeline. The kernel allocates two reusable staging locations, usually shared-memory tiles, [TMEM](https://docs.nvidia.com/cuda/parallel-thread-execution/#tensor-memory) (tensor memory), or symmetric-memory windows. At logical step $t$, the consumer drains slot $b(t)=t\bmod 2$ while the producer fills slot $1-b(t)$ for the next step. This is the smallest schedule that can overlap movement and math without overwriting live data.

For tiled matrix multiplication, the schedule is:

$$
\begin{array}{c|cc|cc}
\text{time} & S_0 & S_1 & \text{consumer} & \text{producer}\\ \hline
t       & A_{r},B_{r} & A_{r+1},B_{r+1} & \operatorname{MMA}(A_r,B_r) & \operatorname{copy}(A_{r+1},B_{r+1})\\
t+1     & A_{r+2},B_{r+2} & A_{r+1},B_{r+1} & \operatorname{MMA}(A_{r+1},B_{r+1}) & \operatorname{copy}(A_{r+2},B_{r+2})\\
t+2     & A_{r+2},B_{r+2} & A_{r+3},B_{r+3} & \operatorname{MMA}(A_{r+2},B_{r+2}) & \operatorname{copy}(A_{r+3},B_{r+3})
\end{array}
$$

Symbols: $S_0$ and $S_1$ are staging buffers; $t$ is the logical pipeline time; $r$ is the current tile index along the reduction dimension; $A_r$ and $B_r$ are operand tiles for reduction tile $r$; $\operatorname{copy}(\cdot)$ is a global-to-shared, global-to-[TMEM](https://docs.nvidia.com/cuda/parallel-thread-execution/#tensor-memory), or peer-memory transfer; $\operatorname{MMA}(\cdot)$ is a matrix-multiply-accumulate instruction group; $b(t)=t\bmod 2$ selects the active consumer slot.

The same primitive appears in attention and communication-heavy kernels with different producer and consumer roles:

| Use | Producer role | Consumer role | Synchronization evidence |
| --- | --- | --- | --- |
| FlashAttention tile loop | [TMA](https://docs.nvidia.com/cuda/parallel-thread-execution/#tensor-memory) (Tensor Memory Accelerator) or vectorized copy loads $K,V$ tiles for step $r+1$ | [MMA](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions) / [WGMMA](https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-instructions) consumes $QK^\top$ and $PV$ tiles for step $r$ | [FA3 blog][S4], [FA3/FA4 helpers][S6], [FA4 report][S5] |
| FP8 GEMM | Producer dequantizes or scales the next tile into a ready slot | Consumer issues tensor-core MMA on the current slot | [DeepGEMM][S11], [Transformer Engine][S14], [CUTLASS/CuTe][S18] |
| MoE dispatch/combine | Producer stages token blocks or peer fragments into slot $1-b(t)$ | Consumer performs expert GEMM, permutation, or combine from slot $b(t)$ | [DeepEP][S12], [MegaBlocks][S31], [Tutel][S38] |
| Native megakernel runtime | Producer warpgroup interprets or fetches the next task record | Consumer warpgroup executes the current task or subgraph state | [Megakernels][S40], [TP Megakernel][S41], [Mirage][S40] |

Correctness is a barrier protocol, not only a buffer layout. A producer must publish "slot ready" after all writes to $S_{1-b(t)}$ are visible; a consumer must publish "slot free" after all reads from $S_{b(t)}$ have retired. On Ampere this often uses `cp.async` wait groups; on Hopper/Blackwell it often uses [mbarrier](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier) (memory barrier) objects, [TMA](https://docs.nvidia.com/cuda/parallel-thread-execution/#tensor-memory) barriers, or CuTe pipeline states; at grid/runtime scale the same pattern can be expressed with [CUDA streams](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#streams), [CUDA events](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#events), or persistent-kernel counters. Two slots are sufficient when producer latency can be hidden by one consumer step. Three or more stages are used when copy latency, remote-memory latency, or softmax-side work exceeds one math tile.

Inspect first: [FA3 blog][S4]; [FA4 report][S5]; [CUTLASS/CuTe docs][S18]; [DeepGEMM][S11]; [Megakernels][S39]; [Mirage][S40]; [CUDA streams][S61].

#### Tensor Memory Staging

[Tensor memory, or TMEM](https://research.colfax-intl.com/cutlass-tutorial-writing-gemm-kernels-using-tensor-memory-for-nvidia-blackwell-gpus/), is a Blackwell on-chip memory space attached to 5th-generation tensor cores. It is separate from ordinary shared memory and registers. In PTX, TMEM is dynamically allocated in column units, addressed as lanes and columns, and accessed by `tcgen05` load, store, copy, and MMA instructions. For kernel authors, TMEM is a new staging tier: accumulators and selected operands can live near tensor cores without consuming the same register file and shared-memory capacity that constrained Hopper schedules.

FA4 uses TMEM to make Blackwell attention pipelines possible at larger tile shapes. Forward can keep score or probability fragments staged while softmax and $PV$ are overlapped. Backward can store transposed $P$ or $dS$ in the layout needed by later MMAs, reducing shared-memory traffic and avoiding some register pressure from carrying multiple full accumulator tiles. This is especially important for training backward, where attention does more MMA work than forward and also has more elementwise plumbing around recomputation and gradients.

The CUDA/platform details are strict. TMEM exists for Blackwell `sm_100`-class tensor-core operations, not for Hopper WGMMA. A kernel must allocate and deallocate TMEM explicitly, respect CTA or CTA-pair ownership, and use the correct synchronization and proxy fences around `tcgen05` operations. TMEM layouts are instruction-specific: accumulator shape, operand packing, scale-factor placement, and lane alignment must match the chosen `tcgen05.mma` kind. These constraints make TMEM powerful but less forgiving than a generic scratchpad.

Inspect first: [FlashAttention-4](https://arxiv.org/abs/2603.05451) and the [FA4 blog](https://tridao.me/blog/2026/flash4/) for how TMEM changes attention forward and backward, the NVIDIA PTX documentation for [Tensor Memory and `tcgen05`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html), the CUTLASS CuTe [`tcgen05` API](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/cute_nvgpu_tcgen05.html), FA4's [Blackwell helpers](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/cute/blackwell_helpers.py), and [SM100 MMA descriptors](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/cute/mma_sm100_desc.py).

#### 2-CTA Cooperation

2-CTA cooperation is Blackwell's ability to let a pair of CTAs in the same cluster cooperate on one tensor-core operation or one logical tile. In `tcgen05` terminology, this appears as `cta_group::2`. The two CTAs remain separate scheduling entities, but the MMA spans their combined resources, including peer CTA participation and, for some patterns, distributed shared memory exchange.

In LLM training, the clearest use is FA4 backward. Backward attention has heavy shared-memory traffic and repeated gradient accumulation, so FA4 uses 2-CTA MMA to partition a larger logical tile across a CTA pair. This can reduce redundant operand staging, reduce per-CTA footprint, and lower the number of global atomic additions for $dQ$. The same principle can apply to large GEMM or MoE tiles when one CTA cannot feed a large enough Blackwell tensor-core tile efficiently by itself.

The platform details are more restrictive than ordinary CTA tiling. The cooperating CTAs must be launched in a compatible cluster configuration and must both remain active while the collective operation is in flight. The CTA group size must be chosen consistently for tensor-memory and tensor-core operations inside the kernel. Because 2-CTA schedules often move data through distributed shared memory, authors must reason about cluster-level barriers, peer visibility, and the shape of the reduction axis; splitting the output tile is not always the same as splitting the gradient reduction dimension.

Inspect first: [FlashAttention-4](https://arxiv.org/abs/2603.05451), the [FA4 blog section on 2-CTA backward](https://tridao.me/blog/2026/flash4/), NVIDIA PTX documentation for [`tcgen05.mma` and `cta_group`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html), the [CUTLASS Blackwell SM100 GEMM notes](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html), and FA4's [2-CTA backward kernel](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/cute/sm100_hd256_2cta_fmha_backward.py).

#### Tile-Shape Retuning For Asymmetric Scaling

Tile-shape retuning is the hardware-specific process of changing block sizes, K-loop depth, stage count, and work partitioning when one hardware resource scales faster than another. Blackwell makes this especially visible: tensor-core throughput grows faster than shared-memory bandwidth, special-function throughput, and some scalar pipelines. A tile shape that was balanced on Hopper can become bottlenecked by softmax exponentials, shared-memory traffic, or synchronization on Blackwell.

Training kernels use retuning to move bottlenecks off the critical path. FA4 forward uses larger tiles, ping-pong query scheduling, staged probability storage, software-assisted exponentials, and conditional rescaling because the attention softmax path can limit the faster tensor cores. FA4 backward retunes around shared-memory traffic by changing tile orientation, keeping selected intermediates in TMEM, and using 2-CTA cooperation. Dense GEMM and MoE kernels make analogous choices for block-scaled FP8/FP4 operands, grouped expert shapes, and epilogue fusion.

The CUDA details are a mix of occupancy math and instruction legality. Larger tiles improve tensor-core efficiency but consume more registers, shared memory, TMEM columns, and barrier slots. Narrow-precision and block-scaled Blackwell MMAs impose alignment, packing, and scale-factor layout constraints. Stage depth and CTA count must be tuned together: a deeper pipeline can hide latency but may reduce resident CTAs enough to hurt tail latency or communication overlap.

Inspect first: [FlashAttention-4](https://arxiv.org/abs/2603.05451) and the [FA4 blog](https://tridao.me/blog/2026/flash4/) for asymmetric scaling analysis, the [CUTLASS Blackwell SM100 GEMM documentation](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html) for legal shapes and data types, FA4's [Blackwell forward kernel](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/cute/flash_fwd_sm100.py) and [Blackwell backward kernel](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/cute/flash_bwd_sm100.py), and [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) for FP8/FP4 GEMM shape choices in LLM workloads.

#### Occupancy And [SM](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#hardware-implementation) (Streaming Multiprocessor)-Count Control

Occupancy and SM-count control deliberately limit how much of the GPU a kernel consumes. Occupancy control can mean choosing register, shared-memory, and CTA shapes that produce a target number of resident CTAs per SM. SM-count control is more explicit: a persistent or communication kernel may be launched with a configured number of SMs so the rest of the GPU remains available to companion kernels, communication progress, or other pipeline stages.

In LLM training, this matters most when compute overlaps with communication. Expert-parallel MoE training often runs dispatch, expert GEMM, and combine phases near each other; if the compute kernel occupies every SM, communication kernels may make poor progress, but if too many SMs are reserved for communication, tensor-core throughput falls. The same tradeoff can appear in pipeline parallelism, tensor-parallel reductions, optimizer sharding, and training kernels that intentionally leave room for asynchronous copy or communication helpers.

CUDA exposes this indirectly through launch geometry, streams, cooperative or persistent kernel design, cluster dimensions, dynamic shared-memory choices, and library-specific configuration knobs. The subtle point is that occupancy is not utilization: lowering occupancy can improve tile efficiency or overlap, while excessive occupancy can increase contention for shared memory, registers, and instruction issue. For distributed training libraries, an SM budget should be treated as a scheduling contract between compute and communication rather than as a local kernel micro-optimization.

Inspect first: [DeepEP](https://github.com/deepseek-ai/DeepEP) for MoE dispatch/combine kernels with SM number control and hook-based overlap, [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) for LLM GEMM kernels and Mega MoE overlap examples, [NVIDIA NCCL documentation](https://docs.nvidia.com/deeplearning/nccl/) for communication behavior, and the CUDA/CUTLASS occupancy and launch-configuration material in the [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html) and [CUTLASS documentation](https://docs.nvidia.com/cutlass/).

### Why It Matters For Training

Training attention includes matmul, softmax, masking, dropout/stat handling, and backward recomputation. If the kernel performs these stages serially, tensor cores sit idle while data moves or softmax executes. Asynchronous pipelining hides part of that non-GEMM work and raises effective utilization.

### Explore Further

- `paper`: [FlashAttention-3](https://arxiv.org/abs/2407.08608)
- `official blog`: [PyTorch FA3 explanation of WGMMA, TMA, and FP8](https://docs.pytorch.org/blog/flashattention-3/)
- `paper`: [FlashAttention-4](https://arxiv.org/abs/2603.05451)
- `official blog/code`: [FA4 blog and code links](https://tridao.me/blog/2026/flash4/)
- `official docs`: [CUTLASS/CuTe documentation](https://docs.nvidia.com/cutlass/)
- `official repo`: [ThunderKittens](https://github.com/HazyResearch/ThunderKittens)

### Implementation Examples

- FA3: WGMMA/TMA producer-consumer warp specialization for Hopper attention.
- FA4: Blackwell UMMA/TMEM pipeline, larger tiles, and CuTe-DSL implementation.
- ThunderKittens: tile-level CUDA primitives wrapping WGMMA/TMA/[TCGEN05](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/cute_nvgpu_tcgen05.html) (Blackwell fifth-generation tensor-core instruction family), useful for studying how a CUDA-embedded DSL exposes overlap templates.
- CUTLASS/CuTe: layout and tiling primitives used by high-performance GEMM and attention kernels.

### Caveats

Pipelining can trade one bottleneck for another. Larger tiles and deeper pipelines may increase register pressure, shared-memory pressure, occupancy loss, or synchronization overhead. The right schedule is hardware-specific.

## 3. Softmax Bottleneck Reduction

### Technique

Attention softmax has non-matmul operations: row max reductions, exponentials, row sums, rescaling, masking, and normalization. As tensor cores get faster, these operations can dominate. [FA3](https://arxiv.org/abs/2407.08608) (FlashAttention-3) overlaps softmax with GEMM. FA4 goes further: it uses conditional online-softmax rescaling, software-emulated exponentials on [FMA](https://en.wikipedia.org/wiki/Multiply%E2%80%93accumulate_operation#Fused_multiply%E2%80%93add) (fused multiply-add) units, and explicit scheduling so softmax worker groups do not contend on scarce exponential units.

### Primitive Techniques

#### Row-Max Reduction

Row-max reduction is the numerically stable first half of softmax. For each query row, the kernel reduces the current score tile to a scalar reference value before exponentiation, so the exponentials are formed as $e^{s-m}$ or, in many CUDA kernels, $2^{(s-m)\log_2 e}$, where $s$ is a score and $m$ is the row maximum. In streaming attention this maximum is not just a local tile statistic. It is a running reference across all K/V blocks visited for the same query row, and it must incorporate causal, padding, window, or score-modification masks before the reduction.

In LLM training kernels the row maximum is part of the forward/backward ABI. The forward pass uses it to normalize each streamed tile without storing the full score matrix; the backward pass recomputes local scores and needs the same log-sum-exp frame to recover probabilities and gradients. CUDA details that matter are the granularity of the row mapping, the reduction scope, and the resident storage. Hopper-style implementations often keep row fragments and maxima in registers while using warp or warpgroup reductions; Blackwell FA4 moves more accumulator traffic through tensor memory but still needs a per-row scalar state that can be updated cheaply. Inspect first: [FA1](https://arxiv.org/abs/2205.14135) for the online-softmax recurrence, [cuDNN SDPA](https://docs.nvidia.com/deeplearning/cudnn/frontend/latest/operations/Attention.html) for the production stats contract, and the [FA4 softmax implementation](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/cute/softmax.py) for `row_max` state and reductions.

#### Row-Sum Accumulation

Row-sum accumulation is the denominator side of softmax. After subtracting the row reference and exponentiating a score tile, the kernel reduces the exponentials to a scalar denominator and adds it into the running row sum. Together, the running maximum/reference and running sum encode the softmax normalization for the entire row without materializing probabilities in HBM.

Training kernels usually keep this accumulator in FP32 even when Q, K, V, P, or O are BF16, FP16, FP8, or lower precision. The reason is simple: denominator error feeds both the forward output scale and the backward gradient scale, and long-context rows can accumulate many tile contributions. CUDA implementations must choose whether row sums live in registers, shared memory, tensor memory, or a compact output stats tensor; they must also decide the reduction width and synchronization point so that the sum is complete before output normalization or backward-stat storage. Inspect first: [FA1](https://arxiv.org/abs/2205.14135) and [FA2](https://arxiv.org/abs/2307.08691) for the probability-free training formulation, [cuDNN SDPA](https://docs.nvidia.com/deeplearning/cudnn/frontend/latest/operations/Attention.html) for the optional stats tensor used by backward, and the [FA4 softmax implementation](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/cute/softmax.py) for the `row_sum` update and finalization path.

#### Conditional Rescaling

Online softmax normally rescales the partial output whenever a later K/V block raises the row reference: the old output is multiplied by $e^{m_{\mathrm{old}}-m_{\mathrm{new}}}$ before adding the new tile contribution. Conditional rescaling observes that small reference changes can be handled by keeping the old reference for the moment and accumulating the new tile in that same scale. The kernel then applies final normalization from the accumulated denominator at the end of the row. This removes many vector-scale operations from the critical path while retaining the same mathematical normalization, subject to the threshold being chosen so the retained reference does not create overflow or excessive roundoff.

In FA4 this is a Blackwell-specific response to asymmetric scaling. Tensor-core throughput increased faster than [SFU](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#arithmetic-instructions) (special-function unit, the hardware path for operations such as exponentials) and ordinary floating-point throughput, so the rescale vector operations around online softmax become visible. The platform details are the threshold $\tau$, whether the decision is made per element, row, warp, or warpgroup, and where the partial output lives while rescaling is deferred. A warp-granular decision reduces divergence, and a final normalization pass still consumes the accumulated softmax statistics. Inspect first: [FA4](https://arxiv.org/abs/2603.05451), the [FA4 blog's conditional-rescaling discussion](https://tridao.me/blog/2026/flash4/), and the [FA4 softmax implementation](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/cute/softmax.py), especially the `SoftmaxSm100` `rescale_threshold` path.

#### Software Exponential

Software exponential replaces part of the hardware `exp2` demand with an FMA-based approximation. A softmax tile needs one exponential per score element, and on Blackwell the [MUFU](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html) (multi-function special-function instruction path in NVIDIA PTX/SASS terminology)/SFU path for transcendental operations does not scale like tensor-core throughput. FA4 therefore computes some $2^x$ values with polynomial approximation and range reduction on FMA units while leaving other values on the hardware `MUFU.EX2` path.

The CUDA issue is resource balancing rather than just instruction count. If tensor cores and FMA pipes have slack while MUFU is saturated, an approximate FMA path can raise effective exponential throughput. The implementation still has to preserve the softmax accuracy envelope: range reduction must keep the polynomial interval small, coefficients must be tuned for relative error, and the kernel must decide which fragments use hardware exponentials versus emulated exponentials. Inspect first: [FA4](https://arxiv.org/abs/2603.05451), the [FA4 blog section on distributing $2^x$ across MUFU and FMA](https://tridao.me/blog/2026/flash4/), and the [FA4 softmax implementation](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/cute/softmax.py) for the `apply_exp2_convert` path and its emulation-frequency parameters.

#### Softmax/[MMA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html) (Matrix Multiply-Accumulate) Interleaving

Softmax/MMA interleaving schedules reductions, exponentials, normalization, and $PV$ work between tensor-core issue windows rather than running the attention stages as a strict serial chain. In the forward pass the kernel alternates score-tile MMA, softmax over that score tile, and value MMA; in the backward pass it recomputes probabilities while issuing gradient MMAs for neighboring tiles.

The important CUDA details are asynchronous tensor-core operations, data-movement engines, and warp specialization. FA3 uses Hopper WGMMA and TMA to overlap block-wise matmul with softmax. FA4 adapts the same idea to Blackwell UMMA-style execution, tensor memory, larger tile shapes, and explicit softmax worker scheduling. Interleaving works only if register pressure, shared-memory or tensor-memory footprint, and barriers are balanced; otherwise the softmax path can still serialize the tensor-core path. Inspect first: [FA3](https://arxiv.org/abs/2407.08608), the [FA3 PyTorch blog](https://docs.pytorch.org/blog/flashattention-3/), [FA4](https://arxiv.org/abs/2603.05451), the FlashAttention Hopper [forward mainloop implementation](https://github.com/Dao-AILab/flash-attention/blob/main/hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp), and the FA4 [Blackwell forward kernel](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/cute/flash_fwd_sm100.py).

#### Dedicated Correction Stage

A dedicated correction stage assigns the online-softmax correction work to a specific warpgroup or pipeline role instead of letting every softmax worker pay the rescale cost inline. The correction work includes applying output rescale factors, coordinating final normalization, and keeping the partial output consistent with the running row statistics.

In FA4 the correction stage is part of the forward pipeline design: two softmax warpgroups process ping-ponged query tiles, and a separate correction warpgroup keeps rescaling off the main tensor-core path. The platform detail that makes this useful is contention. On Blackwell, simultaneous exponential work can contend for `MUFU.EX2`, while rescale/correction vectors can consume ordinary floating-point bandwidth. Named barriers, tensor-memory staging, and warpgroup assignment become algorithmic controls, not just implementation details. Inspect first: the [FA4 blog pipeline section](https://tridao.me/blog/2026/flash4/), [FA4](https://arxiv.org/abs/2603.05451), the FA4 [standard Blackwell forward kernel](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/cute/flash_fwd_sm100.py), and the FA4 [2-CTA forward kernel](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/cute/sm100_hd256_2cta_fmha_forward.py).

#### Mixed-Precision Normalization

Mixed-precision normalization keeps softmax statistics and normalization arithmetic in a wider type than the attention operands or outputs. A modern training kernel may load Q, K, and V as BF16 or FP8, compute tensor-core products with hardware-specific accumulator rules, form exponentials in FP32-like arithmetic, store probabilities or intermediate P tiles in BF16, and still preserve FP32 row maxima, row sums, or log-sum-exp statistics for backward.

This primitive matters because low-precision attention failures often come from scale handling rather than the matmul itself. The CUDA/platform choices include the tensor-core operand format, accumulator type, conversion point for P, scale metadata layout, and whether the stats tensor is stored in a deterministic and backward-compatible format. For long-context training, the denominator and log-sum-exp path should be treated as part of numerical state, not as disposable epilogue math. Inspect first: the [Transformer Engine FP8/FP4 primer](https://nvidia.github.io/TransformerEngine/examples/fp8_primer.html) for scale-management context, [FA3](https://arxiv.org/abs/2407.08608) for low-precision attention, [cuDNN SDPA](https://docs.nvidia.com/deeplearning/cudnn/frontend/latest/operations/Attention.html) for datatype and backward support, and the [FA4 softmax implementation](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/cute/softmax.py) for FP32 row statistics around lower-precision conversions.

### Why It Matters For Training

Backward attention recomputes scores and probabilities, so the softmax bottleneck appears in both forward and backward. On Blackwell, FA4 reports that forward can be limited by exponential throughput and backward by shared-memory traffic, despite the nominal FLOP count being GEMM-heavy.

### Explore Further

- `official blog`: [FA3 PyTorch blog, Asynchrony and FP8 sections](https://docs.pytorch.org/blog/flashattention-3/)
- `paper`: [FlashAttention-4](https://arxiv.org/abs/2603.05451)
- `official blog`: [FA4 Blackwell bottleneck analysis](https://tridao.me/blog/2026/flash4/)

### Implementation Examples

- FA3: interleaves block-wise matmul and softmax to hide non-matmul work.
- FA4 forward: conditional rescaling skips small online-softmax corrections while preserving exact final normalization.
- FA4 forward: software `exp2` path uses polynomial approximation and range reduction to use FMA capacity in addition to hardware exponential units.

### Caveats

Softmax approximations are numerically sensitive. FA4's conditional rescaling is designed to preserve final exactness, while software exponential introduces an approximation that must be bounded and validated.

## 4. Low-Precision Scaling as a Data-Layout Technique

### Technique

FP8 and FP4 are not just data types. They require explicit scale tensors, amax tracking, delayed or current scaling policy, packed scale formats, scale layout transformations, and cast/decast kernels. A production FP8/FP4 kernel often computes:

1. Load low-precision values.
2. Load or reconstruct scale metadata.
3. Descale into the computation format.
4. Accumulate at a wider precision.
5. Track output amax or write new scale metadata.
6. Store in the requested output format.

### Primitive Techniques

#### Per-Tensor Delayed Scaling

Per-tensor delayed scaling stores one scale value, usually FP32, for an entire logical tensor and derives that scale from a history of recent absolute maxima rather than from the tensor being quantized at that instant. In Transformer Engine terms, each module keeps configurable `amax_history` state, computes the next scale from that history, quantizes with a single tensor read, records the current amax, and rotates the history after the pass. The scale is therefore a layout companion to the FP8 payload: the same bytes are only meaningful when the consumer also knows which tensor-level scale and inverse scale belong to them.

Training kernels use this primitive when the extra read required by current scaling would be too expensive or would break a fused path. A cast, transpose, or GEMM epilogue can apply the already-known scale while producing a fresh amax for the next use. This is common around Transformer linear layers, where forward activations and weights are often stored as [E4M3](https://nvidia.github.io/TransformerEngine/examples/fp8_primer.html) (FP8 format with 4 exponent bits and 3 mantissa bits) and backward gradients may use [E5M2](https://nvidia.github.io/TransformerEngine/examples/fp8_primer.html) (FP8 format with 5 exponent bits and 2 mantissa bits) or a hybrid recipe. The price is that the scale may lag an abrupt distribution change, so kernel authors need clipping behavior, amax-window length, and recipe boundaries to be explicit in benchmarks.

CUDA/platform details matter because delayed scaling is only useful if the scale state can be updated with low launch overhead. Transformer Engine batches amax reduction and history rotation at autocast-context boundaries, and its delayed-scaling examples require Ada-class FP8 support or later. In distributed training, delayed scaling also needs a well-defined reduction group when a tensor is sharded across sequence, context, tensor, or data-parallel ranks; otherwise different ranks can quantize the same logical tensor with different scales.

Inspect first: [Transformer Engine FP8 Delayed Scaling](https://nvidia.github.io/TransformerEngine/features/low_precision_training/fp8_delayed_scaling/fp8_delayed_scaling.html), especially quantization, `amax_history`, and distributed-training sections; then inspect the C API entries for `nvte_delayed_scaling_recipe_amax_and_scale_update*` and grouped amax routines in the [Transformer Engine API reference](https://docs.nvidia.com/deeplearning/transformer-engine-releases/release-2.8/user-guide/index.html).

#### Current Scaling / Amax Collection

Current scaling computes amax from the tensor being quantized and derives a scale for that same quantization event or for an immediately adjacent consumer. Conceptually it is the most direct recipe: scan the tensor, compute $s=x_{\max}^{\mathrm{format}}/(a_{\max}+\epsilon)$, scale values into FP8 range, cast, and retain the amax for diagnostics or later recipe state. Because the amax comes from the current data, the scale tracks distribution shifts more quickly than delayed scaling.

In one common scale convention:

$$
a_{\max} = \max_i |x_i|,\qquad
s = \frac{x_{\max}^{\mathrm{format}}}{a_{\max}+\epsilon},\qquad
q_i = \operatorname{clip}_{\mathrm{format}}\left(\operatorname{round}(s x_i)\right),
$$

and dequantization recovers $\hat{x}_i = q_i/s$. Some libraries store the reciprocal scale or apply separate input/output scale factors, so kernel authors must check the convention before composing quantized operators.

Symbols: $x_i$ is a source floating-point value; $a_{\max}$ is the maximum absolute value over the chosen scaling group; $x_{\max}^{\mathrm{format}}$ is the largest finite value representable by the target FP8/FP4 format; $\epsilon$ prevents division by zero; $s$ is the scale factor; $q_i$ is the quantized payload; $\operatorname{round}$ maps to the nearest representable grid point under the selected rounding policy; $\operatorname{clip}_{\mathrm{format}}$ saturates to the target format's valid range; and $\hat{x}_i$ is the dequantized approximation.

In LLM training kernels this primitive appears in cast kernels, fused attention graphs, and GEMM epilogues that must both emit low-precision tensors and expose fresh dynamic-range metadata. It is attractive for unstable tensors, such as activations after nonlinearities, gradients, and outputs whose range changes across curriculum, sequence length, or expert routing. The drawback is memory traffic: a naive implementation reads the source once to compute amax and again to quantize. Production kernels therefore try to fold amax collection into an existing pass, such as a GEMM output epilogue or an attention graph node, so the range statistic is not a standalone reduction pass.

CUDA libraries surface this as explicit scale and amax plumbing. cuDNN FP8 fused attention graphs require quantization scales, dequantization scales, and amax calculations to be present. cuBLASLt tensorwide FP8 matmul accepts scalar scale modes and can report output and auxiliary amax through descriptor pointers such as `CUBLASLT_MATMUL_DESC_AMAX_D_POINTER` and `CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_AMAX_POINTER`. Kernel authors must track whether a library expects a scale or its reciprocal, because dequantization and quantization conventions are both used.

Inspect first: [Transformer Engine FP8 Current Scaling](https://nvidia.github.io/TransformerEngine/features/low_precision_training/fp8_current_scaling/fp8_current_scaling.html) for the two-read reference flow and distributed amax synchronization; [cuDNN Frontend graph FP8 attention notes](https://docs.nvidia.com/deeplearning/cudnn/frontend/v1.12.0/developer/graph-api.html) for scale/descale requirements; and [cuBLASLt narrow-precision scaling](https://docs.nvidia.com/cuda/archive/12.9.1/cublas/index.html#narrow-precision-data-types-usage) for amax and scale descriptor attributes.

#### Per-Block Scaling

Per-block scaling replaces one tensorwide scale with many local scales, each attached to a fixed block of elements. Hopper-oriented FP8 block scaling commonly uses 128-element 1D blocks or 128x128 2D blocks with FP32 scale factors. Blackwell [MXFP8](https://nvidia.github.io/TransformerEngine/features/low_precision_training/mxfp8/mxfp8.html) (microscaling FP8) uses one [E8M0](https://nvidia.github.io/TransformerEngine/features/low_precision_training/mxfp8/mxfp8.html) (8-bit exponent-only scale format) scale per 32 FP8 values. [NVFP4](https://nvidia.github.io/TransformerEngine/features/low_precision_training/nvfp4/nvfp4.html) (NVIDIA FP4 training recipe) uses [E2M1](https://nvidia.github.io/TransformerEngine/features/low_precision_training/nvfp4/nvfp4.html) (4-bit floating-point payload with 2 exponent bits and 1 mantissa bit) values plus hierarchical scaling: an E4M3 block scale for 16-element or 16x16 blocks and a tensor-level FP32 global scale. The point is to make dynamic range a local property of the data layout.

Training kernels use per-block scaling when tensorwide scaling would either clip outliers or waste too much representable range on quiet regions. This is especially relevant for large projection matrices, MoE expert weights, routed activations, and gradients whose per-token or per-expert ranges differ. The scale tensor becomes part of the operand contract: a matmul tile must load the FP8 or FP4 payload and the corresponding block scales, then multiply the scale factors through the dot product or through the quantize/dequantize path.

Platform constraints are not decorative here. Transformer Engine's FP8 block scaling has divisibility requirements on the last dimension and distinguishes 1D and 2D choices for activations, weights, and gradients; it also notes that pure 2D-by-2D GEMM is not supported. cuBLASLt exposes separate scale modes for tensorwide, vector, 128-element, 128x128, 32-element, and 16-element scaling, with architecture gates: Ada/Hopper support tensorwide FP8, Hopper supports 128-element and 128x128 FP8 block scaling, and Blackwell adds native FP4 plus 32-element FP8/16-element FP4 block-scaled modes. These choices affect layout, alignment, output precision, and whether the result can be written directly back to narrow precision.

Inspect first: [Transformer Engine FP8 Blockwise Scaling](https://nvidia.github.io/TransformerEngine/features/low_precision_training/fp8_blockwise_scaling/fp8_blockwise_scaling.html), [Transformer Engine MXFP8](https://nvidia.github.io/TransformerEngine/features/low_precision_training/mxfp8/mxfp8.html), [Transformer Engine NVFP4](https://nvidia.github.io/TransformerEngine/features/low_precision_training/nvfp4/nvfp4.html), [cuBLASLt narrow-precision scaling modes](https://docs.nvidia.com/cuda/archive/12.9.1/cublas/index.html#narrow-precision-data-types-usage), and the [DeepSeek-V3 technical report](https://arxiv.org/abs/2412.19437) for large-scale FP8 training motivation.

#### Scale-Layout Transformation

Scale-layout transformation, often called swizzling in NVIDIA documentation, reorders scale metadata from a communication-friendly or producer-friendly layout into the layout expected by a GEMM or Tensor Core tile. The scale tensor is small relative to the payload, but it is read on the critical path of every tile, so scattered scale loads can erase much of the benefit of narrow precision.

LLM training kernels hit this primitive whenever quantized tensors participate in all-gather, transpose, or grouped GEMM. A compact scale layout is useful for communication because shards can be concatenated without padding or hardware-specific tile order. A GEMM-ready layout is useful for compute because the kernel can load the scales that correspond to a 128x128 or similar tile contiguously. When no communication intervenes, quantization and swizzling can be fused; when all-gather is required, kernels often quantize into compact form, communicate, and swizzle just before GEMM.

The platform details are concrete. Transformer Engine documents separate compact and GEMM-ready scale layouts for FP8 block scaling, including padded and transposed scale shapes. MXFP8 on Blackwell requires scale factors in a hardware layout where 128x4 groups are linearized and permuted for Tensor Core consumption. [cuBLASLt](https://docs.nvidia.com/cuda/cublas/#using-the-cublaslt-api) (the lightweight, descriptor-based cuBLAS matrix-multiply API) documents a tiled layout for 16/32-element block scaling factors with 16-byte alignment, full-tile allocation, and no transposition of the scale layout even when input tensors are transposed. DeepGEMM adds another implementation-level contract: its left-hand-side scale must be TMA-aligned and transposed, with FP32 scales on SM90 and packed [UE8M0](https://docs.nvidia.com/cuda/cublas/#d-block-scaling-factors-layout) (unsigned exponent-only 8-bit block-scale format) scales on SM100.

Inspect first: Transformer Engine developer notes for [FP8 blockwise scale swizzling](https://nvidia.github.io/TransformerEngine/features/low_precision_training/fp8_blockwise_scaling/fp8_blockwise_scaling.html#swizzle-of-scaling-factors), [MXFP8 swizzling](https://nvidia.github.io/TransformerEngine/features/low_precision_training/mxfp8/mxfp8.html#swizzling-scaling-factors), and [NVFP4 layout handling](https://nvidia.github.io/TransformerEngine/features/low_precision_training/nvfp4/nvfp4.html#handling-transposes); then inspect [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) for TMA-aligned scale requirements and [cuBLASLt block-scaling-factor layout](https://docs.nvidia.com/cuda/archive/12.9.1/cublas/index.html#d-block-scaling-factors-layout).

#### Register-Level Descale And Accumulate

Register-level descale and accumulate is the inner-loop act of recovering the mathematical value of a low-precision operand by combining payload bits with scale metadata, then accumulating products in a wider format. In scalar notation the value is $x_{\mathrm{fp8}}s$, $x_{\mathrm{e2m1}}s_{\mathrm{block}}s_{\mathrm{global}}$, or a product of two scaled blocks in which both operands contribute scale factors to the dot product.

A block-scaled dot product can be viewed as:

$$
C_{ij}
= \sum_k
\left(q^A_{ik}\,\alpha_{b_A(k)}\right)
\left(q^B_{kj}\,\beta_{b_B(k)}\right),
$$

where $q^A$ and $q^B$ are narrow payloads, $\alpha$ and $\beta$ are the scale factors attached to the operand blocks, and $b_A(k), b_B(k)$ map a reduction index to the corresponding scale block. The kernel optimization problem is to make those scale loads and multiplies disappear into the tensor-core mainloop or epilogue cost.

Symbols: $C_{ij}$ is one output matrix element; $i$ and $j$ index the output row and column; $k$ indexes the reduction dimension; $q^A_{ik}$ and $q^B_{kj}$ are quantized payload entries from operands $A$ and $B$; $\alpha_{b_A(k)}$ and $\beta_{b_B(k)}$ are the block scales selected by the block-index maps $b_A$ and $b_B$.

The physical layout usually interleaves a dense payload stream with a much smaller scale stream:

$$
\begin{array}{c}
\begin{array}{c|c|c}
\text{payload block 0} & \text{payload block 1} & \text{payload block 2}\\ \hline
q_{0:31} & q_{32:63} & q_{64:95}
\end{array}
\\[-1mm]
\begin{array}{ccc}
\alpha_0 & \alpha_1 & \alpha_2
\end{array}
\end{array}
$$

A GEMM-ready layout tries to ensure that the scale $\alpha_b$ needed for a payload block arrives in the same tile schedule as the payload itself.

Diagram symbols: $q_{0:31}$, $q_{32:63}$, and $q_{64:95}$ are consecutive payload blocks; $\alpha_0$, $\alpha_1$, and $\alpha_2$ are their corresponding scale values; the block size of 32 is illustrative and may be 16, 32, 128, or a hardware-specific tile depending on the format and library.

In LLM training kernels this primitive lives inside dense projection GEMMs, attention projections, weight-gradient GEMMs, and grouped expert MLPs. The kernel wants narrow payloads for memory bandwidth and Tensor Core throughput, but it wants BF16, FP16, [TF32](https://blogs.nvidia.com/blog/tensorfloat-32-precision-format/) (TensorFloat-32, NVIDIA's 10-bit-mantissa tensor-core format), or FP32 accumulation behavior to preserve training signal. Libraries may expose this as a scaled matmul mode, while custom kernels may explicitly load scale tiles, place them in registers, and apply them around MMA instructions or epilogue conversion. For grouped MoE GEMMs, the same kernel may need per-expert offsets, masks, and scale pointers while keeping one schedule for many experts.

CUDA details determine which parts are native and which are emulated. cuBLASLt represents tensorwide scaling multiplicatively around `op(A) op(B)` and defines block-scaled dot products as sums over scaled blocks. Hopper FP8 Tensor Cores support FP8 matrix products but some block-scaled modes are library- or software-managed; Blackwell adds native block-scaled FP4/FP8 paths. DeepGEMM specializes this space for SM90 and SM100 and exposes FP8, grouped GEMM, and FP8xFP4 paths, so it is a useful implementation to inspect when the question is how scale loads, TMA descriptors, and tile scheduling meet in a real kernel.

Inspect first: [cuBLASLt tensorwide and block-scaled matmul definitions](https://docs.nvidia.com/cuda/archive/12.9.1/cublas/index.html#narrow-precision-data-types-usage), [Transformer Engine C API matmul and quantized tensor entries](https://docs.nvidia.com/deeplearning/transformer-engine-releases/release-2.8/user-guide/index.html), and [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) for concrete SM90/SM100 FP8 and FP8xFP4 GEMM interfaces.

#### Quantized Epilogue With Amax Writeback

A quantized epilogue fuses the last arithmetic stage of a kernel with output scaling, cast, clipping, optional auxiliary-output scaling, and amax writeback. Instead of writing BF16/FP32 output and launching a separate quantization kernel, the GEMM or attention kernel computes the output statistic while data is still hot, chooses or applies the relevant scale policy, stores the narrow output, and records the amax needed by the next recipe update.

For LLM training, this primitive is central because projection outputs, activation-gradient outputs, and fused activation paths are written repeatedly and can dominate memory bandwidth. A fused epilogue can emit an FP8 activation for the next layer, a BF16 residual output for accumulation, an auxiliary tensor for a backward activation, and the amax metadata needed by delayed or current scaling. It also localizes clipping behavior: the epilogue is where values outside the FP8 or FP4 representable range are saturated, rounded, or routed to a fallback output type.

The platform surface is visible in cuBLASLt and cuDNN. cuBLASLt describes output and auxiliary amax values as part of FP8 matmul epilogue behavior and provides descriptor pointers for those amax outputs. cuDNN FP8 fused attention requires explicit scale/descale nodes and amax calculations in its graph. Transformer Engine exposes multi-tensor scale, scale-inverse, amax, cast-transpose, and delayed-scale update routines, which is the implementation vocabulary a kernel author should recognize before claiming that an FP8 kernel is fully fused.

Inspect first: [cuBLASLt tensorwide FP8 epilogue and amax descriptor notes](https://docs.nvidia.com/cuda/archive/12.9.1/cublas/index.html#tensorwide-scaling-for-fp8-data-types), [cuDNN Frontend FP8 attention graph requirements](https://docs.nvidia.com/deeplearning/cudnn/frontend/v1.12.0/developer/graph-api.html), and the Transformer Engine API entries for `nvte_multi_tensor_compute_scale_and_scale_inv_cuda`, `nvte_group_amax`, `nvte_compute_amax*`, and cast-transpose routines in the [Transformer Engine user guide](https://docs.nvidia.com/deeplearning/transformer-engine-releases/release-2.8/user-guide/index.html).

#### Stochastic Rounding / Clipping Policy

Stochastic rounding and clipping policy define how a real value becomes a representable FP8 or FP4 value when it lies between two values or outside the narrow-format range. Clipping saturates out-of-range values. Deterministic rounding chooses a fixed nearest representable value. Stochastic rounding chooses between neighboring representable values with probabilities tied to distance, so the expected quantized value tracks the original value more closely over repeated updates.

Training kernels care about this primitive because gradients, activation checkpoints, and optimizer-related tensors are repeatedly quantized and dequantized. Bias in a single cast may be small, but systematic bias across many optimizer steps can change convergence. FP4 makes the issue sharper because the representable grid is very coarse. For this reason, NVFP4 applies stochastic rounding when casting scaled values to FP4, and memory-saving FP8 training systems such as COAT emphasize quantization policy and dynamic-range management for activations and optimizer states rather than only matmul throughput.

CUDA/platform details include where randomness lives and what the hardware can do. Transformer Engine's NVFP4 recipe ties stochastic rounding to gradients and uses Blackwell native support; its C API exposes quantization-config attributes for RNG state and stochastic rounding. A custom kernel must make the rounding stream reproducible enough for debugging, safe under CUDA graph capture if used there, and compatible with distributed training determinism expectations. The clipping threshold must also match the format, such as E4M3, E5M2, E2M1, and any global or block scale applied before casting.

Inspect first: [Transformer Engine NVFP4 stochastic rounding](https://nvidia.github.io/TransformerEngine/features/low_precision_training/nvfp4/nvfp4.html#stochastic-rounding), the C API `NVTEQuantizationConfigAttribute` entries in the [Transformer Engine user guide](https://docs.nvidia.com/deeplearning/transformer-engine-releases/release-2.8/user-guide/index.html), [COAT](https://arxiv.org/abs/2410.19313) for FP8 activation and optimizer-state quantization pressure, and [Scaling FP8 training to trillion-token LLMs](https://arxiv.org/abs/2409.12517) for long-horizon training-instability evidence.

#### Distributed Amax Reduction

Distributed amax reduction synchronizes dynamic-range metadata across ranks that hold shards of one logical tensor. If each rank quantizes from only its local maximum, the same logical activation or gradient can acquire different scales on different ranks. That may be acceptable for block-local scales that never cross rank boundaries, but it is usually wrong for tensorwide scales before quantized all-gather or for global NVFP4 scales.

In LLM training kernels this primitive is tied to sequence parallelism, context parallelism, tensor parallelism, data parallelism, and MoE execution. Inputs or gradients may be gathered in quantized form to avoid BF16 communication, so all ranks that participate in the logical gather must agree on amax before quantization. Expert-parallel models add an edge case: some modules or experts may receive no tokens on a rank, so the scaling-state registration and collective schedule must still match across ranks to avoid hangs and stale scale state.

Platform details differ by recipe. Transformer Engine delayed scaling exposes `reduce_amax` and `amax_reduction_group` for PyTorch and manages broader reduction scope automatically in JAX. Current scaling synchronizes local amax values before quantized all-gather for sharded tensors. FP8 blockwise scaling generally does not need scale synchronization because each scale is local to a 128-element or 128x128 block, while NVFP4 does not need to synchronize block scales but does need synchronization for the global scale of gathered tensors. Kernel authors should treat the reduction group as part of the layout contract, not as a training-loop afterthought.

Inspect first: [Transformer Engine delayed-scaling distributed training](https://nvidia.github.io/TransformerEngine/features/low_precision_training/fp8_delayed_scaling/fp8_delayed_scaling.html#distributed-training), [Transformer Engine current-scaling distributed training](https://nvidia.github.io/TransformerEngine/features/low_precision_training/fp8_current_scaling/fp8_current_scaling.html#distributed-training), [Transformer Engine NVFP4 distributed training](https://nvidia.github.io/TransformerEngine/features/low_precision_training/nvfp4/nvfp4.html#distributed-training), and [Megatron-Core parallelism guidance](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html).

### Why It Matters For Training

Low precision raises throughput and reduces memory traffic, but training is more fragile than inference. Instabilities can come from outliers, activation functions, optimizer states, and gradient statistics. The kernel author must coordinate math, layout, and numerical policy.

### Explore Further

- `official docs`: [NVIDIA Transformer Engine](https://docs.nvidia.com/deeplearning/transformer-engine-releases/release-2.8/user-guide/index.html)
- `official docs`: [Transformer Engine FP8/FP4 primer](https://nvidia.github.io/TransformerEngine/examples/fp8_primer.html)
- `official repo`: [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM)
- `paper`: [COAT: Compressing Optimizer States and Activation for Memory-Efficient FP8 Training](https://arxiv.org/abs/2410.19313)
- `paper`: [Scaling FP8 training to trillion-token LLMs](https://arxiv.org/abs/2409.12517)
- `paper`: [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
- `official docs/blog/report`: [DeepSeek-V4 model card/report](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro)

### Implementation Examples

- Transformer Engine: inspect FP8 modules and recipes for scale/amax management around Transformer layers.
- DeepGEMM: inspect FP8 GEMM APIs, TMA-aligned scale layout requirements, SM90/SM100 support, and FP8xFP4 paths.
- COAT: inspect dynamic range expansion and mixed-granularity activation quantization for training memory reduction.
- DeepSeek-V3: study FP8 mixed-precision training at large MoE scale.
- DeepSeek-V4: treat FP4 expert weights plus FP8 mixed precision as official model-level evidence; detailed pretraining kernels remain partly undisclosed.

### Caveats

Benchmarking low precision without reporting scale policy is incomplete. The same FP8 shape can behave differently under per-tensor, per-channel, per-block, delayed-scaling, or current-scaling recipes.

## 5. Fused Memory-Bound Training Operators

### Technique

Many Transformer operators are bandwidth-bound rather than compute-bound: [RMSNorm](https://arxiv.org/abs/1910.07467) (root-mean-square layer normalization), LayerNorm, [RoPE](https://arxiv.org/abs/2104.09864) (rotary positional embedding), simple activations, cross-entropy, and preference losses. Fusion reduces HBM reads/writes and kernel launch overhead by combining adjacent elementwise/reduction operations. Chunking processes a large dimension, such as vocabulary, in pieces so intermediate tensors never exist in full.

### Primitive Techniques

#### Read-Once/Write-Once Elementwise Fusion

Read-once/write-once elementwise fusion is the simplest form of memory-bound operator fusion. Instead of launching separate kernels for bias addition, residual addition, activation, dropout mask application, scalar scaling, or loss postprocessing, a fused kernel streams each row of the activation tensor through registers and writes the final value once. In LLM training this is useful whenever the operation graph does little computation per byte: residual pathways, activation epilogues, normalization epilogues, and backward scaling steps can be dominated by HBM traffic and launch latency rather than arithmetic.

The CUDA/platform issue is that the win comes from memory behavior, not from making the arithmetic clever. The kernel should issue coalesced loads and stores over the innermost contiguous dimension, use vectorized memory accesses when alignment allows, predicate tail elements cleanly, and avoid adding so many fused terms that register pressure lowers occupancy. For Triton authors, this usually means one program per row or row block with `tl.arange` offsets and masks; for CUDA C++ authors, it means choosing vector widths, block sizes, and shared-memory usage so the operation remains bandwidth-limited instead of becoming occupancy-limited.

Inspect first: [Liger documentation](https://linkedin.github.io/Liger-Kernel/) for the training-operator set and fusion/chunking motivation, the [Liger fused-ops directory](https://github.com/linkedin/Liger-Kernel/tree/main/src/liger_kernel/ops) for concrete fused operators, and the [`element_mul_kernel` use in fused linear cross entropy](https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/fused_linear_cross_entropy.py) as a small example of replacing a separate backward multiply.

#### Fused Reduction-Normalization

Fused reduction-normalization combines the reduction that computes a per-token statistic with affine scaling and output storage. LayerNorm computes mean and variance over the hidden dimension; RMSNorm computes root mean square. In a Transformer training kernel, one CTA or Triton program commonly owns one row or a small group of rows, reduces the hidden vector in FP32 or an explicitly chosen accumulation type, computes inverse standard deviation or reciprocal RMS, applies gamma/beta or RMSNorm weight, and saves the compact statistics needed by backward.

The two common normalization formulas are:

$$
\operatorname{LayerNorm}(x)_i
= \gamma_i\frac{x_i-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta_i,\qquad
\mu=\frac{1}{d}\sum_{j=1}^{d}x_j,\quad
\sigma^2=\frac{1}{d}\sum_{j=1}^{d}(x_j-\mu)^2,
$$

$$
\operatorname{RMSNorm}(x)_i
= w_i\frac{x_i}{\sqrt{\frac{1}{d}\sum_{j=1}^{d}x_j^2+\epsilon}}.
$$

Symbols: $x$ is one hidden-state row of length $d$; $i$ and $j$ index hidden channels; $\mu$ and $\sigma^2$ are the row mean and variance used by LayerNorm; $\gamma_i$ and $\beta_i$ are LayerNorm affine parameters; $w_i$ is the RMSNorm scale parameter; and $\epsilon$ is the numerical-stability constant in the denominator.

The reduction term is small in bytes but latency-sensitive; fusing it with the affine/store path avoids rereading the same row.

In backward the same primitive is more than "normalize again": it must combine `dY`, `X`, the saved statistic, and weight, while also accumulating parameter gradients over tokens. CUDA details that matter include warp/block reductions, shared memory versus register reductions, persistent kernels for supported hidden sizes, contiguous last-dimension layout, statistic save format, and whether gradient accumulation uses multiple programs per row block to avoid atomics. Mixed BF16/FP16 inputs usually still need FP32 accumulation for numerical stability.

Inspect first: Liger's [RMSNorm kernel](https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/rms_norm.py), NVIDIA Apex's [`FusedLayerNorm`](https://nvidia.github.io/apex/layernorm.html) and [source wrapper](https://nvidia.github.io/apex/_modules/apex/normalization/fused_layer_norm.html), and Megatron-Core's [`FusedLayerNorm`](https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.fusions.fused_layer_norm.html), especially the persistent-layer-norm parameter and hidden-size caveat.

#### RoPE-In-Projection Or RoPE-In-Attention

RoPE fusion applies rotary position embeddings to Q and K while those tensors are already resident in registers, shared memory, or a freshly produced projection buffer. Mathematically, RoPE is a position-dependent two-dimensional rotation over paired hidden dimensions; operationally, it is a small amount of multiply-add work plus cos/sin table loads. If it is left as a standalone PyTorch operation, training can pay an extra read and write of Q/K before attention begins.

For a paired channel $(x_{2r}, x_{2r+1})$ at position $p$ and frequency $\theta_r$, RoPE applies:

$$
\begin{bmatrix}
x'_{2r}\\
x'_{2r+1}
\end{bmatrix}
=
\begin{bmatrix}
\cos(p\theta_r) & -\sin(p\theta_r)\\
\sin(p\theta_r) & \cos(p\theta_r)
\end{bmatrix}
\begin{bmatrix}
x_{2r}\\
x_{2r+1}
\end{bmatrix}.
$$

Symbols: $x_{2r}$ and $x_{2r+1}$ are the two channels in rotary pair $r$; $x'_{2r}$ and $x'_{2r+1}$ are the rotated outputs; $p$ is the token position; $\theta_r$ is the rotary frequency for pair $r$; and the $2\times2$ matrix is the position-dependent rotation applied to that channel pair.

In training kernels, RoPE can be placed immediately after [QKV](https://docs.nvidia.com/deeplearning/cudnn/frontend/latest/operations/Attention.html) (query-key-value) projection, inside a Q/K preprocessing kernel, or inside the attention schedule before score computation. The important platform details are tensor layout and position metadata: kernels must handle interleaved versus split-half rotation layouts, partial rotary dimensions, packed variable-length batches, sequence offsets, and GQA/MQA layouts where Q and K head counts differ. If RoPE is fused into attention, the rotation must occur before $QK^\top$ and backward must either save enough intermediate state or apply the conjugate rotation consistently.

Inspect first: Liger's [RoPE kernel](https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/rope.py), FlashAttention's [Triton rotary kernel](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/ops/triton/rotary.py), and FlashAttention's higher-level [rotary layer wrapper](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/layers/rotary.py) for layout, variable-length, and autograd-facing APIs.

#### [GLU](https://arxiv.org/abs/1612.08083) (Gated Linear Unit)/[SwiGLU](https://arxiv.org/abs/2002.05202) (Swish-Gated Linear Unit) Epilogue Fusion

GLU-family [MLPs](https://en.wikipedia.org/wiki/Multilayer_perceptron) (multi-layer perceptrons) produce two projection streams: a gate stream and an up stream. SwiGLU applies $\operatorname{silu}(\mathrm{gate})\odot \mathrm{up}$; [GEGLU](https://arxiv.org/abs/2002.05202) (Gaussian Error Linear Unit-gated GLU) and related variants use a different gate activation but have the same memory problem. If the gate projection, activation output, and product are materialized as separate tensors, the MLP block performs several full-size HBM round trips around a modest amount of elementwise math.

For input $x$, a SwiGLU MLP can be written as:

$$
g = xW_g,\qquad u = xW_u,\qquad
h = \operatorname{silu}(g)\odot u,\qquad
y = hW_o.
$$

Symbols: $x$ is the MLP input row or tile; $W_g$, $W_u$, and $W_o$ are the gate, up, and output projection matrices; $g$ is the gate projection; $u$ is the up projection; $\operatorname{silu}$ is the SiLU activation; $\odot$ is elementwise multiplication; $h$ is the gated hidden activation; and $y$ is the MLP output.

Epilogue fusion tries to compute $h$ while $g$ and $u$ are still in registers or shared memory.

The natural fusion point is the GEMM epilogue or the first kernel after the two projections. In a custom CUDA, CUTLASS/CuTe, cuDNN Frontend, or Triton implementation, the epilogue can load the gate and up fragments, apply bias if present, evaluate the activation in registers, multiply, and store only the final MLP activation. Training backward must recover gradients for both projection streams, so the author must choose between saving gate/up values, saving the post-activation product, or recomputing the activation from projection outputs. The critical CUDA constraints are epilogue register pressure, activation approximation cost, vector alignment, and whether the fusion prevents tensor-core GEMM kernels from using their fastest epilogue path.

Inspect first: Liger's [SwiGLU kernel](https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/swiglu.py), Megatron-Core's [`fused_bias_swiglu`](https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.fusions.fused_bias_swiglu.html), and the cuDNN Frontend [Graph API fusion support](https://docs.nvidia.com/deeplearning/cudnn/frontend/latest/developer/graph-api.html), which distinguishes mainloop and epilogue fusion surfaces for matmul-style graphs.

#### Fused Linear Cross-Entropy

Fused linear cross-entropy combines the final language-model head projection with logit normalization and cross-entropy loss. The unfused computation forms logits with shape `[tokens, vocab]`, then runs softmax and loss kernels over that large matrix. In large-vocabulary LLM training, that logits tensor can be one of the largest transient activations in the step; the fused primitive treats the projection and loss as one streaming computation instead of a materialized interface.

For token representation $h_t$, vocabulary matrix $W$, and target class $y_t$, the fused computation preserves:

$$
z_t = h_tW^\top,\qquad
L_t = -z_{t,y_t} + \log\sum_{v=1}^{|V|} e^{z_{t,v}}.
$$

Symbols: $h_t$ is the hidden representation for token $t$; $W$ is the vocabulary projection matrix; $z_t$ is the logits vector; $y_t$ is the target vocabulary index; $z_{t,y_t}$ is the target logit; $|V|$ is the vocabulary size; $v$ indexes vocabulary entries; and $L_t$ is the per-token cross-entropy loss.

The kernel challenge is to compute the log-sum-exp term over vocabulary tiles without storing all logits $z_t$.

The streaming layout is analogous to attention over the vocabulary axis:

$$
\begin{array}{c}
W^\top =
\left[
W_{V_0}^{\top}\;\middle|\;
W_{V_1}^{\top}\;\middle|\;
\cdots\;\middle|\;
W_{V_R}^{\top}
\right]
\\[2mm]
h_tW_{V_r}^{\top}
\;\longrightarrow\;
\left(m_t^{(r)},\ell_t^{(r)},z_{t,y_t}^{(r)}\right)
\\[2mm]
\operatorname{Reduce}_{r}
\left(m_t^{(r)},\ell_t^{(r)},z_{t,y_t}^{(r)}\right)
\;\longrightarrow\;
L_t
\end{array}
$$

Each $V_r$ is a vocabulary tile; the kernel keeps per-token max/sum statistics and the target logit rather than a full `[tokens, vocab]` matrix.

Diagram symbols: $W^\top$ is partitioned into vocabulary-column tiles $W_{V_0}^\top,\ldots,W_{V_R}^\top$; $r$ indexes a vocabulary tile; $m_t^{(r)}$ and $\ell_t^{(r)}$ are the partial row max and denominator for token $t$ over tile $r$; $z_{t,y_t}^{(r)}$ is the target logit contribution when the target class lies in tile $V_r$; and $\operatorname{Reduce}_r$ merges tile-local statistics into the final loss $L_t$.

A training implementation usually computes chunks or tiles of $hW_{\mathrm{vocab}}^\top$, maintains numerically stable softmax statistics, gathers the target logit, writes per-token loss, and produces gradients for the hidden state and vocabulary weight. The CUDA details resemble attention-style streaming reductions over the vocabulary dimension: use FP32 accumulators for reductions, handle `ignore_index`, class weights, label smoothing, softcaps or z-loss if supported, and decide where to accumulate `grad_weight`. Liger's implementation computes loss and stores gradient buffers through a custom autograd path, so the memory contract is different from a normal `Linear` followed by `CrossEntropyLoss`.

Inspect first: the [Liger paper](https://arxiv.org/abs/2410.10989), Liger's [fused linear cross-entropy kernel](https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/fused_linear_cross_entropy.py), the [Transformers integration wrapper](https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/transformers/fused_linear_cross_entropy.py), and [TRL](https://huggingface.co/docs/trl/index) (Hugging Face Transformer Reinforcement Learning library)'s Liger integration for the training-framework entry point.

#### Vocabulary Chunking

Vocabulary chunking is the scheduling primitive behind memory-bounded cross-entropy and several post-training losses. Rather than allocate the full `[tokens, vocab]` logits tensor, the kernel or wrapper selects a chunk of tokens, a chunk of vocabulary, or both, computes partial logits, and reduces partial results into the same scalar loss and gradients that the dense computation would have produced. Token chunks preserve exactness independently; vocabulary chunks need stable cross-chunk max/sum softmax reductions.

Chunk size is a hardware and model-layout choice. Larger chunks improve GEMM efficiency and reduce Python or launch overhead; smaller chunks lower peak memory and may improve cache behavior. Important CUDA/platform details include the maximum block size or tensor numel limits in the authoring system, whether the vocab weight is row-major and contiguous, how target gathers are handled, and how partial max/sum results are merged. In tensor-parallel vocabularies, chunking also interacts with all-reduce/all-gather boundaries because softmax normalization must cover the logical vocabulary, not only the local shard.

Inspect first: the chunk-size logic in Liger's [fused linear cross-entropy kernel](https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/fused_linear_cross_entropy.py), the [Liger chunked-loss package](https://github.com/linkedin/Liger-Kernel/tree/main/src/liger_kernel/chunked_loss), and [TRL's Liger integration](https://huggingface.co/docs/trl/liger_kernel_integration), which exposes the fused/chunked path to [SFT](https://huggingface.co/docs/trl/sft_trainer) (supervised fine-tuning) and preference-training workflows.

#### Pairwise Preference-Loss Fusion

Pairwise preference-loss fusion targets post-training objectives such as [DPO](https://arxiv.org/abs/2305.18290) (Direct Preference Optimization), [ORPO](https://arxiv.org/abs/2403.07691) (Odds Ratio Preference Optimization), [CPO](https://huggingface.co/docs/trl/cpo_trainer) (Contrastive Preference Optimization), [SimPO](https://arxiv.org/abs/2405.14734) (Simple Preference Optimization), [KTO](https://arxiv.org/abs/2402.01306) (Kahneman-Tversky Optimization), [JSD](https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence) (Jensen-Shannon divergence), and distillation losses. These objectives often compare chosen and rejected responses, subtract reference-model log probabilities, apply masks over response tokens, reduce over sequence positions, and then evaluate a stable scalar loss such as a log-sigmoid or margin expression. Written as framework primitives, those steps can allocate multiple token-level score tensors and launch several small reductions.

A canonical DPO-style pairwise loss is:

$$
\Delta =
\left[\log \pi_\theta(y^+\mid x)-\log \pi_\theta(y^-\mid x)\right]
-
\left[\log \pi_{\mathrm{ref}}(y^+\mid x)-\log \pi_{\mathrm{ref}}(y^-\mid x)\right],
$$

$$
L_{\mathrm{DPO}} = -\log \sigma(\beta \Delta),
$$

where $y^+$ and $y^-$ are chosen and rejected responses, $\pi_\theta$ is the trained policy, $\pi_{\mathrm{ref}}$ is the reference policy, and $\beta$ controls preference strength. Fused kernels reduce memory by computing the needed sequence log-probability sums and the scalar loss together.

Symbols: $x$ is the prompt or conditioning context; $y^+$ and $y^-$ are the preferred and dispreferred completions; $\pi_\theta$ is the model being trained; $\pi_{\mathrm{ref}}$ is the frozen reference model; $\log\pi(\cdot\mid x)$ is the sequence log probability under a policy; $\Delta$ is the preference log-ratio margin; $\sigma$ is the logistic sigmoid; $\beta$ is the temperature or preference-strength coefficient; and $L_{\mathrm{DPO}}$ is the scalar DPO loss.

The fused primitive keeps the dataflow close to the logits or log-probability computation. It can gather target-token log probabilities, apply masks, compute chosen/rejected sums or means, form pairwise deltas, and emit the scalar loss and gradients without retaining every intermediate tensor. CUDA details that matter include variable-length response masks, numerical stability for `logsigmoid`/`softplus` forms, keeping chosen and rejected chunks aligned, and avoiding synchronization between many small reductions. Because these kernels often sit in [RLHF](https://arxiv.org/abs/2203.02155) (reinforcement learning from human feedback) or alignment loops, integration with trainer APIs and autograd correctness matters as much as the raw kernel body.

Inspect first: Liger's [chunked-loss package](https://github.com/linkedin/Liger-Kernel/tree/main/src/liger_kernel/chunked_loss), especially the [DPO loss implementation](https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/chunked_loss/dpo_loss.py), [ORPO loss implementation](https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/chunked_loss/orpo_loss.py), and [fused preference-loss implementation](https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/chunked_loss/fused_linear_preference.py), plus [TRL's supported Liger trainers](https://huggingface.co/docs/trl/liger_kernel_integration).

#### Custom Autograd Backward Fusion

Custom autograd backward fusion makes the training memory savings real. A fused forward-only kernel can still lose most of its benefit if PyTorch backward reconstructs the graph as many primitive operations and saves large tensors for gradient computation. A custom `torch.autograd.Function`, C++/CUDA extension, or Triton-backed wrapper defines exactly which tensors are saved, which statistics are cached, and which gradients are computed by fused kernels or by controlled recomputation.

The platform details are mostly ABI and correctness details: saved tensors must have stable dtype and contiguity assumptions; [AMP](https://pytorch.org/docs/stable/amp.html) (automatic mixed precision) custom forward/backward decorators must preserve mixed-precision behavior; reductions for parameter gradients must be deterministic enough for the training regime; and in-place gradient buffers must not violate PyTorch autograd aliasing rules. Kernel authors also need to decide whether the backward path is a single fused kernel, a short sequence of fused kernels, or a recomputation schedule that spends extra FLOPs to avoid HBM materialization.

Inspect first: Liger's [`LigerFusedLinearCrossEntropyFunction`](https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/fused_linear_cross_entropy.py), Liger's [RMSNorm kernel](https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/rms_norm.py) for a fused forward/backward normalization path, and Apex's [fused layer norm source wrapper](https://nvidia.github.io/apex/_modules/apex/normalization/fused_layer_norm.html), which shows Python autograd functions dispatching to CUDA extension entry points.

### Why It Matters For Training

Large vocabulary logits, long contexts, and post-training objectives can dominate memory even when attention is optimized. Fused linear-cross-entropy avoids materializing full logits. Preference losses such as DPO/ORPO/SimPO can involve multiple sequences, reference scores, or pairwise terms; chunked/fused kernels keep memory bounded.

### Explore Further

- `paper`: [Liger-Kernel: Efficient Triton Kernels for LLM Training](https://arxiv.org/abs/2410.10989)
- `official repo`: [Liger Kernel repository](https://github.com/linkedin/Liger-Kernel)
- `official docs`: [Liger Kernel documentation](https://linkedin.github.io/Liger-Kernel/)
- `official docs`: [Apex FusedLayerNorm](https://nvidia.github.io/apex/layernorm.html)
- `official docs`: [Megatron-Core fused layer norm](https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.fusions.fused_layer_norm.html)

### Implementation Examples

- Liger RMSNorm/RoPE/SwiGLU/CrossEntropy/FusedLinearCrossEntropy: inspect Triton kernels and Hugging Face patching APIs in [`Liger-Kernel`](https://github.com/linkedin/Liger-Kernel).
- Liger post-training kernels: inspect DPO, ORPO, CPO, SimPO, KTO, JSD, and distillation loss implementations.
- Apex/Megatron fused normalization: inspect fused LayerNorm/RMSNorm CUDA paths and persistent layer norm options.

### Caveats

Fusion is not automatically good. It can increase register pressure, lower occupancy, complicate backward, or reduce composability. The best criterion is concrete: does fusion remove an HBM materialization, a launch boundary, or a synchronization point that is visible in the profile?

## 6. Recomputation and Deterministic Backward

### Technique

Recomputation stores compact statistics and recomputes expensive intermediates during backward instead of saving full activations. Attention backward often recomputes $S=QK^\top$ and probabilities from saved row statistics. Deterministic backward constrains reduction order so atomic or parallel accumulation produces reproducible results.

### Primitive Techniques

#### Statistic-Only Activation Saving

Statistic-only activation saving is the forward/backward contract used by memory-efficient attention: the forward pass saves compact row or tile statistics, such as row maxima, sums of exponentials, or log-sum-exp values, instead of saving the full score matrix $S=QK^\top$ or softmax probability matrix $P$. In an LLM attention kernel this changes the activation footprint from quadratic sequence storage toward a small per-row tensor, while still giving backward enough information to reconstruct the same normalized probabilities.

The platform detail that matters is that these statistics are part of the kernel ABI, not a convenient debug output. cuDNN SDPA exposes training-time `softmax_stats`, optional logit-max and sum-exp tensors, and architecture-specific support for fp16, bf16, and fp8 attention on Ampere, Hopper, and Blackwell; FlashAttention exposes the same idea through saved `softmax_lse` tensors in its Python autograd wrapper. Inspect first: [FlashAttention](https://arxiv.org/abs/2205.14135) for the IO argument, [cuDNN Frontend SDPA](https://docs.nvidia.com/deeplearning/cudnn/frontend/latest/operations/Attention.html) for the production API, and the [FlashAttention Python interface](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/flash_attn_interface.py) for the saved tensors.

#### Backward Score Recomputation

Backward score recomputation regenerates the local $QK^\top$ score tile during backpropagation from saved $Q$, $K$, masks, scales, and softmax statistics. The backward kernel then rebuilds the probability tile only long enough to compute $dV$, $dP$, $dS$, $dQ$, and $dK$. This is why FlashAttention-style training can avoid writing the probability matrix during forward: backward spends extra matmul and softmax work to buy back HBM capacity.

CUDA kernel authors must keep the recompute path bitwise-compatible enough with the forward path to respect causal masks, sliding windows, ALiBi, softcap, padding, and variable-length layouts. The recomputed tile also has to be scheduled around tensor-core work and non-matmul softmax work; FA4 describes Blackwell backward as a chain of five MMA operations plus elementwise softmax and `dS` work, with tensor memory used to reduce shared-memory pressure. Inspect first: [FlashAttention-2](https://arxiv.org/abs/2307.08691) for work partitioning, [`flash-attention`](https://github.com/Dao-AILab/flash-attention) for forward/backward bindings, and [FlashAttention-4](https://arxiv.org/abs/2603.05451) plus the [FA4 blog](https://tridao.me/blog/2026/flash4/) for modern Blackwell backward scheduling.

#### [RNG](https://docs.nvidia.com/cuda/curanddx/get_started/philox.html) (Random Number Generator) Replay For Dropout

RNG replay recreates dropout masks during backward from a saved seed, offset, or generator state instead of saving a dense mask tensor. In training attention, the mask is logically applied after softmax and before the $PV$ multiply, so the backward pass must use the same dropped elements when it recomputes probabilities. The same principle applies to checkpointed fused MLPs and other dropout-containing subgraphs: recomputation is only correct if stochastic choices are replayed consistently.

The CUDA detail is counter-based randomness, typically Philox-style seed and offset handling. cuDNN SDPA accepts an RNG seed tensor and offset tensor for dropout, while FlashAttention saves an `rng_state` alongside `softmax_lse`. At the framework level, PyTorch checkpointing stashes and restores RNG state by default for deterministic equivalence, with caveats when tensors move across unexpected device types. Inspect first: [PyTorch checkpointing](https://docs.pytorch.org/docs/stable/checkpoint.html), [cuDNN SDPA dropout options](https://docs.nvidia.com/deeplearning/cudnn/frontend/latest/operations/Attention.html), and the [FlashAttention Python interface](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/flash_attn_interface.py).

#### Checkpoint Granularity Selection

Checkpoint granularity selection decides how much of the forward graph is saved and how much is recomputed: a single attention subgraph, selected operations inside a layer, a full transformer layer, a uniform block of layers, or a whole pipeline segment. The right granularity depends on sequence length, hidden size, batch size, tensor parallelism, and whether the expensive tensors are quadratic attention intermediates or dense linear-layer activations.

For LLM training kernels, granularity determines which kernels are relaunched in backward and which tensors must stay resident while sharded parameters and activations are gathered. Fine-grained self-attention recomputation can save the quadratic softmax/dropout tensors with modest extra compute, while full-layer recomputation saves more memory but replays expensive GEMMs. Under [FSDP](https://docs.pytorch.org/docs/stable/fsdp.html) (Fully Sharded Data Parallel), [ZeRO](https://deepspeed.readthedocs.io/en/stable/zero3.html) (Zero Redundancy Optimizer), tensor parallelism, and pipeline parallelism, the checkpoint boundary also changes when parameter all-gathers, gradient reduce-scatters, and peer-to-peer activation transfers can overlap with useful recompute. Inspect first: [PyTorch checkpointing and selective policies](https://docs.pytorch.org/docs/stable/checkpoint.html), [NeMo activation recomputation](https://docs.nvidia.com/nemo-framework/user-guide/25.04/nemotoolkit/features/optimizations/activation_recomputation.html), [PyTorch FSDP](https://docs.pytorch.org/docs/stable/fsdp.html), and [DeepSpeed ZeRO](https://deepspeed.readthedocs.io/en/stable/zero3.html).

#### Deterministic Reduction Order

Deterministic reduction order fixes the sequence in which partial gradients are added, rather than allowing CTAs or warps to race through global atomics in whatever order the scheduler happens to run them. This matters because floating-point addition is not associative: the same mathematical sum can produce slightly different bits when partials are accumulated in different orders. In long LLM training runs, deterministic backward makes rare instabilities easier to reproduce and compare.

At the CUDA level, ordinary global `atomicAdd` gives atomicity for each read-modify-write operation but does not, by itself, impose a deterministic order among competing CTAs. Attention backward is especially sensitive for gradients such as $dQ$, where multiple tiles reduce into the same output. Deterministic implementations use a fixed reduction tree, a serialized global accumulation order, or a semaphore-style protocol with memory fences, accepting some throughput or memory overhead. Inspect first: [FA4](https://arxiv.org/abs/2603.05451), the [FA4 deterministic-mode discussion](https://tridao.me/blog/2026/flash4/), the [CUDA atomic functions](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomic-functions), and [PyTorch reproducibility notes](https://docs.pytorch.org/docs/stable/notes/randomness.html).

#### Semaphore Or Counter Coordination

Semaphore or counter coordination is an explicit synchronization scheme for ordering producers before final gradient writes. A kernel can use counters, lock words, or turn variables so CTAs publish partial results, wait for a known predecessor, and then perform a final accumulation in a fixed order. In LLM kernels this is mainly a backward-pass tool: it is used when recomputation divides an attention gradient across many work tiles but the final user-visible tensor must be reproducible.

The hard CUDA details are memory visibility, progress, and occupancy. A semaphore must pair atomic updates with the right memory-fence semantics so that a consumer does not observe the counter before the producer's partial gradients are visible. It must also avoid deadlock when only a subset of CTAs can be resident, and it should reduce contention through scheduling choices such as CTA swizzling or shortest-processing-time-first ordering for causal tiles. Inspect first: the [FA4 blog](https://tridao.me/blog/2026/flash4/) for semaphore-style deterministic backward, [CUDA memory fence functions](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#memory-fence-functions), and the [CUDA atomic functions](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomic-functions).

#### Transposed Recompute Layout

Transposed recompute layout changes the backward traversal so the recomputed attention tile is produced in the orientation consumed by the gradient GEMMs. Instead of mirroring the forward tile layout and then transposing through shared memory, the backward pass can compute or store $S^\top$, $P^\top$, and later $dS^\top$ directly in the layout needed by $dV$ and $dK$. This is still exact recomputation; the optimization is the layout of the temporary tile, not a change to the attention equation.

On Blackwell, FA4 uses tensor memory for intermediate storage and describes a backward pipeline where $P^\top$ and $dS^\top$ can be placed in operand layouts consumed by MMA instructions. The same idea generalizes to CUDA kernel design: choose tile ownership, strides, and shared-memory or tensor-memory layouts around the gradient consumers, because backward has different dataflow than forward. Inspect first: [FA4](https://arxiv.org/abs/2603.05451), the [FA4 backward-pass blog section](https://tridao.me/blog/2026/flash4/), and [CUTLASS/CuTe documentation](https://docs.nvidia.com/cutlass/) for layout vocabulary.

#### Recomputation-Aware Communication

Recomputation-aware communication treats activation replay as schedulable work that can hide distributed communication. When a checkpointed backward segment reruns forward kernels, that compute window may overlap with parameter all-gather, gradient reduce-scatter, tensor-parallel activation exchange, context-parallel all-gather/reduce-scatter, or pipeline-parallel sends and receives. The recompute decision therefore changes not just memory and FLOPs, but the communication timeline of the training step.

The platform details are CUDA streams, [NCCL](https://docs.nvidia.com/deeplearning/nccl/) (NVIDIA Collective Communications Library) or peer-to-peer launch order, chunk sizes, and SM occupancy. If recompute consumes all SMs or forces a dependency on parameters that are not yet gathered, it can expose communication rather than hide it. If it is chunked at transformer-layer or attention-block boundaries, it can fill bubbles created by data-parallel, tensor-parallel, context-parallel, or pipeline-parallel collectives. Inspect first: [MegaScale](https://arxiv.org/abs/2402.15627) for large-scale overlap motivation, [NeMo communication overlap](https://docs.nvidia.com/nemo-framework/user-guide/25.04/nemotoolkit/features/optimizations/communication_overlap.html) for concrete [DP](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html) (data parallelism)/[TP](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html) (tensor parallelism)/[PP](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html) (pipeline parallelism)/[CP](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/context_parallel.html) (context parallelism) overlap knobs, and [Megatron-Core](https://developer.nvidia.com/megatron-core) for implementation context.

### Why It Matters For Training

Memory saved in forward can fund longer context, bigger batch, or larger model shards. Determinism matters for debugging long frontier runs, where a rare loss spike or divergence can cost many GPU-days to reproduce.

### Explore Further

- `paper`: [FlashAttention-2](https://arxiv.org/abs/2307.08691)
- `paper`: [FlashAttention-3](https://arxiv.org/abs/2407.08608)
- `paper/blog`: [FlashAttention-4](https://arxiv.org/abs/2603.05451), [FA4 blog](https://tridao.me/blog/2026/flash4/)
- `official docs`: [PyTorch FSDP and checkpointing context](https://docs.pytorch.org/docs/stable/fsdp.html)

### Implementation Examples

- FlashAttention backward: inspect saved softmax statistics and backward recomputation in [`flash-attention`](https://github.com/Dao-AILab/flash-attention).
- FA4 backward: study transposed recomputation, TMEM reuse, 2-CTA MMA, and deterministic semaphore-style reduction.
- FSDP/ZeRO training stacks: inspect how activation checkpointing and parameter sharding shift memory pressure into recompute and communication schedules.

### Caveats

Recomputation saves memory but increases compute and can complicate RNG, dropout, masks, and numerical reproducibility. Deterministic modes may serialize reductions and cost throughput.

## 7. MoE Routing as Data Layout and Communication

### Technique

Mixture-of-Experts layers route tokens to a small subset of experts. The routing operation is a data-layout transformation plus communication: compute top-k experts, permute tokens, all-to-all them across expert-parallel ranks, run grouped expert GEMMs, and combine results back in original token order. Efficient MoE kernels minimize padding, dropped tokens, and interconnect idle time.

### Primitive Techniques

#### Router Logits And Top-k Selection

Router logits are the per-token scores that decide which experts receive a token. A routing layer usually projects the token representation into expert scores, applies a small top-k selection, and emits two compact tensors: expert IDs and routing weights. In top-1 routing each token appears once in the expert workload; in top-k routing it is replicated to several experts and later combined by the routing weights.

For token $x_t$ and router matrix $W_r$, a common top-$k$ routing contract is:

$$
r_t = x_t W_r,\qquad
E_t = \operatorname{TopK}(r_t, k),\qquad
w_{t,e} = \frac{\exp(r_{t,e})}{\sum_{e'\in E_t}\exp(r_{t,e'})}\quad e\in E_t.
$$

The MoE layer output is then a weighted expert sum:

$$
y_t = \sum_{e\in E_t} w_{t,e}\, f_e(x_t),
$$

where $f_e$ is the expert MLP. Dispatch kernels implement the data movement implied by $E_t$; combine kernels implement the final weighted sum.

Symbols: $x_t$ is token $t$'s hidden state; $W_r$ is the router projection; $r_t$ is the vector of expert logits; $k$ is the number of selected experts; $E_t$ is the selected expert set; $e$ and $e'$ index experts; $w_{t,e}$ is token $t$'s normalized routing weight for expert $e$; $f_e$ is expert $e$'s MLP; and $y_t$ is the MoE output for token $t$.

In LLM training kernels, these IDs and weights are the control plane for every later data movement. The IDs determine destination ranks, expert-local offsets, grouped-GEMM M dimensions, and backward routing paths; the weights either scale expert outputs or participate in the combine/reduction path. The CUDA details that matter are small-k row selection, tie-breaking or determinism policy, index dtype, score precision, and whether router bias or load-balancing updates are fused with score computation. Inspect first: the [DeepSeek-V3 technical report](https://arxiv.org/abs/2412.19437) for group-limited routing and auxiliary-loss-free balancing, [Tutel](https://arxiv.org/abs/2206.03382) for routing-driven adaptive execution, and [DeepEP `Buffer.dispatch`](https://github.com/deepseek-ai/DeepEP/blob/main/deep_ep/buffer.py) for the public `topk_idx` and `topk_weights` contract consumed by dispatch.

#### Expert Histogram And Prefix Sum

The expert histogram counts how many routed token copies target each expert, rank, and [RDMA](https://docs.nvidia.com/networking/display/rdmacore50/rdma-core) (remote direct memory access) domain. Prefix sums over those counts turn an irregular set of token-to-expert decisions into deterministic write offsets for packed send buffers and expert-local segments. Conceptually, this is the step where sparse routing becomes an explicit memory layout.

The routing-to-layout transformation is:

$$
\begin{array}{c}
\begin{array}{c|cccccc}
\text{token} & t_0 & t_1 & t_2 & t_3 & t_4 & t_5\\ \hline
\text{expert} & e_2 & e_0 & e_2 & e_1 & e_0 & e_2
\end{array}
\\[3mm]
\Downarrow
\\[-1mm]
\begin{array}{c|ccc}
\text{expert} & e_0 & e_1 & e_2\\ \hline
\text{packed rows} &
[t_1,t_4] &
[t_3] &
[t_0,t_2,t_5]\\
\text{offset} & o_0=0 & o_1=h_0 & o_2=h_0+h_1
\end{array}
\end{array}
$$

The packed tensor is dense within each expert segment even though the original token-to-expert map is sparse.

Diagram symbols: $t_0,\ldots,t_5$ are token rows; $e_0,e_1,e_2$ are experts; $h_e$ is the number of token rows assigned to expert $e$; and $o_e$ is the prefix offset where expert $e$'s contiguous segment begins in the packed buffer.

If $E_t$ is the selected expert set for token $t$, the local expert counts are:

$$
h_e = \sum_t \mathbf{1}[e\in E_t],\qquad
o_e = \sum_{e'<e} h_{e'}.
$$

Symbols: $h_e$ is the local count for expert $e$; $t$ indexes tokens; $E_t$ is token $t$'s selected expert set; $\mathbf{1}[\cdot]$ is an indicator that equals one when its condition is true; $o_e$ is expert $e$'s packed-buffer start offset; and $e'$ ranges over experts ordered before $e$.

The prefix offset $o_e$ gives the start of expert $e$'s packed segment; per-token local ranks within that segment are usually produced by atomics, scans, or segmented counters.

Training kernels use these counts to size communication, build rank-major or expert-major segments, and feed grouped GEMM with the M dimension for each local expert. The same metadata is also needed by backward, where gradients must follow the inverse route without rebuilding inconsistent offsets. CUDA/platform concerns include atomic histogramming versus segmented counting, prefix-scan placement, avoiding CPU synchronization for graph capture, and aligning each expert segment to the block sizes expected by tensor-core GEMM kernels. Inspect first: [DeepEP `get_dispatch_layout`](https://github.com/deepseek-ai/DeepEP/blob/main/deep_ep/buffer.py), which returns token counts per rank, RDMA rank, and expert, the [DeepGEMM README](https://github.com/deepseek-ai/DeepGEMM/blob/main/README.md) for contiguous grouped-GEMM alignment requirements, and [MegaBlocks](https://arxiv.org/abs/2211.15841) for a block-sparse alternative to padded expert segments.

#### Token Permutation / Packing

Token permutation copies or gathers activation rows into the order required by expert-parallel communication and expert GEMM. For top-k routing, the packed buffer contains one row per selected expert, so a single source token may appear multiple times with different destination experts and weights. The packed layout usually groups rows by destination rank first and by local expert second, with side metadata that remembers the original token position.

In LLM training, packing is on the hot path of both forward and backward. Forward packs residual-stream activations before dispatch; backward packs expert-output gradients or uses the stored dispatch handle to undo the layout. CUDA details include vectorized loads over the hidden dimension, coalesced stores into packed buffers, hidden-size alignment, BF16 versus FP8 payload formats, and keeping prefix offsets resident enough that the pack kernel is not dominated by random memory traffic. Inspect first: [DeepEP `Buffer.dispatch` and `Buffer.combine`](https://github.com/deepseek-ai/DeepEP/blob/main/deep_ep/buffer.py) for dispatch/combine handles and packed receive tensors, the [DeepGEMM README](https://github.com/deepseek-ai/DeepGEMM/blob/main/README.md) for the contiguous MoE input layout consumed by grouped FP8 GEMM, and [MegaBlocks](https://arxiv.org/abs/2211.15841) for when the permutation target is a block-sparse layout rather than one dense packed buffer.

#### All-to-All Dispatch

All-to-all dispatch exchanges the packed token buffers across the expert-parallel group so that each GPU receives the tokens for the experts it owns. Standard collectives describe this as each rank sending a distinct chunk to every other rank, but MoE dispatch often wraps or replaces the standard collective because chunk sizes are dynamic and must track per-expert token counts.

In training, this communication sits between the router and the expert MLP, then appears again in reverse for gradients. The performance question is not just bandwidth; it is whether [NVLink](https://www.nvidia.com/en-us/data-center/nvlink/) (NVIDIA's high-bandwidth GPU interconnect), RDMA, buffer registration, rank placement, and stream scheduling allow tensor-core expert work to stay fed. CUDA/platform details include intranode NVLink paths, internode InfiniBand/RDMA paths, [NVSHMEM](https://docs.nvidia.com/nvshmem/) (GPU-side OpenSHMEM-style communication library) or NCCL integration, symmetric communication buffers, rank-to-device mapping, and whether the communication kernel consumes SMs that could otherwise run GEMM. Inspect first: the [DeepEP README](https://github.com/deepseek-ai/DeepEP/blob/main/README.md) and [DeepEP kernels](https://github.com/deepseek-ai/DeepEP/tree/main/csrc/kernels) for high-throughput NVLink/RDMA dispatch and low-latency RDMA kernels, [NCCL AlltoAll](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html#alltoall) for the baseline collective contract, and the [DeepSeek-V3 technical report](https://arxiv.org/abs/2412.19437) for the large-scale expert-parallel setting that motivates custom communication.

#### Expert Combine / Unpermute

Expert combine is the inverse data-layout operation after expert computation. It takes expert outputs in packed expert order, applies or propagates routing weights as required by the framework, and restores a dense tensor in the original token order. With top-k routing, combine is also a reduction because multiple expert outputs may contribute to the same token.

The forward combine equation is the same weighted sum as the MoE layer output:

$$
y_t = \sum_{e\in E_t} w_{t,e}\, y_{t,e}^{\mathrm{expert}}.
$$

Symbols: $y_t$ is the final token output after combine; $E_t$ is the selected expert set for token $t$; $w_{t,e}$ is the routing weight; $y_{t,e}^{\mathrm{expert}}$ is expert $e$'s output for token $t$ before weighted accumulation; $dy_t$ is the upstream output gradient; and $dy_{t,e}^{\mathrm{expert}}$ is the gradient routed back to expert $e$.

Backward must route $dy_t$ back to each selected expert, usually with $dy_{t,e}^{\mathrm{expert}} = w_{t,e}dy_t$ plus an optional gradient for the routing weight.

In training, combine appears in forward after the expert MLP and in backward as the inverse of dispatch. It must preserve enough metadata to send gradients back to the right expert outputs, recover gradients for routing weights when those weights are differentiable, and avoid losing contributions when several routed copies map to the same source token. CUDA details include scatter-add versus gather-reduce schedules, atomic or segmented reductions for top-k accumulation, BF16/FP32 accumulation choices, optional bias application, and event ordering with the preceding grouped GEMM. Inspect first: [DeepEP `Buffer.combine`](https://github.com/deepseek-ai/DeepEP/blob/main/deep_ep/buffer.py) for combine arguments, dispatch handles, and backward examples where dispatch backward is expressed as combine, and [Tutel](https://arxiv.org/abs/2206.03382) for encode/decode style MoE routing operators in an adaptive execution stack.

#### Capacity Or Dropless Policy

Capacity policy decides what happens when experts receive uneven numbers of tokens. A fixed-capacity MoE may pad underfull experts and drop, reroute, or cap overflow tokens; a dropless MoE accepts the dynamic token counts and uses sparse or grouped computation to process every routed token. This policy is not only a modeling choice, because it fixes the shape guarantees available to kernels and collectives.

Training kernels feel the policy directly. Padding wastes GEMM work and bandwidth, dropping changes the optimization problem, and dropless execution pushes more complexity into scans, segmented layouts, and variable-size expert matmuls. CUDA/platform details include CUDA graph compatibility, worst-case buffer sizing, expert-segment alignment, masked grouped GEMM for stable shapes, and block-sparse kernels that avoid both padding and token dropping. Inspect first: [MegaBlocks](https://arxiv.org/abs/2211.15841) for dropless block-sparse MoE training, the [DeepGEMM README](https://github.com/deepseek-ai/DeepGEMM/blob/main/README.md) for contiguous and masked grouped-GEMM layouts, and [DeepEP `Buffer.dispatch`](https://github.com/deepseek-ai/DeepEP/blob/main/deep_ep/buffer.py) for `num_worst_tokens` and alignment options in dispatch.

#### Load-Balancing Control

Load-balancing control is the feedback mechanism that keeps the router from collapsing onto a small set of experts. It can be implemented with auxiliary losses, router bias updates, capacity constraints, group-limited selection, or other policies that steer token counts toward usable expert workloads. The goal is not perfectly equal counts in every microbatch, but enough balance that no rank becomes a persistent straggler.

In LLM training kernels, load balance determines whether grouped GEMM has useful M dimensions and whether all-to-all has a small number of overloaded destinations. The implementation hooks often consume the same histograms used for dispatch, then feed statistics back into the router or loss computation. CUDA/platform concerns include cross-rank reductions of expert counts, asynchronous histogram availability, deterministic updates, and separating load-control work from the latency-critical pack and dispatch kernels. Inspect first: the [DeepSeek-V3 technical report](https://arxiv.org/abs/2412.19437) for auxiliary-loss-free load balancing, [Tutel](https://arxiv.org/abs/2206.03382) for runtime adaptation under dynamic expert workloads, and [DeepEP `get_dispatch_layout`](https://github.com/deepseek-ai/DeepEP/blob/main/deep_ep/buffer.py) for the dispatch-layout counts that expose imbalance to the runtime.

#### FP8-Aware Dispatch

FP8-aware dispatch moves quantized activations together with the scale metadata needed to interpret them. Instead of sending only BF16 activation rows, the dispatch path may send an FP8 payload plus per-token or per-block scales, then let the expert GEMM consume that representation directly. This reduces communication volume, but only if scale layout and quantization boundaries match the GEMM implementation.

In training kernels, FP8 dispatch is most useful when expert-parallel communication is bandwidth-bound and the expert MLP is already written for FP8 tensor cores. The router and pack kernels must keep activation rows and scale rows aligned through top-k replication, all-to-all, expert computation, and combine. CUDA/platform details include E4M3 payload layout, scale granularity such as hidden-dimension blocks, TMA-aligned scale tensors, different scale formats on SM90 versus SM100, and avoiding extra cast kernels that erase the communication savings. Inspect first: the [DeepEP README](https://github.com/deepseek-ai/DeepEP/blob/main/README.md) and [`Buffer.dispatch`](https://github.com/deepseek-ai/DeepEP/blob/main/deep_ep/buffer.py) for FP8 dispatch and BF16 combine interfaces, and the [DeepGEMM README](https://github.com/deepseek-ai/DeepGEMM/blob/main/README.md) for grouped FP8 GEMM, scale-factor layout transforms, and architecture-specific scale formats.

#### Overlap Hooks

Overlap hooks are the event, stream, and runtime callback mechanisms that let communication and expert compute proceed as a staged pipeline. Rather than treating dispatch, GEMM, and combine as three fully serialized phases, the runtime records dependencies and begins communication or computation as soon as the relevant buffer slice is ready.

In LLM training, overlap is essential because MoE introduces all-to-all traffic into the middle of the transformer block. A forward pass may overlap part of dispatch with local preparation or expert GEMM, while backward may overlap combine-style gradient movement with neighboring compute. CUDA/platform details include communication streams, CUDA events, `async_finish` style APIs, allocation ownership on the communication stream, SM reservation or SM-free progress, CUDA graph restrictions, and correctness barriers before packed buffers are reused. Inspect first: the [DeepEP README](https://github.com/deepseek-ai/DeepEP/blob/main/README.md), [`Buffer` APIs](https://github.com/deepseek-ai/DeepEP/blob/main/deep_ep/buffer.py), and [`EventOverlap`](https://github.com/deepseek-ai/DeepEP/blob/main/deep_ep/utils.py) for `previous_event`, communication-stream allocation, and hook-based overlap, plus the [DeepSeek-V3 technical report](https://arxiv.org/abs/2412.19437) for the training-system motivation for overlapping expert-parallel communication.

### Why It Matters For Training

Sparse MoE offers more parameters at roughly fixed active compute, but routing can erase the savings. Expert loads are dynamic, all-to-all communication crosses NVLink/RDMA domains, and grouped GEMMs have variable M dimensions. Training adds backward dispatch, expert weight gradients, and load-balancing constraints.

### Explore Further

- `paper`: [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
- `official repo`: [DeepEP](https://github.com/deepseek-ai/DeepEP)
- `official repo`: [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM)
- `paper`: [MegaBlocks: Efficient Sparse Training with Mixture-of-Experts](https://arxiv.org/abs/2211.15841)
- `paper`: [Tutel: Adaptive Mixture-of-Experts at Scale](https://arxiv.org/abs/2206.03382)

### Implementation Examples

- DeepEP: inspect high-throughput dispatch/combine kernels, low-latency RDMA kernels, FP8 dispatch, SM count control, and hook-based overlap.
- DeepGEMM grouped GEMM: inspect contiguous and masked grouped FP8 GEMM APIs for MoE expert matmuls.
- MegaBlocks: study dropless MoE reformulated as block-sparse GPU kernels.
- Tutel: study adaptive parallelism, adaptive pipelining, and fast encode/decode kernels.
- DeepSeek-V3: study auxiliary-loss-free load balancing and MoE communication overlap at H800 scale.

### Caveats

MoE kernel performance depends heavily on batch size, expert count, top-k, token distribution, interconnect topology, and capacity policy. A microbenchmark of grouped GEMM alone misses dispatch/combine cost.

## 8. Expert Grouped GEMM and MoE Mega-Kernels

### Technique

Grouped GEMM batches many expert matmuls into a single kernel or compact kernel family. A MoE mega-kernel fuses even more: dispatch, expert GEMM 1, activation, expert GEMM 2, and combine. The goal is to keep tensor cores busy while hiding token movement over NVLink or RDMA.

### Primitive Techniques

#### M-Grouped GEMM

M-grouped GEMM is the MoE-specialized case where a kernel batches many expert matrix multiplications that share the same $N$ and $K$ dimensions but have different $M$ dimensions. In an expert MLP, $M$ is the number of routed token rows assigned to a given expert, so it changes every batch even when all experts have identical hidden and intermediate widths.

The grouped workload is a list of same-$K,N$ GEMMs with different row counts:

$$
\begin{array}{c|c|c|c}
\text{expert} & X_e & W_e & Y_e\\ \hline
e_0 & \mathbb{R}^{M_0\times K} & \mathbb{R}^{K\times N} & \mathbb{R}^{M_0\times N}\\
e_1 & \mathbb{R}^{M_1\times K} & \mathbb{R}^{K\times N} & \mathbb{R}^{M_1\times N}\\
\vdots & \vdots & \vdots & \vdots\\
e_{E-1} & \mathbb{R}^{M_{E-1}\times K} & \mathbb{R}^{K\times N} & \mathbb{R}^{M_{E-1}\times N}
\end{array}
$$

Grouped kernels try to schedule tiles across this table so short experts do not leave tensor cores idle while long experts finish.

Diagram symbols: $e_0,\ldots,e_{E-1}$ are local experts; $X_e$ is expert $e$'s packed input activation matrix; $W_e$ is expert $e$'s weight matrix; $Y_e=X_eW_e$ is the expert output; $M_e$ is the routed token count for expert $e$; $K$ is the input/reduction dimension; $N$ is the expert output dimension; and $E$ is the number of experts in the grouped launch.

In training kernels, this primitive is used for expert forward GEMMs and for backward activation-gradient GEMMs. Instead of launching one GEMM per expert, the runtime presents a list of expert segments to one grouped kernel. The kernel scheduler then keeps CTAs or persistent thread blocks busy by pulling tiles from many small expert problems, which amortizes launch overhead and reduces the idle-SM tail caused by imbalanced token counts.

CUDA details that matter are the group metadata path, segment alignment, and scheduler overhead. DeepGEMM documents that its contiguous M-grouped layout concatenates expert token rows and requires each expert segment to be aligned to the GEMM M block size. CUTLASS grouped kernels expose the more general scheduling problem: each CTA repeatedly asks a problem visitor for the next tile, and scheduler mode or problem ordering can matter when problem sizes are small. Inspect first: [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) for `m_grouped_*` APIs and M-block alignment utilities, [Megatron-Core MoE](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/moe.html) for the `--moe-grouped-gemm` training integration, and [CUTLASS grouped kernel schedulers](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/grouped_scheduler.html) for the underlying CUDA scheduling model.

#### K-Grouped GEMM

K-grouped GEMM batches expert matrix multiplications where the reduction dimension varies across experts while the output tile shape is otherwise stable. In MoE training, this appears naturally in expert weight-gradient computation: the number of contributing routed tokens differs per expert, so the reduction over token rows is irregular even though each expert weight matrix has the same logical shape.

The training use case is the backward weight-gradient path. A kernel such as a grouped `tn` GEMM can compute many $dW_e=X_e^\top dY_e$ products in one launch, with each expert $e$ contributing a different number of token rows. This avoids a long sequence of tiny weight-gradient GEMMs and lets the scheduler mix short and long reductions across the GPU.

The main platform issue is load balance in the reduction dimension. Tiles with larger $K$ run longer, so grouped schedulers may need problem sorting, host precomputation, or device-side scheduling that avoids giving one subset of CTAs all long reductions. FP8/FP4 variants add scale-factor layout requirements, and DeepGEMM exposes K-grouped packing utilities for the scale tensors. Inspect first: [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) for `k_grouped_fp8_gemm_*` and `get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor`, plus [CUTLASS grouped scheduler notes on K imbalance](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/grouped_scheduler.html#improving-load-balance-by-sorting-problems).

#### Contiguous Expert Layout

A contiguous expert layout packs routed tokens for all local experts into one buffer and records cumulative expert offsets. Conceptually, token permutation turns an input batch from token-major order into expert-major order: all tokens for expert 0, then all tokens for expert 1, and so on. The grouped GEMM reads those contiguous expert segments directly.

In LLM training, this layout is the bridge between routing, expert-parallel communication, and grouped compute. A dispatcher first exchanges tokens across expert-parallel ranks, local permutation groups the received tokens by expert, grouped GEMM consumes the packed segments, and unpermutation or combine scatters the weighted expert outputs back to token order. Fusion of token permutation and unpermutation is therefore as important as the GEMM itself because otherwise the layout conversion can dominate small expert batches.

CUDA details include coalesced loads, TMA alignment, offset generation, and padding of segment ends to the tile shape expected by the GEMM kernel. cuDNN's grouped GEMM fusion APIs describe MoE groups as contiguous `M` ranges with padded offsets, which is the same metadata idea exposed at a framework level. Inspect first: [DeepGEMM grouped contiguous layout](https://github.com/deepseek-ai/DeepGEMM), [DeepEP](https://github.com/deepseek-ai/DeepEP) for dispatch/combine buffers, [Megatron-Core MoE](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/moe.html) for permute fusion, and [cuDNN Grouped GEMM + dGLU](https://docs.nvidia.com/deeplearning/cudnn/frontend/latest/fe-oss-apis/gemm_fusions/grouped_gemm_dglu.html) for `padded_offsets` style metadata.

#### Masked Expert Layout

A masked expert layout allocates a fixed maximum slot range for each expert and supplies masks or count tensors that identify how much of each expert's slot range is valid work. The physical tensor shape stays stable even when routing sends different token counts to each expert.

This is useful when CUDA Graph capture, inference decoding, or framework scheduling wants invariant launch shapes but the router's actual expert loads remain dynamic. In training, the same idea can be used when a stable graph or ahead-of-time compiled plan is worth some padding overhead. The grouped kernel skips or masks invalid rows while preserving predictable tensor addresses and launch parameters.

The tradeoff is wasted capacity versus reduced CPU synchronization. A masked kernel needs extra predicates in the tile scheduler or epilogue, but it can avoid waiting for the CPU to learn exact expert counts before launching graph-compatible work. DeepGEMM calls out masked M-grouped GEMM for cases where CUDA graphs are enabled and the CPU is unaware of expert token counts, and notes that low-latency DeepEP outputs can feed that layout. Inspect first: [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) for `m_grouped_fp8_gemm_nt_masked`, [DeepEP low-latency dispatch/combine](https://github.com/deepseek-ai/DeepEP), and [Megatron-Core MoE CUDA Graph support](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/moe.html).

#### Scale-Aware Grouped Epilogue

A scale-aware grouped epilogue fuses the arithmetic after the tensor-core matrix multiply with low-precision scale handling. For FP8 and FP4 MoE experts, the GEMM does not simply write an FP16/BF16 matrix. It may dequantize inputs or weights using block scales, accumulate in a wider type, apply router weights or activation derivatives, update amax statistics, and emit either BF16 outputs or re-quantized outputs plus new scale factors.

In training kernels, this primitive removes separate dequantization, activation, quantization, and scale-update launches around the expert MLP. For the first expert projection it can feed a fused SwiGLU or dGLU path; for the second projection it can multiply by per-token gating probabilities and optionally quantize the output for the next layer or communication stage.

Platform details are hardware-specific. DeepGEMM documents different scale formats for SM90 and SM100: SM90 uses FP32 scale factors, while SM100 uses packed UE8M0 scale factors, and both require TMA-aligned scale layouts. cuDNN [FE-OSS](https://docs.nvidia.com/deeplearning/cudnn/frontend/latest/fe-oss-apis/index.html) (Frontend Open-Source Software APIs) grouped GEMM fusions are explicitly SM100-oriented and expose grouped GEMM with quantization or GLU-family epilogues for MoE workloads. Inspect first: [DeepGEMM scale-layout utilities](https://github.com/deepseek-ai/DeepGEMM), [cuDNN GEMM Fusions](https://docs.nvidia.com/deeplearning/cudnn/frontend/latest/fe-oss-apis/gemm_fusions/gemm_fusions.html), and [cuDNN Grouped GEMM + Quant](https://docs.nvidia.com/deeplearning/cudnn/frontend/latest/fe-oss-apis/gemm_fusions/grouped_gemm_quant.html).

#### Expert Weight Transformation

Expert weight transformation pre-packs weights and their scale factors into the physical layout expected by the tensor-core kernel. The logical expert tensor may be stored in a framework-friendly `[num_experts, out, in]` or similar layout, but the fastest kernel may require a tile-major, transposed, interleaved, or scale-adjacent representation.

In MoE training and serving, this transformation is most visible for FP4 expert weights on Blackwell-class paths. DeepGEMM Mega MoE requires transformed FP4 expert weights with UE8M0 scale factors before calling the fused expert path. If the model also needs a standard DeepEP or framework path, the runtime must avoid keeping duplicate resident copies of the same expert weights or must arrange aliases/views carefully.

CUDA details include tensor-core operand layout, TMA alignment, scale packing, and the cost of refreshing transformed weights after optimizer updates. In inference the transform can often be paid once at load time; in training it may need to be integrated with the optimizer, master-weight format, or low-precision weight-casting pipeline. Inspect first: [DeepGEMM Mega MoE](https://github.com/deepseek-ai/DeepGEMM) for `transform_weights_for_mega_moe`, the [DeepGEMM Mega MoE test case](https://github.com/deepseek-ai/DeepGEMM/blob/main/tests/test_mega_moe.py), and the [SGLang/Miles DeepSeek-V4 integration note](https://www.lmsys.org/blog/2026-04-25-deepseek-v4/) describing transformed FP4 expert weights and aliased expert tensors.

#### Fused Expert MLP

A fused expert MLP combines the two expert linear layers and the intervening activation, usually SwiGLU or a GLU derivative, into one larger kernel or tightly coupled kernel family. Instead of materializing the intermediate expert activation to HBM and launching a second GEMM later, the implementation keeps more of the subgraph in registers, shared memory, or a controlled staging buffer.

For LLM training, the forward benefit is reduced launch count and reduced intermediate activation traffic. The backward path is harder: it needs gradients through both linear layers and the activation, may need the forward activation or a recomputed equivalent, and often benefits from dSwiGLU or dGLU epilogue fusion. A practical training stack may therefore fuse different pieces in forward and backward while keeping the same expert-major layout contract.

The platform limit is resource pressure. Fusing linear 1, activation, and linear 2 raises register use, shared-memory pressure, scheduling complexity, and the cost of handling imbalanced expert loads. On Blackwell-oriented paths, FP8xFP4 tensor-core compute, packed scale factors, [PDL](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programmatic-dependent-launch-and-synchronization) (Programmatic Dependent Launch, CUDA's mechanism for device-visible launch dependencies), and persistent scheduling all matter. Inspect first: [DeepGEMM Mega MoE](https://github.com/deepseek-ai/DeepGEMM) for the fused dispatch-linear1-SwiGLU-linear2-combine path, [cuDNN Grouped GEMM + SwiGLU](https://docs.nvidia.com/deeplearning/cudnn/frontend/latest/fe-oss-apis/gemm_fusions/grouped_gemm_swiglu.html) and [cuDNN Grouped GEMM + dGLU](https://docs.nvidia.com/deeplearning/cudnn/frontend/latest/fe-oss-apis/gemm_fusions/grouped_gemm_dglu.html) ([dGLU](https://docs.nvidia.com/deeplearning/cudnn/frontend/latest/fe-oss-apis/gemm_fusions/grouped_gemm_dglu.html) is a GLU-family backward derivative epilogue) for forward/backward epilogue fusions, and [HazyResearch Megakernels](https://github.com/HazyResearch/Megakernels) for the broader persistent megakernel design pattern.

#### Symmetric Communication Buffer

A symmetric communication buffer is a rank-coordinated GPU memory allocation whose addresses and layout are compatible across the processes participating in expert parallelism. Instead of treating dispatch and combine as opaque collectives separated from compute, the fused path can read and write a known communication workspace while expert GEMM work proceeds.

In MoE training, this is the primitive that lets dispatch, expert compute, and combine become one overlapped schedule. Routed tokens, top-k expert indices, top-k weights, per-rank receive buffers, and output slots live in a communication layout that both the network path and the tensor-core path understand. DeepGEMM's Mega MoE API allocates a symmetric buffer and then copies FP8 activations, scale factors, indices, and weights into it before launching the fused kernel.

The platform details are distributed-systems details inside a CUDA kernel boundary: NVLink versus RDMA bandwidth, NVSHMEM or symmetric-memory support, memory ordering, stream/event dependencies, and how many SMs are reserved for communication. DeepEP exposes normal and low-latency dispatch/combine kernels, SM-count controls, NVLink/RDMA buffers, and hook-based overlap where RDMA traffic can progress without occupying compute SMs. Inspect first: [DeepGEMM Mega MoE](https://github.com/deepseek-ai/DeepGEMM), [DeepEP](https://github.com/deepseek-ai/DeepEP), and [SGLang/Miles DeepSeek-V4](https://www.lmsys.org/blog/2026-04-25-deepseek-v4/) for a runtime that combines DeepEP with DeepGEMM Mega MoE.

#### Block-Sparse Expert Reformulation

Block-sparse expert reformulation expresses the irregular token-to-expert computation as sparse matrix products over fixed-size blocks. Rather than padding every expert to a worst-case capacity or dropping overflow tokens, the system builds sparse block metadata that identifies only the active token-expert blocks and runs specialized sparse kernels over those blocks.

In training, this is an alternate answer to the same imbalance problem solved by grouped GEMM. Grouped GEMM keeps expert problems as many dense GEMMs with irregular `M`; MegaBlocks reformulates dropless MoE as block-sparse operations so all routed tokens can be processed without a capacity-factor tradeoff. The approach is especially relevant when the model wants strict dropless routing and when block occupancy is high enough for sparse kernels to approach dense hardware efficiency.

CUDA details include block size, sparse metadata format, transpose metadata for backward, token sorting, and avoiding load imbalance among sparse blocks. Too small a block increases metadata and scheduling overhead; too large a block reintroduces padding. Inspect first: the [MegaBlocks paper](https://arxiv.org/abs/2211.15841), the [MLSys MegaBlocks abstract](https://proceedings.mlsys.org/paper_files/paper/2023/hash/5a54f79333768effe7e8927bcccffe40-Abstract-mlsys2023.html), and the [databricks/megablocks repository](https://github.com/databricks/megablocks) for the dropless MoE implementation and grouped-GEMM option on Hopper-generation GPUs.

### Why It Matters For Training

In sparse MoE, each expert sees a variable number of tokens. Launching separate GEMMs per expert creates overhead and poor occupancy. Grouped GEMM amortizes launch and scheduling. Mega-kernels go further by removing subgraph boundaries and overlapping communication with compute.

### Explore Further

- `official repo`: [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM)
- `official repo`: [DeepEP](https://github.com/deepseek-ai/DeepEP)
- `third-party integration`: [SGLang/Miles DeepSeek-V4 Day-0 support](https://www.lmsys.org/blog/2026-04-25-deepseek-v4/)
- `paper`: [MegaBlocks](https://arxiv.org/abs/2211.15841)

### Implementation Examples

- DeepGEMM grouped GEMM: inspect `m_grouped_fp8_gemm_*` style APIs and scale-layout utilities.
- DeepGEMM Mega MoE: inspect the `fp8_fp4_mega_moe` path described in the README, including symmetric buffer setup and transformed FP4 expert weights.
- SGLang/Miles V4 integration: inspect how the runtime integrates DeepGEMM Mega MoE and DeepEP for DeepSeek-V4 FP4 expert weights.

### Caveats

Mega-kernels are less modular. They may require fixed layouts, symmetric memory, transformed weights, and strict runtime assumptions. They are best used where the fused subgraph is stable and heavily repeated.

## 9. Distributed Communication Overlap

### Technique

Distributed overlap hides all-reduce, reduce-scatter, all-gather, all-to-all, or pipeline transfers under compute. At the CUDA level this can use streams, events, communication hooks, [NCCL](https://docs.nvidia.com/deeplearning/nccl/) (NVIDIA Collective Communications Library) kernels, [NVSHMEM](https://docs.nvidia.com/nvshmem/) (GPU-side OpenSHMEM-style communication library) GPU-initiated operations, or persistent kernels that include communication inside the GPU program.

### Primitive Techniques

#### Stream/Event Staging

Stream/event staging places compute kernels and communication kernels on different CUDA streams, then uses explicit dependencies to make the dataflow legal. A typical pattern records a CUDA event after a GEMM, normalization, or gradient-production kernel finishes writing a buffer; the communication stream waits on that event; and later compute waits on the communication event only at the point where the received or reduced data is actually needed. This is the lowest-level mechanism behind most higher-level overlap features.

In LLM training, stream/event staging is used for gradient all-reduce or reduce-scatter, [FSDP](https://docs.pytorch.org/docs/stable/fsdp.html) (Fully Sharded Data Parallel)/[ZeRO](https://deepspeed.readthedocs.io/en/stable/zero3.html) (Zero Redundancy Optimizer) parameter all-gather, tensor-parallel collectives, pipeline sends and receives, and MoE dispatch/combine. The goal is not merely to launch communication earlier, but to shorten the exposed critical path by letting a ready bucket, activation slice, or token-exchange buffer move while unrelated layer compute continues.

CUDA details that matter include stream priority, allocator lifetime, event placement, and communicator ordering. NCCL operations are issued on CUDA streams and complete according to CUDA stream semantics, but grouped collectives, multiple streams inside one NCCL group, and collectives sharing a process group can introduce ordering constraints that reduce apparent overlap. Framework code also has to make tensors safe across streams, for example by recording stream usage so the caching allocator does not recycle a buffer while a communication kernel is still reading it.

Inspect first: [NCCL CUDA stream semantics](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/streams.html) for the platform contract, [NCCL group calls](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/groups.html) for collective launch ordering, and [DeepEP](https://github.com/deepseek-ai/DeepEP) for `EventOverlap`, `previous_event`, and `allocate_on_comm_stream` patterns in MoE communication.

#### Gradient Bucketing

Gradient bucketing groups many parameter gradients into larger contiguous communication units and launches a collective when a bucket becomes ready during backpropagation. Instead of waiting until the whole backward pass finishes, autograd hooks or runtime graph instrumentation observe that a set of gradients has been produced, copy or view them into a bucket, and enqueue all-reduce or reduce-scatter while earlier layers are still computing their gradients.

In LLM training kernels, bucketing is most visible in data-parallel synchronization and sharded optimizer paths. Dense data parallelism often all-reduces each bucket to replicate full gradients. ZeRO, FSDP, and distributed optimizer variants instead reduce-scatter buckets so each rank retains only the shard it will update. The bucket order is tied to backward readiness: a good order exposes communication under remaining compute, while a poor order creates tail collectives after the last layer's backward kernels finish.

CUDA/platform details include bucket size, gradient dtype, contiguous-gradient layout, autograd-hook overhead, and whether the reduce kernel can run concurrently with the dominant backward GEMMs. Very small buckets become launch- and latency-bound; very large buckets delay the first collective and may leave a long synchronization tail. The communication stream must also respect producer events from the backward stream, and the optimizer must know whether the bucket holds full gradients, reduce-scattered shards, or low-precision communication buffers.

Inspect first: [PyTorch FSDP](https://docs.pytorch.org/docs/stable/fsdp.html) for backward prefetch and reduce-scatter behavior, [DeepSpeed ZeRO](https://deepspeed.readthedocs.io/en/stable/zero3.html) for `reduce_bucket_size`, `contiguous_gradients`, `reduce_scatter`, and `overlap_comm`, and [Megatron Bridge communication overlap](https://docs.nvidia.com/nemo/megatron-bridge/latest/training/communication-overlap.html) for data-parallel overlap guidance.

#### Parameter Prefetch / All-Gather Overlap

Parameter prefetch overlaps the communication needed to materialize sharded parameters with useful computation from the current layer or microbatch. In [ZeRO-3](https://deepspeed.readthedocs.io/en/stable/zero3.html) (the ZeRO stage that shards parameters, gradients, and optimizer states) and FSDP-style training, each rank stores only a shard of the model parameters outside the layer's compute window. Before a layer runs, an all-gather reconstructs the needed parameters, and after the layer is done those full parameters can be released or reshared.

In LLM training, this primitive is essential because large Transformer blocks have repeated parameter materialization points. The runtime can issue the next layer's all-gather before the current layer reaches its synchronization point, turning parameter movement into a staged pipeline. Backward prefetch is especially important because parameter all-gather for the next backward module must be ordered against reduce-scatter of the current module's gradients.

The CUDA details are mostly about issue order and memory pressure. Prefetch usually needs a communication stream, explicit stream dependencies, and at least enough temporary storage for the current parameters plus the next all-gather buffer. PyTorch FSDP exposes this tradeoff through `forward_prefetch`, `backward_prefetch`, and `limit_all_gathers`; DeepSpeed exposes related ZeRO-3 controls such as `stage3_prefetch_bucket_size`, `stage3_max_live_parameters`, and `stage3_max_reuse_distance`. Aggressive prefetch can improve overlap but can also over-allocate HBM or serialize on a shared NCCL ordering path.

Inspect first: [PyTorch FSDP notes](https://docs.pytorch.org/docs/stable/notes/fsdp.html) for forward and backward prefetch nuances, [PyTorch FSDP API docs](https://docs.pytorch.org/docs/stable/fsdp.html) for the rate limiter and `BackwardPrefetch`, and [DeepSpeed ZeRO](https://deepspeed.readthedocs.io/en/stable/zero3.html) for ZeRO-3 parameter prefetch controls.

#### Reduce-Scatter Instead Of All-Reduce

Reduce-scatter performs the reduction and sharding parts of an all-reduce without immediately all-gathering the full result back to every rank. Conceptually, all-reduce can be decomposed into reduce-scatter followed by all-gather. If the next consumer only needs one shard per rank, the all-gather half is unnecessary work and unnecessary memory traffic.

In LLM training, this is the natural collective for sharded data-parallel gradients and distributed optimizers. A rank contributes its local gradients, participates in the reduction, and receives only the gradient shard corresponding to the optimizer state it owns. Tensor-parallel and sequence-parallel paths also use reduce-scatter/all-gather pairs to move partial activations or gradients across ranks while avoiding fully replicated intermediate tensors.

CUDA/platform details include tensor partition alignment, in-place layout constraints, dtype of the reduction, and collective algorithm choice. Reduce-scatter can reduce bytes on the wire and peak memory, but it changes the optimizer and parameter-update contract: the update path must be shard-aware, and any later consumer that needs the full tensor must schedule a matching all-gather. Because NCCL exposes reduce-scatter as a first-class collective, performance tuning follows the same stream, group-call, channel, and topology considerations as all-reduce.

Inspect first: [NCCL collective operations](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html) for reduce-scatter semantics, [PyTorch FSDP](https://docs.pytorch.org/docs/stable/fsdp.html) for full-shard gradient synchronization, [DeepSpeed ZeRO](https://deepspeed.readthedocs.io/en/stable/zero3.html) for `reduce_scatter`, and [Megatron-Core parallelism guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html) for overlap flags such as `--overlap-grad-reduce` and `--overlap-param-gather`.

#### All-to-All Overlap

All-to-all overlap interleaves MoE token exchange with local expert computation. A sparse MoE layer first routes tokens to experts, packs token activations by destination rank, sends them to the ranks that own those experts, runs grouped expert GEMMs, and then combines the outputs back to the original token order. Dispatch and combine are all-to-all-shaped operations, and at scale they can dominate the layer if they are serialized around expert compute.

In LLM training kernels, overlap is created by splitting dispatch/combine into staged chunks, starting communication as soon as routing layout metadata is ready, and scheduling local expert GEMMs against data that has already arrived. The backward pass mirrors this flow: the backward of dispatch behaves like a combine, and the backward of combine behaves like a dispatch. Good MoE overlap therefore treats token layout, communication, and grouped GEMM as one schedule rather than three independent operators.

CUDA/platform details include variable token counts per expert, FP8 or BF16 communication buffers, RDMA versus NVLink paths, stream/event handoff, and SM availability for custom communication kernels. DeepEP exposes normal kernels for training and prefilling, low-latency RDMA kernels, FP8 dispatch, CUDA-event overlap hooks, and SM-count control. On multi-node H800-style systems, the hard part is often forwarding between NVLink and InfiniBand domains while avoiding congestion and keeping enough SMs free for expert GEMMs.

Inspect first: [DeepEP](https://github.com/deepseek-ai/DeepEP) for dispatch/combine APIs, event overlap, FP8 support, RDMA/NVLink forwarding, and `Buffer.set_num_sms`; [DeepSeek-V3](https://arxiv.org/abs/2412.19437) for the MoE training schedule and all-to-all overlap motivation; and [NCCL all-to-all documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html) for the standard collective interface.

#### Pipeline Interleaving

Pipeline interleaving schedules forward and backward microbatches so pipeline communication bubbles are filled with useful work. Instead of running all forward passes, then all backward passes, then waiting at stage boundaries, interleaved schedules split the model into virtual stages or chunks and alternate work so activation sends, gradient sends, and compute on different microbatches proceed concurrently.

In LLM training, this primitive appears in pipeline-parallel Transformer stacks and becomes more important when MoE or tensor/context parallelism adds communication inside each pipeline stage. DeepSeek-V3's DualPipe-style schedule divides a chunk into attention, all-to-all dispatch, MLP, all-to-all combine, and pipeline communication components, then rearranges forward and backward chunks so all-to-all and pipeline communication are hidden under compute as much as possible. Megatron-style virtual pipeline stages pursue the same practical goal: reduce idle bubbles at stage boundaries.

CUDA/platform details include microbatch buffer lifetime, send/receive stream ordering, activation checkpointing interactions, and whether the scheduler can maintain enough independent compute to cover point-to-point latency. Pipeline overlap can be defeated by a single late dependency, a too-small microbatch, an imbalanced stage, or a communication stream that shares scarce SM or network resources with the compute-heavy part of the stage.

Inspect first: [DeepSeek-V3](https://arxiv.org/abs/2412.19437) and the [DualPipe repository](https://github.com/deepseek-ai/DualPipe) for bidirectional pipeline scheduling, [Megatron-Core parallelism guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html) for virtual pipeline configuration, and [MegaScale](https://arxiv.org/abs/2402.15627) for production-scale computation/communication overlap at more than 10,000 GPUs.

#### GPU-Initiated Communication

GPU-initiated communication lets GPU code directly start or participate in data movement instead of relying on the CPU to launch every communication phase. The classic model uses host-side NCCL calls or runtime hooks to enqueue collectives. GPU-initiated models use one-sided put/get/atomic operations, device-side communication APIs, or persistent kernels that include both communication and computation in the same long-running GPU program.

In LLM training kernels, this primitive matters when launch latency, CPU scheduling, or fine-grained message orchestration becomes part of the bottleneck. Persistent MoE dispatch kernels, megakernel schedules, or rank-local work queues can use GPU-side progress to react to data readiness without returning to the host between every small transfer. This is especially attractive for irregular workloads such as MoE routing, pipeline edge sends, or future fused schedules where communication is internal to the GPU program.

Platform details are more restrictive than ordinary stream-based collectives. NVSHMEM requires symmetric memory, processing-element setup, a GPU-visible memory model, and correct ordering/fencing for remote memory operations. NCCL's newer device-side APIs expose device-initiated communication paths for specific connectivity modes, but they come with compatibility and setup constraints. Persistent kernels also reserve registers, shared memory, and CTAs for a long time, so the scheduler must budget occupancy against both compute throughput and communication progress.

Inspect first: [NVSHMEM](https://developer.nvidia.com/nvshmem) and the [NVSHMEM communication model](https://docs.nvidia.com/nvshmem/api/using.html#communication-model) for GPU-initiated put/get/atomic operations, [NCCL device-initiated communication](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/deviceapi.html) for device-side collectives, [DeepEP](https://github.com/deepseek-ai/DeepEP) for an NVSHMEM-dependent MoE communication library, and [Mirage MPK](https://github.com/mirage-project/mirage) for persistent-kernel style scheduling.

#### SM Budgeting For Communication

SM budgeting for communication deliberately controls how many streaming multiprocessors are consumed by communication kernels versus compute kernels. Overlap is only useful if the two classes of work can run at the same time; a collective that occupies too much of the GPU can starve tensor-core kernels, while a compute kernel that consumes every SM can prevent communication progress.

In LLM training, SM budgeting appears in MoE all-to-all kernels, NCCL-heavy tensor/data-parallel collectives, and DualPipe-style schedules that explicitly reserve room for communication while attention or MLP computation runs. DeepEP exposes `Buffer.set_num_sms` for its communication kernels, and DeepSeek-V3 reports manual adjustment of the communication-to-compute SM ratio in its overlap strategy. The tuning target is not maximum standalone bandwidth for communication; it is minimum exposed step time for the overlapped schedule.

CUDA/platform details include CTA count, NCCL channel count, thread count per communication block, stream priority, register pressure, and whether the compute kernel is tensor-core-bound, memory-bandwidth-bound, or latency-bound. NCCL environment variables such as `NCCL_MAX_CTAS`, `NCCL_MIN_CTAS`, `NCCL_NTHREADS`, and channel controls can affect how much GPU resource a collective uses, while custom kernels can expose a direct SM count knob. The right setting is topology- and model-dependent and must be validated with profiler timelines, not just bandwidth microbenchmarks.

Inspect first: [DeepEP](https://github.com/deepseek-ai/DeepEP) for explicit SM-count control in MoE communication, [DeepSeek-V3](https://arxiv.org/abs/2412.19437) for manual SM-ratio tuning in DualPipe, and the [NCCL environment variable reference](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html) for CTA, channel, and thread controls.

#### Topology-Aware Collective Choice

Topology-aware collective choice selects communication algorithms, process groups, and rank mappings based on the actual GPU and network fabric. Rings, trees, hierarchical collectives, NVLink/NVSwitch paths, PCIe paths, InfiniBand rails, and network-offloaded reductions have different cost models. A collective that is ideal inside one node may be poor across nodes, and a rank order that looks arbitrary in code can determine whether traffic stays local or crosses congested links.

In LLM training kernels, topology awareness shapes tensor-parallel groups, data-parallel groups, expert-parallel groups, context-parallel groups, and pipeline stage placement. Dense all-reduce traffic may prefer different grouping than MoE all-to-all traffic. DeepEP's normal kernels distinguish intranode NVLink communication from internode RDMA forwarding, while production systems such as MegaScale treat network tuning and computation/communication overlap as a full-stack problem rather than a library default.

CUDA/platform details include NCCL topology discovery, rank-to-device mapping, [NIC](https://docs.nvidia.com/networking/) (network interface controller) affinity, [GPUDirect RDMA](https://docs.nvidia.com/cuda/gpudirect-rdma/) (direct GPU-to-network-adapter remote memory access), NVLink [SHARP](https://docs.nvidia.com/networking/display/sharpv300) (Scalable Hierarchical Aggregation and Reduction Protocol) or network SHARP availability, cross-NIC behavior, adaptive routing, virtual lanes, and congestion isolation. Topology-aware tuning is also a correctness and reproducibility issue for large jobs: if ranks are remapped by a launcher, scheduler, or fault-recovery path, the collective plan and overlap profile can change even when the model code is identical.

Inspect first: [MegaScale](https://arxiv.org/abs/2402.15627) for large-cluster full-stack network and overlap tuning, [NCCL documentation](https://docs.nvidia.com/deeplearning/nccl/) for topology, [CollNet](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/communicators.html) (NCCL's hierarchical collective-network path), GPUDirect, and environment controls, [DeepEP](https://github.com/deepseek-ai/DeepEP) for NVLink/RDMA-aware MoE kernels and network configuration notes, and [Megatron-Core parallelism guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html) for how DP, TP, PP, CP, and EP groups combine.

### Why It Matters For Training

At thousands of GPUs, interconnect and synchronization become as important as tensor-core FLOPs. Tensor parallelism creates all-reduces and reduce-scatters; FSDP/ZeRO creates parameter all-gathers and gradient reduce-scatters; MoE creates all-to-all dispatch/combine; pipeline/context parallelism creates activation movement. The fastest training systems schedule these transfers with computation.

### Explore Further

- `paper`: [MegaScale: Scaling Large Language Model Training to More Than 10,000 GPUs](https://arxiv.org/abs/2402.15627)
- `paper`: [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
- `official docs`: [NCCL documentation](https://docs.nvidia.com/deeplearning/nccl/)
- `official docs`: [NVSHMEM developer page](https://developer.nvidia.com/nvshmem)
- `official docs`: [DeepSpeed ZeRO](https://deepspeed.readthedocs.io/en/stable/zero3.html)
- `official docs`: [PyTorch FSDP](https://docs.pytorch.org/docs/stable/fsdp.html)
- `official docs/blog/report`: [Megatron-Core](https://developer.nvidia.com/megatron-core), [Dynamic Context Parallelism](https://developer.nvidia.com/blog/speeding-up-variable-length-training-with-dynamic-context-parallelism-and-nvidia-megatron-core/)

### Implementation Examples

- DeepSeek-V3 DualPipe: study pipeline scheduling and reported near full compute-communication overlap.
- DeepEP: inspect hook-based communication-computation overlap for MoE dispatch/combine.
- NCCL: inspect collective primitives and CUDA stream semantics.
- NVSHMEM: inspect GPU-initiated put/get/atomic operations and long-running kernels with communication.
- MegaScale: study full-stack production training overlap, operator optimization, and network tuning.

### Caveats

Overlap is only real if the GPU has independent resources available. A communication kernel that consumes too many SMs can slow compute. A compute kernel that saturates memory bandwidth can leave no bandwidth for communication. Profiling must include both timelines.

## 10. Long-Context Sparse and Hybrid Attention

### Technique

Long-context attention reduces the effective quadratic cost by combining local attention, compressed attention, latent [KV](https://docs.nvidia.com/deeplearning/cudnn/frontend/latest/operations/Attention.html) (key/value) representations, sparse block masks, or indexer-selected keys. The kernel challenge is no longer only a dense [SDPA](https://docs.nvidia.com/deeplearning/cudnn/frontend/latest/operations/Attention.html) (scaled dot-product attention) tile; it also includes metadata construction, sparse index traversal, compressed KV layout, and load balancing across irregular work.

### Primitive Techniques

#### Latent KV Compression

Latent KV compression replaces the usual per-head key and value state with a lower-dimensional latent representation plus projection paths that reconstruct the information needed by attention. In [MLA](https://arxiv.org/abs/2405.04434) (Multi-head Latent Attention)-style designs, the model learns compressed latent K/V state so the attention layer moves and stores fewer bytes than standard [MHA](https://arxiv.org/abs/1706.03762) (multi-head attention)/GQA for the same context length. The primitive is architectural, but it becomes a kernel problem as soon as the compressed state is laid out as a different tensor contract from ordinary `[batch, head, seq, dim]` K/V.

The tensor contract changes from full per-head K/V storage to a smaller latent state plus reconstruction projections:

$$
\begin{array}{c}
\text{standard:}\quad
K,V\in\mathbb{R}^{B\times H\times S\times d_h}
\\[2mm]
\text{latent:}\quad
C\in\mathbb{R}^{B\times S\times d_c},\qquad d_c \ll H d_h
\\[2mm]
C
\xrightarrow{\;W_K,W_V\;}
\widehat{K},\widehat{V}
\xrightarrow{\;\operatorname{Attention}(Q,\widehat{K},\widehat{V})\;}
O
\end{array}
$$

The kernel-design goal is to avoid explicitly expanding $\widehat{K},\widehat{V}$ into full dense K/V tensors unless the schedule can immediately consume the reconstructed tile.

Diagram symbols: $B$ is batch size, $H$ is the number of attention heads, $S$ is sequence length, $d_h$ is per-head dimension, $d_c$ is the compressed latent dimension, $C$ is the latent K/V state, $W_K$ and $W_V$ are reconstruction projections, $\widehat{K}$ and $\widehat{V}$ are reconstructed key/value tiles or logical tensors, $Q$ is the query tensor, and $O$ is the attention output.

In training kernels, latent KV compression changes both projection fusion and attention scheduling. The forward pass can fuse Q/K/V projection, rotary-position handling, latent-cache construction, and the attention call boundary so the compressed state is written once and then consumed by dense, sparse, or hybrid attention. Backward must propagate gradients through the compression projection and any reconstructed key/value paths without accidentally materializing a full uncompressed KV activation for long contexts. Activation checkpointing often recomputes the projection side of this primitive, while the attention kernel consumes only the compact representation and saved softmax statistics.

CUDA details that matter are head-dimension alignment, tensor-core tile shapes, and whether the latent dimension is treated as the MMA $K$ dimension or as a pre/post-processing projection around attention. MLA implementations tend to be bandwidth-sensitive because the whole point is reducing KV traffic; inefficient dequantization, transpose, or head-replication layouts can erase the compression win. Inspect first: [DeepSeek-V2](https://arxiv.org/abs/2405.04434) and [DeepSeek-V3](https://arxiv.org/abs/2412.19437) for the architecture-level MLA motivation, the [FlashMLA repository](https://github.com/deepseek-ai/FlashMLA) for optimized MLA attention kernels, and [TileLang's DeepSeek MLA example](https://github.com/tile-ai/tilelang/blob/main/examples/deepseek_mla/README.md) for a compact kernel-authoring view of the same computation.

#### Local / Sliding-Window Attention

Local or sliding-window attention restricts each query token to a bounded neighborhood of key/value tokens, usually with a left window for causal language modeling and sometimes a small right window for bidirectional or chunked encoders. The sparsity pattern is simple: for each query row, valid columns are a contiguous band rather than the full prefix or full sequence. That makes it one of the easiest sparse-attention forms to map to fast kernels.

At block granularity, a local causal window has a banded pattern:

$$
\begin{array}{c|ccccc}
 & K_0 & K_1 & K_2 & K_3 & K_4\\ \hline
Q_0 & 1 & 0 & 0 & 0 & 0\\
Q_1 & 1 & 1 & 0 & 0 & 0\\
Q_2 & 0 & 1 & 1 & 0 & 0\\
Q_3 & 0 & 0 & 1 & 1 & 0\\
Q_4 & 0 & 0 & 0 & 1 & 1
\end{array}
$$

Here `1` means the score tile is computed and `0` means the scheduler can skip that K/V block before loading it.

Diagram symbols: $Q_0,\ldots,Q_4$ are query blocks, $K_0,\ldots,K_4$ are key/value blocks, and each table entry marks whether that block-pair participates in attention under the local causal window.

For query position $i$, a windowed mask can be written as:

$$
\mathcal{K}(i)=\{j:\ i-w_\ell \le j \le i+w_r\},
$$

with causal decoding usually setting $w_r=0$. The attention formula is unchanged except that the softmax is restricted to $j\in \mathcal{K}(i)$.

Symbols: $i$ is the query position, $j$ is a key/value position, $\mathcal{K}(i)$ is the set of positions visible to query $i$, $w_\ell$ is the left-window size, and $w_r$ is the right-window size.

In LLM training kernels, the local window is usually implemented as tile-local masking plus schedule pruning. Query blocks whose key blocks fall outside the window can be skipped entirely, and partially overlapping blocks apply row/column masks before the online softmax update. Forward stores the same compact softmax statistics used by dense FlashAttention-style kernels, while backward revisits the same window bounds to recompute local probabilities and accumulate gradients only for the touched K/V tiles.

CUDA/platform details are practical: window size determines arithmetic intensity, edge tiles need exact boundary checks, and causal alignment must match the framework's definition of sequence offsets for packed batches. cuDNN's NSA interface exposes sliding-window attention as a component with bounds, sequence-length tensors, stats support for training, and FP32 intermediate/compute controls; FlashAttention-style APIs expose windowed attention as a parameterized variant of the dense IO-aware kernel. Inspect first: [cuDNN Frontend NSA](https://docs.nvidia.com/deeplearning/cudnn/frontend/latest/fe-oss-apis/nsa.html) for the platform API, the [FlashAttention repository](https://github.com/Dao-AILab/flash-attention) for public windowed-attention interfaces, and [TileLang](https://github.com/tile-ai/tilelang) for examples that show how tiled attention schedules are written.

#### Block-Sparse Mask Metadata

Block-sparse mask metadata represents allowed attention regions as block indices, block counts, page tables, or compact masks instead of a dense `[S_q, S_k]` Boolean matrix. The important distinction is that the metadata is part of the kernel input contract, not just a high-level mask object. It tells the kernel which score tiles exist, how many blocks each row or row block should visit, and which blocks can be skipped before loading K/V.

The dense block mask above is normally passed to the kernel in compressed form:

$$
\begin{array}{c|c|c}
\text{query block} & \text{block\_counts} & \text{block\_indices}\\ \hline
Q_0 & 1 & [0]\\
Q_1 & 2 & [0,1]\\
Q_2 & 2 & [1,2]\\
Q_3 & 2 & [2,3]\\
Q_4 & 2 & [3,4]
\end{array}
$$

This is why sparse attention kernels are often metadata-bandwidth and load-balance problems, not merely fewer FLOPs.

Diagram symbols: `query block` names the row block of the sparse attention matrix; `block_counts` is the number of key/value blocks visited by that query block; `block_indices` lists those key/value block IDs; and the table is a compressed representation of the banded block mask above.

In training kernels, this metadata drives both the forward sparse traversal and the backward recomputation path. Forward walks the selected key blocks, computes score tiles, updates online softmax state, and writes output plus statistics. Backward must use equivalent metadata so gradients are computed for the same logical attention graph; otherwise it can silently train a different operator from the forward pass. For dynamic sparsity, metadata construction can become a separate kernel stage that must be captured, checkpointed, or replayed.

CUDA details include metadata dtype, coalescing, bounds checks, and load balance. Block indices are commonly `int32`; block sizes such as 16, 32, 64, or 128 are chosen to align with tensor-core-friendly score tiles and amortize metadata loads. Highly irregular block counts cause warp divergence and uneven CTA runtimes, so kernels often bucket rows, pad block lists, or split large rows across CTAs. Inspect first: [cuDNN Frontend NSA](https://docs.nvidia.com/deeplearning/cudnn/frontend/latest/fe-oss-apis/nsa.html) for `block_indices` and `block_counts` in selection attention, the [Native Sparse Attention paper](https://arxiv.org/abs/2502.11089) for the hardware-aligned sparse pattern, and the [Native Sparse Attention repository](https://github.com/fla-org/native-sparse-attention) for a Triton implementation to compare against vendor APIs.

#### [CSA](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro) (Compressed Sparse Attention)

Compressed Sparse Attention combines a compressed global view of the context with a sparse exact-attention path over selected tokens or blocks. The compressed path gives every query some global information at reduced sequence length, while the sparse path recovers high-resolution detail from the most relevant parts of the original context. It is a hybrid operator rather than a single mask: compression, selection, exact attention, and output combination all have to agree on layout and normalization.

A useful abstract form is:

$$
O = \phi_{\mathrm{combine}}\left(
O_{\mathrm{local}},
O_{\mathrm{compressed}},
O_{\mathrm{selected}}
\right),
$$

where $O_{\mathrm{compressed}}$ attends to compressed K/V summaries and $O_{\mathrm{selected}}$ attends exactly to indexed original blocks. Public V4 material does not fully disclose the production combine function, so this equation should be read as a kernel-contract sketch rather than an implementation claim.

Symbols: $O_{\mathrm{local}}$ is the output of local or sliding-window attention; $O_{\mathrm{compressed}}$ is the output of attention over compressed context summaries; $O_{\mathrm{selected}}$ is the output of exact attention over selected original blocks; $\phi_{\mathrm{combine}}$ is the model's output-combination function; and $O$ is the final hybrid-attention output.

In training kernels, this primitive becomes a pipeline of GPU work. A compression attention stage builds or attends over compressed K/V; a top-k or indexer stage chooses original-context blocks; a selection-attention stage computes exact attention over those blocks; and a combine stage merges compressed, selected, and often local-window outputs. The backward pass may need gradients through all three pieces: compressed K/V construction, selection scores or indexer losses when trainable, and exact selected-block attention.

CUDA/platform details are dominated by staging and ABI boundaries. The compression stage can look like dense attention over a shorter sequence and may support FP8 inputs/outputs on Blackwell-class paths; the selection stage is gather-heavy and depends on block-index locality; the top-k stage needs deterministic-enough index and score tensors if training wants replayable behavior. Inspect first: [cuDNN Frontend NSA](https://docs.nvidia.com/deeplearning/cudnn/frontend/latest/fe-oss-apis/nsa.html), which exposes Compression Attention, Top-K Reduction, Selection Attention, and Sliding Window components, [DeepSeek-V4-Pro's model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro) for CSA as a model-level design, and [NVIDIA's DeepSeek V4 platform note](https://developer.nvidia.com/blog/build-with-deepseek-v4-using-nvidia-blackwell-and-gpu-accelerated-endpoints/) for the reported CSA/[DSA](https://developer.nvidia.com/blog/build-with-deepseek-v4-using-nvidia-blackwell-and-gpu-accelerated-endpoints/) (DeepSeek Sparse Attention deployment framing in NVIDIA's V4 note)/HCA deployment framing.

#### [HCA](https://developer.nvidia.com/blog/build-with-deepseek-v4-using-nvidia-blackwell-and-gpu-accelerated-endpoints/) (Heavily Compressed Attention)

Heavily Compressed Attention is the official DeepSeek/NVIDIA expansion of HCA in the public V4 material. It uses more aggressive KV compression than the CSA path by consolidating sets of tokens into compressed entries, so a query can reach very long-range context without scanning or storing every original K/V position at full resolution. The useful mental model is multi-resolution attention: nearby or selected tokens preserve high resolution, while far context is represented by much smaller compressed summaries.

In training kernels, HCA-style designs require scheduling multiple attention subproblems inside one layer. The kernel stack may run local sliding-window attention, compressed attention over chunk summaries, and a heavier-compression path, then combine their outputs under the model's learned mixing rule. Backward must replay the same hierarchy, preserve the association between original tokens and compressed entries, and avoid expanding the compressed hierarchy into dense full-context activations.

CUDA details include mixed-granularity indexing, output-combine fusion, and memory layout for compressed entries. If compressed blocks are produced by reductions or learned projections, their output locations must be stable across ranks and microbatches. If local and compressed paths are launched separately, kernel-launch overhead and HBM round trips can dominate; if fused, register pressure and divergent metadata paths become the constraint. Inspect first: [DeepSeek-V4-Pro](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro) for the model-level HCA/CSA description, [NVIDIA's DeepSeek V4 platform note](https://developer.nvidia.com/blog/build-with-deepseek-v4-using-nvidia-blackwell-and-gpu-accelerated-endpoints/) for the CSA/DSA/HCA decomposition, the [SGLang/Miles DeepSeek-V4 report](https://www.lmsys.org/blog/2026-04-25-deepseek-v4/) for practical notes on hybrid attention metadata and training support, and [cuDNN Frontend NSA](https://docs.nvidia.com/deeplearning/cudnn/frontend/latest/fe-oss-apis/nsa.html) for the vendor decomposition into compression, selection, top-k, and sliding-window kernels.

#### Indexer-Selected Retrieval

Indexer-selected retrieval uses a learned indexer, auxiliary scoring path, or top-k reduction to choose which K/V blocks a query should attend exactly. Instead of every query scanning every key, the indexer emits a compact list of important blocks. The attention kernel then treats those lists as sparse metadata and computes exact softmax attention over the selected regions.

In training kernels, the indexer is both a model component and a scheduling component. Forward has to compute or consume top-k indices, feed them to selection attention, and save enough state for backward. If the indexer is stochastic or trained with rollout data, training may need index replay so the backward or policy-gradient path uses the same selected blocks observed during generation. This is especially important for [RL](https://en.wikipedia.org/wiki/Reinforcement_learning) (reinforcement learning) or distillation workflows where serving and training kernels must agree on sparse routes.

CUDA/platform details include top-k implementation, metadata construction inside CUDA graphs, and gather locality. Top-k over compressed scores can be memory-traffic-heavy, and selected block lists can destroy coalescing if the indexer scatters attention across the context. Practical kernels use fixed block sizes, `int32` indices, compact count arrays, and sometimes page-table indirection to keep selected K/V loads predictable. Inspect first: [cuDNN Frontend NSA Top-K Reduction and Selection Attention](https://docs.nvidia.com/deeplearning/cudnn/frontend/latest/fe-oss-apis/nsa.html), the [SGLang/Miles DeepSeek-V4 report](https://www.lmsys.org/blog/2026-04-25-deepseek-v4/) for indexer replay and metadata preparation, and the [FlashMLA repository](https://github.com/deepseek-ai/FlashMLA) for sparse and MLA attention kernel code paths.

#### Ragged [THD](https://developer.nvidia.com/blog/speeding-up-variable-length-training-with-dynamic-context-parallelism-and-nvidia-megatron-core/) (Token-Head-Dimension) Layout

Ragged `T,H,D` layout packs variable-length sequences into a token-major tensor where `T` is the sum of actual sequence lengths, with cumulative sequence offsets marking each example boundary. It replaces padded [BSHD](https://developer.nvidia.com/blog/speeding-up-variable-length-training-with-dynamic-context-parallelism-and-nvidia-megatron-core/) (batch-sequence-head-dimension) storage when a batch contains very different context lengths. The layout does not by itself make attention sparse, but it prevents padding tokens from consuming attention compute and memory bandwidth.

In training kernels, ragged layout is the natural companion to long-context batches. Forward attention receives packed Q/K/V plus `cum_seqlen` arrays, computes only real token rows, and uses the offsets to prevent one sequence from attending into another. Backward uses the same offsets to reconstruct row ranges and write gradients into packed tensors. Dynamic batching, activation checkpointing, and context parallelism all become easier when the kernel's sequence dimension is already expressed as real token intervals.

CUDA details include offset dtype, maximum-sequence parameters, and row-to-sequence mapping. Kernels need fast conversion from a global token index to the local sequence span, or they need scheduling that launches CTAs by segment to avoid per-row binary searches. cuDNN NSA explicitly supports `T,H,D` variable-length tensors with cumulative sequence-length inputs; Megatron-Core's dynamic context-parallel work targets the same long-tail sequence-length problem at the distributed scheduler level. Inspect first: [cuDNN Frontend NSA tensor formats](https://docs.nvidia.com/deeplearning/cudnn/frontend/latest/fe-oss-apis/nsa.html), NVIDIA's [Dynamic Context Parallelism blog](https://developer.nvidia.com/blog/speeding-up-variable-length-training-with-dynamic-context-parallelism-and-nvidia-megatron-core/), and [Megatron-Core context parallelism docs](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/context_parallel.html).

#### Context-Parallel Attention Shards

Context-parallel attention shards the sequence dimension across ranks so each GPU owns only a slice of the tokens and activations. Non-attention layers can usually operate on their local sequence shard, but attention needs access to K/V from other shards or a distributed sparse/compressed equivalent. The primitive trades local memory pressure for communication and coordination.

In training kernels, [CP](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/context_parallel.html) (context parallelism) is what makes very long contexts fit without relying only on full activation recomputation. Forward attention may all-gather K/V, ring through remote K/V blocks, or use specialized distributed attention schedules; backward performs the complementary reduce-scatter or all-gather patterns for gradients. Sparse and hybrid attention add another layer: selected blocks, compressed entries, and page tables must be reindexed per rank so each kernel sees local metadata while the logical context remains global.

CUDA/platform details involve overlap, communication granularity, and rank-local metadata. Attention kernels must be shaped so NCCL or NVSHMEM transfers can overlap with local QK/PV work, and the per-rank sequence shard should still contain enough rows to keep tensor cores busy. Variable-length batches complicate CP because long samples dominate memory and compute; Dynamic-CP addresses this by choosing CP size per microbatch. Inspect first: [Megatron-Core context parallelism](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/context_parallel.html), the [Megatron-Core parallelism guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html), NVIDIA's [Dynamic Context Parallelism blog](https://developer.nvidia.com/blog/speeding-up-variable-length-training-with-dynamic-context-parallelism-and-nvidia-megatron-core/), and the [SGLang/Miles DeepSeek-V4 report](https://www.lmsys.org/blog/2026-04-25-deepseek-v4/) for CP reindexing notes in a hybrid-attention stack.

#### Sparse Backward Metadata Reuse

Sparse backward metadata reuse means the backward pass consumes the same sparse structure produced or used by forward: block masks, selected block indices, block counts, compressed-token mappings, top-k results, sequence offsets, and sometimes RNG or indexer replay state. The purpose is correctness as much as speed. A sparse forward operator defines a smaller attention graph; backward must differentiate that graph, not a freshly sampled or approximately reconstructed one.

In training kernels, this primitive prevents expensive metadata rebuilds and avoids forward/backward mismatch. Forward can save compact index tensors and softmax statistics instead of dense probabilities. Backward then recomputes score tiles only for the saved sparse blocks, applies the same mask and compression mapping, and accumulates gradients into Q/K/V, compressor projections, and indexer-related tensors where applicable. For RL or online serving-to-training pipelines, metadata reuse may cross process boundaries as route replay rather than a simple saved tensor.

CUDA details include lifetime management, determinism, and memory placement. Metadata should be compact enough to save, aligned enough for coalesced reads, and stable enough to survive graph capture or activation checkpoint recomputation. If block lists are regenerated on the CPU or scheduler stream, they can become a launch bottleneck; device-side metadata preparation and replay buffers can keep the sparse backward path inside CUDA graphs. Inspect first: [cuDNN Frontend NSA](https://docs.nvidia.com/deeplearning/cudnn/frontend/latest/fe-oss-apis/nsa.html) for the explicit selection/top-k metadata tensors, the [Native Sparse Attention repository](https://github.com/fla-org/native-sparse-attention) for a training-oriented sparse attention implementation, and the [SGLang/Miles DeepSeek-V4 report](https://www.lmsys.org/blog/2026-04-25-deepseek-v4/) for rollout routing and indexer replay in a hybrid sparse-attention training stack.

### Why It Matters For Training

Million-token contexts are infeasible with dense full attention at every layer. Training and post-training need sparse or compressed attention variants plus context parallelism and activation checkpointing. These features make the attention kernel a compound operator: projection, compression, indexing, local/dense/sparse attention, metadata, and backward.

### Explore Further

- `paper`: [DeepSeek-V2](https://arxiv.org/abs/2405.04434) for MLA and DeepSeekMoE background.
- `paper`: [DeepSeek-V3](https://arxiv.org/abs/2412.19437) for MLA and large-scale MoE training.
- `official docs/blog/report`: [DeepSeek-V4 model card/report](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro) for CSA/HCA and 1M context.
- `third-party integration`: [SGLang/Miles V4 support](https://www.lmsys.org/blog/2026-04-25-deepseek-v4/)
- `official repo/docs`: [TileLang](https://github.com/tile-ai/tilelang), [cuDNN Frontend Native Sparse Attention APIs](https://docs.nvidia.com/deeplearning/cudnn/frontend/)

### Implementation Examples

- DeepSeek-V2/V3 MLA: study latent KV compression as an architecture-level reduction in attention state.
- DeepSeek-V4 CSA/HCA: official model-level evidence; kernel implementation details are partly inferred unless using third-party runtime integrations.
- SGLang/Miles: inspect reported TileLang attention and mHC kernels, FlashMLA integration, DeepEP, and DeepGEMM Mega MoE.
- TileLang: inspect MLA, FlashAttention, and Native Sparse Attention examples.
- cuDNN Frontend: inspect Native Sparse Attention and block-mask APIs.

### Caveats

Sparse attention performance depends on sparsity structure. Random sparse patterns, local windows, block masks, compressed dense paths, and top-k indexers require different memory layouts and schedules.

## 11. [mHC](https://arxiv.org/abs/2512.24880) (Manifold-Constrained Hyper-Connections) and Residual-Path Mixing Kernels

### Technique

Manifold-Constrained Hyper-Connections extend residual-stream connectivity while constraining mixing matrices to preserve stability. Public mHC sources frame the method as a way to regain identity-like stability properties while using richer multi-stream connections.

### Primitive Techniques

#### Multi-Stream Residual State

Multi-stream residual state replaces the single Transformer residual vector with a small residual-stream axis. In the mHC formulation, each token carries an `hc_mult`-wide set of residual branches, while learned pre-, post-, and residual mappings decide how a sublayer reads from those branches, writes back into them, and mixes information across depth. The branch axis is small compared with the hidden dimension, but it changes the activation contract from a conventional `[sequence, batch, hidden]` residual tensor into a layout that may need to carry `[sequence, batch, hc_mult, hidden]` across recomputation blocks and pipeline-parallel stage boundaries.

A useful tensor view is:

$$
\begin{array}{c}
\text{ordinary residual:}\quad
H\in\mathbb{R}^{T\times D}
\\[2mm]
\text{mHC residual streams:}\quad
\mathcal{H}\in\mathbb{R}^{T\times R\times D}
=
\begin{bmatrix}
h_{1}^{(0)} & h_{1}^{(1)} & \cdots & h_{1}^{(R-1)}\\
h_{2}^{(0)} & h_{2}^{(1)} & \cdots & h_{2}^{(R-1)}\\
\vdots & \vdots & \ddots & \vdots\\
h_{T}^{(0)} & h_{T}^{(1)} & \cdots & h_{T}^{(R-1)}
\end{bmatrix}
\end{array}
$$

The hidden dimension $D$ remains the vectorized memory dimension; the stream dimension $R$ is small and should usually stay close to registers or shared memory during mixing.

Diagram symbols: $T$ is the number of packed tokens, $D$ is hidden width, $R$ is the number of residual streams or `hc_mult`, $H$ is an ordinary single-stream residual tensor, $\mathcal{H}$ is the multi-stream residual tensor, and $h_t^{(r)}$ is token $t$'s hidden vector in residual stream $r$.

In LLM training kernels, this primitive adds branch-local reads and writes around every attention or MLP sublayer. The kernel author should preserve a contiguous hidden dimension for coalesced vector loads, keep the small branch dimension in registers or shared memory when possible, and avoid materializing separate branch tensors between the pre-map, layer function, post-map, and residual merge. The first evidence to inspect is the [mHC paper](https://arxiv.org/abs/2512.24880), especially its method and infrastructure sections, followed by the [DeepSeek-V4 report](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro) for model-level use and the [SGLang/Miles V4 integration](https://www.lmsys.org/blog/2026-04-25-deepseek-v4/) for the practical pipeline boundary note that mHC streams must survive pipeline-parallel transfers.

#### Constrained Mixing Matrix

The constrained mixing matrix is the stability mechanism in mHC. Instead of allowing the residual-stream mixing matrix to be arbitrary, mHC projects the residual mapping onto a doubly stochastic manifold: entries are non-negative, each row sums to one, and each column sums to one. This makes the residual update behave like a convex mixture of streams rather than an unconstrained amplifier, preserving the identity-like signal path that ordinary residual networks rely on.

The constraint set is the Birkhoff polytope:

$$
\mathcal{B}_n =
\left\{
B\in\mathbb{R}^{n\times n}:
B_{ij}\ge 0,\quad
B\mathbf{1}=\mathbf{1},\quad
B^\top\mathbf{1}=\mathbf{1}
\right\}.
$$

If $H$ stacks the residual streams, a constrained residual mix has the abstract form $H' = BH$ or its batched/token-wise analogue.

Symbols: $\mathcal{B}_n$ is the Birkhoff polytope of $n\times n$ doubly stochastic matrices; $B$ is a residual-stream mixing matrix; $B_{ij}$ is its row $i$, column $j$ entry; $\mathbf{1}$ is an all-ones vector; $B\mathbf{1}=\mathbf{1}$ enforces row sums of one; $B^\top\mathbf{1}=\mathbf{1}$ enforces column sums of one; $H$ stacks residual streams; and $H'$ is the mixed residual state.

In training kernels, the constrained matrix is a small per-token or per-position coefficient object that sits on the residual path rather than in the large attention or MLP GEMMs. Its arithmetic is small, but it is invoked at every layer and participates in backward propagation, so launch overhead, coefficient storage, and numerical precision matter. CUDA implementations should treat the matrix as a batched small-matrix workload, accumulate row and column sums in higher precision when practical, and make the tensor layout friendly to both forward mixing and coefficient-gradient reductions. Inspect the [mHC paper](https://arxiv.org/abs/2512.24880) first for the Birkhoff-polytope constraint, the non-negativity constraints on the pre/post mappings, and the stability analysis comparing unconstrained HC with mHC.

#### Sinkhorn-Style Normalization

Sinkhorn-style normalization is the projection step that turns positive residual-mixing logits into an approximately doubly stochastic matrix. mHC makes the raw matrix positive, then alternates row and column normalization for a fixed number of iterations; the paper reports 20 iterations as its practical training setting. The important point for kernel authors is that Sinkhorn is not a generic large GEMM. It is a repeated small reduction and rescaling procedure over the branch matrix, executed many times across tokens and layers.

Starting from positive matrix $A^{(0)}$, one Sinkhorn iteration is:

$$
\tilde{A}^{(t)}_{ij}
=
\frac{A^{(t)}_{ij}}{\sum_j A^{(t)}_{ij}},
\qquad
A^{(t+1)}_{ij}
=
\frac{\tilde{A}^{(t)}_{ij}}{\sum_i \tilde{A}^{(t)}_{ij}}.
$$

The CUDA opportunity is that both normalizations are small reductions over the branch axis and can often stay on chip.

Symbols: $A^{(t)}$ is the positive mixing matrix estimate at Sinkhorn iteration $t$; $\tilde{A}^{(t)}$ is the row-normalized intermediate; $i$ and $j$ index rows and columns; $\sum_j A^{(t)}_{ij}$ is a row sum; $\sum_i \tilde{A}^{(t)}_{ij}$ is a column sum; and $A^{(t+1)}$ is the matrix after one row-normalization and column-normalization pass.

In LLM training kernels, Sinkhorn produces the weights consumed by residual-stream mixing and must either expose gradients through the normalization loop or follow an explicitly frozen/proxy rule in a given training stack. CUDA details that matter include fixed iteration counts for graph capture, FP32 accumulation for row and column sums when low-precision inputs are used, register/shared-memory residency for the small matrix, and avoiding one launch per row/column pass. The first links to inspect are the [mHC paper](https://arxiv.org/abs/2512.24880) for the projection definition and custom backward discussion, and the [SGLang/Miles V4 integration](https://www.lmsys.org/blog/2026-04-25-deepseek-v4/) for reported fused TileLang Sinkhorn support in DeepSeek-V4 training and serving paths.

#### Fused Norm-And-Mix

Fused norm-and-mix combines RMSNorm-like normalization, residual coefficient generation, Sinkhorn projection, and branch mixing into fewer kernel boundaries. The motivation is bandwidth: a naive implementation writes normalized residual streams, reads them back to form mHC coefficients, writes projected coefficients, then reads everything again for residual merging. mHC's own infrastructure section describes reordering the normalization arithmetic and fusing shared-memory-access operations to reduce those trips through HBM.

In training kernels, this primitive is used around pre-norm Transformer sublayers so that the normalized hidden state, coefficient calculation, and residual update can share loaded data. CUDA/platform details include vectorized loads over the hidden dimension, mixed-precision cast points, reductions for RMSNorm, small-matrix operations for branch mixing, and backward kernels that avoid reloading the same activation several times. Inspect the [mHC paper](https://arxiv.org/abs/2512.24880) for the kernel-fusion design, the [SGLang/Miles V4 integration](https://www.lmsys.org/blog/2026-04-25-deepseek-v4/) for its reported fused `mhc_pre_big_fuse_tilelang` path combining RMSNorm, Sinkhorn, and residual mixing, and [`Liger-Kernel`](https://github.com/linkedin/Liger-Kernel) for the public `LigerMHC` API entry and adjacent Triton training-kernel style.

#### Split-K Small-Batch GEMM

Split-K small-batch GEMM partitions the reduction dimension of a GEMM across multiple CTAs so more thread blocks can work on a problem whose `M` dimension is too small to fill the GPU. In mHC paths, the relevant GEMMs are the coefficient-generation projections around the residual stream. They can become under-occupied in small-batch decoding, RL, or post-training regimes even though the same model uses large hidden dimensions.

In training-oriented kernels, split-K is a utilization tradeoff: it increases parallelism, but it also introduces partial-output reductions, extra synchronization or reduction buffers, and more tuning knobs. CUDA implementations need to choose split counts based on hidden size, batch/token count, tensor-core tile shape, and the cost of reducing partial sums. The first implementation evidence to inspect is the [SGLang/Miles V4 integration](https://www.lmsys.org/blog/2026-04-25-deepseek-v4/), which reports extending a two-stage split-K TileLang mHC pre-GEMM kernel, `mhc_pre_gemm_sqrsum_splitk_kernel`, to improve small-batch utilization.

#### Residual Gating / Weighted Sum

Residual gating and weighted sum are the actual application of the mHC coefficients to branch activations. The pre-map reads the multi-stream residual state into the layer input, the post-map writes the sublayer output back to the stream, and the residual map carries the stream forward through a constrained mixture. The gates are learned or input-dependent coefficients, with mHC constraining the residual map and applying non-negativity to the input/output maps to avoid cancellation-heavy mixing.

Kernel-wise, this primitive is usually memory-bound unless it is fused with normalization, coefficient generation, or residual add. The branch dimension is small enough to unroll, but the hidden dimension is large enough that coalesced loads, vectorized stores, and avoiding extra reads dominate. Backward propagation also needs reductions from hidden elements back into the small coefficient tensors, so the implementation should plan for deterministic or at least stable accumulation behavior under mixed precision. Inspect the [mHC paper](https://arxiv.org/abs/2512.24880) for the pre-, post-, and residual mapping equations; the [DeepSeek-V4 report](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro) for architecture-level adoption; and [`Liger-Kernel`](https://github.com/linkedin/Liger-Kernel) for a public training-kernel surface that names mHC as a model kernel.

#### Constraint-Gradient Handling

Constraint-gradient handling covers the backward pass through constrained mappings: gradients must be propagated through non-negativity transforms, Sinkhorn projection, and residual mixing without silently violating the forward parameterization. The mHC paper describes custom backward kernels that recompute Sinkhorn intermediates on chip and traverse the normalization iterations, which avoids storing every intermediate from every layer and token.

For LLM training kernels, this primitive is where mathematical correctness and memory pressure meet. Storing all Sinkhorn states is expensive; recomputing them saves activation memory but adds compute in the backward pass. CUDA details that matter include fixed control flow for graph capture, FP32 accumulation for sensitive reductions, careful cast placement for BF16/FP8 stacks, and integration with activation checkpointing or stage-local recomputation boundaries. Inspect the [mHC paper](https://arxiv.org/abs/2512.24880) for the custom backward and recomputing design, then inspect the [SGLang/Miles V4 integration](https://www.lmsys.org/blog/2026-04-25-deepseek-v4/) for recent training-stack notes on fused Sinkhorn kernels, numerical precision, and selectively frozen unstable paths in RL-oriented workflows.

### Why It Matters For Training

If residual mixing adds GEMMs, normalizations, Sinkhorn-style constraints, or branch mixing around each layer, it creates new small-to-mid-sized kernels in the training step. These kernels can become launch-bound or memory-bound if implemented naively, especially at small batch or during post-training.

### Explore Further

- `paper`: [mHC: Manifold-Constrained Hyper-Connections](https://arxiv.org/abs/2512.24880)
- `official docs/blog/report`: [DeepSeek-V4 model card/report](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro)
- `third-party integration`: [SGLang/Miles DeepSeek-V4 support](https://www.lmsys.org/blog/2026-04-25-deepseek-v4/)

### Implementation Examples

- DeepSeek-V4: official model-level evidence that mHC is part of the architecture.
- SGLang/Miles: reports TileLang mHC kernels, split-K for small-batch utilization, and fused RMSNorm/Sinkhorn/residual mixing paths.
- Liger Kernel: includes an mHC API entry in its public kernel list, useful to inspect as mHC becomes standardized.

### Caveats

DeepSeek-V4's official model card does not disclose the full pretraining mHC kernel implementation. Treat detailed V4 mHC kernel claims as `third-party integration` or `inferred` unless backed by released code.

## 12. Optimizer and Optimizer-State Kernels

### Technique

Optimizer kernels update parameters and optimizer states with many elementwise operations. Traditional AdamW optimization is memory-bandwidth heavy and launch-heavy across many tensors. Newer matrix optimizers such as Muon add matrix-level operations, including Newton-Schulz orthogonalization, that require distributed matrix kernels rather than only elementwise updates. FP8 optimizer-state compression adds quantization kernels and dynamic range management.

### Primitive Techniques

#### Multi-Tensor Apply

Multi-tensor apply is the optimizer equivalent of batching many tiny pointwise kernels into one larger launch. Instead of launching one Adam update per parameter tensor, the framework builds lists of gradient, parameter, first-moment, and second-moment tensors, partitions those lists by dtype and hyperparameter group, and sends the lists to a CUDA extension that iterates over many tensor pointers inside one or a few kernels. The mathematical operation is still per element; the primitive changes the launch and memory-access envelope.

In LLM training, this matters because transformer models contain many parameter tensors whose individual optimizer updates are too small to saturate a GPU. A multi-tensor path lets the optimizer amortize Python dispatch, autograd bookkeeping, and CUDA launch overhead across all parameters in a group. It also makes fused AdamW practical: once the kernel is walking a flat stream of tensor chunks, it can update `m`, `v`, and `p` for many tensors without returning to the host between tensors.

CUDA details that matter are pointer-list construction, dtype bucketing, chunk size, alignment, sparse-gradient rejection, and whether the launch is graph-capture friendly. The kernel must handle tensors with different shapes while still producing mostly coalesced vectorized memory accesses; large tensors are split into chunks while small tensors are packed together. Apex is the cleanest first implementation to inspect because its docs explicitly describe the two fusions in `FusedAdam`: elementwise Adam fusion plus a multi-tensor apply launch. Inspect first: [Apex FusedAdam docs](https://nvidia.github.io/apex/optimizers.html), [Apex `fused_adam` source](https://nvidia.github.io/apex/_modules/apex/optimizers/fused_adam.html), and Megatron Core's [`core.optimizer.optimizer`](https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.optimizer.optimizer.html) for multi-tensor copying around mixed-precision optimizer state.

#### Fused AdamW Elementwise Update

The fused AdamW update combines the pointwise parts of an optimizer step into one memory pass: optional gradient scaling/unscaling policy, first moment update, second moment update, bias correction, decoupled weight decay, and parameter writeback. The primitive is not a new optimizer algorithm; it is the kernel form of the same AdamW recurrence, arranged so each parameter element and its state are loaded, transformed, and stored with minimal intermediate traffic.

For gradient $g_t$ and parameter $\theta_t$, AdamW is commonly written as:

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\qquad
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2,
$$

$$
\hat{m}_t=\frac{m_t}{1-\beta_1^t},\qquad
\hat{v}_t=\frac{v_t}{1-\beta_2^t},
$$

$$
\theta_{t+1}
= \theta_t
-\eta\left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}
+\lambda\theta_t\right),
$$

where $\eta$ is the learning rate and $\lambda$ is the decoupled weight-decay coefficient.

Symbols: $t$ is the optimizer step; $g_t$ is the current gradient; $\theta_t$ and $\theta_{t+1}$ are the parameter before and after the update; $m_t$ and $v_t$ are first- and second-moment estimates; $\beta_1$ and $\beta_2$ are exponential-decay coefficients; $\hat{m}_t$ and $\hat{v}_t$ are bias-corrected moments; $\epsilon$ prevents division by zero; and $\lambda\theta_t$ is the decoupled weight-decay term.

In LLM training kernels, fused AdamW is mainly a bandwidth optimization. A naive optimizer step can read and write gradients, parameters, `exp_avg`, `exp_avg_sq`, and sometimes master parameters through several separate kernels. Fusing those operations reduces HBM round trips and launch count, which becomes visible when attention, GEMM, and normalization kernels have already been heavily optimized. It is also the baseline against which newer optimizer-state compression and matrix-optimizer kernels should be judged: if a proposed optimizer adds FLOPs or communication, it must beat a very efficient memory-bound fused AdamW implementation in end-to-end training.

CUDA/platform details include accumulation dtype, vectorized load/store width, scheduler step handling, sparse gradient behavior, and the exact AdamW convention for decoupled weight decay. Implementations generally bucket tensors by dtype, use FP32 or higher-precision intermediates for update math, and keep hyperparameters uniform within a launched tensor list. Inspect first: [Apex FusedAdam docs](https://nvidia.github.io/apex/optimizers.html) for the public contract, [Apex `fused_adam` source](https://nvidia.github.io/apex/_modules/apex/optimizers/fused_adam.html) for the call into `multi_tensor_adam`, and [Megatron Core optimizer config](https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.optimizer.optimizer_config.html) for how large training stacks expose Adam, distributed optimizer, and precision-aware optimizer settings.

#### Master-Weight Mixed Precision

Master-weight mixed precision keeps an optimizer-owned higher-precision copy of parameters, often called the main parameter or master parameter, while the model participates in forward and backward with lower-precision tensors such as FP16, BF16, FP8, MXFP8, or FP4. The optimizer step updates the higher-precision value, then casts or copies the result back to the model parameter representation used by the next forward pass.

In LLM training, this primitive preserves update fidelity while the expensive matrix multiplications run in lower precision. It is especially important when the compute path and the optimizer-state path use different formats: a Transformer Engine layer may use FP8 formats and scaling metadata for GEMMs, while the optimizer may keep main parameters, gradients, and moments in BF16/FP32 or a precision-aware mixture. The optimizer kernel therefore sits at a format boundary: it consumes gradients from the backward pass, updates the stable copy of the weights, and publishes the dtype/layout expected by distributed parameter gather and the next iteration.

CUDA/platform details include cast kernels, extra parameter buffers, loss-scaling interaction for FP16, BF16 behavior without dynamic loss scaling, and whether low-precision parameter gather is enabled. Transformer Engine documentation is the best starting point for FP8 format and scaling behavior, while Megatron Core shows the optimizer-side knobs for main parameter dtype, main gradient dtype, and moment dtype. Inspect first: [Transformer Engine FP8 primer](https://nvidia.github.io/TransformerEngine/examples/fp8_primer.html), [Transformer Engine current scaling](https://nvidia.github.io/TransformerEngine/features/low_precision_training/fp8_current_scaling/fp8_current_scaling.html), [Megatron Core optimizer config](https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.optimizer.optimizer_config.html), and [Megatron Core optimizer internals](https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.optimizer.optimizer.html).

#### Sharded Optimizer State

Sharded optimizer state partitions parameters, gradients, and/or optimizer moments across data-parallel ranks instead of replicating every state tensor on every GPU. ZeRO-style optimizers and FSDP-style training differ in exact lifecycle, but the core primitive is the same: each rank owns a shard of optimizer state, updates that shard locally, and relies on collectives to make full parameters or full gradients available at the moments when computation requires them.

In LLM training, this is the difference between fitting the model and running out of memory. AdamW normally needs parameter storage, gradient storage, first moment, second moment, and often master weights. Sharding those states across data-parallel ranks reduces per-rank memory, but it turns the optimizer step into a communication schedule: gradients are reduced and scattered to owning ranks, local optimizer updates run on shards, and updated parameters are all-gathered or unsharded before the next forward or backward region that needs them.

CUDA/platform details are mostly distributed-systems details with direct kernel consequences. Reduce-scatter and all-gather timing determines whether optimizer kernels can overlap with communication; contiguous flattened buffers make collective calls efficient but complicate mapping between model tensors and shard ranges; NCCL stream ordering affects overlap; checkpointing must reassemble or save sharded state consistently. Inspect first: [DeepSpeed ZeRO](https://deepspeed.readthedocs.io/en/stable/zero3.html), [PyTorch FSDP `ShardingStrategy`](https://docs.pytorch.org/docs/stable/fsdp.html), and [Megatron Core distributed optimizer](https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.optimizer.distrib_optimizer.html), especially its param/grad buffer range maps and local shard ownership model.

#### FP8 State Compression

FP8 state compression stores optimizer state, and sometimes saved activations, in eight-bit floating-point formats with scale metadata rather than in BF16 or FP32. COAT is the main public example in this section: it targets optimizer moments whose dynamic range does not naturally fill FP8's representable range, then applies dynamic range expansion and per-group quantization so first- and second-order moments can be stored compactly with lower quantization error.

In LLM training kernels, FP8 optimizer state changes the update from a simple fused AdamW pass into a quantized-state pipeline. The kernel must dequantize or interpret the stored moment, update it with the new gradient, compute fresh scale or expansion metadata, requantize the moment, and update the parameter. The memory win can be large because AdamW moments are model-size tensors, but the benefit depends on whether scale reductions, casts, and metadata traffic are fused tightly enough to avoid replacing state bandwidth with extra kernel overhead.

CUDA/platform details include FP8 format choice, group size, scale storage dtype, amax or min/max reductions, rounding policy, metadata layout, and whether the state format works with checkpointing and distributed sharding. Transformer Engine's current-scaling documentation is useful for generic FP8 scaling mechanics: quantization needs an amax reduction, a scale computation, and a cast. COAT's implementation is the more optimizer-specific anchor: `CoatAdamW` stores FP8 moment tensors plus scale and expansion tensors, then calls CUDA extension functions such as `qoptim_cuda.fp8_adamw_expand_step`. Inspect first: [COAT paper/project](https://research.nvidia.com/labs/eai/publication/coat/), [NVlabs/COAT](https://github.com/NVlabs/COAT), the [COAT FP8 AdamW implementation](https://raw.githubusercontent.com/NVlabs/COAT/main/coat/optimizer/fp8_adamw.py), and [Transformer Engine current scaling](https://nvidia.github.io/TransformerEngine/features/low_precision_training/fp8_current_scaling/fp8_current_scaling.html).

#### Muon Newton-Schulz Iteration

Muon-style optimizers move part of the optimizer step from scalar elementwise recurrence to matrix preconditioning. The core primitive is a Newton-Schulz iteration that orthogonalizes or whitens a momentum/update matrix. Instead of updating every parameter element independently, the optimizer treats a weight tensor as a matrix, normalizes the update, and applies a small fixed number of matrix multiplications to approximate an orthogonalized direction.

A common quintic Newton-Schulz-style update for a normalized matrix $X_i$ has the form:

$$
X_{i+1}
= aX_i
+ bX_iX_i^\top X_i
+ cX_iX_i^\top X_iX_i^\top X_i,
$$

with coefficients $(a,b,c)$ chosen by the optimizer implementation. This turns the optimizer step into a sequence of GEMM/SYRK-like matrix kernels rather than only elementwise updates.

Symbols: $X_i$ is the normalized matrix iterate at Newton-Schulz step $i$; $X_{i+1}$ is the next iterate; $X_i^\top$ is the transpose; and $a$, $b$, and $c$ are implementation-chosen polynomial coefficients that control the orthogonalization approximation.

In LLM training kernels, this changes the optimizer's hardware profile. AdamW is dominated by HBM bandwidth over parameter-sized tensors; Muon adds GEMM/[SYRK](https://docs.nvidia.com/cuda/cublas/#cublas-t-syrk) (symmetric rank-k update) work over layer matrices, FP32 or configurable FP32-matmul precision choices, and additional synchronization when the matrix is sharded. This can be acceptable or even efficient on modern GPUs because tensor-core throughput is high, but it means optimizer time can no longer be modeled as a pure elementwise memory pass. It also means the optimizer kernel mix depends on model architecture: large MLP and attention projection matrices have different preconditioning costs than biases, embeddings, or normalization parameters.

CUDA/platform details include matrix orientation, spectral or p2 normalization, coefficient set, iteration count, matmul precision, exclusion of one-dimensional parameters, and how fused [QKV](https://docs.nvidia.com/deeplearning/cudnn/frontend/latest/operations/Attention.html) (query-key-value) parameters are split before preconditioning. NVIDIA's emerging optimizer docs describe `OrthogonalizedOptimizer` and Newton-Schulz utility functions, including coefficient choices and FP32 matmul precision, while the Megatron integration exposes Muon-specific configuration such as number of Newton-Schulz steps and tensor-parallel mode. Inspect first: [Scaling Muon](https://arxiv.org/abs/2502.16982), [NVIDIA Emerging Optimizers docs](https://docs.nvidia.com/nemo/emerging-optimizers/latest/apidocs/orthogonalized-optimizers.html), [Muon utility source docs](https://docs.nvidia.com/nemo/emerging-optimizers/latest/_modules/emerging_optimizers/orthogonalized_optimizers/muon_utils.html), and [Megatron emerging optimizer integration](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/optimizer/emerging_optimizers.py).

#### Distributed Gram / SYRK Reduction

Distributed Gram/SYRK reduction is the communication-heavy subprimitive inside distributed matrix optimizers. Newton-Schulz iterations need matrix products such as $XX^\top$ or $X^\top X$, depending on orientation. When the update matrix is sharded across tensor-parallel ranks, each rank can compute a local contribution to the Gram-like product, but ranks must reduce those contributions before subsequent matrix multiplications see the same global preconditioner information.

For row-sharded blocks $X_r$, the global Gram matrix is:

$$
G = X^\top X = \sum_{r=1}^{R} X_r^\top X_r.
$$

Symbols: $X$ is the optimizer update or momentum matrix being preconditioned; $X_r$ is rank $r$'s row shard; $R$ is the number of participating ranks or shards; $G$ is the global Gram matrix; and $X_r^\top X_r$ is the rank-local symmetric-rank-k contribution.

Equivalently, if the matrix is distributed by rows,

$$
\begin{array}{c}
X =
\begin{bmatrix}
X_1\\
X_2\\
\vdots\\
X_R
\end{bmatrix}
\quad\Longrightarrow\quad
\begin{array}{c|c}
\text{rank} & \text{local contribution}\\ \hline
1 & X_1^\top X_1\\
2 & X_2^\top X_2\\
\vdots & \vdots\\
R & X_R^\top X_R
\end{array}
\quad\xrightarrow{\;\operatorname{AllReduce}\;}\quad
G
\end{array}
$$

Diagram symbols: the vertical stack shows $X$ partitioned by rows into $X_1,\ldots,X_R$; the table shows each rank's local Gram contribution; $\operatorname{AllReduce}$ sums those local matrices across ranks; and $G$ is the replicated global result consumed by later Newton-Schulz matrix multiplications.

That summation is the collective-reduction boundary; fusing or overlapping it with SYRK tiles is the systems opportunity.

In LLM training kernels, this primitive appears when Muon-like updates are applied to tensor-parallel weights. A duplicated mode can all-gather the full momentum matrix once and run all Newton-Schulz work redundantly on each rank. A distributed mode can keep the matrix partitioned, but each Newton-Schulz iteration introduces collectives around the matrix products. NVIDIA's 2026 Megatron writeup notes that the first two of three matrix multiplications in a Newton-Schulz iteration can be mapped to SYRK, saving roughly triangular work, but distributed mode still needs all-reduce behavior unless communication is fused into the SYRK kernel.

CUDA/platform details include whether the local matrix tile is row- or column-sharded, whether SYRK writes a full matrix or compact triangular form, whether all-reduce is launched after a kernel or fused tile-by-tile, and whether communication uses the same NCCL stream that other optimizer collectives use. On Hopper/Blackwell-era systems, the interesting implementation question is not only "which matmul is fastest" but "where can the reduction be hidden behind tiles of symmetric-rank-k work." Inspect first: [NVIDIA Megatron emerging optimizers blog](https://developer.nvidia.com/blog/advancing-emerging-optimizers-for-accelerated-llm-training-with-nvidia-megatron/), [NVIDIA Emerging Optimizers](https://github.com/NVIDIA-NeMo/Emerging-Optimizers), [Megatron layer-wise optimizer](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/optimizer/layer_wise_optimizer.py), and [cuBLAS SYRK reference](https://docs.nvidia.com/cuda/cublas/#cublas-t-syrk).

#### Blockwise Or Duplicated Optimizer Modes

Blockwise and duplicated optimizer modes are placement choices for matrix-optimizer work under tensor parallelism. Duplicated mode gathers enough of the matrix state that every rank can run the full Newton-Schulz iteration locally, then each rank applies the slice of the resulting update that it owns. Distributed mode keeps the matrix sharded and communicates during the iteration. Blockwise mode orthogonalizes only each rank's local block, avoiding communication but no longer matching full-matrix orthogonalization exactly.

In LLM training kernels, the right mode depends on topology, matrix shape, tensor-parallel degree, and the relative cost of tensor-core work versus network latency. Duplicated mode spends extra compute to reduce communication frequency; distributed mode reduces redundant compute but adds collectives within each Newton-Schulz iteration; blockwise mode is the cheapest communication path but changes the optimizer's mathematical approximation. At the data-parallel level, layer-wise distributed optimizers add another placement choice: whole layers can be assigned to ranks so the rank that updates a layer has the full layer-shaped object required for preconditioning.

CUDA/platform details include all-gather versus all-reduce count, `all_gatherv` for variable-size layer ownership, load balancing across layers with different matrix sizes, parameter layout under [TP](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html) (tensor parallelism) sharding, and how QKV or fused parameters are split before preconditioning. These choices should be benchmarked with optimizer FLOPs and communication counted explicitly; otherwise Muon-like methods can appear faster or slower depending on accounting. Inspect first: [NVIDIA Megatron emerging optimizers blog](https://developer.nvidia.com/blog/advancing-emerging-optimizers-for-accelerated-llm-training-with-nvidia-megatron/), [Emerging Optimizers tensor-parallel Newton-Schulz docs](https://docs.nvidia.com/nemo/emerging-optimizers/latest/apidocs/orthogonalized-optimizers.html), [Megatron Core optimizer config](https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.optimizer.optimizer_config.html), and [Megatron layer-wise optimizer](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/optimizer/layer_wise_optimizer.py).

### Why It Matters For Training

As attention and GEMM get faster, optimizer overhead becomes more visible. Optimizer states can dominate memory. At scale, optimizer communication and matrix iterations can affect [MFU](https://arxiv.org/abs/2104.04473) (Model FLOP Utilization, the fraction of theoretical hardware FLOPs achieved by model training) unless fused, distributed, or overlapped.

### Explore Further

- `official docs`: [Apex FusedAdam](https://nvidia.github.io/apex/optimizers.html)
- `paper`: [COAT](https://arxiv.org/abs/2410.19313)
- `paper`: [Scaling FP8 training to trillion-token LLMs](https://arxiv.org/abs/2409.12517)
- `official blog`: [NVIDIA Megatron emerging optimizers / Muon](https://developer.nvidia.com/blog/advancing-emerging-optimizers-for-accelerated-llm-training-with-nvidia-megatron/)
- `paper`: [Scaling Muon](https://arxiv.org/abs/2502.16982)

### Implementation Examples

- Apex FusedAdam: inspect multi-tensor batching and fused elementwise Adam updates.
- COAT: inspect FP8 compression of optimizer states and activations.
- NVIDIA Megatron Muon integration: inspect duplicated, distributed, and blockwise Newton-Schulz modes and planned fused SYRK/all-reduce style optimizations.
- DeepSeek-V4: official model card says Muon is used; exact training optimizer kernels are not fully disclosed.

### Caveats

Optimizer benchmarking must include communication, state precision, parameter sharding, and whether optimizer FLOPs are counted in MFU. Muon-like methods can look expensive or cheap depending on accounting.

## 13. Variable-Length Scheduling and Load Balancing

### Technique

Variable-length scheduling reorders or repartitions work so that long sequences, causal masks, sparse blocks, or expert load imbalance do not leave a tail of slow tiles. Examples include longest-processing-time-first scheduling, shortest-processing-time-first scheduling for deterministic reductions, microbatch-specific context parallelism, and auxiliary-loss-free MoE load balancing.

### Primitive Techniques

#### [THD](https://developer.nvidia.com/blog/speeding-up-variable-length-training-with-dynamic-context-parallelism-and-nvidia-megatron-core/) (Token-Head-Dimension) Packed Layout

THD packed layout collapses the conventional batch-by-sequence axes into a token-major `T,H,D` tensor plus sequence-length metadata. Instead of launching kernels over a padded rectangular `B,S,H,D` region, the training step carries the exact number of valid tokens and the per-sequence offsets needed to reconstruct sequence boundaries. In variable-length training this is the data-contract primitive that makes later scheduling useful: microbatches can contain a variable number of original sequences, attention kernels can avoid padding tiles, and loss kernels can normalize by valid-token count rather than by `max_seqlen`.

The conversion from padded BSHD to packed THD can be sketched as:

$$
\begin{array}{c}
\begin{array}{c|cccc}
\text{batch row} & s=0 & s=1 & s=2 & s=3\\ \hline
b_0 & x_{0,0} & x_{0,1} & x_{0,2} & \varnothing\\
b_1 & x_{1,0} & x_{1,1} & \varnothing & \varnothing\\
b_2 & x_{2,0} & x_{2,1} & x_{2,2} & x_{2,3}
\end{array}
\\[3mm]
\Downarrow
\\[-1mm]
\begin{array}{c|ccccccccc}
T & 0&1&2&3&4&5&6&7&8\\ \hline
\text{token} &
x_{0,0}&x_{0,1}&x_{0,2}&
x_{1,0}&x_{1,1}&
x_{2,0}&x_{2,1}&x_{2,2}&x_{2,3}
\\
\text{cu\_seqlens} &
0&&&3&&5&&&9
\end{array}
\end{array}
$$

The attention kernel receives the compact token axis plus `cu_seqlens`; padding symbols $\varnothing$ do not become work.

Diagram symbols: $b_0,b_1,b_2$ are batch rows, $s$ is the padded sequence-position index, $x_{b,s}$ is a real token, $\varnothing$ is padding, $T$ is the packed token-axis index, and `cu_seqlens` stores cumulative sequence starts and ends so the kernel can recover each original sequence boundary.

CUDA details are mostly about preserving regular memory access while exposing irregular boundaries. The hidden dimension should remain contiguous for vectorized loads and tensor-core-friendly fragments, while `cu_seqlens`, `seq_len_q`, `seq_len_kv`, or ragged offset tensors define legal query/key ranges. Attention forward and backward must keep softmax statistics, dropout/RNG replay, and gradient scatter aligned with packed token order. Inspect the [Dynamic CP post](https://developer.nvidia.com/blog/speeding-up-variable-length-training-with-dynamic-context-parallelism-and-nvidia-megatron-core/) first for Megatron-Core's move from BSHD to THD and its `PackedSeqParams` contract, then inspect [cuDNN NSA](https://docs.nvidia.com/deeplearning/cudnn/frontend/latest/fe-oss-apis/nsa.html) for concrete `T,H,D` variable-length API requirements.

#### Shape/Length Metadata Probing

Shape/length metadata probing is a lightweight pass over the next global batch that gathers sequence lengths, modality shapes, and packed-sample bounds before the real training step consumes the data. It is not the same as loading every token twice; the scheduler needs just enough metadata to estimate which samples are expensive, which samples fit together, and which microbatches need extra context-parallel capacity. For multimodal and long-context training, the probe can include image/video token counts as well as text sequence lengths.

In LLM training kernels, this metadata becomes launch geometry and boundary metadata: maximum sequence lengths, cumulative sequence offsets, valid-token counts, and per-microbatch context-parallel settings. Platform details matter because the probing path can become a CPU and storage bottleneck if it performs full deserialization or blocks the main data loader. Distributed probing should gather compact integer metadata, preserve deterministic sample order for reproducibility, and broadcast the resulting `num_micro_batches`, `max_seqlen`, and sequence offsets to pipeline stages that did not build the schedule locally. Inspect the [Dynamic CP post](https://developer.nvidia.com/blog/speeding-up-variable-length-training-with-dynamic-context-parallelism-and-nvidia-megatron-core/) first, especially its "I/O pressure" and pipeline-broadcast discussion.

#### Cost Model For FLOPs And Memory

A cost model for FLOPs and memory assigns each sample or packed sequence an estimated runtime and activation footprint before kernel launch. Attention work grows roughly quadratically with sequence length, while activation storage and many non-attention paths grow closer to linearly, so a pack that balances tokens can still be badly imbalanced in attention FLOPs. The model should also account for communication exposure: a short packed sample sharded across a large context-parallel group may not have enough compute to hide CP communication.

For a microbatch with sequence lengths $\{s_b\}$, a crude dense-attention proxy is:

$$
C_{\mathrm{attn}} \propto \sum_b H\,s_b^2\,d_h,\qquad
M_{\mathrm{act}} \propto \sum_b s_b\,d_{\mathrm{model}},
$$

where $H$ is the head count and $d_h$ is the per-head dimension. This is why equal token counts do not imply equal attention cost.

Symbols: $C_{\mathrm{attn}}$ is a proxy for dense-attention arithmetic cost, $M_{\mathrm{act}}$ is a proxy for activation memory, $b$ indexes sequences in the microbatch, $s_b$ is sequence $b$'s length, $H$ is the number of attention heads, $d_h$ is the per-head dimension, and $d_{\mathrm{model}}$ is the model hidden width.

Training systems use this primitive to choose microbatch packs, target per-rank work quotas, and CP sizes that reduce data-parallel waiting and pipeline bubbles. CUDA/platform calibration is important: the model should be checked against actual backend kernels, recomputation settings, precision modes, and collective topology, because a simple $S^2$ proxy will miss launch overhead, sparse/tiled attention effects, and exposed NCCL time. Inspect the [Dynamic CP post](https://developer.nvidia.com/blog/speeding-up-variable-length-training-with-dynamic-context-parallelism-and-nvidia-megatron-core/) for its solver/cost-model/simulator split, then inspect [MegaScale](https://arxiv.org/abs/2402.15627) for production-scale observability and straggler diagnosis at more than 10,000 GPUs.

#### Microbatch-Specific CP Size

Microbatch-specific context-parallel size selects the context-parallel degree per scheduled microbatch instead of using a single static CP size for the whole job. Long packed sequences can be sharded to fit memory, while short packed sequences can stay on fewer ranks and avoid unnecessary communication. The important contract is that the attention stack receives the selected `cp_size` and `cp_group` with the packed sequence metadata, so kernels and position-embedding code do not silently read stale global CP settings.

CUDA and distributed-runtime details dominate this primitive. Communication groups should be constructed ahead of time for the allowed CP sizes, because creating NCCL groups in the hot path would erase the scheduling benefit. Kernel launch parameters, sequence-slice offsets, rotary-position offsets, and Transformer Engine or cuDNN attention calls must all agree on the microbatch's CP group. Pipeline-parallel stages also need the same `num_micro_batches`, `max_seqlen`, and packed offsets or the stage schedule will diverge. Inspect the [Dynamic CP post](https://developer.nvidia.com/blog/speeding-up-variable-length-training-with-dynamic-context-parallelism-and-nvidia-megatron-core/) first for the multiple-CP-group design and `PackedSeqParams` changes, and then its linked Megatron-LM Hybrid CP work for implementation shape.

#### Tile List Scheduling

Tile list scheduling builds an explicit list of work tiles instead of deriving work solely from a rectangular grid. A tile entry can encode a batch or packed-sequence index, head group, query block, key/value block, causal/sparse mask information, and sometimes an estimated cost. This is useful when causal masking, variable sequence length, or sparse attention causes different tiles in the same nominal tensor shape to run different mainloop lengths.

In training kernels, tile lists feed attention forward and backward schedulers, sparse-attention selection paths, and deterministic backward variants that need a fixed reduction order. CUDA details include keeping the tile metadata compact enough to stay cache-friendly, avoiding per-tile binary searches through cumulative lengths, and making the virtual-to-actual index mapping cheap for every CTA. On Blackwell and Hopper-class kernels, tile ordering also interacts with L2 locality, cluster placement, and whether CTAs are paired for larger tensor-core tiles. Inspect [FlashAttention-4](https://arxiv.org/abs/2603.05451) and the [FA4 blog](https://tridao.me/blog/2026/flash4/) for causal and variable-length attention tile scheduling, then inspect [cuDNN NSA](https://docs.nvidia.com/deeplearning/cudnn/frontend/latest/fe-oss-apis/nsa.html) for block-index and top-k metadata used by sparse attention APIs.

#### [LPT](https://en.wikipedia.org/wiki/Longest-processing-time-first_scheduling) (Longest-Processing-Time-First)/[SPT](https://en.wikipedia.org/wiki/Shortest_job_next) (Shortest-Processing-Time-First) Ordering

LPT/SPT ordering chooses the order in which irregular work units are issued. Longest-processing-time-first (LPT) launches the expensive tiles or batches early so the tail of the kernel is less likely to be dominated by a few late heavy CTAs. Shortest-processing-time-first (SPT) is useful when the goal is not only throughput but also deterministic or serialized progress, because short work units can clear semaphore-protected regions without waiting behind the largest reductions.

Given tile costs $c_i$, the policies are simply:

$$
\mathrm{LPT}: c_{\pi(1)}\ge c_{\pi(2)}\ge\cdots,\qquad
\mathrm{SPT}: c_{\pi(1)}\le c_{\pi(2)}\le\cdots.
$$

The hard part is estimating $c_i$ cheaply enough and keeping the sorted tile list consistent across forward/backward or across ranks when determinism matters.

Symbols: $c_i$ is the estimated cost of tile or work item $i$; $\pi$ is the permutation that orders work items; $\pi(1)$ is the first scheduled item; LPT sorts costs from largest to smallest; and SPT sorts costs from smallest to largest.

For LLM training kernels, LPT applies naturally to causal attention tiles whose valid key range grows with position, packed batches whose sequence lengths differ, and sparse attention blocks whose selected block counts differ. SPT appears in deterministic backward paths where global atomic accumulation is replaced or constrained by locks, fences, or semaphore-style ordering. Platform details include stable sorting of virtual tile IDs, caching the sorted metadata when sequence lengths repeat, preserving reproducibility across ranks, and avoiding CPU-side sorts in the inner iteration when a small preprocessing kernel can generate the mapping. Inspect the [FA4 blog](https://tridao.me/blog/2026/flash4/) first for its LPT scheduling and SPT deterministic-mode discussion.

#### Expert-Load-Aware Routing

Expert-load-aware routing changes MoE routing, capacity, or bias state so that expert work is balanced enough for the GPU kernels and all-to-all exchanges that follow. The router still chooses experts based on token affinity, but the training system monitors per-expert token counts and can apply load-balancing signals or routing biases to prevent a few experts from becoming stragglers. This primitive is distinct from the grouped GEMM implementation: it shapes the row counts that grouped GEMM and expert dispatch kernels must process.

In training kernels, expert imbalance shows up as uneven token permutation buffers, skewed all-to-all payloads, and grouped GEMM problems where a few experts have large `M` while others are empty. CUDA details include fast per-expert histograms, prefix sums for packed expert buffers, capacity or dropless behavior, stable unpermute/combine metadata for backward, and scale metadata if the MoE path uses FP8 communication or expert GEMMs. Inspect [DeepSeek-V3](https://arxiv.org/abs/2412.19437) first for its auxiliary-loss-free expert-bias strategy and no-token-dropping claim, then inspect [Tutel](https://arxiv.org/abs/2206.03382) for adaptive MoE parallelism and pipelining under dynamic expert workloads.

#### Asynchronous Solver / Sampler

An asynchronous solver or sampler moves schedule construction out of the critical training step. The solver consumes probed metadata, evaluates packing and CP choices, and prepares the plan for a future iteration while the GPUs execute the current one. This makes variable-length scheduling viable at scale: a strong scheduling plan is only useful if its CPU-side search, sorting, metadata exchange, and simulator pass do not create a new bubble before the next GPU launch.

Platform details are about overlap and determinism. The solver should run in the data-sampling path or a bounded background worker, use compact metadata rather than full tensors, and publish plans early enough for pipeline stages and data-parallel ranks to agree on microbatch counts. It also needs fallbacks when the next plan is late, because blocking all ranks on a slow CPU worker can be worse than using a simpler pack. Inspect the [Dynamic CP post](https://developer.nvidia.com/blog/speeding-up-variable-length-training-with-dynamic-context-parallelism-and-nvidia-megatron-core/) first for asynchronous `data_sampler` execution, small grid search around the microbatch-count knee, and the practical constraint that scheduling overhead should be effectively hidden.

### Why It Matters For Training

Real pretraining and post-training data has long-tailed sequence lengths. A few very long samples can dominate memory and runtime. Causal attention tiles have different work depending on position. MoE expert loads vary with routing. Without scheduling, GPUs wait on stragglers.

### Explore Further

- `paper/blog`: [FlashAttention-4](https://arxiv.org/abs/2603.05451), [FA4 blog](https://tridao.me/blog/2026/flash4/)
- `official blog`: [Dynamic Context Parallelism in Megatron-Core](https://developer.nvidia.com/blog/speeding-up-variable-length-training-with-dynamic-context-parallelism-and-nvidia-megatron-core/)
- `paper`: [DeepSeek-V3](https://arxiv.org/abs/2412.19437)
- `paper`: [MegaScale](https://arxiv.org/abs/2402.15627)

### Implementation Examples

- FA4: tile scheduler for causal masking and variable sequence length.
- Megatron-Core Dynamic Context Parallelism: chooses CP size per microbatch for variable-length datasets.
- DeepSeek-V3: auxiliary-loss-free load balancing for MoE routing and DualPipe scheduling.
- MegaScale: production observability and straggler mitigation at over 10,000 GPUs.

### Caveats

Scheduling preprocessing can cost time and memory. It only pays off if metadata can be reused, cached, or computed cheaply relative to the imbalance it removes.

## 14. Persistent and Megakernel Scheduling

### Technique

Persistent kernels and megakernels keep a GPU program resident and execute many logical operations internally. Instead of launching a sequence of small kernels, a large kernel uses an instruction set, task queue, counters, shared-memory pages, or symmetric memory to coordinate work.

### Primitive Techniques

#### Resident [CTA](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#thread-hierarchy) (Cooperative Thread Array) Work Queue

A resident CTA work queue is persistent-thread scheduling at CUDA thread-block granularity. Instead of launching one grid per operation, the kernel launches a bounded set of long-lived CTAs, often sized to the intended SM budget, and each CTA repeatedly pulls task descriptors until the current epoch is complete. A task descriptor may name an operation, tensor tile, expert bucket, sequence block, rank shard, scratch allocation, and dependency counter.

In LLM training kernels, this pattern is useful when subgraph work is irregular or too fine-grained to justify many separate launches. MoE layers can enqueue token-pack, expert-GEMM, activation, and combine tasks whose sizes vary by router output. Variable-length attention can enqueue only real tiles. Optimizer and loss kernels can batch many small parameter or vocabulary chunks. The persistent queue amortizes launch overhead and gives the device a way to smooth tail effects when one SM receives slower work than another.

CUDA details matter because the queue itself is synchronization-heavy. A global atomic dequeue is tolerable only when each task does enough work to hide the atomic latency; otherwise, per-SM queues, batched dequeues, or static schedules are preferable. The resident grid also reduces CUDA's normal ability to oversubscribe blocks, so the launch must leave enough occupancy for latency hiding and, in distributed training, enough SM or copy-engine headroom for communication. If grid-wide synchronization is required, cooperative launch constraints apply; if the queue uses ordinary global memory, task publication needs explicit memory-ordering discipline with atomics or fences.

Inspect first: Hazy's tensor-parallel megakernel discussion of a [global work queue](https://hazyresearch.stanford.edu/blog/2025-09-28-tp-llama-main), Mirage [MPK](https://github.com/mirage-project/mirage) (Mirage Persistent Kernel)'s [`PersistentKernel`](https://github.com/mirage-project/mirage) worker/scheduler API, and NVIDIA's [Cooperative Groups](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cooperative-groups.html) launch and synchronization rules.

#### On-GPU Interpreter

An on-GPU interpreter encodes a fused subgraph as a compact instruction stream that the persistent kernel executes on device. Each instruction describes an operation family, tensor pointers, shape metadata, tile coordinates, dependency counters, and scratch-memory needs. The implementation is usually not a general virtual machine; it is a small instruction dispatcher whose operation bodies are still specialized CUDA templates or generated code.

For LLM training, an interpreter lets one launch execute many logical operations while retaining device-side scheduling control. A forward subgraph might run RMSNorm, QKV projection, RoPE, attention, output projection, residual update, MLP, and normalization. A training-adjacent subgraph might run MoE dispatch, two expert GEMMs, SwiGLU, and combine. Backward use is harder, but the same abstraction can encode recomputation, statistic reloads, gradient reductions, and saved-activation handoffs without returning to the CPU between every small stage.

The platform risk is decode overhead and resource inflation. Instruction descriptors should be cache-friendly and preferably read-only during execution. Branching on operation type can cause warp divergence if too many unrelated instruction kinds are mixed inside one CTA. Carrying all operation variants in one translation unit can increase code size and register pressure, so practical implementations often use a small instruction set with templated bodies and a stable ABI for layouts, scratch pages, and counters. CPU-side schedule construction can be reused across steps when tensor shapes are stable, but training workloads with changing sequence lengths or MoE loads need either dynamic queues or cheap rescheduling.

Inspect first: Hazy's low-latency megakernel writeup on the [on-GPU interpreter](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles), the [Hazy Megakernels repository](https://github.com/HazyResearch/Megakernels), and Mirage MPK's graph-building examples in the [Mirage repository README](https://github.com/mirage-project/mirage).

#### Shared-Memory Paging

Shared-memory paging treats a persistent CTA's dynamic shared-memory allocation as a small managed cache divided into fixed-size pages. Instructions request pages for input tiles, weights, scale metadata, or temporary accumulators, then release them when downstream consumers no longer need the data. The important idea is not virtual memory; it is explicit lifetime management for scarce on-chip SRAM.

In LLM training kernels, this helps fused stages overlap data movement and compute. A megakernel can start loading weights or activation tiles for the next state while the current state is storing results, provided a page is available. MoE expert MLPs can reuse pages across dispatch, GEMM, activation, and combine. Attention backward or recomputation-heavy training subgraphs can keep score statistics, partial gradients, or reduction fragments on chip long enough to avoid avoidable HBM traffic.

CUDA shared memory is allocated per CTA, backed by the physical shared-memory pool on an SM. Large page pools can therefore reduce the number of resident CTAs and make a persistent kernel less tolerant of memory latency. Page sizes must respect alignment, bank-conflict behavior, and the data movement path being used; on Hopper and Blackwell, TMA-style bulk copies and asynchronous barriers add their own alignment and completion requirements. A page allocator also has to be conservative about lifetimes, because a premature release is a data race and a late release creates artificial memory bubbles.

Inspect first: Hazy's [shared-memory paging](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles) description, NVIDIA's Hopper [Tensor Memory Accelerator](https://docs.nvidia.com/cuda/archive/13.0.2/hopper-tuning-guide/index.html) notes, and the CUDA [Cooperative Groups barrier](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cooperative-groups.html) APIs for asynchronous shared-memory workflows.

#### Native GPU-Side Ping-Pong Scheduling

Native GPU-side ping-pong scheduling moves the double-buffering decision from a host stream graph into a resident kernel. A CTA, warpgroup, or small scheduler loop alternates between two slots while the kernel stays live: one role fills slot $1-b(e)$ for epoch $e+1$, and another role consumes slot $b(e)$ for epoch $e$. The slots may be shared-memory pages, [TMEM](https://docs.nvidia.com/cuda/parallel-thread-execution/#tensor-memory) columns, peer-visible symmetric buffers, or task descriptors in global memory.

The device-side schedule is:

$$
\begin{array}{c|ccc}
\text{epoch} & \text{producer role} & \text{consumer role} & \text{dependency object}\\ \hline
e     & \operatorname{fill}(S_{1-b(e)},T_{e+1}) & \operatorname{run}(S_{b(e)},T_e) & C_{b(e)}=\operatorname{ready}\\
e+1   & \operatorname{fill}(S_{b(e)},T_{e+2}) & \operatorname{run}(S_{1-b(e)},T_{e+1}) & C_{1-b(e)}=\operatorname{ready}\\
e+2   & \operatorname{fill}(S_{1-b(e)},T_{e+3}) & \operatorname{run}(S_{b(e)},T_{e+2}) & C_{b(e)}=\operatorname{ready}
\end{array}
$$

Symbols: $e$ is a persistent-kernel scheduling epoch; $S_0$ and $S_1$ are reusable slots; $b(e)=e\bmod 2$ selects the consumer slot; $T_e$ is the tile, task, or subgraph state for epoch $e$; $C_0$ and $C_1$ are counters, memory barriers, or semaphores that publish slot state; $\operatorname{fill}(\cdot)$ moves data or task metadata into a slot; $\operatorname{run}(\cdot)$ executes the compute or communication state associated with the slot.

This pattern is "native" when the alternation is expressed inside CUDA code rather than as separate host launches on two streams. FlashAttention-3 uses producer/consumer warp specialization and ping-pong overlap inside one attention kernel. FlashAttention-4 extends the idea to Blackwell schedules with [UMMA](https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-instructions) (ultra matrix multiply-accumulate), [TMEM](https://docs.nvidia.com/cuda/parallel-thread-execution/#tensor-memory), softmax/correction worker groups, and 2-CTA backward. Hazy megakernels and Mirage MPK use the same primitive at a coarser granularity: while one resident worker group executes the current instruction or subgraph state, another group can prepare the next state, page, or dependency record. DeepGEMM Mega MoE applies the idea to training-adjacent expert-parallel dispatch, grouped GEMMs, activation, and combine, where the slot may be a symmetric communication buffer rather than a pure on-chip tile.

The hard part is progress. A native schedule cannot rely on kernel launch order to drain all earlier work, so every slot transition needs an acquire/release protocol: producers publish after writes are visible; consumers release after reads and dependent MMAs are complete. Counter placement and polling frequency must avoid saturating memory pipelines. The resident grid must reserve enough active CTAs or warpgroups that a producer for epoch $e+1$ can run while consumers for epoch $e$ are waiting. If all resident workers wait on slots that require unscheduled workers, the ping-pong protocol deadlocks. For multi-rank kernels, communication progress also competes with tensor-core work for SMs, so native ping-pong becomes a global scheduling contract rather than a local latency-hiding trick.

Inspect first: [FA3 blog][S4] for ping-pong warpgroup overlap; [FA4 report][S5] for Blackwell native attention scheduling; Hazy's [no-bubbles megakernel][S39] and [tensor-parallel megakernel][S41]; [Mirage MPK][S40]; [DeepGEMM][S11]; [NVSHMEM][S29]; CUDA [Cooperative Groups and barriers][S61]; CUDA [Programmatic Dependent Launch][S61].

#### Counter Barriers / Semaphores

Counter barriers and semaphores are device-side dependency objects. A producer increments or stores a counter after finishing a tile, chunk, or instruction. A consumer waits until the counter reaches the expected value before reading the produced data. In a megakernel, these counters replace the implicit global ordering normally supplied by ending one kernel launch before starting the next.

Training kernels need this because the useful dependence granularity is often smaller than a whole tensor. Down-projection work can begin as soon as a chunk of an MLP intermediate is ready. Attention backward can consume per-row softmax statistics as soon as they are produced or recomputed. MoE combine can start for experts whose outputs have arrived while other experts are still running. Chunk-level counters expose this overlap without forcing a coarse stream or kernel boundary.

Correctness depends on CUDA memory visibility, not just integer values. Producers must make global or shared-memory writes visible before publishing completion, and consumers must use an acquire-style read or an equivalent fence before consuming. Busy-wait loops burn issue slots and can deadlock if all resident CTAs wait for work that can only be produced by CTAs that were never scheduled, so queue design and launch bounds must guarantee progress. CUDA Programmatic Dependent Launch can start a later kernel early on compute capability 9.0 and newer, but its synchronization point is still coarser than fine-grained in-kernel counters.

Inspect first: Hazy's [counter synchronization](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles) mechanism, NVIDIA's [Programmatic Dependent Launch](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html) documentation, and Mirage MPK's [`PersistentKernel`](https://github.com/mirage-project/mirage) scheduler model.

#### Fused Subgraph State Machine

A fused subgraph state machine represents a multi-operation region as states inside one kernel: dispatch, load, GEMM, activation, reduction, communication, combine, and epilogue. State transitions are guarded by counters, queue entries, or statically scheduled instruction order. Compared with a narrow fused epilogue, the state machine owns the whole subgraph schedule.

This is especially natural for LLM training subgraphs that repeat every layer and include both compute and movement. DeepGEMM Mega MoE is the clearest public training-adjacent example: it fuses expert-parallel dispatch, FP8xFP4 expert linears, SwiGLU, the second expert linear, and expert-parallel combine into one mega-kernel. The same idea can apply to post-training losses, small-batch long-context routines, optimizer shards, or activation-recompute regions where intermediate tensors would otherwise be written to HBM between short kernels.

The CUDA cost is that every state contributes to the same launch's resource envelope. If a communication state needs persistent buffers, a GEMM state needs tensor-core fragments, and an activation state needs special-function throughput, the compiled kernel must budget registers, shared memory, warp roles, and CTA residency across all of them. State machines also complicate autograd integration: training code must define which tensors or statistics are saved, which are recomputed, and where deterministic ordering is required for debugging or gradient checks.

Inspect first: DeepGEMM's [Mega MoE](https://github.com/deepseek-ai/DeepGEMM) README section and [Mega MoE test case](https://github.com/deepseek-ai/DeepGEMM/blob/main/tests/test_mega_moe.py), Hazy's [Megakernels implementation](https://github.com/HazyResearch/Megakernels), and Mirage MPK's [RMSNorm-linear graph example](https://github.com/mirage-project/mirage) style fused graph API.

#### Symmetric Multi-Rank Memory

Symmetric multi-rank memory allocates communication buffers with matching layout, and often matching virtual-address assumptions, across the participating GPU ranks. In a partitioned global address space such as NVSHMEM, device code can compute or name peer locations directly instead of asking the host to set up every transfer. For megakernels, this makes peer buffers addressable from the long-running kernel itself.

LLM training uses this pattern wherever communication is inside the subgraph rather than around it. Expert-parallel MoE needs dispatch and combine buffers across ranks. Tensor-parallel layers need all-gather, reduce-scatter, or custom transposes. Pipeline-parallel or context-parallel variants may need activation fragments to move while local computation continues. Symmetric buffers let a persistent kernel write remote rank slots predictably and signal completion with device-side counters.

Platform details are central. Peer access and addressability depend on the topology, runtime, and allocator. NVLink or NVSwitch can make fine-grained peer stores practical; PCIe or cross-node paths may prefer bulk collectives. Symmetric buffers need identical sizing, padding, alignment, and rank ordering across processes, and remote writes still need explicit completion and visibility protocols. DeepGEMM's Mega MoE path calls out multi-process launch with symmetric memory, while NVSHMEM provides in-kernel communication APIs over a partitioned global address space.

Inspect first: DeepGEMM's [Mega MoE](https://github.com/deepseek-ai/DeepGEMM) symmetric-buffer API, NVIDIA [NVSHMEM](https://developer.nvidia.com/nvshmem), and Mirage MPK's [`nvshmem_tensor`](https://github.com/mirage-project/mirage) allocation mode.

#### Persistent Communication Progress

Persistent communication progress keeps communication work inside, or tightly coupled to, the resident kernel. Instead of launching NCCL or copy operations at every subgraph boundary, some resident warps or CTAs issue remote stores, advance collectives, or poll completion counters while other warps continue compute. This turns communication into another scheduled state rather than a separate host-visible phase.

In LLM training, this is useful for stable repeated communication patterns: MoE all-to-all dispatch and combine, tensor-parallel all-gather and reduce-scatter, sequence or context-parallel activation exchange, and post-attention distributed transposes. When the compute state has enough independent work, communication workers can hide remote movement behind GEMM, normalization, or reduction work. This is also a way to express nonstandard communication patterns that do not map cleanly onto a stock collective.

The tradeoff is that communication progress consumes SM issue bandwidth and memory-system resources. Copy engines plus NCCL may outperform device-side communication for large contiguous transfers, while in-kernel peer stores can win for fine-grained or layout-transforming traffic. Completion protocols must avoid all ranks waiting on each other with no active producer, and the kernel should reserve enough workers for both local compute and remote progress. Topology awareness matters because NVLink, NVSwitch, and cross-node fabrics have very different latency, ordering, and bandwidth behavior.

Inspect first: Hazy's tensor-parallel megakernel section on [dedicated storer threads and cross-GPU overlap](https://hazyresearch.stanford.edu/blog/2025-09-28-tp-llama-main), NVIDIA [NVSHMEM](https://developer.nvidia.com/nvshmem) in-kernel communication, and Mirage [MPK](https://github.com/mirage-project/mirage) multi-GPU support.

#### Resource Budgeting Across Fused Stages

Resource budgeting across fused stages is the act of choosing how many CTAs, warps, registers, shared-memory pages, tensor-core issue slots, communication workers, and SMs each state may consume. A megakernel's throughput is limited by the slowest or most over-resourced state, so the scheduling problem is not simply "fuse more." It is to fuse only as much as the device can keep balanced.

Training kernels make this difficult because adjacent stages stress different hardware. Attention may be limited by softmax or shared-memory traffic; expert GEMMs by tensor cores; routing and masking by integer and memory operations; communication by NVLink or NVSwitch; optimizer steps by HBM bandwidth. A fused training subgraph must avoid letting a high-register GEMM state reduce occupancy so much that communication polling stalls, or letting a large shared-memory page pool prevent enough CTAs from residing to cover latency.

CUDA and architecture details change the right budget. Hopper and Blackwell expose different ratios of tensor-core throughput to shared-memory bandwidth, special-function throughput, and asynchronous-copy capability. Persistent kernels may also need explicit SM-count control so they can coexist with NCCL, copy-engine work, or other pipeline stages. Practical systems therefore expose knobs such as worker counts, scheduler counts, SM limits, tensor-core utilization targets, tile shapes, and page sizes, then tune those knobs per model, microbatch, precision, and topology.

Inspect first: Hazy's low-latency megakernel discussion of [resource sharing](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles), Hazy's tensor-parallel [interleaving](https://hazyresearch.stanford.edu/blog/2025-09-28-tp-llama-main) analysis, DeepGEMM's [`set_num_sms` and `set_tc_util`](https://github.com/deepseek-ai/DeepGEMM) utilities, and the [FlashAttention-4](https://arxiv.org/abs/2603.05451) paper for Blackwell-era asymmetric hardware-scaling constraints.

### Why It Matters For Training

Public megakernel work is mostly inference-focused, but the technique is relevant to training subgraphs that have many small operations or communication bubbles: MoE dispatch/expert/combines, post-training losses, small-batch long-context routines, and optimizer substeps. DeepGEMM Mega MoE is already a training-adjacent example of fusing an expert-parallel subgraph.

### Explore Further

- `official blog`: [Look Ma, No Bubbles! Designing a Low-Latency Megakernel for Llama-1B](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles)
- `official repo`: [Hazy Megakernels repository](https://github.com/HazyResearch/Megakernels)
- `official blog`: [Tensor-parallel megakernel post](https://hazyresearch.stanford.edu/blog/2025-09-28-tp-llama-main)
- `official repo`: [Mirage Persistent Kernel](https://github.com/mirage-project/mirage)
- `official repo`: [DeepGEMM Mega MoE](https://github.com/deepseek-ai/DeepGEMM)

### Implementation Examples

- Hazy low-latency megakernel: inspect on-GPU interpreter, shared-memory paging, and counter-based synchronization.
- Hazy tensor-parallel megakernel: inspect cross-GPU compute/memory/communication overlap design.
- Mirage MPK: inspect compiler/runtime approach for generating persistent megakernels.
- DeepGEMM Mega MoE: inspect fused dispatch plus FP8xFP4 expert MLP plus combine.

### Caveats

Training backward is much harder than inference forward. Dropout/RNG, activation saving, recomputation, gradient accumulation, optimizer state, and deterministic debugging complicate whole-model megakernels. Subgraph megakernels are a more realistic near-term target.

## 15. Authoring Abstraction and Automated Search

### Technique

Kernel authoring now often separates the schedule from the low-level CUDA thread details. A kernel author chooses the lowest abstraction that still exposes the needed resources: block-level tensor [DSLs](https://en.wikipedia.org/wiki/Domain-specific_language) (domain-specific languages) for productivity, CuTe/CUTLASS for layout-heavy tensor-core kernels, CUDA-embedded tile libraries for maximum control, or generated search loops for candidate kernels.

### Primitive Techniques

#### Tile [DSL](https://en.wikipedia.org/wiki/Domain-specific_language) (Domain-Specific Language) Schedule Abstraction

A tile DSL schedule abstraction expresses a kernel as a grid of tile programs rather than as individual CUDA threads. The author names block shapes, program IDs, memory scopes, reduction axes, pipeline stages, and tensor operations; the compiler and runtime map those choices onto CTAs, warps, shared memory, registers, and vectorized memory operations. Triton, Pallas, and TileLang all use this idea, with different host-language and backend assumptions.

In LLM training, this is the fastest path for fused elementwise and reduction kernels: RMSNorm/LayerNorm, RoPE, SwiGLU/GeGLU, cross entropy, fused linear-cross-entropy, attention variants, optimizer-adjacent reductions, and shape-specialized post-training losses. It is especially useful when research code changes tensor shapes or fusion boundaries faster than vendor libraries can expose new kernels.

CUDA and platform details matter because tile size controls occupancy, coalescing, register pressure, and shared-memory use. In Triton, `num_warps` determines how many warps cooperate on one program instance, `num_stages` controls software pipelining, and `num_ctas` exposes block-cluster scheduling on SM90+ GPUs. Masked loads are routine for ragged sequence lengths, but they can hide poor memory layout. Pallas has backend-specific behavior: Mosaic GPU targets Hopper and newer GPUs, while TPU kernels require different assumptions about grid execution, VMEM/SMEM, and supported block shapes. TileLang exposes explicit `T.Kernel`, block/thread bindings, copy, GEMM, and pipeline primitives across CUDA, HIP, and CPU targets.

Evidence and implementation links to inspect first: [Triton matmul tutorial](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html), [`triton.Config`](https://triton-lang.org/main/python-api/generated/triton.Config.html), [JAX Pallas](https://docs.jax.dev/en/latest/pallas/index.html), [Pallas quickstart](https://docs.jax.dev/en/latest/pallas/quickstart.html), [TileLang paper](https://arxiv.org/abs/2504.17577), the [TileLang repository](https://github.com/tile-ai/tilelang), and the [Liger Kernel repository](https://github.com/linkedin/Liger-Kernel).

#### Layout Algebra

Layout algebra treats tensor layouts, thread layouts, swizzles, tilers, and tile shapes as programmable objects rather than implicit address arithmetic. In CuTe, a `Layout` maps logical coordinates to linear indices, and operations such as composition, product, and divide build the transformations needed to partition data across tiles, warps, and tensor-core fragments.

LLM training kernels use layout algebra wherever the performance-critical object is not simply a row-major tensor. Examples include FP8/FP4 scale metadata, block-scaled GEMMs, tensor-core operand fragments, shared-memory tiles, TMA descriptors, attention score tiles, and Blackwell tensor-memory paths. Attention kernels need this especially badly: the logical sequence/head layout, the shared-memory swizzle, and the MMA operand layout all have to agree while masking, softmax, and backward accumulation run around them.

CUDA details include operand majorness, shared-memory bank conflicts, static versus dynamic shape information, alignment, and architecture-specific MMA layout requirements. Hopper-era WGMMA and TMA require carefully described shared-memory tiles and descriptors; Blackwell SM100 adds `tcgen05.mma`, tensor memory, and new low-precision formats. A layout object is useful only if it preserves the invariants demanded by the instruction being generated, such as 16-byte shared-memory pointer alignment or the supported swizzle variants for UMMA shared-memory descriptors.

Evidence and implementation links to inspect first: [CuTe layouts](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/01_layout.html), [CuTe layout algebra](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/02_layout_algebra.html), [CUTLASS overview](https://docs.nvidia.com/cutlass/latest/overview.html), [Blackwell SM100 GEMMs](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html), [tcgen05 submodule](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/cute_nvgpu_tcgen05.html), and [FlashAttention-4](https://arxiv.org/abs/2603.05451).

#### Autotuning Parameter Search

Autotuning parameter search compiles and benchmarks a set of candidate schedules, then selects the fastest correct configuration for a shape family and target GPU. The search space usually includes tile sizes, group sizes, warp counts, pipeline stages, block-cluster dimensions, register caps, vectorization width, and sometimes the choice between different algorithmic decompositions.

In LLM training, autotuning is most useful when one operator must serve many hidden sizes, vocabulary sizes, batch sizes, context lengths, or expert counts. A fused loss might prefer one tile at short sequence length and another at large vocabulary; an FP8 GEMM may need a different `BLOCK_K` and scale layout than BF16; an attention variant may need separate schedules for causal, packed, grouped-query, or sliding-window cases. Automated search reduces the amount of hand-testing required when adding a new model family.

CUDA details are easy to get wrong. Benchmarking must include warmup, synchronization, stable input allocation, and enough repetitions to defeat timer noise. Tuning in-place kernels may need reset hooks because each candidate can mutate tensors multiple times. The fastest microbenchmark can still be the wrong training choice if it increases peak memory, causes spills under autograd fusion, overuses shared memory and reduces occupancy, or depends on an architecture-specific feature such as SM90 block clusters.

Evidence and implementation links to inspect first: [`triton.autotune`](https://triton-lang.org/main/python-api/generated/triton.autotune.html), [`triton.Config`](https://triton-lang.org/main/python-api/generated/triton.Config.html), [Triton matmul autotuning examples](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html), [KernelBench](https://arxiv.org/abs/2502.10517), and the [KernelBench repository](https://github.com/ScalingIntelligence/KernelBench).

#### Generated-Code Inspection

Generated-code inspection checks the [IR](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-1/) (intermediate representation), PTX, [SASS](https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html) (NVIDIA native GPU assembly), or compiler logs produced after a DSL lowers a kernel. It answers a different question than a benchmark: not just "is it fast?" but "did the compiler emit the hardware operations the schedule was intended to use, and did it avoid pathological lowering?"

In LLM training kernels, inspection is how teams confirm that a high-level schedule actually becomes tensor-core MMA, asynchronous copy/TMA, vectorized loads, expected barriers, and a sane register allocation. It is also how they catch accidental scalarization of reductions, local-memory spills, excessive predicate work for masks, redundant global loads, missing asynchronous overlap, or unsupported low-precision paths that silently fall back to slower code.

CUDA details include the lowering stack and the instruction names to look for. Triton exposes compiler IR before PTX; CuTe DSL lowers Python/CuTe abstractions to PTX before the CUDA toolkit emits machine code; CUDA C++ and embedded tile libraries can be inspected with `ptxas`, `cuobjdump`, `nvdisasm`, Nsight Compute, and occupancy reports. On Hopper, expected instructions may include WGMMA and TMA; on Blackwell, `tcgen05`/UMMA and tensor-memory operations become important. Spills often show up as local-memory loads/stores, and barrier mistakes may be visible only when reading generated synchronization sequences together with profiler counters.

Evidence and implementation links to inspect first: [CuTe DSL](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl.html), [CUTLASS/CuTe](https://docs.nvidia.com/cutlass/), [FlashAttention-4 blog](https://tridao.me/blog/2026/flash4/), and [ThunderKittens](https://github.com/HazyResearch/ThunderKittens).

#### Custom-Op Integration

Custom-op integration registers a generated, DSL-authored, or CUDA-authored kernel as a framework operator with dispatch, shape checking, dtype/device validation, and training support. The goal is for the model code to keep calling normal PyTorch, JAX, or framework APIs while the runtime routes selected operations to fused kernels.

For LLM training, this is what turns a fast kernel into a usable training primitive. Fused norms, RoPE, SwiGLU, cross entropy, fused linear-cross-entropy, preference-optimization losses, and attention variants need to compose with autograd, `torch.compile`, distributed wrappers, mixed precision, gradient checkpointing, and model patching APIs. Without the integration layer, a kernel remains a benchmark artifact rather than something a trainer can adopt.

CUDA and platform details include stream semantics, tensor contiguity and strides, memory format assumptions, dispatch keys, autograd formulas, saved tensors, meta/fake-tensor behavior for compilation, and ABI/build compatibility. PyTorch custom operators should register backend implementations and training formulas through the operator system rather than bypassing framework dispatch with raw pointer calls. JAX and cuDNN graph integrations have different constraints, but the same principle applies: the framework must understand the operator boundary well enough to schedule, differentiate, and compile around it.

Evidence and implementation links to inspect first: [PyTorch custom C++ and CUDA operators](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html), [PyTorch custom operators landing page](https://docs.pytorch.org/tutorials/advanced/custom_ops_landing_page.html), [`Liger-Kernel`](https://github.com/linkedin/Liger-Kernel), and [cuDNN Frontend graph API](https://docs.nvidia.com/deeplearning/cudnn/frontend/latest/developer/graph-api.html).

#### Correctness Oracle And Benchmark Harness

A correctness oracle and benchmark harness compares an optimized kernel against a trusted reference implementation across randomized shapes, dtypes, strides, masks, and training modes. The oracle establishes numerical validity; the benchmark establishes whether the kernel is actually faster under realistic measurement rules.

In LLM training, the harness must test forward and backward paths, gradient formulas, accumulation dtype, masking behavior, ragged sequence metadata, vocabulary tails, expert routing edge cases, and deterministic modes where required. Many training kernels are memory-saving transformations, so correctness includes not only final values but also whether the autograd graph saves the right tensors and whether recomputation changes numerics within acceptable tolerance.

CUDA details include synchronization around timing, warmup, CUDA Graph capture effects, separate measurement of compile time versus run time, tolerance choices for BF16/FP8/FP4, seeded randomness, and race-sensitive tests. A harness should test non-contiguous inputs when the operator claims stride support, adversarial small and tail shapes, and multiple GPU architectures when the schedule contains SM-specific paths. It should also distinguish a true speedup from a measurement artifact caused by asynchronous launches or missing reference synchronization.

Evidence and implementation links to inspect first: [KernelBench](https://arxiv.org/abs/2502.10517), the [KernelBench repository](https://github.com/ScalingIntelligence/KernelBench), [GPU Kernel Scientist](https://arxiv.org/abs/2506.20807), and [PyTorch `opcheck` guidance](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html).

#### Hardware-Specific Escape Hatch

A hardware-specific escape hatch is the ability to drop below the DSL when a new GPU feature, instruction, memory path, or synchronization primitive is not yet exposed cleanly. The escape hatch can be CUDA C++, CuTe/CUTLASS, inline PTX, embedded tile libraries, vendor graph APIs, or architecture-specific intrinsics.

In LLM training kernels, this is necessary for features that arrive first in hardware and low-level libraries: Hopper WGMMA/TMA, Blackwell `tcgen05.mma`, tensor memory, 2-CTA MMA modes, low-precision scale-factor paths, cluster-level synchronization, and specialized attention backward pipelines. It also matters when a compiler can express the algorithm but not the exact schedule needed to avoid shared-memory pressure, SFU bottlenecks, atomics, or register spills.

CUDA details include compute capability targeting, CUDA toolkit and PTX ISA versions, shared-memory swizzles, `mbarrier` usage, CTA clusters, warpgroup roles, register redistribution, tensor-memory descriptors, and fallback paths for older GPUs. Escape hatches improve peak control but increase maintenance cost: every architecture-specific branch needs tests, build flags, and profiling evidence that it still beats the portable path.

Evidence and implementation links to inspect first: [FlashAttention-4 blog](https://tridao.me/blog/2026/flash4/), [FlashAttention-4 paper](https://arxiv.org/abs/2603.05451), [CUTLASS Blackwell SM100 GEMMs](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html), [tcgen05 submodule](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/cute_nvgpu_tcgen05.html), and [ThunderKittens](https://github.com/HazyResearch/ThunderKittens).

#### LLM-Assisted Candidate Generation

LLM-assisted candidate generation uses a language model to propose CUDA, Triton, [HIP](https://rocm.docs.amd.com/projects/HIP/en/latest/) (Heterogeneous-computing Interface for Portability, AMD ROCm's CUDA-like programming layer), CuTe DSL, or TileLang kernels and schedule variants, then runs those candidates through compile, correctness, and benchmark loops. The useful unit of work is not a single generated kernel but an iterative search process with feedback, filters, and human review.

In LLM training, this can accelerate exploration for compact operators whose reference implementation is clear: activations, reductions, fused losses, layout transforms, small GEMMs, or narrow attention variants. It is less reliable for broad training subgraphs, distributed collectives, or kernels requiring subtle synchronization, but it can still produce useful boilerplate, alternate tilings, and surprising schedule ideas that an expert can profile.

CUDA and platform details shape the guardrails. Generated kernels often fail on address-space qualifiers, synchronization, aliasing, shared-memory sizing, vector alignment, stream behavior, or dtype-specific accumulation. The loop must compile in a sandbox, reject undefined behavior, run broad correctness tests before timing, and store every candidate with metadata so performance improvements are reproducible. Timing-only feedback can guide search, but profiler counters and generated-code inspection are needed before trusting a result in training.

Evidence and implementation links to inspect first: [KernelBench](https://arxiv.org/abs/2502.10517), the [KernelBench repository](https://github.com/ScalingIntelligence/KernelBench), [GPU Kernel Scientist](https://arxiv.org/abs/2506.20807), and the GPU Kernel Scientist [supplementary automation files](https://arxiv.org/abs/2506.20807).

### Why It Matters For Training

Training kernels change frequently as model architectures evolve. Researchers need to prototype fused losses, sparse attention, custom norms, or low-precision layouts without waiting for vendor libraries. But the newest hardware features still require precise control. A mature workflow lets teams prototype in a DSL, benchmark, inspect generated code, and drop lower when needed.

### Explore Further

- `official blog`: [Triton](https://openai.com/research/triton)
- `official docs`: [JAX Pallas](https://docs.jax.dev/en/latest/pallas/index.html)
- `paper/repo`: [TileLang](https://arxiv.org/abs/2504.17577), [TileLang repository](https://github.com/tile-ai/tilelang)
- `official repo`: [ThunderKittens](https://github.com/HazyResearch/ThunderKittens)
- `official docs`: [CUTLASS/CuTe](https://docs.nvidia.com/cutlass/)
- `paper/repo`: [KernelBench](https://arxiv.org/abs/2502.10517), [KernelBench repository](https://github.com/ScalingIntelligence/KernelBench)
- `paper`: [GPU Kernel Scientist](https://arxiv.org/abs/2506.20807)

### Implementation Examples

- Liger Kernel: Triton for memory-bound training operators and patchable model integrations.
- FA4: CuTe-DSL for Blackwell attention while preserving access to architecture-specific primitives.
- TileLang: Pythonic tiled kernels for GEMM, FlashAttention, MLA, and Native Sparse Attention examples.
- ThunderKittens: CUDA-embedded tiles for WGMMA/TMA/TCGEN05 and networking-aware kernels.
- KernelBench: benchmark harness for generated CUDA/DSL kernels with correctness and speed checks.
- GPU Kernel Scientist: iterative LLM-driven optimization loop using timing feedback.

### Caveats

LLM-generated kernels are not yet a replacement for expert kernel engineering. KernelBench reports that even frontier reasoning models often fail to beat PyTorch baselines. The most realistic current use is assisted exploration, boilerplate generation, and schedule search around well-scoped kernels.

## 16. LLM-Assisted Kernel Generation and Optimization Workflows

### Technique

This section uses "generation" to mean LLM-assisted generation of GPU kernel code and optimization hypotheses, not token generation during model inference. The top-level technique is to wrap an LLM or coding agent in a disciplined systems loop: specify a kernel target, retrieve relevant implementation context, generate a candidate in an appropriate authoring stack, compile it, check numerical correctness, benchmark it, profile it, and feed the evidence back into the next attempt.

The key shift is from one-shot code generation to supervised search. KernelBench shows that raw model generations often fail to beat PyTorch baselines, while iterative feedback improves outcomes but remains challenging. GPU Kernel Scientist frames the process as evolutionary experimentation using prior versions and timing feedback. CudaForge frames it as a multi-agent Coder/Judge workflow with hardware feedback such as Nsight Compute metrics. Claude Code and Codex provide practical workflow surfaces for applying the same loop to real repositories: local agents, cloud sandboxes, subagents, hooks, worktrees, approvals, skills, and background tasks.

### Primitive Techniques

#### Kernel Task Factoring

Kernel task factoring decomposes an ambiguous request such as "make attention faster" into a target operator, shape family, dtype set, baseline, correctness oracle, and measurable objective. For training kernels, the factorization must include forward and backward behavior, saved activations, layout constraints, distributed sharding assumptions, and whether speed, memory, determinism, or compile time is the primary objective.

In an LLM-assisted workflow, task factoring is the difference between useful generation and noise. A good task says, for example, "replace a PyTorch `torch.softmax(x @ w)` baseline for shapes $B\in\{1,4\}$, $N=8192$, $D=128$, BF16 on H100, tolerance $10^{-2}$, no dropout, and compare against this PyTorch reference." It should also state disallowed shortcuts, such as changing numerics, assuming contiguous tensors when the caller does not, or hardcoding one hidden size if the training stack needs several.

CUDA details matter because kernel tasks are not interchangeable. A reduction kernel, a fused [CE](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) (cross-entropy) kernel, an attention backward kernel, and a grouped GEMM require different oracles, tolerances, and benchmark shapes. The first evidence to inspect is [KernelBench](https://arxiv.org/abs/2502.10517) and the [KernelBench repository](https://github.com/ScalingIntelligence/KernelBench), because its task format makes the generation problem concrete: a reference PyTorch workload, generated CUDA/DSL kernels, correctness checks, and speed metrics.

#### Evidence Retrieval and Context Packing

Evidence retrieval and context packing assemble the small set of facts that the agent needs before writing code. For GPU kernels this context usually includes the reference implementation, expected shapes and strides, dtype and tolerance policy, hardware target, relevant architecture docs, known neighboring kernels, and a small number of examples in the same authoring style.

Training-kernel generation benefits from retrieval because many failures are caused by missing implicit constraints: launch signatures, `torch.library` registration details, autograd saved tensors, stream semantics, scale metadata layout, or tensor-core alignment. The context should be compact and operational: one good local kernel, one reference test, one benchmark harness, and a short architecture note are often more valuable than a huge dump of CUDA documentation.

Claude Code's project memory and `CLAUDE.md` mechanism are a practical way to keep project-specific build commands, style rules, and invariants near the agent across sessions. Codex Skills serve a similar role for repeatable workflows: package instructions, scripts, and resources so the agent does not rediscover the same process each time. Inspect first: Claude Code [memory](https://code.claude.com/docs/en/memory), Claude Code [settings](https://code.claude.com/docs/en/settings), and the Codex app discussion of [Skills](https://openai.com/index/introducing-the-codex-app/).

#### Candidate Generation Across Authoring Levels

Candidate generation across authoring levels means asking the LLM to choose or propose an implementation layer: PyTorch composition, Triton, CUDA C++, CuTe-DSL, CUTLASS customization, ThunderKittens, TileLang, or a vendor graph API. The workflow should start at the highest level that exposes the needed resources and drop lower only when the optimization requires explicit hardware control.

For LLM training kernels, the best initial candidates are usually narrow: a fused loss, a reduction, a small layout transform, a RoPE/RMSNorm/SwiGLU kernel, or a specialized forward-only prototype. Harder candidates include attention backward, FP8/FP4 scale-layout kernels, MoE dispatch/combine, and anything with cross-rank communication or deterministic reductions. The model may generate syntactically plausible code for all of them, but the validation burden rises sharply as synchronization, mixed precision, and distributed state enter the kernel.

CUDA details include ABI stability and fallback paths. A generated Triton kernel might be ideal for rapid testing, but a Blackwell TMEM/UMMA kernel may require CuTe-DSL or CUDA-level code. A candidate should be tagged by authoring level, hardware target, supported dtypes, shape specialization, and missing backward support. Inspect first: [KernelBench](https://arxiv.org/abs/2502.10517) for generated CUDA/DSL evaluation, [CudaForge](https://arxiv.org/abs/2511.01884) for iterative CUDA generation, and Codex [cloud task delegation](https://developers.openai.com/codex/cloud) for repository-level implementation tasks.

#### Compile-Run-Correctness Loop

The compile-run-correctness loop rejects most bad generated kernels before performance is considered. It compiles the candidate, runs deterministic and randomized tests against a reference implementation, checks tolerances across dtypes, exercises edge shapes, and verifies forward/backward behavior when the operator participates in autograd.

This loop is mandatory for training kernels. A kernel that passes one forward shape can still corrupt gradients, mishandle ragged sequence lengths, race under high occupancy, break CUDA Graph capture, or fail when tensor strides differ. For low precision, correctness checks must explicitly test scale/amax behavior and accepted error envelopes; for stochastic operations such as dropout or stochastic rounding, the oracle must separate distributional behavior from deterministic replay expectations.

Codex CLI approval modes are useful here because they let a user choose how much autonomy the agent has while reading files, editing code, and running commands. Claude Code hooks are useful because they can deterministically run formatters, tests, or blockers after edits rather than relying on the model to remember. Inspect first: Codex CLI [approval modes](https://developers.openai.com/codex/cli), Claude Code [hooks](https://code.claude.com/docs/en/hooks), and PyTorch [custom operator testing](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html).

#### Benchmark and Profiler Feedback

Benchmark and profiler feedback turn a correct candidate into an optimization experiment. Timing alone can identify whether a candidate is worth keeping, but profiler feedback explains why: memory throughput, occupancy, warp stalls, tensor-core utilization, shared-memory bank conflicts, instruction mix, register spills, and kernel launch overhead.

In LLM-assisted workflows, profiler feedback should be summarized into actionable hypotheses rather than pasted raw into a prompt. Examples include "global loads are uncoalesced because the K dimension is stride-major," "register spills begin when `BLOCK_M=128`," "softmax is SFU-bound," or "the generated kernel is launch-bound and should be fused with the preceding cast." CudaForge explicitly integrates hardware feedback such as Nsight Compute metrics, while GPU Kernel Scientist uses timing feedback in an evolutionary loop.

CUDA details include stable benchmarking and attribution. The harness must synchronize correctly, separate compile time from runtime, warm up caches and [JIT](https://en.wikipedia.org/wiki/Just-in-time_compilation) (just-in-time) compilers, and report shapes, hardware, precision, and baseline. Profiling generated kernels on the wrong shape or comparing an eager PyTorch baseline to a graph-captured custom kernel can mislead the agent. Inspect first: [GPU Kernel Scientist](https://arxiv.org/abs/2506.20807), [CudaForge](https://arxiv.org/abs/2511.01884), and the [KernelBench repository](https://github.com/ScalingIntelligence/KernelBench).

#### Iterative Hypothesis Search

Iterative hypothesis search records candidate versions, explains the next experiment, applies one or a few changes, then compares against a baseline and prior best. The agent should not simply keep rewriting the kernel; it should maintain a search log of shapes, code deltas, correctness status, timing, profiler observations, and why a candidate was kept or discarded.

For training kernels, useful hypotheses often map to the primitive techniques earlier in this paper: increase tile size, change vector width, fuse an epilogue, split reductions, store only statistics, alter scale layout, prepack expert tokens, reserve SMs for communication, or specialize for a ragged layout. Hypothesis search is also where agents should learn when to stop: if a PyTorch or vendor library baseline is already near roofline, the right result may be "do not replace this kernel."

CUDA details include search-space control. Letting an agent modify tile shape, dtype, algorithm, and API all at once makes regressions hard to attribute. A better loop changes one dimension at a time, caches compile artifacts where possible, and validates the best candidate on held-out shapes. Inspect first: [GPU Kernel Scientist](https://arxiv.org/abs/2506.20807) for evolutionary refinement and [CudaForge](https://arxiv.org/abs/2511.01884) for Coder/Judge iteration with hardware feedback.

#### Multi-Agent Role Decomposition

Multi-agent role decomposition assigns different agents to different parts of the kernel workflow. A practical split is: Explorer gathers references and constraints; Generator writes the candidate; Verifier owns tests and numerical checks; Profiler interprets timing and Nsight data; Reviewer looks for race conditions, unsupported shapes, and maintainability risks. The benefit is context isolation and specialization, not magic parallelism.

Claude Code exposes this pattern through subagents with separate context windows, descriptions, tool permissions, and project-level configuration. Codex exposes it through multi-agent work in the Codex app, cloud tasks, and worktrees, where agents can work on isolated copies of a repository. For GPU kernels, disjoint write scopes are essential: one agent should not rewrite the same `.cu` file that another agent is benchmarking unless there is an explicit integration step.

CUDA/platform details include build cost and hardware scarcity. Profiling agents should not monopolize expensive GPUs with low-quality candidates; verifier agents should be able to run CPU-side shape checks or compile-only tests before escalating to GPU benchmarking. Inspect first: Claude Code [subagents](https://code.claude.com/docs/en/sub-agents), Claude Code [common workflows](https://code.claude.com/docs/en/common-workflows), the Codex app [multi-agent and worktree workflow](https://openai.com/index/introducing-the-codex-app/), and Codex [cloud tasks](https://developers.openai.com/codex/cloud).

#### Worktree and Sandbox Isolation

Worktree and sandbox isolation keep speculative kernel experiments from corrupting the main training stack. Each candidate branch or worktree can modify build files, generated kernels, and tests independently. The integration branch only receives the candidate after it passes correctness, benchmark, and review gates.

This is particularly important for CUDA kernels because failing candidates can leave generated build artifacts, change compiler flags, alter environment variables, or introduce subtle ABI changes. Isolation also lets several agents test different authoring levels in parallel: one worktree tries a Triton fused op, another tries CUDA C++, and a third modifies the benchmark harness. The result is a controlled tournament rather than a tangled edit history.

Claude Code's common workflows explicitly recommend Git worktrees for parallel sessions. Codex app similarly emphasizes built-in worktrees, isolated agent work, reviewable diffs, and cloud sandboxes. Codex CLI also exposes local approval modes and sandboxed full-auto operation. Inspect first: Claude Code [parallel worktree workflow](https://code.claude.com/docs/en/common-workflows), Codex app [worktrees](https://developers.openai.com/codex/app/worktrees), Codex [cloud sandbox environments](https://developers.openai.com/codex/cloud), and Codex CLI [sandbox/approval modes](https://developers.openai.com/codex/cli).

#### Deterministic Guardrails and Hooks

Deterministic guardrails are non-LLM mechanisms that force repeated checks: format, lint, static analysis, compile, unit tests, benchmark smoke tests, forbidden-file policies, and source-annotation requirements. Hooks, CI jobs, and preconfigured agent permissions reduce reliance on the model's memory or good habits.

In training-kernel work, guardrails should include at least compile checks for every targeted compute capability, correctness tests for forward/backward, tolerance tables for BF16/FP8/FP4, benchmark metadata capture, and checks that performance numbers include hardware and shape. For multi-agent workflows, guardrails should also prevent agents from editing generated benchmark baselines or deleting failing tests to make a candidate look good.

Claude Code hooks provide explicit lifecycle events such as `PreToolUse`, `PostToolUse`, `Stop`, and `SubagentStop`, which can run commands and block progress on failures. Codex emphasizes approval modes, sandboxing, reviewable diffs, and code review workflows. Inspect first: Claude Code [hooks reference](https://code.claude.com/docs/en/hooks), Claude Code [hooks guide](https://code.claude.com/docs/en/hooks-guide), Codex CLI [approval modes](https://developers.openai.com/codex/cli), and Codex [GitHub/code review workflows](https://developers.openai.com/codex/cloud).

#### Repository Memory and Workflow Codification

Repository memory and workflow codification convert successful agent behavior into durable project assets. Instead of repeatedly prompting "run these tests, use this benchmark, do not touch this baseline," teams store build commands, benchmark scripts, known-good examples, source conventions, and safety rules in repository files or reusable skills.

For kernel work, useful memory includes target GPU inventory, compiler versions, supported architectures, canonical shape sets, tolerance policy, profiling commands, expected metric names, and "known traps" such as unsupported tensor strides or required scale swizzles. This makes Claude/Codex sessions more reproducible and lets new agents start from the same operational contract.

Claude Code supports project memory through `CLAUDE.md`, settings files, and project-level subagents. Codex supports reusable Skills and app/CLI/project configuration. In both cases, the memory should be concise and testable; a ten-page instruction file that never runs a check is weaker than a short rule plus a hook or script. Inspect first: Claude Code [memory](https://code.claude.com/docs/en/memory), Claude Code [settings](https://code.claude.com/docs/en/settings), Codex app [Skills](https://openai.com/index/introducing-the-codex-app/), and OpenAI Academy [Codex workflows](https://openai.com/academy/codex/).

#### GitHub and CI Agent Workflows

GitHub and [CI](https://docs.github.com/actions) (continuous integration) agent workflows move generation and optimization from an interactive terminal into reviewable pull requests, issue-driven tasks, and scheduled checks. An agent can create a candidate branch, add benchmark artifacts, open a [PR](https://docs.github.com/pull-requests) (pull request), respond to review comments, or automatically inspect failures after CI runs.

For LLM training kernels, GitHub/CI workflows are most useful after the local candidate loop has a plausible winner. CI can compile against multiple CUDA versions, test multiple GPUs if runners exist, run correctness suites, and preserve benchmark output. The PR becomes the evidence bundle: what changed, which shapes improved, which shapes regressed, which hardware was used, and what risks remain.

Claude Code GitHub Actions supports `@claude`-style issue and PR workflows and can follow project standards such as `CLAUDE.md`. Codex cloud can read, modify, run code, draft PRs, review diffs, add tests, and fix bugs inside isolated task containers. Inspect first: [Claude Code GitHub Actions](https://code.claude.com/docs/en/github-actions), Codex [cloud overview](https://developers.openai.com/codex/cloud), and [Introducing Codex](https://openai.com/index/introducing-codex/).

#### Human Review and Acceptance Criteria

Human review and acceptance criteria define when an LLM-generated kernel is allowed into a training stack. A candidate should pass correctness, benchmark, maintainability, portability, and observability gates. It should also explain why it is faster in terms of the earlier techniques in this paper: fewer HBM round trips, better tensor-core occupancy, fewer launches, improved scale layout, or better communication overlap.

For training kernels, acceptance criteria should include backward correctness, determinism expectations, supported shapes and dtypes, integration with checkpointing and distributed wrappers, failure fallback, and source-level readability. Public LLM-kernel work supports a conservative stance: LLMs can generate useful candidates and optimization hypotheses, but they still need expert validation before they become infrastructure.

CUDA details include long-tail failure modes. A generated kernel may pass small tests and fail at production sequence length, may be faster only because it skips an edge case, or may regress after a CUDA toolkit update. Human review should inspect generated code, profiler evidence, and benchmark methodology. Inspect first: [KernelBench](https://arxiv.org/abs/2502.10517) for benchmark difficulty, [CudaForge](https://arxiv.org/abs/2511.01884) for multi-agent claims and hardware feedback, and official Claude/Codex workflow docs for review, permissions, and PR handling.

### Why It Matters For Training

Training-kernel authoring is expensive because the search space is both algorithmic and hardware-specific. LLM-assisted workflows can widen exploration: they can try more tilings, generate more tests, summarize profiler evidence, and keep several implementation paths alive in parallel. The value is especially high for narrow kernels, new fused losses, custom post-training objectives, layout transforms, and benchmark harness work.

The risk is that generated kernels are easy to trust too early. A model can produce code that compiles, passes one shape, and still violates a training invariant. The right workflow treats LLMs as search accelerators under deterministic verification, not as replacements for correctness or performance evidence.

### Explore Further

- `paper/repo`: [KernelBench](https://arxiv.org/abs/2502.10517), [KernelBench repository](https://github.com/ScalingIntelligence/KernelBench)
- `paper`: [GPU Kernel Scientist](https://arxiv.org/abs/2506.20807)
- `paper/repo`: [CudaForge](https://arxiv.org/abs/2511.01884), [CudaForge repository](https://github.com/OptimAI-Lab/CudaForge)
- `official docs`: [Claude Code common workflows](https://code.claude.com/docs/en/common-workflows), [subagents](https://code.claude.com/docs/en/sub-agents), [hooks](https://code.claude.com/docs/en/hooks), [GitHub Actions](https://code.claude.com/docs/en/github-actions)
- `official docs/blog/report`: [Codex cloud](https://developers.openai.com/codex/cloud), [Codex app](https://openai.com/index/introducing-the-codex-app/), [Codex CLI](https://developers.openai.com/codex/cli), [Introducing Codex](https://openai.com/index/introducing-codex/)

### Implementation Examples

- Claude Code local kernel loop: store GPU/build instructions in `CLAUDE.md`, create project subagents such as `kernel-explorer`, `kernel-generator`, `kernel-verifier`, and `kernel-profiler`, add hooks that run compile/test/format commands after edits, and use worktrees for parallel candidate branches.
- Claude Code CI loop: use GitHub Actions for issue-to-PR or PR-review flows, then require benchmark artifacts and hardware metadata before accepting a generated kernel.
- Codex cloud loop: delegate a narrow kernel task to an isolated cloud environment, have Codex add tests and draft a PR, and review the diff plus benchmark output before merge.
- Codex app multi-agent loop: run one agent per candidate strategy in isolated worktrees, use Skills for repeatable benchmark/profiling procedure, and keep an integration thread that compares candidate evidence.
- Codex [CLI](https://developers.openai.com/codex/cli) (command-line interface) local loop: use Suggest mode for exploration/review, Auto Edit for bounded refactors, and Full Auto only inside a sandboxed repo with tests and benchmark commands already codified.
- Research loop: use KernelBench-style tasks for benchmark comparability, GPU Kernel Scientist-style version archives for evolutionary refinement, and CudaForge-style Coder/Judge separation when profiler feedback is available.

### Caveats

Claude and Codex workflow documentation describes agent capabilities and recommended usage patterns, not peer-reviewed evidence that those tools can autonomously produce production-grade LLM training kernels. Treat Claude/Codex sections as practical workflow design, not as benchmark claims. For performance claims, rely on kernel-generation papers, public benchmark repositories, and your own reproducible hardware measurements.

LLM-assisted optimization also has cost and safety limits. Running many generated CUDA kernels can consume expensive GPU time, trigger driver instability, or hide subtle numerical failures behind loose tolerances. The safest production pattern is staged autonomy: allow broad exploration in sandboxes, require deterministic correctness gates, require human review for CUDA synchronization and numerical assumptions, and promote only evidence-backed candidates.

## 17. Technique Cross-Reference Matrix

| Training Kernel / Subsystem | Primary Techniques | Explore First | Implementation Paths | Evidence Grade |
|---|---|---|---|---|
| Dense projection GEMMs | Low-precision scaling, asynchronous tensor-core tiling, authoring abstraction | [CUTLASS/CuTe](https://docs.nvidia.com/cutlass/), [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) | DeepGEMM $\mathrm{FP8}$ GEMM, Transformer Engine, CUTLASS/CuTe kernels | `official docs/repo` |
| Attention forward | IO-aware tiling, online softmax, softmax bottleneck reduction, async pipelining | [FA1](https://arxiv.org/abs/2205.14135), [FA3](https://arxiv.org/abs/2407.08608), [FA4](https://arxiv.org/abs/2603.05451) | `flash-attention`, cuDNN SDPA, TileLang attention | `paper` + `official repo/docs` |
| Attention backward | Recomputation, deterministic reductions, TMEM/$2$-CTA scheduling | [FA2](https://arxiv.org/abs/2307.08691), [FA4 blog](https://tridao.me/blog/2026/flash4/) | FA backward kernels, cuDNN SDPA backward | `paper` + `official blog/docs` |
| Hybrid/sparse attention | Long-context sparse attention, metadata scheduling, variable-length load balancing | [DeepSeek-V2](https://arxiv.org/abs/2405.04434), [DeepSeek-V4](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro) | TileLang MLA/NSA examples, SGLang/Miles integrations, cuDNN NSA | `paper` + `official docs/blog/report` + `third-party integration` |
| Norm/RoPE/SwiGLU | Fused memory-bound operators, low-precision stability | [Liger](https://github.com/linkedin/Liger-Kernel), [Apex LayerNorm](https://nvidia.github.io/apex/layernorm.html) | Liger Triton kernels, Apex/Megatron fused norms | `official repo/docs` + `paper` |
| Cross-entropy and post-training losses | Fusion, chunking, logits avoidance | [Liger paper](https://arxiv.org/abs/2410.10989) | Liger fused linear CE, [DPO](https://arxiv.org/abs/2305.18290) (Direct Preference Optimization)/[ORPO](https://arxiv.org/abs/2403.07691) (Odds Ratio Preference Optimization)/[SimPO](https://arxiv.org/abs/2405.14734) (Simple Preference Optimization)/[KTO](https://arxiv.org/abs/2402.01306) (Kahneman-Tversky Optimization)/[JSD](https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence) (Jensen-Shannon divergence) kernels | `paper` + `official repo` |
| MoE dispatch/combine | Routing as data layout, communication overlap | [DeepEP](https://github.com/deepseek-ai/DeepEP), [DeepSeek-V3](https://arxiv.org/abs/2412.19437) | DeepEP all-to-all kernels, NCCL/NVSHMEM substrate | `official repo` + `paper` |
| Expert matmul | Grouped GEMM, low-precision scaling, MoE mega-kernel fusion | [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM), [MegaBlocks](https://arxiv.org/abs/2211.15841) | DeepGEMM grouped GEMM/Mega MoE, MegaBlocks sparse kernels | `official repo` + `paper` |
| Distributed training step | Communication overlap, variable-length scheduling | [MegaScale](https://arxiv.org/abs/2402.15627), [Megatron-Core](https://developer.nvidia.com/megatron-core) | DualPipe, Dynamic [CP](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/context_parallel.html) (context parallelism), NCCL/NVSHMEM, [ZeRO](https://deepspeed.readthedocs.io/en/stable/zero3.html) (Zero Redundancy Optimizer)/[FSDP](https://docs.pytorch.org/docs/stable/fsdp.html) (Fully Sharded Data Parallel) | `paper` + `official docs/repo` |
| Optimizer step | Fused optimizer, state compression, matrix optimizer kernels | [Apex FusedAdam](https://nvidia.github.io/apex/optimizers.html), [COAT](https://arxiv.org/abs/2410.19313), [Muon post](https://developer.nvidia.com/blog/advancing-emerging-optimizers-for-accelerated-llm-training-with-nvidia-megatron/) | Apex multi-tensor Adam, COAT, Megatron Muon | `official docs/blog/report` + `paper` |
| Repeated subgraphs | Persistent/megakernel scheduling | [Hazy Megakernels](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles), Mirage [MPK](https://github.com/mirage-project/mirage) (Mirage Persistent Kernel) | Hazy Megakernels, Mirage, DeepGEMM Mega MoE | `official blog/repo` |
| LLM-assisted kernel workflow | Generation loop, correctness oracle, profiler feedback, multi-agent review | [KernelBench](https://arxiv.org/abs/2502.10517), [GPU Kernel Scientist](https://arxiv.org/abs/2506.20807), [CudaForge](https://arxiv.org/abs/2511.01884) | Claude Code subagents/hooks/GitHub Actions, Codex cloud/app/CLI, worktrees, [CI](https://docs.github.com/actions) (continuous integration) benchmark gates | `paper` + `official docs/blog/report` |

## 18. Open Questions and Evidence Boundaries

| Question | Public Evidence | Current Status |
|---|---|---|
| How do closed frontier labs author full production training kernels beyond attention? | OpenAI released Triton publicly, but not detailed frontier training kernels. Meta's Llama 3 report discusses infrastructure and collectives, not a complete custom-kernel release. | Mostly unpublished |
| Do DeepSeek-V4 pretraining kernels match public DeepGEMM/DeepEP implementations? | Official V4 card establishes architecture and precision. DeepGEMM/DeepEP expose related kernels. SGLang/Miles reports runtime integrations. | Partly inferred |
| How is $\mathrm{FP4}$ expert-weight training handled? | V4 official card reports $\mathrm{FP4}$ expert parameters for instruct models. DeepGEMM exposes $\mathrm{FP8}\times\mathrm{FP4}$ kernels. | Emerging and not fully disclosed |
| Will megakernels move into full training backward passes? | Hazy/Mirage focus mostly inference. DeepGEMM Mega MoE fuses a distributed MoE subgraph. | Research |
| How should Muon kernels be standardized? | NVIDIA reports Megatron integration and distributed Newton-Schulz modes. | Emerging |
| How portable can high-end kernels be across CUDA, ROCm, TPU, and [NPU](https://en.wikipedia.org/wiki/AI_accelerator) (neural processing unit) backends? | Pallas, TileLang, ROCm FlashAttention backends, and SGLang/Miles integrations show movement, but FA4-class kernels remain hardware-specific. | Open |
| Can Claude, Codex, or other coding agents independently author production training kernels? | Public workflow docs show agentic coding patterns. KernelBench, GPU Kernel Scientist, and CudaForge show measurable progress on generated kernels. | Useful but not autonomous enough for unsupervised production adoption |

## 19. Conclusion

The most important LLM training kernel advances are techniques, not brand names. The public evidence points to a coherent playbook.

First, reduce HBM traffic before chasing FLOPs. IO-aware attention, fused losses, norm fusion, and chunked post-training objectives all win by deleting intermediate tensors.

Second, exploit asynchronous hardware deliberately. Hopper and Blackwell require kernels that overlap tensor cores, TMA/UMMA, softmax, shared-memory traffic, and sometimes communication.

Third, treat precision as layout. FP8 and FP4 require scale metadata, amax policy, packed formats, scale transformations, and stability-aware activation/optimizer design.

Fourth, make communication part of the operator. Sparse MoE training turns all-to-all into a data-layout primitive, and expert GEMM cannot be optimized independently from dispatch/combine.

Fifth, move scheduling closer to the GPU when launch boundaries or pipeline bubbles dominate. Megakernels are not yet the default for training, but MoE mega-kernels and persistent scheduling show where the field is going.

Finally, choose authoring tools by resource requirement. Triton and TileLang are strong for productive fused kernels. CuTe/CUTLASS, CuTe-DSL, ThunderKittens, and CUDA remain necessary for the newest hardware features. LLM-assisted kernel search is useful as an exploration loop, especially when paired with Claude/Codex-style subagents, hooks, worktrees, skills, sandboxes, CI, and code review. The acceptance bar stays the same: a generated kernel is only as good as its correctness oracle, profiling evidence, and human review of synchronization and numerical assumptions.

## Source Notes and Bibliography

- <a id="S1"></a>**[S1]** Tri Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness," arXiv:2205.14135, 2022. https://arxiv.org/abs/2205.14135
- <a id="S2"></a>**[S2]** Tri Dao, "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning," arXiv:2307.08691, 2023. https://arxiv.org/abs/2307.08691
- <a id="S3"></a>**[S3]** Jay Shah et al., "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision," arXiv:2407.08608, 2024. https://arxiv.org/abs/2407.08608
- <a id="S4"></a>**[S4]** PyTorch Blog, "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision," 2024. https://docs.pytorch.org/blog/flashattention-3/
- <a id="S5"></a>**[S5]** Ted Zadouri et al., "FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling," arXiv:2603.05451, 2026. https://arxiv.org/abs/2603.05451
- <a id="S6"></a>**[S6]** Dao-AILab, `flash-attention` repository and FA4 blog/code links. https://github.com/Dao-AILab/flash-attention and https://tridao.me/blog/2026/flash4/
- <a id="S7"></a>**[S7]** Pin-Lun Hsu et al., "Liger-Kernel: Efficient Triton Kernels for LLM Training," arXiv:2410.10989, 2024. https://arxiv.org/abs/2410.10989
- <a id="S8"></a>**[S8]** LinkedIn, `Liger-Kernel` repository and documentation. https://github.com/linkedin/Liger-Kernel and https://linkedin.github.io/Liger-Kernel/
- <a id="S9"></a>**[S9]** DeepSeek-AI, "DeepSeek-V3 Technical Report," arXiv:2412.19437, 2024/2025. https://arxiv.org/abs/2412.19437
- <a id="S10"></a>**[S10]** DeepSeek-AI, `DeepSeek-V3` repository. https://github.com/deepseek-ai/DeepSeek-V3
- <a id="S11"></a>**[S11]** DeepSeek-AI, `DeepGEMM`: FP8/FP4 GEMM and Mega MoE kernels. https://github.com/deepseek-ai/DeepGEMM
- <a id="S12"></a>**[S12]** DeepSeek-AI, `DeepEP`: expert-parallel communication library. https://github.com/deepseek-ai/DeepEP
- <a id="S13"></a>**[S13]** DeepSeek-AI, "DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence," official model card/report. https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro
- <a id="S14"></a>**[S14]** NVIDIA Transformer Engine documentation, release 2.8. https://docs.nvidia.com/deeplearning/transformer-engine-releases/release-2.8/user-guide/index.html
- <a id="S15"></a>**[S15]** NVIDIA Transformer Engine developer docs, FP8/FP4 primer. https://nvidia.github.io/TransformerEngine/examples/fp8_primer.html
- <a id="S16"></a>**[S16]** Haocheng Xi et al., "COAT: Compressing Optimizer states and Activation for Memory-Efficient FP8 Training," arXiv:2410.19313, 2024; NVIDIA project page. https://arxiv.org/abs/2410.19313 and https://research.nvidia.com/labs/eai/publication/coat/
- <a id="S17"></a>**[S17]** NVIDIA cuDNN Frontend, Attention operation documentation. https://docs.nvidia.com/deeplearning/cudnn/frontend/latest/operations/Attention.html
- <a id="S18"></a>**[S18]** NVIDIA CUTLASS/CuTe documentation and cuDNN FE-OSS (Frontend Open-Source Software) APIs. https://docs.nvidia.com/cutlass/ and https://docs.nvidia.com/deeplearning/cudnn/frontend/
- <a id="S19"></a>**[S19]** OpenAI, "Introducing Triton: Open-source GPU programming for neural networks," 2021. https://openai.com/research/triton
- <a id="S20"></a>**[S20]** JAX documentation, "Pallas: a JAX kernel language." https://docs.jax.dev/en/latest/pallas/index.html
- <a id="S21"></a>**[S21]** JAX documentation, Pallas Mosaic GPU and TPU kernel guides. https://docs.jax.dev/en/latest/pallas/gpu/reference.html and https://docs.jax.dev/en/latest/pallas/tpu/
- <a id="S22"></a>**[S22]** Tile-AI, `tilelang` repository/docs and "TileLang: A Composable Tiled Programming Model for AI Systems," arXiv:2504.17577. https://github.com/tile-ai/tilelang and https://arxiv.org/abs/2504.17577
- <a id="S23"></a>**[S23]** HazyResearch, `ThunderKittens` repository. https://github.com/HazyResearch/ThunderKittens
- <a id="S24"></a>**[S24]** NVIDIA Megatron-Core developer information. https://developer.nvidia.com/megatron-core
- <a id="S25"></a>**[S25]** NVIDIA NCCL documentation. https://docs.nvidia.com/deeplearning/nccl/
- <a id="S26"></a>**[S26]** Ziheng Jiang et al., "MegaScale: Scaling Large Language Model Training to More Than 10,000 GPUs," arXiv:2402.15627, 2024. https://arxiv.org/abs/2402.15627
- <a id="S27"></a>**[S27]** PyTorch Fully Sharded Data Parallel documentation. https://docs.pytorch.org/docs/stable/fsdp.html
- <a id="S28"></a>**[S28]** DeepSpeed ZeRO documentation. https://deepspeed.readthedocs.io/en/stable/zero3.html
- <a id="S29"></a>**[S29]** NVIDIA NVSHMEM documentation/developer page. https://developer.nvidia.com/nvshmem and https://docs.nvidia.com/nvshmem/
- <a id="S30"></a>**[S30]** NVIDIA Technical Blog, "Speeding Up Variable-Length Training with Dynamic Context Parallelism and NVIDIA Megatron Core," 2026. https://developer.nvidia.com/blog/speeding-up-variable-length-training-with-dynamic-context-parallelism-and-nvidia-megatron-core/
- <a id="S31"></a>**[S31]** Trevor Gale et al., "MegaBlocks: Efficient Sparse Training with Mixture-of-Experts," arXiv:2211.15841 / MLSys 2023. https://arxiv.org/abs/2211.15841
- <a id="S32"></a>**[S32]** MLSys proceedings page for MegaBlocks. https://proceedings.mlsys.org/paper_files/paper/2023/hash/5a54f79333768effe7e8927bcccffe40-Abstract-mlsys2023.html
- <a id="S33"></a>**[S33]** LMSYS/SGLang/Miles, "DeepSeek-V4 on Day 0: From Fast Inference to Verified RL with SGLang and Miles," 2026. https://www.lmsys.org/blog/2026-04-25-deepseek-v4/
- <a id="S34"></a>**[S34]** NVIDIA Technical Blog, "Build with DeepSeek V4 Using NVIDIA Blackwell and GPU-Accelerated Endpoints," 2026. https://developer.nvidia.com/blog/build-with-deepseek-v4-using-nvidia-blackwell-and-gpu-accelerated-endpoints/
- <a id="S35"></a>**[S35]** NVIDIA Apex optimizer documentation, FusedAdam. https://nvidia.github.io/apex/optimizers.html
- <a id="S36"></a>**[S36]** NVIDIA Apex fused layer norm documentation. https://nvidia.github.io/apex/layernorm.html
- <a id="S37"></a>**[S37]** Megatron-Core fused layer norm API documentation. https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.fusions.fused_layer_norm.html
- <a id="S38"></a>**[S38]** Changho Hwang et al., "Tutel: Adaptive Mixture-of-Experts at Scale," arXiv:2206.03382 / MLSys 2023. https://arxiv.org/abs/2206.03382
- <a id="S39"></a>**[S39]** HazyResearch, "Look Ma, No Bubbles! Designing a Low-Latency Megakernel for Llama-1B," 2025. https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles
- <a id="S40"></a>**[S40]** HazyResearch `Megakernels` repository and Mirage MPK (Mirage Persistent Kernel) repository. https://github.com/HazyResearch/Megakernels and https://github.com/mirage-project/mirage
- <a id="S41"></a>**[S41]** HazyResearch, "We Bought the Whole GPU, So We're Damn Well Going to Use the Whole GPU," tensor-parallel megakernel post, 2025. https://hazyresearch.stanford.edu/blog/2025-09-28-tp-llama-main
- <a id="S42"></a>**[S42]** NVIDIA Blackwell/CUDA architecture and CUTLASS documentation, including CuTe and Blackwell kernel abstractions. https://docs.nvidia.com/cutlass/
- <a id="S43"></a>**[S43]** NVIDIA cuDNN Frontend FE-OSS (Frontend Open-Source Software) Native Sparse Attention and attention APIs. https://docs.nvidia.com/deeplearning/cudnn/frontend/
- <a id="S44"></a>**[S44]** NVIDIA Technical Blog, "Advancing Emerging Optimizers for Accelerated LLM Training with NVIDIA Megatron," 2026. https://developer.nvidia.com/blog/advancing-emerging-optimizers-for-accelerated-llm-training-with-nvidia-megatron/
- <a id="S45"></a>**[S45]** "Scaling Muon: A 2x More Efficient Optimizer for Large Language Models," arXiv:2502.16982, 2025. https://arxiv.org/abs/2502.16982
- <a id="S46"></a>**[S46]** Muon optimizer public research summaries and scalable LLM training reports; use original arXiv where possible. https://arxiv.org/abs/2502.16982
- <a id="S47"></a>**[S47]** DeepSeek-AI, "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model," arXiv:2405.04434, 2024. https://arxiv.org/abs/2405.04434
- <a id="S48"></a>**[S48]** DeepSeek-AI, DeepSeek-V2 model card. https://huggingface.co/deepseek-ai/DeepSeek-V2
- <a id="S49"></a>**[S49]** Zhenda Xie et al., "mHC: Manifold-Constrained Hyper-Connections," arXiv:2512.24880, 2025. https://arxiv.org/abs/2512.24880
- <a id="S50"></a>**[S50]** Benjamin F. Spector et al., "ThunderKittens: Simple, Fast, and Adorable AI Kernels," arXiv:2410.20399, 2024. https://arxiv.org/abs/2410.20399
- <a id="S51"></a>**[S51]** Meta AI, "The Llama 3 Herd of Models," arXiv:2407.21783, 2024. https://arxiv.org/abs/2407.21783
- <a id="S52"></a>**[S52]** Anne Ouyang et al., "KernelBench: Can LLMs Write Efficient GPU Kernels?", arXiv:2502.10517, 2025; ScalingIntelligence repository. https://arxiv.org/abs/2502.10517 and https://github.com/ScalingIntelligence/KernelBench
- <a id="S53"></a>**[S53]** Martin Andrews and Sam Witteveen, "GPU Kernel Scientist: An LLM-Driven Framework for Iterative Kernel Optimization," arXiv:2506.20807, 2025. https://arxiv.org/abs/2506.20807
- <a id="S54"></a>**[S54]** NVIDIA Transformer Engine, "FP8 Delayed Scaling" documentation. https://nvidia.github.io/TransformerEngine/features/low_precision_training/fp8_delayed_scaling/fp8_delayed_scaling.html
- <a id="S55"></a>**[S55]** NVIDIA cuDNN Frontend FE-OSS (Frontend Open-Source Software) Native Sparse Attention API documentation. https://docs.nvidia.com/deeplearning/cudnn/frontend/latest/fe-oss-apis/nsa.html
- <a id="S56"></a>**[S56]** NVIDIA Megatron-Core, "Parallelism Strategies Guide." https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html
- <a id="S57"></a>**[S57]** Hugging Face TRL documentation, "Liger Kernel Integration." https://huggingface.co/docs/trl/liger_kernel_integration
- <a id="S58"></a>**[S58]** PyTorch documentation, "torch.utils.checkpoint." https://docs.pytorch.org/docs/stable/checkpoint.html
- <a id="S59"></a>**[S59]** Native Sparse Attention public material: paper arXiv:2502.11089 and `fla-org/native-sparse-attention`. https://arxiv.org/abs/2502.11089 and https://github.com/fla-org/native-sparse-attention
- <a id="S60"></a>**[S60]** DeepSeek-AI, `FlashMLA` optimized MLA attention repository. https://github.com/deepseek-ai/FlashMLA
- <a id="S61"></a>**[S61]** NVIDIA CUDA platform documentation used for low-level primitives: PTX ISA, CUDA C Programming Guide, cuBLASLt narrow precision, Cooperative Groups, and Programmatic Dependent Launch. https://docs.nvidia.com/cuda/
- <a id="S62"></a>**[S62]** NVIDIA NeMo and Megatron Bridge documentation for activation recomputation, communication overlap, and emerging optimizers. https://docs.nvidia.com/nemo-framework/ and https://docs.nvidia.com/nemo/megatron-bridge/
- <a id="S63"></a>**[S63]** Triton language documentation: matmul tutorial, `triton.Config`, and `triton.autotune`. https://triton-lang.org/main/
- <a id="S64"></a>**[S64]** PyTorch custom operator and reproducibility documentation. https://docs.pytorch.org/tutorials/advanced/custom_ops_landing_page.html and https://docs.pytorch.org/docs/stable/notes/randomness.html
- <a id="S65"></a>**[S65]** DeepSeek-AI, `DualPipe` pipeline-overlap repository. https://github.com/deepseek-ai/DualPipe
- <a id="S66"></a>**[S66]** Databricks, `megablocks` dropless MoE implementation repository. https://github.com/databricks/megablocks
- <a id="S67"></a>**[S67]** Zijian Zhang et al., "CudaForge: An Agent Framework with Hardware Feedback for CUDA Kernel Optimization," arXiv:2511.01884, 2025; `OptimAI-Lab/CudaForge` repository. https://arxiv.org/abs/2511.01884 and https://github.com/OptimAI-Lab/CudaForge
- <a id="S68"></a>**[S68]** Anthropic, "Best Practices for Claude Code." https://www.anthropic.com/engineering/claude-code-best-practices
- <a id="S69"></a>**[S69]** Anthropic Claude Code documentation: common workflows, subagents, hooks, settings, memory, and GitHub Actions. https://code.claude.com/docs/en/common-workflows, https://code.claude.com/docs/en/sub-agents, https://code.claude.com/docs/en/hooks, https://code.claude.com/docs/en/settings, https://code.claude.com/docs/en/memory, and https://code.claude.com/docs/en/github-actions
- <a id="S70"></a>**[S70]** OpenAI Codex cloud documentation. https://developers.openai.com/codex/cloud
- <a id="S71"></a>**[S71]** OpenAI, "Introducing the Codex app," 2026. https://openai.com/index/introducing-the-codex-app/
- <a id="S72"></a>**[S72]** OpenAI Developers, "Codex CLI (command-line interface)." https://developers.openai.com/codex/cli
- <a id="S73"></a>**[S73]** OpenAI, "Introducing Codex," 2025. https://openai.com/index/introducing-codex/

## Verification Notes

- The document is technique-first: there is no standalone framework catalog. Frameworks appear only as exploration links or implementation examples under a technique.
- Each top-level numbered technique section includes a primitive-technique decomposition with descriptions, training usage, and evidence or implementation links.
- Primitive-technique subsections use the evidence-grade convention in the "How to Read This Paper" section; architecture-implied DeepSeek-V4 and mHC kernel details are kept conservative.
- The LLM-assisted generation section treats Claude/Codex content as workflow evidence from official product documentation, not as benchmark evidence of autonomous kernel quality.
- DeepSeek-V4 is treated conservatively. Model-level claims use the official Hugging Face model card/report. Runtime kernel claims use third-party integration evidence or are labeled as inferred.
- Benchmark numbers are included only where a cited source reports enough context at a high level. The paper avoids comparing numbers across sources as if they used identical shapes or methodology.
- Inference-only megakernel projects are included only for transferable scheduling patterns, not as evidence of current full-training megakernels.
- Review pass on 2026-04-27 added an evidence-and-benchmark normalization table, normalized evidence-grade labels, corrected HCA to "Heavily Compressed Attention," and updated Claude Code/Codex documentation links to current canonical URLs.
- Acronym pass on 2026-04-27 added inline definitions and learning links for less-common kernel, architecture, precision, distributed-systems, optimizer, and agent-workflow acronyms while leaving common terms such as LLM and GPU readable.
- Math-formula pass on 2026-04-27 added LaTeX notation for dense attention, online softmax, attention backward, quantization scales, normalization, RoPE, SwiGLU, fused cross-entropy, DPO, MoE routing, sparse attention, mHC/Sinkhorn, AdamW, Muon/Newton-Schulz, distributed Gram reductions, and variable-length scheduling.
- Inline math normalization pass on 2026-04-27 uses `$...$` for inline math and `$$...$$` for displayed equations/diagrams, so tensor terms such as `$O_{I,:}$` are explicitly marked rather than appearing as plain parenthesized notation.
- Code-link readability pass on 2026-04-27 replaced path-like source labels with descriptive labels such as "FA4 pipeline helpers," "Liger RMSNorm kernel," and "FlashAttention Python interface" while preserving the original target URLs.
- Tensor-diagram pass on 2026-04-27 added LaTeX array diagrams for attention tile streaming, tiled GEMM, shared-memory pipelining, block-scaled payload/scale layout, fused vocabulary loss streaming, MoE packing, grouped expert GEMM, sparse-attention metadata, latent KV compression, mHC residual streams, distributed Gram reductions, and THD packing.
- Symbol-explanation pass on 2026-04-27 added explicit `Symbols:` or `Diagram symbols:` notes for displayed formula and diagram groups, including optimizer notation such as $\theta_{t+1}$, $m_t$, and $\hat{m}_t$.
- Ping-pong scheduling pass on 2026-04-28 added explicit two-slot double-buffering and native GPU-side ping-pong coverage, including tile schedules, persistent-kernel schedules, synchronization contracts, and evidence links.
