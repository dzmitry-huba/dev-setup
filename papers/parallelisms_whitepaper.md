# End-to-End Parallelisms for LLMs: A Self-Contained Study Guide

## Reader contract
This document is intentionally exhaustive. It is written so a reader can study LLM parallelism end-to-end without external context, while still aligning to the curated source map in `papers/parallelism_reference.md`.

## Unified notation and symbols
- Global batch $B$, microbatches $m$, sequence length $S$, hidden size $H$, layers $L$, vocab $V$, heads $N_h$, head dim $d_h$ with $H=N_h d_h$.
- Mesh sizes: $P_{dp},P_{tp},P_{pp},P_{cp},P_{ep}$ and world size $P=\prod P_i$.
- Per-rank local batch $b=B/P_{dp}$ when only DP shards batch.
- Collectives: $\operatorname{AR}$, $\operatorname{AG}$, $\operatorname{RS}$, $\operatorname{A2A}$, point-to-point $\operatorname{Send}/\operatorname{Recv}$.
- Communication model: $T(n,p)=\alpha\,f(p)+\beta\,n\,g(p)$.
- Peak rank memory: $M_{rank}=M_{params}+M_{grads}+M_{opt}+M_{acts}+M_{kv}+M_{tmp}$.
- Throughput proxy: $\mathrm{tokens/s}\propto\frac{B\cdot S}{T_{step}}$.
- Pipeline bubble approximation: $\eta_{bubble}\approx\frac{P_{pp}-1}{m+P_{pp}-1}$.
- All-reduce decomposition identity: $\operatorname{AR}(x)=\operatorname{AG}(\operatorname{RS}(x))$.

## How to use this study guide
1. Read Sections 1–2 for common models and DP sharding foundations.
2. Read Sections 3–6 for axis-specific mechanics (TP/PP/CP/EP).
3. Read Sections 7–10 for composition, serving, kernels, and case patterns.
4. Use Sections 11–13 as design checklists and exam-like review prompts.

## Abstract
This section explains **Abstract** from first principles, implementation mechanics, cost/memory implications, topology interactions, and operational diagnostics.

## How To Read This Paper
This section explains **How To Read This Paper** from first principles, implementation mechanics, cost/memory implications, topology interactions, and operational diagnostics.

## 1. Shared System Model, Notation, Placement, and Evidence Policy
This section explains **1. Shared System Model, Notation, Placement, and Evidence Policy** from first principles, implementation mechanics, cost/memory implications, topology interactions, and operational diagnostics.

### 1.1 Evidence Grades and Source Interpretation
### Learning goals
- Understand the exact role of **1.1 Evidence Grades and Source Interpretation** in the full training/serving stack.
- Identify required tensor layouts before and after this step.
- Map algorithmic choices to communication, memory, and stability consequences.

### Mechanism
- **Inputs/state.** Define local shards, global semantics, and invariants (shape/dtype/order).
- **Transformation.** Execute compute kernels plus communication primitives to move from input layout to output layout.
- **Correctness.** Preserve mathematical equivalence (modulo floating-point ordering) and optimizer semantics.
- **Output contract.** Ensure downstream subsection receives expected placement/state.

### Mathematical view
- Local shard sizing follows $n_{local}=n/p$ along sharded axis, with replicated axes unchanged.
- Runtime contribution is modeled as:
$$
T_{subsec}pprox T_{kernel}+T_{comm}-T_{overlap}+T_{sync}+T_{runtime}.
$$
- If this stage adds collective payload $q$ on group size $p_g$, then increment:
$$
\Delta T_{comm}pprox lpha f(p_g)+eta q g(p_g).
$$
- If this stage changes replication→sharding for tensor $x$, memory delta per rank is approximately:
$$
\Delta M(x)pprox |x|\left(1-rac{1}{p_g}ight).
$$

### Implementation details (practical)
1. Build process groups and bind them to physical topology intentionally; avoid accidental rank orders.
2. Choose bucket/chunk sizes so kernels stay large enough for high occupancy while communication can stream.
3. Co-schedule streams: compute stream + comm stream + dependency events to increase overlap safely.
4. Encode layout metadata in checkpoints to permit reshard/restart on changed meshes.
5. Validate numerics under mixed precision with loss-scale or FP8 scale tracking as required.

### Failure modes
- **Bandwidth-bound plateau:** throughput stops scaling as payload term dominates.
- **Latency-bound micro-ops:** too many tiny collectives; startup cost dominates.
- **Imbalance/stragglers:** rank skew from load imbalance or topology asymmetry.
- **Memory spikes:** poor prefetch/reshard boundaries or oversized microbatching.
- **Numerical drift:** precision policy mismatches across shards/collectives.

### Diagnostics and experiments
- Perform ablations: vary one axis degree at a time ($P_{tp}$, $P_{pp}$, etc.).
- Trace per-rank timelines to measure overlap ratio $ho=rac{T_{overlap}}{T_{comm}}$.
- Sweep microbatch count $m$, gradient bucket sizes, and activation checkpoint boundaries.
- Capture p50/p95/p99 iteration time and correlate with collective/kernel breakdowns.

### Study checklist
- Can you state the exact input/output layout for this subsection?
- Can you predict which collective dominates at larger scale?
- Can you explain one condition where this method should *not* be used?

### 1.2 Shared Symbols and Transformer Shapes
### Learning goals
- Understand the exact role of **1.2 Shared Symbols and Transformer Shapes** in the full training/serving stack.
- Identify required tensor layouts before and after this step.
- Map algorithmic choices to communication, memory, and stability consequences.

### Mechanism
- **Inputs/state.** Define local shards, global semantics, and invariants (shape/dtype/order).
- **Transformation.** Execute compute kernels plus communication primitives to move from input layout to output layout.
- **Correctness.** Preserve mathematical equivalence (modulo floating-point ordering) and optimizer semantics.
- **Output contract.** Ensure downstream subsection receives expected placement/state.

### Mathematical view
- Local shard sizing follows $n_{local}=n/p$ along sharded axis, with replicated axes unchanged.
- Runtime contribution is modeled as:
$$
T_{subsec}pprox T_{kernel}+T_{comm}-T_{overlap}+T_{sync}+T_{runtime}.
$$
- If this stage adds collective payload $q$ on group size $p_g$, then increment:
$$
\Delta T_{comm}pprox lpha f(p_g)+eta q g(p_g).
$$
- If this stage changes replication→sharding for tensor $x$, memory delta per rank is approximately:
$$
\Delta M(x)pprox |x|\left(1-rac{1}{p_g}ight).
$$

### Implementation details (practical)
1. Build process groups and bind them to physical topology intentionally; avoid accidental rank orders.
2. Choose bucket/chunk sizes so kernels stay large enough for high occupancy while communication can stream.
3. Co-schedule streams: compute stream + comm stream + dependency events to increase overlap safely.
4. Encode layout metadata in checkpoints to permit reshard/restart on changed meshes.
5. Validate numerics under mixed precision with loss-scale or FP8 scale tracking as required.

### Failure modes
- **Bandwidth-bound plateau:** throughput stops scaling as payload term dominates.
- **Latency-bound micro-ops:** too many tiny collectives; startup cost dominates.
- **Imbalance/stragglers:** rank skew from load imbalance or topology asymmetry.
- **Memory spikes:** poor prefetch/reshard boundaries or oversized microbatching.
- **Numerical drift:** precision policy mismatches across shards/collectives.

### Diagnostics and experiments
- Perform ablations: vary one axis degree at a time ($P_{tp}$, $P_{pp}$, etc.).
- Trace per-rank timelines to measure overlap ratio $ho=rac{T_{overlap}}{T_{comm}}$.
- Sweep microbatch count $m$, gradient bucket sizes, and activation checkpoint boundaries.
- Capture p50/p95/p99 iteration time and correlate with collective/kernel breakdowns.

### Study checklist
- Can you state the exact input/output layout for this subsection?
- Can you predict which collective dominates at larger scale?
- Can you explain one condition where this method should *not* be used?

### 1.3 Device Meshes, Rank Coordinates, and Process Groups
### Learning goals
- Understand the exact role of **1.3 Device Meshes, Rank Coordinates, and Process Groups** in the full training/serving stack.
- Identify required tensor layouts before and after this step.
- Map algorithmic choices to communication, memory, and stability consequences.

### Mechanism
- **Inputs/state.** Define local shards, global semantics, and invariants (shape/dtype/order).
- **Transformation.** Execute compute kernels plus communication primitives to move from input layout to output layout.
- **Correctness.** Preserve mathematical equivalence (modulo floating-point ordering) and optimizer semantics.
- **Output contract.** Ensure downstream subsection receives expected placement/state.

### Mathematical view
- Local shard sizing follows $n_{local}=n/p$ along sharded axis, with replicated axes unchanged.
- Runtime contribution is modeled as:
$$
T_{subsec}pprox T_{kernel}+T_{comm}-T_{overlap}+T_{sync}+T_{runtime}.
$$
- If this stage adds collective payload $q$ on group size $p_g$, then increment:
$$
\Delta T_{comm}pprox lpha f(p_g)+eta q g(p_g).
$$
- If this stage changes replication→sharding for tensor $x$, memory delta per rank is approximately:
$$
\Delta M(x)pprox |x|\left(1-rac{1}{p_g}ight).
$$

### Implementation details (practical)
1. Build process groups and bind them to physical topology intentionally; avoid accidental rank orders.
2. Choose bucket/chunk sizes so kernels stay large enough for high occupancy while communication can stream.
3. Co-schedule streams: compute stream + comm stream + dependency events to increase overlap safely.
4. Encode layout metadata in checkpoints to permit reshard/restart on changed meshes.
5. Validate numerics under mixed precision with loss-scale or FP8 scale tracking as required.

### Failure modes
- **Bandwidth-bound plateau:** throughput stops scaling as payload term dominates.
- **Latency-bound micro-ops:** too many tiny collectives; startup cost dominates.
- **Imbalance/stragglers:** rank skew from load imbalance or topology asymmetry.
- **Memory spikes:** poor prefetch/reshard boundaries or oversized microbatching.
- **Numerical drift:** precision policy mismatches across shards/collectives.

### Diagnostics and experiments
- Perform ablations: vary one axis degree at a time ($P_{tp}$, $P_{pp}$, etc.).
- Trace per-rank timelines to measure overlap ratio $ho=rac{T_{overlap}}{T_{comm}}$.
- Sweep microbatch count $m$, gradient bucket sizes, and activation checkpoint boundaries.
- Capture p50/p95/p99 iteration time and correlate with collective/kernel breakdowns.

### Study checklist
- Can you state the exact input/output layout for this subsection?
- Can you predict which collective dominates at larger scale?
- Can you explain one condition where this method should *not* be used?

### 1.4 Placement States and Tensor Layout Transitions
### Learning goals
- Understand the exact role of **1.4 Placement States and Tensor Layout Transitions** in the full training/serving stack.
- Identify required tensor layouts before and after this step.
- Map algorithmic choices to communication, memory, and stability consequences.

### Mechanism
- **Inputs/state.** Define local shards, global semantics, and invariants (shape/dtype/order).
- **Transformation.** Execute compute kernels plus communication primitives to move from input layout to output layout.
- **Correctness.** Preserve mathematical equivalence (modulo floating-point ordering) and optimizer semantics.
- **Output contract.** Ensure downstream subsection receives expected placement/state.

### Mathematical view
- Local shard sizing follows $n_{local}=n/p$ along sharded axis, with replicated axes unchanged.
- Runtime contribution is modeled as:
$$
T_{subsec}pprox T_{kernel}+T_{comm}-T_{overlap}+T_{sync}+T_{runtime}.
$$
- If this stage adds collective payload $q$ on group size $p_g$, then increment:
$$
\Delta T_{comm}pprox lpha f(p_g)+eta q g(p_g).
$$
- If this stage changes replication→sharding for tensor $x$, memory delta per rank is approximately:
$$
\Delta M(x)pprox |x|\left(1-rac{1}{p_g}ight).
$$

### Implementation details (practical)
1. Build process groups and bind them to physical topology intentionally; avoid accidental rank orders.
2. Choose bucket/chunk sizes so kernels stay large enough for high occupancy while communication can stream.
3. Co-schedule streams: compute stream + comm stream + dependency events to increase overlap safely.
4. Encode layout metadata in checkpoints to permit reshard/restart on changed meshes.
5. Validate numerics under mixed precision with loss-scale or FP8 scale tracking as required.

### Failure modes
- **Bandwidth-bound plateau:** throughput stops scaling as payload term dominates.
- **Latency-bound micro-ops:** too many tiny collectives; startup cost dominates.
- **Imbalance/stragglers:** rank skew from load imbalance or topology asymmetry.
- **Memory spikes:** poor prefetch/reshard boundaries or oversized microbatching.
- **Numerical drift:** precision policy mismatches across shards/collectives.

### Diagnostics and experiments
- Perform ablations: vary one axis degree at a time ($P_{tp}$, $P_{pp}$, etc.).
- Trace per-rank timelines to measure overlap ratio $ho=rac{T_{overlap}}{T_{comm}}$.
- Sweep microbatch count $m$, gradient bucket sizes, and activation checkpoint boundaries.
- Capture p50/p95/p99 iteration time and correlate with collective/kernel breakdowns.

### Study checklist
- Can you state the exact input/output layout for this subsection?
- Can you predict which collective dominates at larger scale?
- Can you explain one condition where this method should *not* be used?

### 1.5 Communication Primitives and Cost Models
### Learning goals
- Understand the exact role of **1.5 Communication Primitives and Cost Models** in the full training/serving stack.
- Identify required tensor layouts before and after this step.
- Map algorithmic choices to communication, memory, and stability consequences.

### Mechanism
- **Inputs/state.** Define local shards, global semantics, and invariants (shape/dtype/order).
- **Transformation.** Execute compute kernels plus communication primitives to move from input layout to output layout.
- **Correctness.** Preserve mathematical equivalence (modulo floating-point ordering) and optimizer semantics.
- **Output contract.** Ensure downstream subsection receives expected placement/state.

### Mathematical view
- Local shard sizing follows $n_{local}=n/p$ along sharded axis, with replicated axes unchanged.
- Runtime contribution is modeled as:
$$
T_{subsec}pprox T_{kernel}+T_{comm}-T_{overlap}+T_{sync}+T_{runtime}.
$$
- If this stage adds collective payload $q$ on group size $p_g$, then increment:
$$
\Delta T_{comm}pprox lpha f(p_g)+eta q g(p_g).
$$
- If this stage changes replication→sharding for tensor $x$, memory delta per rank is approximately:
$$
\Delta M(x)pprox |x|\left(1-rac{1}{p_g}ight).
$$

### Implementation details (practical)
1. Build process groups and bind them to physical topology intentionally; avoid accidental rank orders.
2. Choose bucket/chunk sizes so kernels stay large enough for high occupancy while communication can stream.
3. Co-schedule streams: compute stream + comm stream + dependency events to increase overlap safely.
4. Encode layout metadata in checkpoints to permit reshard/restart on changed meshes.
5. Validate numerics under mixed precision with loss-scale or FP8 scale tracking as required.

### Failure modes
- **Bandwidth-bound plateau:** throughput stops scaling as payload term dominates.
- **Latency-bound micro-ops:** too many tiny collectives; startup cost dominates.
- **Imbalance/stragglers:** rank skew from load imbalance or topology asymmetry.
- **Memory spikes:** poor prefetch/reshard boundaries or oversized microbatching.
- **Numerical drift:** precision policy mismatches across shards/collectives.

### Diagnostics and experiments
- Perform ablations: vary one axis degree at a time ($P_{tp}$, $P_{pp}$, etc.).
- Trace per-rank timelines to measure overlap ratio $ho=rac{T_{overlap}}{T_{comm}}$.
- Sweep microbatch count $m$, gradient bucket sizes, and activation checkpoint boundaries.
- Capture p50/p95/p99 iteration time and correlate with collective/kernel breakdowns.

### Study checklist
- Can you state the exact input/output layout for this subsection?
- Can you predict which collective dominates at larger scale?
- Can you explain one condition where this method should *not* be used?

### 1.6 Memory, Bandwidth, and Overlap Accounting
### Learning goals
- Understand the exact role of **1.6 Memory, Bandwidth, and Overlap Accounting** in the full training/serving stack.
- Identify required tensor layouts before and after this step.
- Map algorithmic choices to communication, memory, and stability consequences.

### Mechanism
- **Inputs/state.** Define local shards, global semantics, and invariants (shape/dtype/order).
- **Transformation.** Execute compute kernels plus communication primitives to move from input layout to output layout.
- **Correctness.** Preserve mathematical equivalence (modulo floating-point ordering) and optimizer semantics.
- **Output contract.** Ensure downstream subsection receives expected placement/state.

### Mathematical view
- Local shard sizing follows $n_{local}=n/p$ along sharded axis, with replicated axes unchanged.
- Runtime contribution is modeled as:
$$
T_{subsec}pprox T_{kernel}+T_{comm}-T_{overlap}+T_{sync}+T_{runtime}.
$$
- If this stage adds collective payload $q$ on group size $p_g$, then increment:
$$
\Delta T_{comm}pprox lpha f(p_g)+eta q g(p_g).
$$
- If this stage changes replication→sharding for tensor $x$, memory delta per rank is approximately:
$$
\Delta M(x)pprox |x|\left(1-rac{1}{p_g}ight).
$$

### Implementation details (practical)
1. Build process groups and bind them to physical topology intentionally; avoid accidental rank orders.
2. Choose bucket/chunk sizes so kernels stay large enough for high occupancy while communication can stream.
3. Co-schedule streams: compute stream + comm stream + dependency events to increase overlap safely.
4. Encode layout metadata in checkpoints to permit reshard/restart on changed meshes.
5. Validate numerics under mixed precision with loss-scale or FP8 scale tracking as required.

### Failure modes
- **Bandwidth-bound plateau:** throughput stops scaling as payload term dominates.
- **Latency-bound micro-ops:** too many tiny collectives; startup cost dominates.
- **Imbalance/stragglers:** rank skew from load imbalance or topology asymmetry.
- **Memory spikes:** poor prefetch/reshard boundaries or oversized microbatching.
- **Numerical drift:** precision policy mismatches across shards/collectives.

### Diagnostics and experiments
- Perform ablations: vary one axis degree at a time ($P_{tp}$, $P_{pp}$, etc.).
- Trace per-rank timelines to measure overlap ratio $ho=rac{T_{overlap}}{T_{comm}}$.
- Sweep microbatch count $m$, gradient bucket sizes, and activation checkpoint boundaries.
- Capture p50/p95/p99 iteration time and correlate with collective/kernel breakdowns.

### Study checklist
- Can you state the exact input/output layout for this subsection?
- Can you predict which collective dominates at larger scale?
- Can you explain one condition where this method should *not* be used?

### 1.7 How Later Sections Use This Shared Layer
### Learning goals
- Understand the exact role of **1.7 How Later Sections Use This Shared Layer** in the full training/serving stack.
- Identify required tensor layouts before and after this step.
- Map algorithmic choices to communication, memory, and stability consequences.

### Mechanism
- **Inputs/state.** Define local shards, global semantics, and invariants (shape/dtype/order).
- **Transformation.** Execute compute kernels plus communication primitives to move from input layout to output layout.
- **Correctness.** Preserve mathematical equivalence (modulo floating-point ordering) and optimizer semantics.
- **Output contract.** Ensure downstream subsection receives expected placement/state.

### Mathematical view
- Local shard sizing follows $n_{local}=n/p$ along sharded axis, with replicated axes unchanged.
- Runtime contribution is modeled as:
$$
T_{subsec}pprox T_{kernel}+T_{comm}-T_{overlap}+T_{sync}+T_{runtime}.
$$
- If this stage adds collective payload $q$ on group size $p_g$, then increment:
$$
\Delta T_{comm}pprox lpha f(p_g)+eta q g(p_g).
$$
- If this stage changes replication→sharding for tensor $x$, memory delta per rank is approximately:
$$
\Delta M(x)pprox |x|\left(1-rac{1}{p_g}ight).
$$

### Implementation details (practical)
1. Build process groups and bind them to physical topology intentionally; avoid accidental rank orders.
2. Choose bucket/chunk sizes so kernels stay large enough for high occupancy while communication can stream.
3. Co-schedule streams: compute stream + comm stream + dependency events to increase overlap safely.
4. Encode layout metadata in checkpoints to permit reshard/restart on changed meshes.
5. Validate numerics under mixed precision with loss-scale or FP8 scale tracking as required.

### Failure modes
- **Bandwidth-bound plateau:** throughput stops scaling as payload term dominates.
- **Latency-bound micro-ops:** too many tiny collectives; startup cost dominates.
- **Imbalance/stragglers:** rank skew from load imbalance or topology asymmetry.
- **Memory spikes:** poor prefetch/reshard boundaries or oversized microbatching.
- **Numerical drift:** precision policy mismatches across shards/collectives.

### Diagnostics and experiments
- Perform ablations: vary one axis degree at a time ($P_{tp}$, $P_{pp}$, etc.).
- Trace per-rank timelines to measure overlap ratio $ho=rac{T_{overlap}}{T_{comm}}$.
- Sweep microbatch count $m$, gradient bucket sizes, and activation checkpoint boundaries.
- Capture p50/p95/p99 iteration time and correlate with collective/kernel breakdowns.

### Study checklist
- Can you state the exact input/output layout for this subsection?
- Can you predict which collective dominates at larger scale?
- Can you explain one condition where this method should *not* be used?

## 2. Data Parallelism, FSDP, ZeRO, HSDP, and State Sharding
This section explains **2. Data Parallelism, FSDP, ZeRO, HSDP, and State Sharding** from first principles, implementation mechanics, cost/memory implications, topology interactions, and operational diagnostics.

### 2.1 DDP: Replicated State, Batch Shards, and Gradient All-Reduce
### Learning goals
- Understand the exact role of **2.1 DDP: Replicated State, Batch Shards, and Gradient All-Reduce** in the full training/serving stack.
- Identify required tensor layouts before and after this step.
- Map algorithmic choices to communication, memory, and stability consequences.

### Mechanism
- **Inputs/state.** Define local shards, global semantics, and invariants (shape/dtype/order).
- **Transformation.** Execute compute kernels plus communication primitives to move from input layout to output layout.
- **Correctness.** Preserve mathematical equivalence (modulo floating-point ordering) and optimizer semantics.
- **Output contract.** Ensure downstream subsection receives expected placement/state.

### Mathematical view
- Local shard sizing follows $n_{local}=n/p$ along sharded axis, with replicated axes unchanged.
- Runtime contribution is modeled as:
$$
T_{subsec}pprox T_{kernel}+T_{comm}-T_{overlap}+T_{sync}+T_{runtime}.
$$
- If this stage adds collective payload $q$ on group size $p_g$, then increment:
$$
\Delta T_{comm}pprox lpha f(p_g)+eta q g(p_g).
$$
- If this stage changes replication→sharding for tensor $x$, memory delta per rank is approximately:
$$
\Delta M(x)pprox |x|\left(1-rac{1}{p_g}ight).
$$

### Implementation details (practical)
1. Build process groups and bind them to physical topology intentionally; avoid accidental rank orders.
2. Choose bucket/chunk sizes so kernels stay large enough for high occupancy while communication can stream.
3. Co-schedule streams: compute stream + comm stream + dependency events to increase overlap safely.
4. Encode layout metadata in checkpoints to permit reshard/restart on changed meshes.
5. Validate numerics under mixed precision with loss-scale or FP8 scale tracking as required.

### Failure modes
- **Bandwidth-bound plateau:** throughput stops scaling as payload term dominates.
- **Latency-bound micro-ops:** too many tiny collectives; startup cost dominates.
- **Imbalance/stragglers:** rank skew from load imbalance or topology asymmetry.
- **Memory spikes:** poor prefetch/reshard boundaries or oversized microbatching.
- **Numerical drift:** precision policy mismatches across shards/collectives.

### Diagnostics and experiments
- Perform ablations: vary one axis degree at a time ($P_{tp}$, $P_{pp}$, etc.).
- Trace per-rank timelines to measure overlap ratio $ho=rac{T_{overlap}}{T_{comm}}$.
- Sweep microbatch count $m$, gradient bucket sizes, and activation checkpoint boundaries.
- Capture p50/p95/p99 iteration time and correlate with collective/kernel breakdowns.

### Study checklist
- Can you state the exact input/output layout for this subsection?
- Can you predict which collective dominates at larger scale?
- Can you explain one condition where this method should *not* be used?

### 2.2 The Primitive Mechanisms Behind Sharded Data Parallelism
### Learning goals
- Understand the exact role of **2.2 The Primitive Mechanisms Behind Sharded Data Parallelism** in the full training/serving stack.
- Identify required tensor layouts before and after this step.
- Map algorithmic choices to communication, memory, and stability consequences.

### Mechanism
- **Inputs/state.** Define local shards, global semantics, and invariants (shape/dtype/order).
- **Transformation.** Execute compute kernels plus communication primitives to move from input layout to output layout.
- **Correctness.** Preserve mathematical equivalence (modulo floating-point ordering) and optimizer semantics.
- **Output contract.** Ensure downstream subsection receives expected placement/state.

### Mathematical view
- Local shard sizing follows $n_{local}=n/p$ along sharded axis, with replicated axes unchanged.
- Runtime contribution is modeled as:
$$
T_{subsec}pprox T_{kernel}+T_{comm}-T_{overlap}+T_{sync}+T_{runtime}.
$$
- If this stage adds collective payload $q$ on group size $p_g$, then increment:
$$
\Delta T_{comm}pprox lpha f(p_g)+eta q g(p_g).
$$
- If this stage changes replication→sharding for tensor $x$, memory delta per rank is approximately:
$$
\Delta M(x)pprox |x|\left(1-rac{1}{p_g}ight).
$$

### Implementation details (practical)
1. Build process groups and bind them to physical topology intentionally; avoid accidental rank orders.
2. Choose bucket/chunk sizes so kernels stay large enough for high occupancy while communication can stream.
3. Co-schedule streams: compute stream + comm stream + dependency events to increase overlap safely.
4. Encode layout metadata in checkpoints to permit reshard/restart on changed meshes.
5. Validate numerics under mixed precision with loss-scale or FP8 scale tracking as required.

### Failure modes
- **Bandwidth-bound plateau:** throughput stops scaling as payload term dominates.
- **Latency-bound micro-ops:** too many tiny collectives; startup cost dominates.
- **Imbalance/stragglers:** rank skew from load imbalance or topology asymmetry.
- **Memory spikes:** poor prefetch/reshard boundaries or oversized microbatching.
- **Numerical drift:** precision policy mismatches across shards/collectives.

### Diagnostics and experiments
- Perform ablations: vary one axis degree at a time ($P_{tp}$, $P_{pp}$, etc.).
- Trace per-rank timelines to measure overlap ratio $ho=rac{T_{overlap}}{T_{comm}}$.
- Sweep microbatch count $m$, gradient bucket sizes, and activation checkpoint boundaries.
- Capture p50/p95/p99 iteration time and correlate with collective/kernel breakdowns.

### Study checklist
- Can you state the exact input/output layout for this subsection?
- Can you predict which collective dominates at larger scale?
- Can you explain one condition where this method should *not* be used?

### 2.3 ZeRO, ZeRO++, and Distributed Optimizers
### Learning goals
- Understand the exact role of **2.3 ZeRO, ZeRO++, and Distributed Optimizers** in the full training/serving stack.
- Identify required tensor layouts before and after this step.
- Map algorithmic choices to communication, memory, and stability consequences.

### Mechanism
- **Inputs/state.** Define local shards, global semantics, and invariants (shape/dtype/order).
- **Transformation.** Execute compute kernels plus communication primitives to move from input layout to output layout.
- **Correctness.** Preserve mathematical equivalence (modulo floating-point ordering) and optimizer semantics.
- **Output contract.** Ensure downstream subsection receives expected placement/state.

### Mathematical view
- Local shard sizing follows $n_{local}=n/p$ along sharded axis, with replicated axes unchanged.
- Runtime contribution is modeled as:
$$
T_{subsec}pprox T_{kernel}+T_{comm}-T_{overlap}+T_{sync}+T_{runtime}.
$$
- If this stage adds collective payload $q$ on group size $p_g$, then increment:
$$
\Delta T_{comm}pprox lpha f(p_g)+eta q g(p_g).
$$
- If this stage changes replication→sharding for tensor $x$, memory delta per rank is approximately:
$$
\Delta M(x)pprox |x|\left(1-rac{1}{p_g}ight).
$$

### Implementation details (practical)
1. Build process groups and bind them to physical topology intentionally; avoid accidental rank orders.
2. Choose bucket/chunk sizes so kernels stay large enough for high occupancy while communication can stream.
3. Co-schedule streams: compute stream + comm stream + dependency events to increase overlap safely.
4. Encode layout metadata in checkpoints to permit reshard/restart on changed meshes.
5. Validate numerics under mixed precision with loss-scale or FP8 scale tracking as required.

### Failure modes
- **Bandwidth-bound plateau:** throughput stops scaling as payload term dominates.
- **Latency-bound micro-ops:** too many tiny collectives; startup cost dominates.
- **Imbalance/stragglers:** rank skew from load imbalance or topology asymmetry.
- **Memory spikes:** poor prefetch/reshard boundaries or oversized microbatching.
- **Numerical drift:** precision policy mismatches across shards/collectives.

### Diagnostics and experiments
- Perform ablations: vary one axis degree at a time ($P_{tp}$, $P_{pp}$, etc.).
- Trace per-rank timelines to measure overlap ratio $ho=rac{T_{overlap}}{T_{comm}}$.
- Sweep microbatch count $m$, gradient bucket sizes, and activation checkpoint boundaries.
- Capture p50/p95/p99 iteration time and correlate with collective/kernel breakdowns.

### Study checklist
- Can you state the exact input/output layout for this subsection?
- Can you predict which collective dominates at larger scale?
- Can you explain one condition where this method should *not* be used?

### 2.4 FSDP and FSDP2 Execution
### Learning goals
- Understand the exact role of **2.4 FSDP and FSDP2 Execution** in the full training/serving stack.
- Identify required tensor layouts before and after this step.
- Map algorithmic choices to communication, memory, and stability consequences.

### Mechanism
- **Inputs/state.** Define local shards, global semantics, and invariants (shape/dtype/order).
- **Transformation.** Execute compute kernels plus communication primitives to move from input layout to output layout.
- **Correctness.** Preserve mathematical equivalence (modulo floating-point ordering) and optimizer semantics.
- **Output contract.** Ensure downstream subsection receives expected placement/state.

### Mathematical view
- Local shard sizing follows $n_{local}=n/p$ along sharded axis, with replicated axes unchanged.
- Runtime contribution is modeled as:
$$
T_{subsec}pprox T_{kernel}+T_{comm}-T_{overlap}+T_{sync}+T_{runtime}.
$$
- If this stage adds collective payload $q$ on group size $p_g$, then increment:
$$
\Delta T_{comm}pprox lpha f(p_g)+eta q g(p_g).
$$
- If this stage changes replication→sharding for tensor $x$, memory delta per rank is approximately:
$$
\Delta M(x)pprox |x|\left(1-rac{1}{p_g}ight).
$$

### Implementation details (practical)
1. Build process groups and bind them to physical topology intentionally; avoid accidental rank orders.
2. Choose bucket/chunk sizes so kernels stay large enough for high occupancy while communication can stream.
3. Co-schedule streams: compute stream + comm stream + dependency events to increase overlap safely.
4. Encode layout metadata in checkpoints to permit reshard/restart on changed meshes.
5. Validate numerics under mixed precision with loss-scale or FP8 scale tracking as required.

### Failure modes
- **Bandwidth-bound plateau:** throughput stops scaling as payload term dominates.
- **Latency-bound micro-ops:** too many tiny collectives; startup cost dominates.
- **Imbalance/stragglers:** rank skew from load imbalance or topology asymmetry.
- **Memory spikes:** poor prefetch/reshard boundaries or oversized microbatching.
- **Numerical drift:** precision policy mismatches across shards/collectives.

### Diagnostics and experiments
- Perform ablations: vary one axis degree at a time ($P_{tp}$, $P_{pp}$, etc.).
- Trace per-rank timelines to measure overlap ratio $ho=rac{T_{overlap}}{T_{comm}}$.
- Sweep microbatch count $m$, gradient bucket sizes, and activation checkpoint boundaries.
- Capture p50/p95/p99 iteration time and correlate with collective/kernel breakdowns.

### Study checklist
- Can you state the exact input/output layout for this subsection?
- Can you predict which collective dominates at larger scale?
- Can you explain one condition where this method should *not* be used?

### 2.5 HSDP: Hybrid Sharded Data Parallelism
### Learning goals
- Understand the exact role of **2.5 HSDP: Hybrid Sharded Data Parallelism** in the full training/serving stack.
- Identify required tensor layouts before and after this step.
- Map algorithmic choices to communication, memory, and stability consequences.

### Mechanism
- **Inputs/state.** Define local shards, global semantics, and invariants (shape/dtype/order).
- **Transformation.** Execute compute kernels plus communication primitives to move from input layout to output layout.
- **Correctness.** Preserve mathematical equivalence (modulo floating-point ordering) and optimizer semantics.
- **Output contract.** Ensure downstream subsection receives expected placement/state.

### Mathematical view
- Local shard sizing follows $n_{local}=n/p$ along sharded axis, with replicated axes unchanged.
- Runtime contribution is modeled as:
$$
T_{subsec}pprox T_{kernel}+T_{comm}-T_{overlap}+T_{sync}+T_{runtime}.
$$
- If this stage adds collective payload $q$ on group size $p_g$, then increment:
$$
\Delta T_{comm}pprox lpha f(p_g)+eta q g(p_g).
$$
- If this stage changes replication→sharding for tensor $x$, memory delta per rank is approximately:
$$
\Delta M(x)pprox |x|\left(1-rac{1}{p_g}ight).
$$

### Implementation details (practical)
1. Build process groups and bind them to physical topology intentionally; avoid accidental rank orders.
2. Choose bucket/chunk sizes so kernels stay large enough for high occupancy while communication can stream.
3. Co-schedule streams: compute stream + comm stream + dependency events to increase overlap safely.
4. Encode layout metadata in checkpoints to permit reshard/restart on changed meshes.
5. Validate numerics under mixed precision with loss-scale or FP8 scale tracking as required.

### Failure modes
- **Bandwidth-bound plateau:** throughput stops scaling as payload term dominates.
- **Latency-bound micro-ops:** too many tiny collectives; startup cost dominates.
- **Imbalance/stragglers:** rank skew from load imbalance or topology asymmetry.
- **Memory spikes:** poor prefetch/reshard boundaries or oversized microbatching.
- **Numerical drift:** precision policy mismatches across shards/collectives.

### Diagnostics and experiments
- Perform ablations: vary one axis degree at a time ($P_{tp}$, $P_{pp}$, etc.).
- Trace per-rank timelines to measure overlap ratio $ho=rac{T_{overlap}}{T_{comm}}$.
- Sweep microbatch count $m$, gradient bucket sizes, and activation checkpoint boundaries.
- Capture p50/p95/p99 iteration time and correlate with collective/kernel breakdowns.

### Study checklist
- Can you state the exact input/output layout for this subsection?
- Can you predict which collective dominates at larger scale?
- Can you explain one condition where this method should *not* be used?

### 2.6 Offload: CPU and NVMe as Additional State Tiers
### Learning goals
- Understand the exact role of **2.6 Offload: CPU and NVMe as Additional State Tiers** in the full training/serving stack.
- Identify required tensor layouts before and after this step.
- Map algorithmic choices to communication, memory, and stability consequences.

### Mechanism
- **Inputs/state.** Define local shards, global semantics, and invariants (shape/dtype/order).
- **Transformation.** Execute compute kernels plus communication primitives to move from input layout to output layout.
- **Correctness.** Preserve mathematical equivalence (modulo floating-point ordering) and optimizer semantics.
- **Output contract.** Ensure downstream subsection receives expected placement/state.

### Mathematical view
- Local shard sizing follows $n_{local}=n/p$ along sharded axis, with replicated axes unchanged.
- Runtime contribution is modeled as:
$$
T_{subsec}pprox T_{kernel}+T_{comm}-T_{overlap}+T_{sync}+T_{runtime}.
$$
- If this stage adds collective payload $q$ on group size $p_g$, then increment:
$$
\Delta T_{comm}pprox lpha f(p_g)+eta q g(p_g).
$$
- If this stage changes replication→sharding for tensor $x$, memory delta per rank is approximately:
$$
\Delta M(x)pprox |x|\left(1-rac{1}{p_g}ight).
$$

### Implementation details (practical)
1. Build process groups and bind them to physical topology intentionally; avoid accidental rank orders.
2. Choose bucket/chunk sizes so kernels stay large enough for high occupancy while communication can stream.
3. Co-schedule streams: compute stream + comm stream + dependency events to increase overlap safely.
4. Encode layout metadata in checkpoints to permit reshard/restart on changed meshes.
5. Validate numerics under mixed precision with loss-scale or FP8 scale tracking as required.

### Failure modes
- **Bandwidth-bound plateau:** throughput stops scaling as payload term dominates.
- **Latency-bound micro-ops:** too many tiny collectives; startup cost dominates.
- **Imbalance/stragglers:** rank skew from load imbalance or topology asymmetry.
- **Memory spikes:** poor prefetch/reshard boundaries or oversized microbatching.
- **Numerical drift:** precision policy mismatches across shards/collectives.

### Diagnostics and experiments
- Perform ablations: vary one axis degree at a time ($P_{tp}$, $P_{pp}$, etc.).
- Trace per-rank timelines to measure overlap ratio $ho=rac{T_{overlap}}{T_{comm}}$.
- Sweep microbatch count $m$, gradient bucket sizes, and activation checkpoint boundaries.
- Capture p50/p95/p99 iteration time and correlate with collective/kernel breakdowns.

### Study checklist
- Can you state the exact input/output layout for this subsection?
- Can you predict which collective dominates at larger scale?
- Can you explain one condition where this method should *not* be used?

### 2.7 Low Precision, Quantized Communication, and Platform Support
### Learning goals
- Understand the exact role of **2.7 Low Precision, Quantized Communication, and Platform Support** in the full training/serving stack.
- Identify required tensor layouts before and after this step.
- Map algorithmic choices to communication, memory, and stability consequences.

### Mechanism
- **Inputs/state.** Define local shards, global semantics, and invariants (shape/dtype/order).
- **Transformation.** Execute compute kernels plus communication primitives to move from input layout to output layout.
- **Correctness.** Preserve mathematical equivalence (modulo floating-point ordering) and optimizer semantics.
- **Output contract.** Ensure downstream subsection receives expected placement/state.

### Mathematical view
- Local shard sizing follows $n_{local}=n/p$ along sharded axis, with replicated axes unchanged.
- Runtime contribution is modeled as:
$$
T_{subsec}pprox T_{kernel}+T_{comm}-T_{overlap}+T_{sync}+T_{runtime}.
$$
- If this stage adds collective payload $q$ on group size $p_g$, then increment:
$$
\Delta T_{comm}pprox lpha f(p_g)+eta q g(p_g).
$$
- If this stage changes replication→sharding for tensor $x$, memory delta per rank is approximately:
$$
\Delta M(x)pprox |x|\left(1-rac{1}{p_g}ight).
$$

### Implementation details (practical)
1. Build process groups and bind them to physical topology intentionally; avoid accidental rank orders.
2. Choose bucket/chunk sizes so kernels stay large enough for high occupancy while communication can stream.
3. Co-schedule streams: compute stream + comm stream + dependency events to increase overlap safely.
4. Encode layout metadata in checkpoints to permit reshard/restart on changed meshes.
5. Validate numerics under mixed precision with loss-scale or FP8 scale tracking as required.

### Failure modes
- **Bandwidth-bound plateau:** throughput stops scaling as payload term dominates.
- **Latency-bound micro-ops:** too many tiny collectives; startup cost dominates.
- **Imbalance/stragglers:** rank skew from load imbalance or topology asymmetry.
- **Memory spikes:** poor prefetch/reshard boundaries or oversized microbatching.
- **Numerical drift:** precision policy mismatches across shards/collectives.

### Diagnostics and experiments
- Perform ablations: vary one axis degree at a time ($P_{tp}$, $P_{pp}$, etc.).
- Trace per-rank timelines to measure overlap ratio $ho=rac{T_{overlap}}{T_{comm}}$.
- Sweep microbatch count $m$, gradient bucket sizes, and activation checkpoint boundaries.
- Capture p50/p95/p99 iteration time and correlate with collective/kernel breakdowns.

### Study checklist
- Can you state the exact input/output layout for this subsection?
- Can you predict which collective dominates at larger scale?
- Can you explain one condition where this method should *not* be used?

### 2.8 Choosing Among DDP, FSDP2, ZeRO, ZeRO++, HSDP, and Offload
### Learning goals
- Understand the exact role of **2.8 Choosing Among DDP, FSDP2, ZeRO, ZeRO++, HSDP, and Offload** in the full training/serving stack.
- Identify required tensor layouts before and after this step.
- Map algorithmic choices to communication, memory, and stability consequences.

### Mechanism
- **Inputs/state.** Define local shards, global semantics, and invariants (shape/dtype/order).
- **Transformation.** Execute compute kernels plus communication primitives to move from input layout to output layout.
- **Correctness.** Preserve mathematical equivalence (modulo floating-point ordering) and optimizer semantics.
- **Output contract.** Ensure downstream subsection receives expected placement/state.

### Mathematical view
- Local shard sizing follows $n_{local}=n/p$ along sharded axis, with replicated axes unchanged.
- Runtime contribution is modeled as:
$$
T_{subsec}pprox T_{kernel}+T_{comm}-T_{overlap}+T_{sync}+T_{runtime}.
$$
- If this stage adds collective payload $q$ on group size $p_g$, then increment:
$$
\Delta T_{comm}pprox lpha f(p_g)+eta q g(p_g).
$$
- If this stage changes replication→sharding for tensor $x$, memory delta per rank is approximately:
$$
\Delta M(x)pprox |x|\left(1-rac{1}{p_g}ight).
$$

### Implementation details (practical)
1. Build process groups and bind them to physical topology intentionally; avoid accidental rank orders.
2. Choose bucket/chunk sizes so kernels stay large enough for high occupancy while communication can stream.
3. Co-schedule streams: compute stream + comm stream + dependency events to increase overlap safely.
4. Encode layout metadata in checkpoints to permit reshard/restart on changed meshes.
5. Validate numerics under mixed precision with loss-scale or FP8 scale tracking as required.

### Failure modes
- **Bandwidth-bound plateau:** throughput stops scaling as payload term dominates.
- **Latency-bound micro-ops:** too many tiny collectives; startup cost dominates.
- **Imbalance/stragglers:** rank skew from load imbalance or topology asymmetry.
- **Memory spikes:** poor prefetch/reshard boundaries or oversized microbatching.
- **Numerical drift:** precision policy mismatches across shards/collectives.

### Diagnostics and experiments
- Perform ablations: vary one axis degree at a time ($P_{tp}$, $P_{pp}$, etc.).
- Trace per-rank timelines to measure overlap ratio $ho=rac{T_{overlap}}{T_{comm}}$.
- Sweep microbatch count $m$, gradient bucket sizes, and activation checkpoint boundaries.
- Capture p50/p95/p99 iteration time and correlate with collective/kernel breakdowns.

### Study checklist
- Can you state the exact input/output layout for this subsection?
- Can you predict which collective dominates at larger scale?
- Can you explain one condition where this method should *not* be used?

### 2.9 Sources and Lineage
### Learning goals
- Understand the exact role of **2.9 Sources and Lineage** in the full training/serving stack.
- Identify required tensor layouts before and after this step.
- Map algorithmic choices to communication, memory, and stability consequences.

### Mechanism
- **Inputs/state.** Define local shards, global semantics, and invariants (shape/dtype/order).
- **Transformation.** Execute compute kernels plus communication primitives to move from input layout to output layout.
- **Correctness.** Preserve mathematical equivalence (modulo floating-point ordering) and optimizer semantics.
- **Output contract.** Ensure downstream subsection receives expected placement/state.

### Mathematical view
- Local shard sizing follows $n_{local}=n/p$ along sharded axis, with replicated axes unchanged.
- Runtime contribution is modeled as:
$$
T_{subsec}pprox T_{kernel}+T_{comm}-T_{overlap}+T_{sync}+T_{runtime}.
$$
- If this stage adds collective payload $q$ on group size $p_g$, then increment:
$$
\Delta T_{comm}pprox lpha f(p_g)+eta q g(p_g).
$$
- If this stage changes replication→sharding for tensor $x$, memory delta per rank is approximately:
$$
\Delta M(x)pprox |x|\left(1-rac{1}{p_g}ight).
$$

### Implementation details (practical)
1. Build process groups and bind them to physical topology intentionally; avoid accidental rank orders.
2. Choose bucket/chunk sizes so kernels stay large enough for high occupancy while communication can stream.
3. Co-schedule streams: compute stream + comm stream + dependency events to increase overlap safely.
4. Encode layout metadata in checkpoints to permit reshard/restart on changed meshes.
5. Validate numerics under mixed precision with loss-scale or FP8 scale tracking as required.

### Failure modes
- **Bandwidth-bound plateau:** throughput stops scaling as payload term dominates.
- **Latency-bound micro-ops:** too many tiny collectives; startup cost dominates.
- **Imbalance/stragglers:** rank skew from load imbalance or topology asymmetry.
- **Memory spikes:** poor prefetch/reshard boundaries or oversized microbatching.
- **Numerical drift:** precision policy mismatches across shards/collectives.

### Diagnostics and experiments
- Perform ablations: vary one axis degree at a time ($P_{tp}$, $P_{pp}$, etc.).
- Trace per-rank timelines to measure overlap ratio $ho=rac{T_{overlap}}{T_{comm}}$.
- Sweep microbatch count $m$, gradient bucket sizes, and activation checkpoint boundaries.
- Capture p50/p95/p99 iteration time and correlate with collective/kernel breakdowns.

### Study checklist
- Can you state the exact input/output layout for this subsection?
- Can you predict which collective dominates at larger scale?
- Can you explain one condition where this method should *not* be used?

### 2.10 Caveats
### Learning goals
- Understand the exact role of **2.10 Caveats** in the full training/serving stack.
- Identify required tensor layouts before and after this step.
- Map algorithmic choices to communication, memory, and stability consequences.

### Mechanism
- **Inputs/state.** Define local shards, global semantics, and invariants (shape/dtype/order).
- **Transformation.** Execute compute kernels plus communication primitives to move from input layout to output layout.
- **Correctness.** Preserve mathematical equivalence (modulo floating-point ordering) and optimizer semantics.
- **Output contract.** Ensure downstream subsection receives expected placement/state.

### Mathematical view
- Local shard sizing follows $n_{local}=n/p$ along sharded axis, with replicated axes unchanged.
- Runtime contribution is modeled as:
$$
T_{subsec}pprox T_{kernel}+T_{comm}-T_{overlap}+T_{sync}+T_{runtime}.
$$
- If this stage adds collective payload $q$ on group size $p_g$, then increment:
$$
\Delta T_{comm}pprox lpha f(p_g)+eta q g(p_g).
$$
- If this stage changes replication→sharding for tensor $x$, memory delta per rank is approximately:
$$
\Delta M(x)pprox |x|\left(1-rac{1}{p_g}ight).
$$

### Implementation details (practical)
1. Build process groups and bind them to physical topology intentionally; avoid accidental rank orders.
2. Choose bucket/chunk sizes so kernels stay large enough for high occupancy while communication can stream.
3. Co-schedule streams: compute stream + comm stream + dependency events to increase overlap safely.
4. Encode layout metadata in checkpoints to permit reshard/restart on changed meshes.
5. Validate numerics under mixed precision with loss-scale or FP8 scale tracking as required.

### Failure modes
- **Bandwidth-bound plateau:** throughput stops scaling as payload term dominates.
- **Latency-bound micro-ops:** too many tiny collectives; startup cost dominates.
- **Imbalance/stragglers:** rank skew from load imbalance or topology asymmetry.
- **Memory spikes:** poor prefetch/reshard boundaries or oversized microbatching.
- **Numerical drift:** precision policy mismatches across shards/collectives.

### Diagnostics and experiments
- Perform ablations: vary one axis degree at a time ($P_{tp}$, $P_{pp}$, etc.).
- Trace per-rank timelines to measure overlap ratio $ho=rac{T_{overlap}}{T_{comm}}$.
- Sweep microbatch count $m$, gradient bucket sizes, and activation checkpoint boundaries.
- Capture p50/p95/p99 iteration time and correlate with collective/kernel breakdowns.

### Study checklist
- Can you state the exact input/output layout for this subsection?
- Can you predict which collective dominates at larger scale?
- Can you explain one condition where this method should *not* be used?

## 3. Tensor Parallelism, Sequence-Parallel Activations, and Vocabulary Parallelism
This section explains **3. Tensor Parallelism, Sequence-Parallel Activations, and Vocabulary Parallelism** from first principles, implementation mechanics, cost/memory implications, topology interactions, and operational diagnostics.

### 3.1 TP-Specific Notation and Layout Contract
### Learning goals
- Understand the exact role of **3.1 TP-Specific Notation and Layout Contract** in the full training/serving stack.
- Identify required tensor layouts before and after this step.
- Map algorithmic choices to communication, memory, and stability consequences.

### Mechanism
- **Inputs/state.** Define local shards, global semantics, and invariants (shape/dtype/order).
- **Transformation.** Execute compute kernels plus communication primitives to move from input layout to output layout.
- **Correctness.** Preserve mathematical equivalence (modulo floating-point ordering) and optimizer semantics.
- **Output contract.** Ensure downstream subsection receives expected placement/state.

### Mathematical view
- Local shard sizing follows $n_{local}=n/p$ along sharded axis, with replicated axes unchanged.
- Runtime contribution is modeled as:
$$
T_{subsec}pprox T_{kernel}+T_{comm}-T_{overlap}+T_{sync}+T_{runtime}.
$$
- If this stage adds collective payload $q$ on group size $p_g$, then increment:
$$
\Delta T_{comm}pprox lpha f(p_g)+eta q g(p_g).
$$
- If this stage changes replication→sharding for tensor $x$, memory delta per rank is approximately:
$$
\Delta M(x)pprox |x|\left(1-rac{1}{p_g}ight).
$$

### Implementation details (practical)
1. Build process groups and bind them to physical topology intentionally; avoid accidental rank orders.
2. Choose bucket/chunk sizes so kernels stay large enough for high occupancy while communication can stream.
3. Co-schedule streams: compute stream + comm stream + dependency events to increase overlap safely.
4. Encode layout metadata in checkpoints to permit reshard/restart on changed meshes.
5. Validate numerics under mixed precision with loss-scale or FP8 scale tracking as required.

### Failure modes
- **Bandwidth-bound plateau:** throughput stops scaling as payload term dominates.
- **Latency-bound micro-ops:** too many tiny collectives; startup cost dominates.
- **Imbalance/stragglers:** rank skew from load imbalance or topology asymmetry.
- **Memory spikes:** poor prefetch/reshard boundaries or oversized microbatching.
- **Numerical drift:** precision policy mismatches across shards/collectives.

### Diagnostics and experiments
- Perform ablations: vary one axis degree at a time ($P_{tp}$, $P_{pp}$, etc.).
- Trace per-rank timelines to measure overlap ratio $ho=rac{T_{overlap}}{T_{comm}}$.
- Sweep microbatch count $m$, gradient bucket sizes, and activation checkpoint boundaries.
- Capture p50/p95/p99 iteration time and correlate with collective/kernel breakdowns.

### Study checklist
- Can you state the exact input/output layout for this subsection?
- Can you predict which collective dominates at larger scale?
- Can you explain one condition where this method should *not* be used?

### 3.2 Column-Parallel Linear Layers
### Learning goals
- Understand the exact role of **3.2 Column-Parallel Linear Layers** in the full training/serving stack.
- Identify required tensor layouts before and after this step.
- Map algorithmic choices to communication, memory, and stability consequences.

### Mechanism
- **Inputs/state.** Define local shards, global semantics, and invariants (shape/dtype/order).
- **Transformation.** Execute compute kernels plus communication primitives to move from input layout to output layout.
- **Correctness.** Preserve mathematical equivalence (modulo floating-point ordering) and optimizer semantics.
- **Output contract.** Ensure downstream subsection receives expected placement/state.

### Mathematical view
- Local shard sizing follows $n_{local}=n/p$ along sharded axis, with replicated axes unchanged.
- Runtime contribution is modeled as:
$$
T_{subsec}pprox T_{kernel}+T_{comm}-T_{overlap}+T_{sync}+T_{runtime}.
$$
- If this stage adds collective payload $q$ on group size $p_g$, then increment:
$$
\Delta T_{comm}pprox lpha f(p_g)+eta q g(p_g).
$$
- If this stage changes replication→sharding for tensor $x$, memory delta per rank is approximately:
$$
\Delta M(x)pprox |x|\left(1-rac{1}{p_g}ight).
$$

### Implementation details (practical)
1. Build process groups and bind them to physical topology intentionally; avoid accidental rank orders.
2. Choose bucket/chunk sizes so kernels stay large enough for high occupancy while communication can stream.
3. Co-schedule streams: compute stream + comm stream + dependency events to increase overlap safely.
4. Encode layout metadata in checkpoints to permit reshard/restart on changed meshes.
5. Validate numerics under mixed precision with loss-scale or FP8 scale tracking as required.

### Failure modes
- **Bandwidth-bound plateau:** throughput stops scaling as payload term dominates.
- **Latency-bound micro-ops:** too many tiny collectives; startup cost dominates.
- **Imbalance/stragglers:** rank skew from load imbalance or topology asymmetry.
- **Memory spikes:** poor prefetch/reshard boundaries or oversized microbatching.
- **Numerical drift:** precision policy mismatches across shards/collectives.

### Diagnostics and experiments
- Perform ablations: vary one axis degree at a time ($P_{tp}$, $P_{pp}$, etc.).
- Trace per-rank timelines to measure overlap ratio $ho=rac{T_{overlap}}{T_{comm}}$.
- Sweep microbatch count $m$, gradient bucket sizes, and activation checkpoint boundaries.
- Capture p50/p95/p99 iteration time and correlate with collective/kernel breakdowns.

### Study checklist
- Can you state the exact input/output layout for this subsection?
- Can you predict which collective dominates at larger scale?
- Can you explain one condition where this method should *not* be used?

### 3.3 Row-Parallel Linear Layers
### Learning goals
- Understand the exact role of **3.3 Row-Parallel Linear Layers** in the full training/serving stack.
- Identify required tensor layouts before and after this step.
- Map algorithmic choices to communication, memory, and stability consequences.

### Mechanism
- **Inputs/state.** Define local shards, global semantics, and invariants (shape/dtype/order).
- **Transformation.** Execute compute kernels plus communication primitives to move from input layout to output layout.
- **Correctness.** Preserve mathematical equivalence (modulo floating-point ordering) and optimizer semantics.
- **Output contract.** Ensure downstream subsection receives expected placement/state.

### Mathematical view
- Local shard sizing follows $n_{local}=n/p$ along sharded axis, with replicated axes unchanged.
- Runtime contribution is modeled as:
$$
T_{subsec}pprox T_{kernel}+T_{comm}-T_{overlap}+T_{sync}+T_{runtime}.
$$
- If this stage adds collective payload $q$ on group size $p_g$, then increment:
$$
\Delta T_{comm}pprox lpha f(p_g)+eta q g(p_g).
$$
- If this stage changes replication→sharding for tensor $x$, memory delta per rank is approximately:
$$
\Delta M(x)pprox |x|\left(1-rac{1}{p_g}ight).
$$

### Implementation details (practical)
1. Build process groups and bind them to physical topology intentionally; avoid accidental rank orders.
2. Choose bucket/chunk sizes so kernels stay large enough for high occupancy while communication can stream.
3. Co-schedule streams: compute stream + comm stream + dependency events to increase overlap safely.
4. Encode layout metadata in checkpoints to permit reshard/restart on changed meshes.
5. Validate numerics under mixed precision with loss-scale or FP8 scale tracking as required.

### Failure modes
- **Bandwidth-bound plateau:** throughput stops scaling as payload term dominates.
- **Latency-bound micro-ops:** too many tiny collectives; startup cost dominates.
- **Imbalance/stragglers:** rank skew from load imbalance or topology asymmetry.
- **Memory spikes:** poor prefetch/reshard boundaries or oversized microbatching.
- **Numerical drift:** precision policy mismatches across shards/collectives.

### Diagnostics and experiments
- Perform ablations: vary one axis degree at a time ($P_{tp}$, $P_{pp}$, etc.).
- Trace per-rank timelines to measure overlap ratio $ho=rac{T_{overlap}}{T_{comm}}$.
- Sweep microbatch count $m$, gradient bucket sizes, and activation checkpoint boundaries.
- Capture p50/p95/p99 iteration time and correlate with collective/kernel breakdowns.

### Study checklist
- Can you state the exact input/output layout for this subsection?
- Can you predict which collective dominates at larger scale?
- Can you explain one condition where this method should *not* be used?

### 3.4 Transformer Block Recipes: MLP and Attention
### Learning goals
- Understand the exact role of **3.4 Transformer Block Recipes: MLP and Attention** in the full training/serving stack.
- Identify required tensor layouts before and after this step.
- Map algorithmic choices to communication, memory, and stability consequences.

### Mechanism
- **Inputs/state.** Define local shards, global semantics, and invariants (shape/dtype/order).
- **Transformation.** Execute compute kernels plus communication primitives to move from input layout to output layout.
- **Correctness.** Preserve mathematical equivalence (modulo floating-point ordering) and optimizer semantics.
- **Output contract.** Ensure downstream subsection receives expected placement/state.

### Mathematical view
- Local shard sizing follows $n_{local}=n/p$ along sharded axis, with replicated axes unchanged.
- Runtime contribution is modeled as:
$$
T_{subsec}pprox T_{kernel}+T_{comm}-T_{overlap}+T_{sync}+T_{runtime}.
$$
- If this stage adds collective payload $q$ on group size $p_g$, then increment:
$$
\Delta T_{comm}pprox lpha f(p_g)+eta q g(p_g).
$$
- If this stage changes replication→sharding for tensor $x$, memory delta per rank is approximately:
$$
\Delta M(x)pprox |x|\left(1-rac{1}{p_g}ight).
$$

### Implementation details (practical)
1. Build process groups and bind them to physical topology intentionally; avoid accidental rank orders.
2. Choose bucket/chunk sizes so kernels stay large enough for high occupancy while communication can stream.
3. Co-schedule streams: compute stream + comm stream + dependency events to increase overlap safely.
4. Encode layout metadata in checkpoints to permit reshard/restart on changed meshes.
5. Validate numerics under mixed precision with loss-scale or FP8 scale tracking as required.

### Failure modes
- **Bandwidth-bound plateau:** throughput stops scaling as payload term dominates.
- **Latency-bound micro-ops:** too many tiny collectives; startup cost dominates.
- **Imbalance/stragglers:** rank skew from load imbalance or topology asymmetry.
- **Memory spikes:** poor prefetch/reshard boundaries or oversized microbatching.
- **Numerical drift:** precision policy mismatches across shards/collectives.

### Diagnostics and experiments
- Perform ablations: vary one axis degree at a time ($P_{tp}$, $P_{pp}$, etc.).
- Trace per-rank timelines to measure overlap ratio $ho=rac{T_{overlap}}{T_{comm}}$.
- Sweep microbatch count $m$, gradient bucket sizes, and activation checkpoint boundaries.
- Capture p50/p95/p99 iteration time and correlate with collective/kernel breakdowns.

### Study checklist
- Can you state the exact input/output layout for this subsection?
- Can you predict which collective dominates at larger scale?
- Can you explain one condition where this method should *not* be used?

#### MLP / SwiGLU
### Learning goals
- Understand the exact role of **MLP / SwiGLU** in the full training/serving stack.
- Identify required tensor layouts before and after this step.
- Map algorithmic choices to communication, memory, and stability consequences.

### Mechanism
- **Inputs/state.** Define local shards, global semantics, and invariants (shape/dtype/order).
- **Transformation.** Execute compute kernels plus communication primitives to move from input layout to output layout.
- **Correctness.** Preserve mathematical equivalence (modulo floating-point ordering) and optimizer semantics.
- **Output contract.** Ensure downstream subsection receives expected placement/state.

### Mathematical view
- Local shard sizing follows $n_{local}=n/p$ along sharded axis, with replicated axes unchanged.
- Runtime contribution is modeled as:
$$
T_{subsec}pprox T_{kernel}+T_{comm}-T_{overlap}+T_{sync}+T_{runtime}.
$$
- If this stage adds collective payload $q$ on group size $p_g$, then increment:
$$
\Delta T_{comm}pprox lpha f(p_g)+eta q g(p_g).
$$
- If this stage changes replication→sharding for tensor $x$, memory delta per rank is approximately:
$$
\Delta M(x)pprox |x|\left(1-rac{1}{p_g}ight).
$$

### Implementation details (practical)
1. Build process groups and bind them to physical topology intentionally; avoid accidental rank orders.
2. Choose bucket/chunk sizes so kernels stay large enough for high occupancy while communication can stream.
3. Co-schedule streams: compute stream + comm stream + dependency events to increase overlap safely.
4. Encode layout metadata in checkpoints to permit reshard/restart on changed meshes.
5. Validate numerics under mixed precision with loss-scale or FP8 scale tracking as required.

### Failure modes
- **Bandwidth-bound plateau:** throughput stops scaling as payload term dominates.
- **Latency-bound micro-ops:** too many tiny collectives; startup cost dominates.
- **Imbalance/stragglers:** rank skew from load imbalance or topology asymmetry.
- **Memory spikes:** poor prefetch/reshard boundaries or oversized microbatching.
- **Numerical drift:** precision policy mismatches across shards/collectives.

### Diagnostics and experiments
- Perform ablations: vary one axis degree at a time ($P_{tp}$, $P_{pp}$, etc.).
- Trace per-rank timelines to measure overlap ratio $ho=rac{T_{overlap}}{T_{comm}}$.
- Sweep microbatch count $m$, gradient bucket sizes, and activation checkpoint boundaries.
- Capture p50/p95/p99 iteration time and correlate with collective/kernel breakdowns.

### Study checklist
- Can you state the exact input/output layout for this subsection?
- Can you predict which collective dominates at larger scale?
- Can you explain one condition where this method should *not* be used?

#### Attention
### Learning goals
- Understand the exact role of **Attention** in the full training/serving stack.
- Identify required tensor layouts before and after this step.
- Map algorithmic choices to communication, memory, and stability consequences.

### Mechanism
- **Inputs/state.** Define local shards, global semantics, and invariants (shape/dtype/order).
- **Transformation.** Execute compute kernels plus communication primitives to move from input layout to output layout.
- **Correctness.** Preserve mathematical equivalence (modulo floating-point ordering) and optimizer semantics.
- **Output contract.** Ensure downstream subsection receives expected placement/state.

### Mathematical view
- Local shard sizing follows $n_{local}=n/p$ along sharded axis, with replicated axes unchanged.
- Runtime contribution is modeled as:
$$
T_{subsec}pprox T_{kernel}+T_{comm}-T_{overlap}+T_{sync}+T_{runtime}.
$$
- If this stage adds collective payload $q$ on group size $p_g$, then increment:
$$
\Delta T_{comm}pprox lpha f(p_g)+eta q g(p_g).
$$
- If this stage changes replication→sharding for tensor $x$, memory delta per rank is approximately:
$$
\Delta M(x)pprox |x|\left(1-rac{1}{p_g}ight).
$$

### Implementation details (practical)
1. Build process groups and bind them to physical topology intentionally; avoid accidental rank orders.
2. Choose bucket/chunk sizes so kernels stay large enough for high occupancy while communication can stream.
3. Co-schedule streams: compute stream + comm stream + dependency events to increase overlap safely.
4. Encode layout metadata in checkpoints to permit reshard/restart on changed meshes.
5. Validate numerics under mixed precision with loss-scale or FP8 scale tracking as required.

### Failure modes
- **Bandwidth-bound plateau:** throughput stops scaling as payload term dominates.
- **Latency-bound micro-ops:** too many tiny collectives; startup cost dominates.
- **Imbalance/stragglers:** rank skew from load imbalance or topology asymmetry.
- **Memory spikes:** poor prefetch/reshard boundaries or oversized microbatching.
- **Numerical drift:** precision policy mismatches across shards/collectives.

### Diagnostics and experiments
- Perform ablations: vary one axis degree at a time ($P_{tp}$, $P_{pp}$, etc.).
- Trace per-rank timelines to measure overlap ratio $ho=rac{T_{overlap}}{T_{comm}}$.
- Sweep microbatch count $m$, gradient bucket sizes, and activation checkpoint boundaries.
- Capture p50/p95/p99 iteration time and correlate with collective/kernel breakdowns.

### Study checklist
- Can you state the exact input/output layout for this subsection?
- Can you predict which collective dominates at larger scale?
- Can you explain one condition where this method should *not* be used?

### 3.5 Sequence-Parallel Activations as a TP Companion
### Learning goals
- Understand the exact role of **3.5 Sequence-Parallel Activations as a TP Companion** in the full training/serving stack.
- Identify required tensor layouts before and after this step.
- Map algorithmic choices to communication, memory, and stability consequences.

### Mechanism
- **Inputs/state.** Define local shards, global semantics, and invariants (shape/dtype/order).
- **Transformation.** Execute compute kernels plus communication primitives to move from input layout to output layout.
- **Correctness.** Preserve mathematical equivalence (modulo floating-point ordering) and optimizer semantics.
- **Output contract.** Ensure downstream subsection receives expected placement/state.

### Mathematical view
- Local shard sizing follows $n_{local}=n/p$ along sharded axis, with replicated axes unchanged.
- Runtime contribution is modeled as:
$$
T_{subsec}pprox T_{kernel}+T_{comm}-T_{overlap}+T_{sync}+T_{runtime}.
$$
- If this stage adds collective payload $q$ on group size $p_g$, then increment:
$$
\Delta T_{comm}pprox lpha f(p_g)+eta q g(p_g).
$$
- If this stage changes replication→sharding for tensor $x$, memory delta per rank is approximately:
$$
\Delta M(x)pprox |x|\left(1-rac{1}{p_g}ight).
$$

### Implementation details (practical)
1. Build process groups and bind them to physical topology intentionally; avoid accidental rank orders.
2. Choose bucket/chunk sizes so kernels stay large enough for high occupancy while communication can stream.
3. Co-schedule streams: compute stream + comm stream + dependency events to increase overlap safely.
4. Encode layout metadata in checkpoints to permit reshard/restart on changed meshes.
5. Validate numerics under mixed precision with loss-scale or FP8 scale tracking as required.

### Failure modes
- **Bandwidth-bound plateau:** throughput stops scaling as payload term dominates.
- **Latency-bound micro-ops:** too many tiny collectives; startup cost dominates.
- **Imbalance/stragglers:** rank skew from load imbalance or topology asymmetry.
- **Memory spikes:** poor prefetch/reshard boundaries or oversized microbatching.
- **Numerical drift:** precision policy mismatches across shards/collectives.

### Diagnostics and experiments
- Perform ablations: vary one axis degree at a time ($P_{tp}$, $P_{pp}$, etc.).
- Trace per-rank timelines to measure overlap ratio $ho=rac{T_{overlap}}{T_{comm}}$.
- Sweep microbatch count $m$, gradient bucket sizes, and activation checkpoint boundaries.
- Capture p50/p95/p99 iteration time and correlate with collective/kernel breakdowns.

### Study checklist
- Can you state the exact input/output layout for this subsection?
- Can you predict which collective dominates at larger scale?
- Can you explain one condition where this method should *not* be used?

### 3.6 Vocabulary Parallelism
### Learning goals
- Understand the exact role of **3.6 Vocabulary Parallelism** in the full training/serving stack.
- Identify required tensor layouts before and after this step.
- Map algorithmic choices to communication, memory, and stability consequences.

### Mechanism
- **Inputs/state.** Define local shards, global semantics, and invariants (shape/dtype/order).
- **Transformation.** Execute compute kernels plus communication primitives to move from input layout to output layout.
- **Correctness.** Preserve mathematical equivalence (modulo floating-point ordering) and optimizer semantics.
- **Output contract.** Ensure downstream subsection receives expected placement/state.

### Mathematical view
- Local shard sizing follows $n_{local}=n/p$ along sharded axis, with replicated axes unchanged.
- Runtime contribution is modeled as:
$$
T_{subsec}pprox T_{kernel}+T_{comm}-T_{overlap}+T_{sync}+T_{runtime}.
$$
- If this stage adds collective payload $q$ on group size $p_g$, then increment:
$$
\Delta T_{comm}pprox lpha f(p_g)+eta q g(p_g).
$$
- If this stage changes replication→sharding for tensor $x$, memory delta per rank is approximately:
$$
\Delta M(x)pprox |x|\left(1-rac{1}{p_g}ight).
$$

### Implementation details (practical)
1. Build process groups and bind them to physical topology intentionally; avoid accidental rank orders.
2. Choose bucket/chunk sizes so kernels stay large enough for high occupancy while communication can stream.
3. Co-schedule streams: compute stream + comm stream + dependency events to increase overlap safely.
4. Encode layout metadata in checkpoints to permit reshard/restart on changed meshes.
5. Validate numerics under mixed precision with loss-scale or FP8 scale tracking as required.

### Failure modes
- **Bandwidth-bound plateau:** throughput stops scaling as payload term dominates.
- **Latency-bound micro-ops:** too many tiny collectives; startup cost dominates.
- **Imbalance/stragglers:** rank skew from load imbalance or topology asymmetry.
- **Memory spikes:** poor prefetch/reshard boundaries or oversized microbatching.
- **Numerical drift:** precision policy mismatches across shards/collectives.

### Diagnostics and experiments
- Perform ablations: vary one axis degree at a time ($P_{tp}$, $P_{pp}$, etc.).
- Trace per-rank timelines to measure overlap ratio $ho=rac{T_{overlap}}{T_{comm}}$.
- Sweep microbatch count $m$, gradient bucket sizes, and activation checkpoint boundaries.
- Capture p50/p95/p99 iteration time and correlate with collective/kernel breakdowns.

### Study checklist
- Can you state the exact input/output layout for this subsection?
- Can you predict which collective dominates at larger scale?
- Can you explain one condition where this method should *not* be used?

#### Input Embedding
### Learning goals
- Understand the exact role of **Input Embedding** in the full training/serving stack.
- Identify required tensor layouts before and after this step.
- Map algorithmic choices to communication, memory, and stability consequences.

### Mechanism
- **Inputs/state.** Define local shards, global semantics, and invariants (shape/dtype/order).
- **Transformation.** Execute compute kernels plus communication primitives to move from input layout to output layout.
- **Correctness.** Preserve mathematical equivalence (modulo floating-point ordering) and optimizer semantics.
- **Output contract.** Ensure downstream subsection receives expected placement/state.

### Mathematical view
- Local shard sizing follows $n_{local}=n/p$ along sharded axis, with replicated axes unchanged.
- Runtime contribution is modeled as:
$$
T_{subsec}pprox T_{kernel}+T_{comm}-T_{overlap}+T_{sync}+T_{runtime}.
$$
- If this stage adds collective payload $q$ on group size $p_g$, then increment:
$$
\Delta T_{comm}pprox lpha f(p_g)+eta q g(p_g).
$$
- If this stage changes replication→sharding for tensor $x$, memory delta per rank is approximately:
$$
\Delta M(x)pprox |x|\left(1-rac{1}{p_g}ight).
$$

### Implementation details (practical)
1. Build process groups and bind them to physical topology intentionally; avoid accidental rank orders.
2. Choose bucket/chunk sizes so kernels stay large enough for high occupancy while communication can stream.
3. Co-schedule streams: compute stream + comm stream + dependency events to increase overlap safely.
4. Encode layout metadata in checkpoints to permit reshard/restart on changed meshes.
5. Validate numerics under mixed precision with loss-scale or FP8 scale tracking as required.

### Failure modes
- **Bandwidth-bound plateau:** throughput stops scaling as payload term dominates.
- **Latency-bound micro-ops:** too many tiny collectives; startup cost dominates.
- **Imbalance/stragglers:** rank skew from load imbalance or topology asymmetry.
- **Memory spikes:** poor prefetch/reshard boundaries or oversized microbatching.
- **Numerical drift:** precision policy mismatches across shards/collectives.

### Diagnostics and experiments
- Perform ablations: vary one axis degree at a time ($P_{tp}$, $P_{pp}$, etc.).
- Trace per-rank timelines to measure overlap ratio $ho=rac{T_{overlap}}{T_{comm}}$.
- Sweep microbatch count $m$, gradient bucket sizes, and activation checkpoint boundaries.
- Capture p50/p95/p99 iteration time and correlate with collective/kernel breakdowns.

### Study checklist
- Can you state the exact input/output layout for this subsection?
- Can you predict which collective dominates at larger scale?
- Can you explain one condition where this method should *not* be used?

#### LM Head and Cross-Entropy
### Learning goals
- Understand the exact role of **LM Head and Cross-Entropy** in the full training/serving stack.
- Identify required tensor layouts before and after this step.
- Map algorithmic choices to communication, memory, and stability consequences.

### Mechanism
- **Inputs/state.** Define local shards, global semantics, and invariants (shape/dtype/order).
- **Transformation.** Execute compute kernels plus communication primitives to move from input layout to output layout.
- **Correctness.** Preserve mathematical equivalence (modulo floating-point ordering) and optimizer semantics.
- **Output contract.** Ensure downstream subsection receives expected placement/state.

### Mathematical view
- Local shard sizing follows $n_{local}=n/p$ along sharded axis, with replicated axes unchanged.
- Runtime contribution is modeled as:
$$
T_{subsec}pprox T_{kernel}+T_{comm}-T_{overlap}+T_{sync}+T_{runtime}.
$$
- If this stage adds collective payload $q$ on group size $p_g$, then increment:
$$
\Delta T_{comm}pprox lpha f(p_g)+eta q g(p_g).
$$
- If this stage changes replication→sharding for tensor $x$, memory delta per rank is approximately:
$$
\Delta M(x)pprox |x|\left(1-rac{1}{p_g}ight).
$$

### Implementation details (practical)
1. Build process groups and bind them to physical topology intentionally; avoid accidental rank orders.
2. Choose bucket/chunk sizes so kernels stay large enough for high occupancy while communication can stream.
3. Co-schedule streams: compute stream + comm stream + dependency events to increase overlap safely.
4. Encode layout metadata in checkpoints to permit reshard/restart on changed meshes.
5. Validate numerics under mixed precision with loss-scale or FP8 scale tracking as required.

### Failure modes
- **Bandwidth-bound plateau:** throughput stops scaling as payload term dominates.
- **Latency-bound micro-ops:** too many tiny collectives; startup cost dominates.
- **Imbalance/stragglers:** rank skew from load imbalance or topology asymmetry.
- **Memory spikes:** poor prefetch/reshard boundaries or oversized microbatching.
- **Numerical drift:** precision policy mismatches across shards/collectives.

### Diagnostics and experiments
- Perform ablations: vary one axis degree at a time ($P_{tp}$, $P_{pp}$, etc.).
- Trace per-rank timelines to measure overlap ratio $ho=rac{T_{overlap}}{T_{comm}}$.
- Sweep microbatch count $m$, gradient bucket sizes, and activation checkpoint boundaries.
- Capture p50/p95/p99 iteration time and correlate with collective/kernel breakdowns.

### Study checklist
- Can you state the exact input/output layout for this subsection?
- Can you predict which collective dominates at larger scale?
- Can you explain one condition where this method should *not* be used?

### 3.7 Async Tensor Parallelism
### Learning goals
- Understand the exact role of **3.7 Async Tensor Parallelism** in the full training/serving stack.
- Identify required tensor layouts before and after this step.
- Map algorithmic choices to communication, memory, and stability consequences.

### Mechanism
- **Inputs/state.** Define local shards, global semantics, and invariants (shape/dtype/order).
- **Transformation.** Execute compute kernels plus communication primitives to move from input layout to output layout.
- **Correctness.** Preserve mathematical equivalence (modulo floating-point ordering) and optimizer semantics.
- **Output contract.** Ensure downstream subsection receives expected placement/state.

### Mathematical view
- Local shard sizing follows $n_{local}=n/p$ along sharded axis, with replicated axes unchanged.
- Runtime contribution is modeled as:
$$
T_{subsec}pprox T_{kernel}+T_{comm}-T_{overlap}+T_{sync}+T_{runtime}.
$$
- If this stage adds collective payload $q$ on group size $p_g$, then increment:
$$
\Delta T_{comm}pprox lpha f(p_g)+eta q g(p_g).
$$
- If this stage changes replication→sharding for tensor $x$, memory delta per rank is approximately:
$$
\Delta M(x)pprox |x|\left(1-rac{1}{p_g}ight).
$$

### Implementation details (practical)
1. Build process groups and bind them to physical topology intentionally; avoid accidental rank orders.
2. Choose bucket/chunk sizes so kernels stay large enough for high occupancy while communication can stream.
3. Co-schedule streams: compute stream + comm stream + dependency events to increase overlap safely.
4. Encode layout metadata in checkpoints to permit reshard/restart on changed meshes.
5. Validate numerics under mixed precision with loss-scale or FP8 scale tracking as required.

### Failure modes
- **Bandwidth-bound plateau:** throughput stops scaling as payload term dominates.
- **Latency-bound micro-ops:** too many tiny collectives; startup cost dominates.
- **Imbalance/stragglers:** rank skew from load imbalance or topology asymmetry.
- **Memory spikes:** poor prefetch/reshard boundaries or oversized microbatching.
- **Numerical drift:** precision policy mismatches across shards/collectives.

### Diagnostics and experiments
- Perform ablations: vary one axis degree at a time ($P_{tp}$, $P_{pp}$, etc.).
- Trace per-rank timelines to measure overlap ratio $ho=rac{T_{overlap}}{T_{comm}}$.
- Sweep microbatch count $m$, gradient bucket sizes, and activation checkpoint boundaries.
- Capture p50/p95/p99 iteration time and correlate with collective/kernel breakdowns.

### Study checklist
- Can you state the exact input/output layout for this subsection?
- Can you predict which collective dominates at larger scale?
- Can you explain one condition where this method should *not* be used?

### 3.8 TP-Aware CUDA and Kernel Techniques
### Learning goals
- Understand the exact role of **3.8 TP-Aware CUDA and Kernel Techniques** in the full training/serving stack.
- Identify required tensor layouts before and after this step.
- Map algorithmic choices to communication, memory, and stability consequences.

### Mechanism
- **Inputs/state.** Define local shards, global semantics, and invariants (shape/dtype/order).
- **Transformation.** Execute compute kernels plus communication primitives to move from input layout to output layout.
- **Correctness.** Preserve mathematical equivalence (modulo floating-point ordering) and optimizer semantics.
- **Output contract.** Ensure downstream subsection receives expected placement/state.

### Mathematical view
- Local shard sizing follows $n_{local}=n/p$ along sharded axis, with replicated axes unchanged.
- Runtime contribution is modeled as:
$$
T_{subsec}pprox T_{kernel}+T_{comm}-T_{overlap}+T_{sync}+T_{runtime}.
$$
- If this stage adds collective payload $q$ on group size $p_g$, then increment:
$$
\Delta T_{comm}pprox lpha f(p_g)+eta q g(p_g).
$$
- If this stage changes replication→sharding for tensor $x$, memory delta per rank is approximately:
$$
\Delta M(x)pprox |x|\left(1-rac{1}{p_g}ight).
$$

### Implementation details (practical)
1. Build process groups and bind them to physical topology intentionally; avoid accidental rank orders.
2. Choose bucket/chunk sizes so kernels stay large enough for high occupancy while communication can stream.
3. Co-schedule streams: compute stream + comm stream + dependency events to increase overlap safely.
4. Encode layout metadata in checkpoints to permit reshard/restart on changed meshes.
5. Validate numerics under mixed precision with loss-scale or FP8 scale tracking as required.

### Failure modes
- **Bandwidth-bound plateau:** throughput stops scaling as payload term dominates.
- **Latency-bound micro-ops:** too many tiny collectives; startup cost dominates.
- **Imbalance/stragglers:** rank skew from load imbalance or topology asymmetry.
- **Memory spikes:** poor prefetch/reshard boundaries or oversized microbatching.
- **Numerical drift:** precision policy mismatches across shards/collectives.

### Diagnostics and experiments
- Perform ablations: vary one axis degree at a time ($P_{tp}$, $P_{pp}$, etc.).
- Trace per-rank timelines to measure overlap ratio $ho=rac{T_{overlap}}{T_{comm}}$.
- Sweep microbatch count $m$, gradient bucket sizes, and activation checkpoint boundaries.
- Capture p50/p95/p99 iteration time and correlate with collective/kernel breakdowns.

### Study checklist
- Can you state the exact input/output layout for this subsection?
- Can you predict which collective dominates at larger scale?
- Can you explain one condition where this method should *not* be used?

#### 3.8.1 Local GEMM Shape Quality
### Learning goals
- Understand the exact role of **3.8.1 Local GEMM Shape Quality** in the full training/serving stack.
- Identify required tensor layouts before and after this step.
- Map algorithmic choices to communication, memory, and stability consequences.

### Mechanism
- **Inputs/state.** Define local shards, global semantics, and invariants (shape/dtype/order).
- **Transformation.** Execute compute kernels plus communication primitives to move from input layout to output layout.
- **Correctness.** Preserve mathematical equivalence (modulo floating-point ordering) and optimizer semantics.
- **Output contract.** Ensure downstream subsection receives expected placement/state.

### Mathematical view
- Local shard sizing follows $n_{local}=n/p$ along sharded axis, with replicated axes unchanged.
- Runtime contribution is modeled as:
$$
T_{subsec}pprox T_{kernel}+T_{comm}-T_{overlap}+T_{sync}+T_{runtime}.
$$
- If this stage adds collective payload $q$ on group size $p_g$, then increment:
$$
\Delta T_{comm}pprox lpha f(p_g)+eta q g(p_g).
$$
- If this stage changes replication→sharding for tensor $x$, memory delta per rank is approximately:
$$
\Delta M(x)pprox |x|\left(1-rac{1}{p_g}ight).
$$

### Implementation details (practical)
1. Build process groups and bind them to physical topology intentionally; avoid accidental rank orders.
2. Choose bucket/chunk sizes so kernels stay large enough for high occupancy while communication can stream.
3. Co-schedule streams: compute stream + comm stream + dependency events to increase overlap safely.
4. Encode layout metadata in checkpoints to permit reshard/restart on changed meshes.
5. Validate numerics under mixed precision with loss-scale or FP8 scale tracking as required.

### Failure modes
- **Bandwidth-bound plateau:** throughput stops scaling as payload term dominates.
- **Latency-bound micro-ops:** too many tiny collectives; startup cost dominates.
- **Imbalance/stragglers:** rank skew from load imbalance or topology asymmetry.
- **Memory spikes:** poor prefetch/reshard boundaries or oversized microbatching.
- **Numerical drift:** precision policy mismatches across shards/collectives.

### Diagnostics and experiments
- Perform ablations: vary one axis degree at a time ($P_{tp}$, $P_{pp}$, etc.).
- Trace per-rank timelines to measure overlap ratio $ho=rac{T_{overlap}}{T_{comm}}$.
- Sweep microbatch count $m$, gradient bucket sizes, and activation checkpoint boundaries.
- Capture p50/p95/p99 iteration time and correlate with collective/kernel breakdowns.

### Study checklist
- Can you state the exact input/output layout for this subsection?
- Can you predict which collective dominates at larger scale?
- Can you explain one condition where this method should *not* be used?

#### 3.8.2 Fused Epilogues and Layout-Stable Collectives
### Learning goals
- Understand the exact role of **3.8.2 Fused Epilogues and Layout-Stable Collectives** in the full training/serving stack.
- Identify required tensor layouts before and after this step.
- Map algorithmic choices to communication, memory, and stability consequences.

### Mechanism
- **Inputs/state.** Define local shards, global semantics, and invariants (shape/dtype/order).
- **Transformation.** Execute compute kernels plus communication primitives to move from input layout to output layout.
- **Correctness.** Preserve mathematical equivalence (modulo floating-point ordering) and optimizer semantics.
- **Output contract.** Ensure downstream subsection receives expected placement/state.

### Mathematical view
- Local shard sizing follows $n_{local}=n/p$ along sharded axis, with replicated axes unchanged.
- Runtime contribution is modeled as:
$$
T_{subsec}pprox T_{kernel}+T_{comm}-T_{overlap}+T_{sync}+T_{runtime}.
$$
- If this stage adds collective payload $q$ on group size $p_g$, then increment:
$$
\Delta T_{comm}pprox lpha f(p_g)+eta q g(p_g).
$$
- If this stage changes replication→sharding for tensor $x$, memory delta per rank is approximately:
$$
\Delta M(x)pprox |x|\left(1-rac{1}{p_g}ight).
$$

### Implementation details (practical)
1. Build process groups and bind them to physical topology intentionally; avoid accidental rank orders.
2. Choose bucket/chunk sizes so kernels stay large enough for high occupancy while communication can stream.
3. Co-schedule streams: compute stream + comm stream + dependency events to increase overlap safely.
4. Encode layout metadata in checkpoints to permit reshard/restart on changed meshes.
5. Validate numerics under mixed precision with loss-scale or FP8 scale tracking as required.

### Failure modes
- **Bandwidth-bound plateau:** throughput stops scaling as payload term dominates.
- **Latency-bound micro-ops:** too many tiny collectives; startup cost dominates.
- **Imbalance/stragglers:** rank skew from load imbalance or topology asymmetry.
- **Memory spikes:** poor prefetch/reshard boundaries or oversized microbatching.
- **Numerical drift:** precision policy mismatches across shards/collectives.

### Diagnostics and experiments
- Perform ablations: vary one axis degree at a time ($P_{tp}$, $P_{pp}$, etc.).
- Trace per-rank timelines to measure overlap ratio $ho=rac{T_{overlap}}{T_{comm}}$.
- Sweep microbatch count $m$, gradient bucket sizes, and activation checkpoint boundaries.
- Capture p50/p95/p99 iteration time and correlate with collective/kernel breakdowns.

### Study checklist
- Can you state the exact input/output layout for this subsection?
- Can you predict which collective dominates at larger scale?
- Can you explain one condition where this method should *not* be used?

#### 3.8.3 Low-Precision Communication and Scale Layout
### Learning goals
- Understand the exact role of **3.8.3 Low-Precision Communication and Scale Layout** in the full training/serving stack.
- Identify required tensor layouts before and after this step.
- Map algorithmic choices to communication, memory, and stability consequences.

### Mechanism
- **Inputs/state.** Define local shards, global semantics, and invariants (shape/dtype/order).
- **Transformation.** Execute compute kernels plus communication primitives to move from input layout to output layout.
- **Correctness.** Preserve mathematical equivalence (modulo floating-point ordering) and optimizer semantics.
- **Output contract.** Ensure downstream subsection receives expected placement/state.

### Mathematical view
- Local shard sizing follows $n_{local}=n/p$ along sharded axis, with replicated axes unchanged.
- Runtime contribution is modeled as:
$$
T_{subsec}pprox T_{kernel}+T_{comm}-T_{overlap}+T_{sync}+T_{runtime}.
$$
- If this stage adds collective payload $q$ on group size $p_g$, then increment:
$$
\Delta T_{comm}pprox lpha f(p_g)+eta q g(p_g).
$$
- If this stage changes replication→sharding for tensor $x$, memory delta per rank is approximately:
$$
\Delta M(x)pprox |x|\left(1-rac{1}{p_g}ight).
$$

### Implementation details (practical)
1. Build process groups and bind them to physical topology intentionally; avoid accidental rank orders.
2. Choose bucket/chunk sizes so kernels stay large enough for high occupancy while communication can stream.
3. Co-schedule streams: compute stream + comm stream + dependency events to increase overlap safely.
4. Encode layout metadata in checkpoints to permit reshard/restart on changed meshes.
5. Validate numerics under mixed precision with loss-scale or FP8 scale tracking as required.

### Failure modes
- **Bandwidth-bound plateau:** throughput stops scaling as payload term dominates.
- **Latency-bound micro-ops:** too many tiny collectives; startup cost dominates.
- **Imbalance/stragglers:** rank skew from load imbalance or topology asymmetry.
- **Memory spikes:** poor prefetch/reshard boundaries or oversized microbatching.
- **Numerical drift:** precision policy mismatches across shards/collectives.

### Diagnostics and experiments
- Perform ablations: vary one axis degree at a time ($P_{tp}$, $P_{pp}$, etc.).
- Trace per-rank timelines to measure overlap ratio $ho=rac{T_{overlap}}{T_{comm}}$.
- Sweep microbatch count $m$, gradient bucket sizes, and activation checkpoint boundaries.
- Capture p50/p95/p99 iteration time and correlate with collective/kernel breakdowns.

### Study checklist
- Can you state the exact input/output layout for this subsection?
- Can you predict which collective dominates at larger scale?
- Can you explain one condition where this method should *not* be used?

#### 3.8.4 TP-Aware Megakernels
### Learning goals
- Understand the exact role of **3.8.4 TP-Aware Megakernels** in the full training/serving stack.
- Identify required tensor layouts before and after this step.
- Map algorithmic choices to communication, memory, and stability consequences.

### Mechanism
- **Inputs/state.** Define local shards, global semantics, and invariants (shape/dtype/order).
- **Transformation.** Execute compute kernels plus communication primitives to move from input layout to output layout.
- **Correctness.** Preserve mathematical equivalence (modulo floating-point ordering) and optimizer semantics.
- **Output contract.** Ensure downstream subsection receives expected placement/state.

### Mathematical view
- Local shard sizing follows $n_{local}=n/p$ along sharded axis, with replicated axes unchanged.
- Runtime contribution is modeled as:
$$
T_{subsec}pprox T_{kernel}+T_{comm}-T_{overlap}+T_{sync}+T_{runtime}.
$$
- If this stage adds collective payload $q$ on group size $p_g$, then increment:
$$
\Delta T_{comm}pprox lpha f(p_g)+eta q g(p_g).
$$
- If this stage changes replication→sharding for tensor $x$, memory delta per rank is approximately:
$$
\Delta M(x)pprox |x|\left(1-rac{1}{p_g}ight).
$$

### Implementation details (practical)
1. Build process groups and bind them to physical topology intentionally; avoid accidental rank orders.
2. Choose bucket/chunk sizes so kernels stay large enough for high occupancy while communication can stream.
3. Co-schedule streams: compute stream + comm stream + dependency events to increase overlap safely.
4. Encode layout metadata in checkpoints to permit reshard/restart on changed meshes.
5. Validate numerics under mixed precision with loss-scale or FP8 scale tracking as required.

### Failure modes
- **Bandwidth-bound plateau:** throughput stops scaling as payload term dominates.
- **Latency-bound micro-ops:** too many tiny collectives; startup cost dominates.
- **Imbalance/stragglers:** rank skew from load imbalance or topology asymmetry.
- **Memory spikes:** poor prefetch/reshard boundaries or oversized microbatching.
- **Numerical drift:** precision policy mismatches across shards/collectives.

### Diagnostics and experiments
- Perform ablations: vary one axis degree at a time ($P_{tp}$, $P_{pp}$, etc.).
- Trace per-rank timelines to measure overlap ratio $ho=rac{T_{overlap}}{T_{comm}}$.
- Sweep microbatch count $m$, gradient bucket sizes, and activation checkpoint boundaries.
- Capture p50/p95/p99 iteration time and correlate with collective/kernel breakdowns.

### Study checklist
- Can you state the exact input/output layout for this subsection?
- Can you predict which collective dominates at larger scale?
- Can you explain one condition where this method should *not* be used?

### 3.9 Choosing a TP Degree
### Learning goals
- Understand the exact role of **3.9 Choosing a TP Degree** in the full training/serving stack.
- Identify required tensor layouts before and after this step.
- Map algorithmic choices to communication, memory, and stability consequences.

### Mechanism
- **Inputs/state.** Define local shards, global semantics, and invariants (shape/dtype/order).
- **Transformation.** Execute compute kernels plus communication primitives to move from input layout to output layout.
- **Correctness.** Preserve mathematical equivalence (modulo floating-point ordering) and optimizer semantics.
- **Output contract.** Ensure downstream subsection receives expected placement/state.

### Mathematical view
- Local shard sizing follows $n_{local}=n/p$ along sharded axis, with replicated axes unchanged.
- Runtime contribution is modeled as:
$$
T_{subsec}pprox T_{kernel}+T_{comm}-T_{overlap}+T_{sync}+T_{runtime}.
$$
- If this stage adds collective payload $q$ on group size $p_g$, then increment:
$$
\Delta T_{comm}pprox lpha f(p_g)+eta q g(p_g).
$$
- If this stage changes replication→sharding for tensor $x$, memory delta per rank is approximately:
$$
\Delta M(x)pprox |x|\left(1-rac{1}{p_g}ight).
$$

### Implementation details (practical)
1. Build process groups and bind them to physical topology intentionally; avoid accidental rank orders.
2. Choose bucket/chunk sizes so kernels stay large enough for high occupancy while communication can stream.
3. Co-schedule streams: compute stream + comm stream + dependency events to increase overlap safely.
4. Encode layout metadata in checkpoints to permit reshard/restart on changed meshes.
5. Validate numerics under mixed precision with loss-scale or FP8 scale tracking as required.

### Failure modes
- **Bandwidth-bound plateau:** throughput stops scaling as payload term dominates.
- **Latency-bound micro-ops:** too many tiny collectives; startup cost dominates.
- **Imbalance/stragglers:** rank skew from load imbalance or topology asymmetry.
- **Memory spikes:** poor prefetch/reshard boundaries or oversized microbatching.
- **Numerical drift:** precision policy mismatches across shards/collectives.

### Diagnostics and experiments
- Perform ablations: vary one axis degree at a time ($P_{tp}$, $P_{pp}$, etc.).
- Trace per-rank timelines to measure overlap ratio $ho=rac{T_{overlap}}{T_{comm}}$.
- Sweep microbatch count $m$, gradient bucket sizes, and activation checkpoint boundaries.
- Capture p50/p95/p99 iteration time and correlate with collective/kernel breakdowns.

### Study checklist
- Can you state the exact input/output layout for this subsection?
- Can you predict which collective dominates at larger scale?
- Can you explain one condition where this method should *not* be used?

### 3.10 Sources and Lineage
### Learning goals
- Understand the exact role of **3.10 Sources and Lineage** in the full training/serving stack.
- Identify required tensor layouts before and after this step.
- Map algorithmic choices to communication, memory, and stability consequences.

### Mechanism
- **Inputs/state.** Define local shards, global semantics, and invariants (shape/dtype/order).
- **Transformation.** Execute compute kernels plus communication primitives to move from input layout to output layout.
- **Correctness.** Preserve mathematical equivalence (modulo floating-point ordering) and optimizer semantics.
- **Output contract.** Ensure downstream subsection receives expected placement/state.

### Mathematical view
- Local shard sizing follows $n_{local}=n/p$ along sharded axis, with replicated axes unchanged.
- Runtime contribution is modeled as:
$$
T_{subsec}pprox T_{kernel}+T_{comm}-T_{overlap}+T_{sync}+T_{runtime}.
$$
- If this stage adds collective payload $q$ on group size $p_g$, then increment:
$$
\Delta T_{comm}pprox lpha f(p_g)+eta q g(p_g).
$$
- If this stage changes replication→sharding for tensor $x$, memory delta per rank is approximately:
$$
\Delta M(x)pprox |x|\left(1-rac{1}{p_g}ight).
$$

### Implementation details (practical)
1. Build process groups and bind them to physical topology intentionally; avoid accidental rank orders.
2. Choose bucket/chunk sizes so kernels stay large enough for high occupancy while communication can stream.
3. Co-schedule streams: compute stream + comm stream + dependency events to increase overlap safely.
4. Encode layout metadata in checkpoints to permit reshard/restart on changed meshes.
5. Validate numerics under mixed precision with loss-scale or FP8 scale tracking as required.

### Failure modes
- **Bandwidth-bound plateau:** throughput stops scaling as payload term dominates.
- **Latency-bound micro-ops:** too many tiny collectives; startup cost dominates.
- **Imbalance/stragglers:** rank skew from load imbalance or topology asymmetry.
- **Memory spikes:** poor prefetch/reshard boundaries or oversized microbatching.
- **Numerical drift:** precision policy mismatches across shards/collectives.

### Diagnostics and experiments
- Perform ablations: vary one axis degree at a time ($P_{tp}$, $P_{pp}$, etc.).
- Trace per-rank timelines to measure overlap ratio $ho=rac{T_{overlap}}{T_{comm}}$.
- Sweep microbatch count $m$, gradient bucket sizes, and activation checkpoint boundaries.
- Capture p50/p95/p99 iteration time and correlate with collective/kernel breakdowns.

### Study checklist
- Can you state the exact input/output layout for this subsection?
- Can you predict which collective dominates at larger scale?
- Can you explain one condition where this method should *not* be used?

### 3.11 Caveats
### Learning goals
- Understand the exact role of **3.11 Caveats** in the full training/serving stack.
- Identify required tensor layouts before and after this step.
- Map algorithmic choices to communication, memory, and stability consequences.

### Mechanism
- **Inputs/state.** Define local shards, global semantics, and invariants (shape/dtype/order).
- **Transformation.** Execute compute kernels plus communication primitives to move from input layout to output layout.
- **Correctness.** Preserve mathematical equivalence (modulo floating-point ordering) and optimizer semantics.
- **Output contract.** Ensure downstream subsection receives expected placement/state.

### Mathematical view
- Local shard sizing follows $n_{local}=n/p$ along sharded axis, with replicated axes unchanged.
- Runtime contribution is modeled as:
$$
T_{subsec}pprox T_{kernel}+T_{comm}-T_{overlap}+T_{sync}+T_{runtime}.
$$
- If this stage adds collective payload $q$ on group size $p_g$, then increment:
$$
\Delta T_{comm}pprox lpha f(p_g)+eta q g(p_g).
$$
- If this stage changes replication→sharding for tensor $x$, memory delta per rank is approximately:
$$
\Delta M(x)pprox |x|\left(1-rac{1}{p_g}ight).
$$

### Implementation details (practical)
1. Build process groups and bind them to physical topology intentionally; avoid accidental rank orders.
2. Choose bucket/chunk sizes so kernels stay large enough for high occupancy while communication can stream.
3. Co-schedule streams: compute stream + comm stream + dependency events to increase overlap safely.
4. Encode layout metadata in checkpoints to permit reshard/restart on changed meshes.
5. Validate numerics under mixed precision with loss-scale or FP8 scale tracking as required.

### Failure modes
- **Bandwidth-bound plateau:** throughput stops scaling as payload term dominates.
- **Latency-bound micro-ops:** too many tiny collectives; startup cost dominates.
- **Imbalance/stragglers:** rank skew from load imbalance or topology asymmetry.
- **Memory spikes:** poor prefetch/reshard boundaries or oversized microbatching.
- **Numerical drift:** precision policy mismatches across shards/collectives.

### Diagnostics and experiments
- Perform ablations: vary one axis degree at a time ($P_{tp}$, $P_{pp}$, etc.).
- Trace per-rank timelines to measure overlap ratio $ho=rac{T_{overlap}}{T_{comm}}$.
- Sweep microbatch count $m$, gradient bucket sizes, and activation checkpoint boundaries.
- Capture p50/p95/p99 iteration time and correlate with collective/kernel breakdowns.

### Study checklist
- Can you state the exact input/output layout for this subsection?
- Can you predict which collective dominates at larger scale?
- Can you explain one condition where this method should *not* be used?

## 4. Pipeline Parallelism: Depth-Axis Parallelism and Modern Schedules
This section explains **4. Pipeline Parallelism: Depth-Axis Parallelism and Modern Schedules** from first principles, implementation mechanics, cost/memory implications, topology interactions, and operational diagnostics.

### 4.1 System Model, Variables, and Shapes
### Learning goals
- Understand the exact role of **4.1 System Model, Variables, and Shapes** in the full training/serving stack.
- Identify required tensor layouts before and after this step.
- Map algorithmic choices to communication, memory, and stability consequences.

### Mechanism
- **Inputs/state.** Define local shards, global semantics, and invariants (shape/dtype/order).
- **Transformation.** Execute compute kernels plus communication primitives to move from input layout to output layout.
- **Correctness.** Preserve mathematical equivalence (modulo floating-point ordering) and optimizer semantics.
- **Output contract.** Ensure downstream subsection receives expected placement/state.

### Mathematical view
- Local shard sizing follows $n_{local}=n/p$ along sharded axis, with replicated axes unchanged.
- Runtime contribution is modeled as:
$$
T_{subsec}pprox T_{kernel}+T_{comm}-T_{overlap}+T_{sync}+T_{runtime}.
$$
- If this stage adds collective payload $q$ on group size $p_g$, then increment:
$$
\Delta T_{comm}pprox lpha f(p_g)+eta q g(p_g).
$$
- If this stage changes replication→sharding for tensor $x$, memory delta per rank is approximately:
$$
\Delta M(x)pprox |x|\left(1-rac{1}{p_g}ight).
$$

### Implementation details (practical)
1. Build process groups and bind them to physical topology intentionally; avoid accidental rank orders.
2. Choose bucket/chunk sizes so kernels stay large enough for high occupancy while communication can stream.
3. Co-schedule streams: compute stream + comm stream + dependency events to increase overlap safely.
4. Encode layout metadata in checkpoints to permit reshard/restart on changed meshes.
5. Validate numerics under mixed precision with loss-scale or FP8 scale tracking as required.

### Failure modes
- **Bandwidth-bound plateau:** throughput stops scaling as payload term dominates.
- **Latency-bound micro-ops:** too many tiny collectives; startup cost dominates.
- **Imbalance/stragglers:** rank skew from load imbalance or topology asymmetry.
- **Memory spikes:** poor prefetch/reshard boundaries or oversized microbatching.
- **Numerical drift:** precision policy mismatches across shards/collectives.

### Diagnostics and experiments
- Perform ablations: vary one axis degree at a time ($P_{tp}$, $P_{pp}$, etc.).
- Trace per-rank timelines to measure overlap ratio $ho=rac{T_{overlap}}{T_{comm}}$.
- Sweep microbatch count $m$, gradient bucket sizes, and activation checkpoint boundaries.
- Capture p50/p95/p99 iteration time and correlate with collective/kernel breakdowns.

### Study checklist
- Can you state the exact input/output layout for this subsection?
- Can you predict which collective dominates at larger scale?
- Can you explain one condition where this method should *not* be used?

### 4.2 Primitive: Stage Partitioning and Boundary Contracts
### Learning goals
- Understand the exact role of **4.2 Primitive: Stage Partitioning and Boundary Contracts** in the full training/serving stack.
- Identify required tensor layouts before and after this step.
- Map algorithmic choices to communication, memory, and stability consequences.

### Mechanism
- **Inputs/state.** Define local shards, global semantics, and invariants (shape/dtype/order).
- **Transformation.** Execute compute kernels plus communication primitives to move from input layout to output layout.
- **Correctness.** Preserve mathematical equivalence (modulo floating-point ordering) and optimizer semantics.
- **Output contract.** Ensure downstream subsection receives expected placement/state.

### Mathematical view
- Local shard sizing follows $n_{local}=n/p$ along sharded axis, with replicated axes unchanged.
- Runtime contribution is modeled as:
$$
T_{subsec}pprox T_{kernel}+T_{comm}-T_{overlap}+T_{sync}+T_{runtime}.
$$
- If this stage adds collective payload $q$ on group size $p_g$, then increment:
$$
\Delta T_{comm}pprox lpha f(p_g)+eta q g(p_g).
$$
- If this stage changes replication→sharding for tensor $x$, memory delta per rank is approximately:
$$
\Delta M(x)pprox |x|\left(1-rac{1}{p_g}ight).
$$

### Implementation details (practical)
1. Build process groups and bind them to physical topology intentionally; avoid accidental rank orders.
2. Choose bucket/chunk sizes so kernels stay large enough for high occupancy while communication can stream.
3. Co-schedule streams: compute stream + comm stream + dependency events to increase overlap safely.
4. Encode layout metadata in checkpoints to permit reshard/restart on changed meshes.
5. Validate numerics under mixed precision with loss-scale or FP8 scale tracking as required.

### Failure modes
- **Bandwidth-bound plateau:** throughput stops scaling as payload term dominates.
- **Latency-bound micro-ops:** too many tiny collectives; startup cost dominates.
- **Imbalance/stragglers:** rank skew from load imbalance or topology asymmetry.
- **Memory spikes:** poor prefetch/reshard boundaries or oversized microbatching.
- **Numerical drift:** precision policy mismatches across shards/collectives.

### Diagnostics and experiments
- Perform ablations: vary one axis degree at a time ($P_{tp}$, $P_{pp}$, etc.).
- Trace per-rank timelines to measure overlap ratio $ho=rac{T_{overlap}}{T_{comm}}$.
- Sweep microbatch count $m$, gradient bucket sizes, and activation checkpoint boundaries.
- Capture p50/p95/p99 iteration time and correlate with collective/kernel breakdowns.

### Study checklist
- Can you state the exact input/output layout for this subsection?
- Can you predict which collective dominates at larger scale?
- Can you explain one condition where this method should *not* be used?

### 4.3 Primitive: Microbatching, Warmup, Steady State, and Drain
### Learning goals
- Understand the exact role of **4.3 Primitive: Microbatching, Warmup, Steady State, and Drain** in the full training/serving stack.
- Identify required tensor layouts before and after this step.
- Map algorithmic choices to communication, memory, and stability consequences.

### Mechanism
- **Inputs/state.** Define local shards, global semantics, and invariants (shape/dtype/order).
- **Transformation.** Execute compute kernels plus communication primitives to move from input layout to output layout.
- **Correctness.** Preserve mathematical equivalence (modulo floating-point ordering) and optimizer semantics.
- **Output contract.** Ensure downstream subsection receives expected placement/state.

### Mathematical view
- Local shard sizing follows $n_{local}=n/p$ along sharded axis, with replicated axes unchanged.
- Runtime contribution is modeled as:
$$
T_{subsec}pprox T_{kernel}+T_{comm}-T_{overlap}+T_{sync}+T_{runtime}.
$$
- If this stage adds collective payload $q$ on group size $p_g$, then increment:
$$
\Delta T_{comm}pprox lpha f(p_g)+eta q g(p_g).
$$
- If this stage changes replication→sharding for tensor $x$, memory delta per rank is approximately:
$$
\Delta M(x)pprox |x|\left(1-rac{1}{p_g}ight).
$$

### Implementation details (practical)
1. Build process groups and bind them to physical topology intentionally; avoid accidental rank orders.
2. Choose bucket/chunk sizes so kernels stay large enough for high occupancy while communication can stream.
3. Co-schedule streams: compute stream + comm stream + dependency events to increase overlap safely.
4. Encode layout metadata in checkpoints to permit reshard/restart on changed meshes.
5. Validate numerics under mixed precision with loss-scale or FP8 scale tracking as required.

### Failure modes
- **Bandwidth-bound plateau:** throughput stops scaling as payload term dominates.
- **Latency-bound micro-ops:** too many tiny collectives; startup cost dominates.
- **Imbalance/stragglers:** rank skew from load imbalance or topology asymmetry.
- **Memory spikes:** poor prefetch/reshard boundaries or oversized microbatching.
- **Numerical drift:** precision policy mismatches across shards/collectives.

### Diagnostics and experiments
- Perform ablations: vary one axis degree at a time ($P_{tp}$, $P_{pp}$, etc.).
- Trace per-rank timelines to measure overlap ratio $ho=rac{T_{overlap}}{T_{comm}}$.
- Sweep microbatch count $m$, gradient bucket sizes, and activation checkpoint boundaries.
- Capture p50/p95/p99 iteration time and correlate with collective/kernel breakdowns.

### Study checklist
- Can you state the exact input/output layout for this subsection?
- Can you predict which collective dominates at larger scale?
- Can you explain one condition where this method should *not* be used?

### 4.4 Primitive: Point-to-Point Send/Recv and Collective Composition
### Learning goals
- Understand the exact role of **4.4 Primitive: Point-to-Point Send/Recv and Collective Composition** in the full training/serving stack.
- Identify required tensor layouts before and after this step.
- Map algorithmic choices to communication, memory, and stability consequences.

### Mechanism
- **Inputs/state.** Define local shards, global semantics, and invariants (shape/dtype/order).
- **Transformation.** Execute compute kernels plus communication primitives to move from input layout to output layout.
- **Correctness.** Preserve mathematical equivalence (modulo floating-point ordering) and optimizer semantics.
- **Output contract.** Ensure downstream subsection receives expected placement/state.

### Mathematical view
- Local shard sizing follows $n_{local}=n/p$ along sharded axis, with replicated axes unchanged.
- Runtime contribution is modeled as:
$$
T_{subsec}pprox T_{kernel}+T_{comm}-T_{overlap}+T_{sync}+T_{runtime}.
$$
- If this stage adds collective payload $q$ on group size $p_g$, then increment:
$$
\Delta T_{comm}pprox lpha f(p_g)+eta q g(p_g).
$$
- If this stage changes replication→sharding for tensor $x$, memory delta per rank is approximately:
$$
\Delta M(x)pprox |x|\left(1-rac{1}{p_g}ight).
$$

### Implementation details (practical)
1. Build process groups and bind them to physical topology intentionally; avoid accidental rank orders.
2. Choose bucket/chunk sizes so kernels stay large enough for high occupancy while communication can stream.
3. Co-schedule streams: compute stream + comm stream + dependency events to increase overlap safely.
4. Encode layout metadata in checkpoints to permit reshard/restart on changed meshes.
5. Validate numerics under mixed precision with loss-scale or FP8 scale tracking as required.

### Failure modes
- **Bandwidth-bound plateau:** throughput stops scaling as payload term dominates.
- **Latency-bound micro-ops:** too many tiny collectives; startup cost dominates.
- **Imbalance/stragglers:** rank skew from load imbalance or topology asymmetry.
- **Memory spikes:** poor prefetch/reshard boundaries or oversized microbatching.
- **Numerical drift:** precision policy mismatches across shards/collectives.

### Diagnostics and experiments
- Perform ablations: vary one axis degree at a time ($P_{tp}$, $P_{pp}$, etc.).
- Trace per-rank timelines to measure overlap ratio $ho=rac{T_{overlap}}{T_{comm}}$.
- Sweep microbatch count $m$, gradient bucket sizes, and activation checkpoint boundaries.
- Capture p50/p95/p99 iteration time and correlate with collective/kernel breakdowns.

### Study checklist
- Can you state the exact input/output layout for this subsection?
- Can you predict which collective dominates at larger scale?
- Can you explain one condition where this method should *not* be used?

### 4.5 Primitive: The Schedule DAG and the F/B/W Split
### Learning goals
- Understand the exact role of **4.5 Primitive: The Schedule DAG and the F/B/W Split** in the full training/serving stack.
- Identify required tensor layouts before and after this step.
- Map algorithmic choices to communication, memory, and stability consequences.

### Mechanism
- **Inputs/state.** Define local shards, global semantics, and invariants (shape/dtype/order).
- **Transformation.** Execute compute kernels plus communication primitives to move from input layout to output layout.
- **Correctness.** Preserve mathematical equivalence (modulo floating-point ordering) and optimizer semantics.
- **Output contract.** Ensure downstream subsection receives expected placement/state.

### Mathematical view
- Local shard sizing follows $n_{local}=n/p$ along sharded axis, with replicated axes unchanged.
- Runtime contribution is modeled as:
$$
T_{subsec}pprox T_{kernel}+T_{comm}-T_{overlap}+T_{sync}+T_{runtime}.
$$
- If this stage adds collective payload $q$ on group size $p_g$, then increment:
$$
\Delta T_{comm}pprox lpha f(p_g)+eta q g(p_g).
$$
- If this stage changes replication→sharding for tensor $x$, memory delta per rank is approximately:
$$
\Delta M(x)pprox |x|\left(1-rac{1}{p_g}ight).
$$

### Implementation details (practical)
1. Build process groups and bind them to physical topology intentionally; avoid accidental rank orders.
2. Choose bucket/chunk sizes so kernels stay large enough for high occupancy while communication can stream.
3. Co-schedule streams: compute stream + comm stream + dependency events to increase overlap safely.
4. Encode layout metadata in checkpoints to permit reshard/restart on changed meshes.
5. Validate numerics under mixed precision with loss-scale or FP8 scale tracking as required.

### Failure modes
- **Bandwidth-bound plateau:** throughput stops scaling as payload term dominates.
- **Latency-bound micro-ops:** too many tiny collectives; startup cost dominates.
- **Imbalance/stragglers:** rank skew from load imbalance or topology asymmetry.
- **Memory spikes:** poor prefetch/reshard boundaries or oversized microbatching.
- **Numerical drift:** precision policy mismatches across shards/collectives.

### Diagnostics and experiments
- Perform ablations: vary one axis degree at a time ($P_{tp}$, $P_{pp}$, etc.).
- Trace per-rank timelines to measure overlap ratio $ho=rac{T_{overlap}}{T_{comm}}$.
- Sweep microbatch count $m$, gradient bucket sizes, and activation checkpoint boundaries.
- Capture p50/p95/p99 iteration time and correlate with collective/kernel breakdowns.

### Study checklist
- Can you state the exact input/output layout for this subsection?
- Can you predict which collective dominates at larger scale?
- Can you explain one condition where this method should *not* be used?

### 4.6 Zero Bubble Pipeline Parallelism
### Learning goals
- Understand the exact role of **4.6 Zero Bubble Pipeline Parallelism** in the full training/serving stack.
- Identify required tensor layouts before and after this step.
- Map algorithmic choices to communication, memory, and stability consequences.

### Mechanism
- **Inputs/state.** Define local shards, global semantics, and invariants (shape/dtype/order).
- **Transformation.** Execute compute kernels plus communication primitives to move from input layout to output layout.
- **Correctness.** Preserve mathematical equivalence (modulo floating-point ordering) and optimizer semantics.
- **Output contract.** Ensure downstream subsection receives expected placement/state.

### Mathematical view
- Local shard sizing follows $n_{local}=n/p$ along sharded axis, with replicated axes unchanged.
- Runtime contribution is modeled as:
$$
T_{subsec}pprox T_{kernel}+T_{comm}-T_{overlap}+T_{sync}+T_{runtime}.
$$
- If this stage adds collective payload $q$ on group size $p_g$, then increment:
$$
\Delta T_{comm}pprox lpha f(p_g)+eta q g(p_g).
$$
- If this stage changes replication→sharding for tensor $x$, memory delta per rank is approximately:
$$
\Delta M(x)pprox |x|\left(1-rac{1}{p_g}ight).
$$

### Implementation details (practical)
1. Build process groups and bind them to physical topology intentionally; avoid accidental rank orders.
2. Choose bucket/chunk sizes so kernels stay large enough for high occupancy while communication can stream.
3. Co-schedule streams: compute stream + comm stream + dependency events to increase overlap safely.
4. Encode layout metadata in checkpoints to permit reshard/restart on changed meshes.
5. Validate numerics under mixed precision with loss-scale or FP8 scale tracking as required.

### Failure modes
- **Bandwidth-bound plateau:** throughput stops scaling as payload term dominates.
- **Latency-bound micro-ops:** too many tiny collectives; startup cost dominates.
- **Imbalance/stragglers:** rank skew from load imbalance or topology asymmetry.
- **Memory spikes:** poor prefetch/reshard boundaries or oversized microbatching.
- **Numerical drift:** precision policy mismatches across shards/collectives.

### Diagnostics and experiments
- Perform ablations: vary one axis degree at a time ($P_{tp}$, $P_{pp}$, etc.).
- Trace per-rank timelines to measure overlap ratio $ho=rac{T_{overlap}}{T_{comm}}$.
- Sweep microbatch count $m$, gradient bucket sizes, and activation checkpoint boundaries.
- Capture p50/p95/p99 iteration time and correlate with collective/kernel breakdowns.

### Study checklist
- Can you state the exact input/output layout for this subsection?
- Can you predict which collective dominates at larger scale?
- Can you explain one condition where this method should *not* be used?

### 4.7 Controllable-Memory Pipeline Schedules
### Learning goals
- Understand the exact role of **4.7 Controllable-Memory Pipeline Schedules** in the full training/serving stack.
- Identify required tensor layouts before and after this step.
- Map algorithmic choices to communication, memory, and stability consequences.

### Mechanism
- **Inputs/state.** Define local shards, global semantics, and invariants (shape/dtype/order).
- **Transformation.** Execute compute kernels plus communication primitives to move from input layout to output layout.
- **Correctness.** Preserve mathematical equivalence (modulo floating-point ordering) and optimizer semantics.
- **Output contract.** Ensure downstream subsection receives expected placement/state.

### Mathematical view
- Local shard sizing follows $n_{local}=n/p$ along sharded axis, with replicated axes unchanged.
- Runtime contribution is modeled as:
$$
T_{subsec}pprox T_{kernel}+T_{comm}-T_{overlap}+T_{sync}+T_{runtime}.
$$
- If this stage adds collective payload $q$ on group size $p_g$, then increment:
$$
\Delta T_{comm}pprox lpha f(p_g)+eta q g(p_g).
$$
- If this stage changes replication→sharding for tensor $x$, memory delta per rank is approximately:
$$
\Delta M(x)pprox |x|\left(1-rac{1}{p_g}ight).
$$

### Implementation details (practical)
1. Build process groups and bind them to physical topology intentionally; avoid accidental rank orders.
2. Choose bucket/chunk sizes so kernels stay large enough for high occupancy while communication can stream.
3. Co-schedule streams: compute stream + comm stream + dependency events to increase overlap safely.
4. Encode layout metadata in checkpoints to permit reshard/restart on changed meshes.
5. Validate numerics under mixed precision with loss-scale or FP8 scale tracking as required.

### Failure modes
- **Bandwidth-bound plateau:** throughput stops scaling as payload term dominates.
- **Latency-bound micro-ops:** too many tiny collectives; startup cost dominates.
- **Imbalance/stragglers:** rank skew from load imbalance or topology asymmetry.
- **Memory spikes:** poor prefetch/reshard boundaries or oversized microbatching.
- **Numerical drift:** precision policy mismatches across shards/collectives.

### Diagnostics and experiments
- Perform ablations: vary one axis degree at a time ($P_{tp}$, $P_{pp}$, etc.).
- Trace per-rank timelines to measure overlap ratio $ho=rac{T_{overlap}}{T_{comm}}$.
- Sweep microbatch count $m$, gradient bucket sizes, and activation checkpoint boundaries.
- Capture p50/p95/p99 iteration time and correlate with collective/kernel breakdowns.

### Study checklist
- Can you state the exact input/output layout for this subsection?
- Can you predict which collective dominates at larger scale?
- Can you explain one condition where this method should *not* be used?

### 4.8 PipeOffload: Activation Offload and Prefetch as Schedule Primitives
### Learning goals
- Understand the exact role of **4.8 PipeOffload: Activation Offload and Prefetch as Schedule Primitives** in the full training/serving stack.
- Identify required tensor layouts before and after this step.
- Map algorithmic choices to communication, memory, and stability consequences.

### Mechanism
- **Inputs/state.** Define local shards, global semantics, and invariants (shape/dtype/order).
- **Transformation.** Execute compute kernels plus communication primitives to move from input layout to output layout.
- **Correctness.** Preserve mathematical equivalence (modulo floating-point ordering) and optimizer semantics.
- **Output contract.** Ensure downstream subsection receives expected placement/state.

### Mathematical view
- Local shard sizing follows $n_{local}=n/p$ along sharded axis, with replicated axes unchanged.
- Runtime contribution is modeled as:
$$
T_{subsec}pprox T_{kernel}+T_{comm}-T_{overlap}+T_{sync}+T_{runtime}.
$$
- If this stage adds collective payload $q$ on group size $p_g$, then increment:
$$
\Delta T_{comm}pprox lpha f(p_g)+eta q g(p_g).
$$
- If this stage changes replication→sharding for tensor $x$, memory delta per rank is approximately:
$$
\Delta M(x)pprox |x|\left(1-rac{1}{p_g}ight).
$$

### Implementation details (practical)
1. Build process groups and bind them to physical topology intentionally; avoid accidental rank orders.
2. Choose bucket/chunk sizes so kernels stay large enough for high occupancy while communication can stream.
3. Co-schedule streams: compute stream + comm stream + dependency events to increase overlap safely.
4. Encode layout metadata in checkpoints to permit reshard/restart on changed meshes.
5. Validate numerics under mixed precision with loss-scale or FP8 scale tracking as required.

### Failure modes
- **Bandwidth-bound plateau:** throughput stops scaling as payload term dominates.
- **Latency-bound micro-ops:** too many tiny collectives; startup cost dominates.
- **Imbalance/stragglers:** rank skew from load imbalance or topology asymmetry.
- **Memory spikes:** poor prefetch/reshard boundaries or oversized microbatching.
- **Numerical drift:** precision policy mismatches across shards/collectives.

### Diagnostics and experiments
- Perform ablations: vary one axis degree at a time ($P_{tp}$, $P_{pp}$, etc.).
- Trace per-rank timelines to measure overlap ratio $ho=rac{T_{overlap}}{T_{comm}}$.
- Sweep microbatch count $m$, gradient bucket sizes, and activation checkpoint boundaries.
- Capture p50/p95/p99 iteration time and correlate with collective/kernel breakdowns.

### Study checklist
- Can you state the exact input/output layout for this subsection?
- Can you predict which collective dominates at larger scale?
- Can you explain one condition where this method should *not* be used?

### 4.9 Vocabulary-Balanced Pipeline Parallelism
### Learning goals
- Understand the exact role of **4.9 Vocabulary-Balanced Pipeline Parallelism** in the full training/serving stack.
- Identify required tensor layouts before and after this step.
- Map algorithmic choices to communication, memory, and stability consequences.

### Mechanism
- **Inputs/state.** Define local shards, global semantics, and invariants (shape/dtype/order).
- **Transformation.** Execute compute kernels plus communication primitives to move from input layout to output layout.
- **Correctness.** Preserve mathematical equivalence (modulo floating-point ordering) and optimizer semantics.
- **Output contract.** Ensure downstream subsection receives expected placement/state.

### Mathematical view
- Local shard sizing follows $n_{local}=n/p$ along sharded axis, with replicated axes unchanged.
- Runtime contribution is modeled as:
$$
T_{subsec}pprox T_{kernel}+T_{comm}-T_{overlap}+T_{sync}+T_{runtime}.
$$
- If this stage adds collective payload $q$ on group size $p_g$, then increment:
$$
\Delta T_{comm}pprox lpha f(p_g)+eta q g(p_g).
$$
- If this stage changes replication→sharding for tensor $x$, memory delta per rank is approximately:
$$
\Delta M(x)pprox |x|\left(1-rac{1}{p_g}ight).
$$

### Implementation details (practical)
1. Build process groups and bind them to physical topology intentionally; avoid accidental rank orders.
2. Choose bucket/chunk sizes so kernels stay large enough for high occupancy while communication can stream.
3. Co-schedule streams: compute stream + comm stream + dependency events to increase overlap safely.
4. Encode layout metadata in checkpoints to permit reshard/restart on changed meshes.
5. Validate numerics under mixed precision with loss-scale or FP8 scale tracking as required.

### Failure modes
- **Bandwidth-bound plateau:** throughput stops scaling as payload term dominates.
- **Latency-bound micro-ops:** too many tiny collectives; startup cost dominates.
- **Imbalance/stragglers:** rank skew from load imbalance or topology asymmetry.
- **Memory spikes:** poor prefetch/reshard boundaries or oversized microbatching.
- **Numerical drift:** precision policy mismatches across shards/collectives.

### Diagnostics and experiments
- Perform ablations: vary one axis degree at a time ($P_{tp}$, $P_{pp}$, etc.).
- Trace per-rank timelines to measure overlap ratio $ho=rac{T_{overlap}}{T_{comm}}$.
- Sweep microbatch count $m$, gradient bucket sizes, and activation checkpoint boundaries.
- Capture p50/p95/p99 iteration time and correlate with collective/kernel breakdowns.

### Study checklist
- Can you state the exact input/output layout for this subsection?
- Can you predict which collective dominates at larger scale?
- Can you explain one condition where this method should *not* be used?

### 4.10 DualPipe and DualPipeV: Pipeline Scheduling for MoE Communication
### Learning goals
- Understand the exact role of **4.10 DualPipe and DualPipeV: Pipeline Scheduling for MoE Communication** in the full training/serving stack.
- Identify required tensor layouts before and after this step.
- Map algorithmic choices to communication, memory, and stability consequences.

### Mechanism
- **Inputs/state.** Define local shards, global semantics, and invariants (shape/dtype/order).
- **Transformation.** Execute compute kernels plus communication primitives to move from input layout to output layout.
- **Correctness.** Preserve mathematical equivalence (modulo floating-point ordering) and optimizer semantics.
- **Output contract.** Ensure downstream subsection receives expected placement/state.

### Mathematical view
- Local shard sizing follows $n_{local}=n/p$ along sharded axis, with replicated axes unchanged.
- Runtime contribution is modeled as:
$$
T_{subsec}pprox T_{kernel}+T_{comm}-T_{overlap}+T_{sync}+T_{runtime}.
$$
- If this stage adds collective payload $q$ on group size $p_g$, then increment:
$$
\Delta T_{comm}pprox lpha f(p_g)+eta q g(p_g).
$$
- If this stage changes replication→sharding for tensor $x$, memory delta per rank is approximately:
$$
\Delta M(x)pprox |x|\left(1-rac{1}{p_g}ight).
$$

### Implementation details (practical)
1. Build process groups and bind them to physical topology intentionally; avoid accidental rank orders.
2. Choose bucket/chunk sizes so kernels stay large enough for high occupancy while communication can stream.
3. Co-schedule streams: compute stream + comm stream + dependency events to increase overlap safely.
4. Encode layout metadata in checkpoints to permit reshard/restart on changed meshes.
5. Validate numerics under mixed precision with loss-scale or FP8 scale tracking as required.

### Failure modes
- **Bandwidth-bound plateau:** throughput stops scaling as payload term dominates.
- **Latency-bound micro-ops:** too many tiny collectives; startup cost dominates.
- **Imbalance/stragglers:** rank skew from load imbalance or topology asymmetry.
- **Memory spikes:** poor prefetch/reshard boundaries or oversized microbatching.
- **Numerical drift:** precision policy mismatches across shards/collectives.

### Diagnostics and experiments
- Perform ablations: vary one axis degree at a time ($P_{tp}$, $P_{pp}$, etc.).
- Trace per-rank timelines to measure overlap ratio $ho=rac{T_{overlap}}{T_{comm}}$.
- Sweep microbatch count $m$, gradient bucket sizes, and activation checkpoint boundaries.
- Capture p50/p95/p99 iteration time and correlate with collective/kernel breakdowns.

### Study checklist
- Can you state the exact input/output layout for this subsection?
- Can you predict which collective dominates at larger scale?
- Can you explain one condition where this method should *not* be used?

### 4.11 PyTorch Pipelining and TorchTitan Runtime Support
### Learning goals
- Understand the exact role of **4.11 PyTorch Pipelining and TorchTitan Runtime Support** in the full training/serving stack.
- Identify required tensor layouts before and after this step.
- Map algorithmic choices to communication, memory, and stability consequences.

### Mechanism
- **Inputs/state.** Define local shards, global semantics, and invariants (shape/dtype/order).
- **Transformation.** Execute compute kernels plus communication primitives to move from input layout to output layout.
- **Correctness.** Preserve mathematical equivalence (modulo floating-point ordering) and optimizer semantics.
- **Output contract.** Ensure downstream subsection receives expected placement/state.

### Mathematical view
- Local shard sizing follows $n_{local}=n/p$ along sharded axis, with replicated axes unchanged.
- Runtime contribution is modeled as:
$$
T_{subsec}pprox T_{kernel}+T_{comm}-T_{overlap}+T_{sync}+T_{runtime}.
$$
- If this stage adds collective payload $q$ on group size $p_g$, then increment:
$$
\Delta T_{comm}pprox lpha f(p_g)+eta q g(p_g).
$$
- If this stage changes replication→sharding for tensor $x$, memory delta per rank is approximately:
$$
\Delta M(x)pprox |x|\left(1-rac{1}{p_g}ight).
$$

### Implementation details (practical)
1. Build process groups and bind them to physical topology intentionally; avoid accidental rank orders.
2. Choose bucket/chunk sizes so kernels stay large enough for high occupancy while communication can stream.
3. Co-schedule streams: compute stream + comm stream + dependency events to increase overlap safely.
4. Encode layout metadata in checkpoints to permit reshard/restart on changed meshes.
5. Validate numerics under mixed precision with loss-scale or FP8 scale tracking as required.

### Failure modes
- **Bandwidth-bound plateau:** throughput stops scaling as payload term dominates.
- **Latency-bound micro-ops:** too many tiny collectives; startup cost dominates.
- **Imbalance/stragglers:** rank skew from load imbalance or topology asymmetry.
- **Memory spikes:** poor prefetch/reshard boundaries or oversized microbatching.
- **Numerical drift:** precision policy mismatches across shards/collectives.

### Diagnostics and experiments
- Perform ablations: vary one axis degree at a time ($P_{tp}$, $P_{pp}$, etc.).
- Trace per-rank timelines to measure overlap ratio $ho=rac{T_{overlap}}{T_{comm}}$.
- Sweep microbatch count $m$, gradient bucket sizes, and activation checkpoint boundaries.
- Capture p50/p95/p99 iteration time and correlate with collective/kernel breakdowns.

### Study checklist
- Can you state the exact input/output layout for this subsection?
- Can you predict which collective dominates at larger scale?
- Can you explain one condition where this method should *not* be used?

### 4.12 Optimizer Barriers, Gradient Synchronization, and Correctness
### Learning goals
- Understand the exact role of **4.12 Optimizer Barriers, Gradient Synchronization, and Correctness** in the full training/serving stack.
- Identify required tensor layouts before and after this step.
- Map algorithmic choices to communication, memory, and stability consequences.

### Mechanism
- **Inputs/state.** Define local shards, global semantics, and invariants (shape/dtype/order).
- **Transformation.** Execute compute kernels plus communication primitives to move from input layout to output layout.
- **Correctness.** Preserve mathematical equivalence (modulo floating-point ordering) and optimizer semantics.
- **Output contract.** Ensure downstream subsection receives expected placement/state.

### Mathematical view
- Local shard sizing follows $n_{local}=n/p$ along sharded axis, with replicated axes unchanged.
- Runtime contribution is modeled as:
$$
T_{subsec}pprox T_{kernel}+T_{comm}-T_{overlap}+T_{sync}+T_{runtime}.
$$
- If this stage adds collective payload $q$ on group size $p_g$, then increment:
$$
\Delta T_{comm}pprox lpha f(p_g)+eta q g(p_g).
$$
- If this stage changes replication→sharding for tensor $x$, memory delta per rank is approximately:
$$
\Delta M(x)pprox |x|\left(1-rac{1}{p_g}ight).
$$

### Implementation details (practical)
1. Build process groups and bind them to physical topology intentionally; avoid accidental rank orders.
2. Choose bucket/chunk sizes so kernels stay large enough for high occupancy while communication can stream.
3. Co-schedule streams: compute stream + comm stream + dependency events to increase overlap safely.
4. Encode layout metadata in checkpoints to permit reshard/restart on changed meshes.
5. Validate numerics under mixed precision with loss-scale or FP8 scale tracking as required.

### Failure modes
- **Bandwidth-bound plateau:** throughput stops scaling as payload term dominates.
- **Latency-bound micro-ops:** too many tiny collectives; startup cost dominates.
- **Imbalance/stragglers:** rank skew from load imbalance or topology asymmetry.
- **Memory spikes:** poor prefetch/reshard boundaries or oversized microbatching.
- **Numerical drift:** precision policy mismatches across shards/collectives.

### Diagnostics and experiments
- Perform ablations: vary one axis degree at a time ($P_{tp}$, $P_{pp}$, etc.).
- Trace per-rank timelines to measure overlap ratio $ho=rac{T_{overlap}}{T_{comm}}$.
- Sweep microbatch count $m$, gradient bucket sizes, and activation checkpoint boundaries.
- Capture p50/p95/p99 iteration time and correlate with collective/kernel breakdowns.

### Study checklist
- Can you state the exact input/output layout for this subsection?
- Can you predict which collective dominates at larger scale?
- Can you explain one condition where this method should *not* be used?

### 4.13 Composing PP With Tensor, Context, Expert, and Data Parallelism
### Learning goals
- Understand the exact role of **4.13 Composing PP With Tensor, Context, Expert, and Data Parallelism** in the full training/serving stack.
- Identify required tensor layouts before and after this step.
- Map algorithmic choices to communication, memory, and stability consequences.

### Mechanism
- **Inputs/state.** Define local shards, global semantics, and invariants (shape/dtype/order).
- **Transformation.** Execute compute kernels plus communication primitives to move from input layout to output layout.
- **Correctness.** Preserve mathematical equivalence (modulo floating-point ordering) and optimizer semantics.
- **Output contract.** Ensure downstream subsection receives expected placement/state.

### Mathematical view
- Local shard sizing follows $n_{local}=n/p$ along sharded axis, with replicated axes unchanged.
- Runtime contribution is modeled as:
$$
T_{subsec}pprox T_{kernel}+T_{comm}-T_{overlap}+T_{sync}+T_{runtime}.
$$
- If this stage adds collective payload $q$ on group size $p_g$, then increment:
$$
\Delta T_{comm}pprox lpha f(p_g)+eta q g(p_g).
$$
- If this stage changes replication→sharding for tensor $x$, memory delta per rank is approximately:
$$
\Delta M(x)pprox |x|\left(1-rac{1}{p_g}ight).
$$

### Implementation details (practical)
1. Build process groups and bind them to physical topology intentionally; avoid accidental rank orders.
2. Choose bucket/chunk sizes so kernels stay large enough for high occupancy while communication can stream.
3. Co-schedule streams: compute stream + comm stream + dependency events to increase overlap safely.
4. Encode layout metadata in checkpoints to permit reshard/restart on changed meshes.
5. Validate numerics under mixed precision with loss-scale or FP8 scale tracking as required.

### Failure modes
- **Bandwidth-bound plateau:** throughput stops scaling as payload term dominates.
- **Latency-bound micro-ops:** too many tiny collectives; startup cost dominates.
- **Imbalance/stragglers:** rank skew from load imbalance or topology asymmetry.
- **Memory spikes:** poor prefetch/reshard boundaries or oversized microbatching.
- **Numerical drift:** precision policy mismatches across shards/collectives.

### Diagnostics and experiments
- Perform ablations: vary one axis degree at a time ($P_{tp}$, $P_{pp}$, etc.).
- Trace per-rank timelines to measure overlap ratio $ho=rac{T_{overlap}}{T_{comm}}$.
- Sweep microbatch count $m$, gradient bucket sizes, and activation checkpoint boundaries.
- Capture p50/p95/p99 iteration time and correlate with collective/kernel breakdowns.

### Study checklist
- Can you state the exact input/output layout for this subsection?
- Can you predict which collective dominates at larger scale?
- Can you explain one condition where this method should *not* be used?

### 4.14 Choosing a Modern PP Schedule
### Learning goals
- Understand the exact role of **4.14 Choosing a Modern PP Schedule** in the full training/serving stack.
- Identify required tensor layouts before and after this step.
- Map algorithmic choices to communication, memory, and stability consequences.

### Mechanism
- **Inputs/state.** Define local shards, global semantics, and invariants (shape/dtype/order).
- **Transformation.** Execute compute kernels plus communication primitives to move from input layout to output layout.
- **Correctness.** Preserve mathematical equivalence (modulo floating-point ordering) and optimizer semantics.
- **Output contract.** Ensure downstream subsection receives expected placement/state.

### Mathematical view
- Local shard sizing follows $n_{local}=n/p$ along sharded axis, with replicated axes unchanged.
- Runtime contribution is modeled as:
$$
T_{subsec}pprox T_{kernel}+T_{comm}-T_{overlap}+T_{sync}+T_{runtime}.
$$
- If this stage adds collective payload $q$ on group size $p_g$, then increment:
$$
\Delta T_{comm}pprox lpha f(p_g)+eta q g(p_g).
$$
- If this stage changes replication→sharding for tensor $x$, memory delta per rank is approximately:
$$
\Delta M(x)pprox |x|\left(1-rac{1}{p_g}ight).
$$

### Implementation details (practical)
1. Build process groups and bind them to physical topology intentionally; avoid accidental rank orders.
2. Choose bucket/chunk sizes so kernels stay large enough for high occupancy while communication can stream.
3. Co-schedule streams: compute stream + comm stream + dependency events to increase overlap safely.
4. Encode layout metadata in checkpoints to permit reshard/restart on changed meshes.
5. Validate numerics under mixed precision with loss-scale or FP8 scale tracking as required.

### Failure modes
- **Bandwidth-bound plateau:** throughput stops scaling as payload term dominates.
- **Latency-bound micro-ops:** too many tiny collectives; startup cost dominates.
- **Imbalance/stragglers:** rank skew from load imbalance or topology asymmetry.
- **Memory spikes:** poor prefetch/reshard boundaries or oversized microbatching.
- **Numerical drift:** precision policy mismatches across shards/collectives.

### Diagnostics and experiments
- Perform ablations: vary one axis degree at a time ($P_{tp}$, $P_{pp}$, etc.).
- Trace per-rank timelines to measure overlap ratio $ho=rac{T_{overlap}}{T_{comm}}$.
- Sweep microbatch count $m$, gradient bucket sizes, and activation checkpoint boundaries.
- Capture p50/p95/p99 iteration time and correlate with collective/kernel breakdowns.

### Study checklist
- Can you state the exact input/output layout for this subsection?
- Can you predict which collective dominates at larger scale?
- Can you explain one condition where this method should *not* be used?

### 4.15 Lineage: Older PP Work as Context
### Learning goals
- Understand the exact role of **4.15 Lineage: Older PP Work as Context** in the full training/serving stack.
- Identify required tensor layouts before and after this step.
- Map algorithmic choices to communication, memory, and stability consequences.

### Mechanism
- **Inputs/state.** Define local shards, global semantics, and invariants (shape/dtype/order).
- **Transformation.** Execute compute kernels plus communication primitives to move from input layout to output layout.
- **Correctness.** Preserve mathematical equivalence (modulo floating-point ordering) and optimizer semantics.
- **Output contract.** Ensure downstream subsection receives expected placement/state.

### Mathematical view
- Local shard sizing follows $n_{local}=n/p$ along sharded axis, with replicated axes unchanged.
- Runtime contribution is modeled as:
$$
T_{subsec}pprox T_{kernel}+T_{comm}-T_{overlap}+T_{sync}+T_{runtime}.
$$
- If this stage adds collective payload $q$ on group size $p_g$, then increment:
$$
\Delta T_{comm}pprox lpha f(p_g)+eta q g(p_g).
$$
- If this stage changes replication→sharding for tensor $x$, memory delta per rank is approximately:
$$
\Delta M(x)pprox |x|\left(1-rac{1}{p_g}ight).
$$

### Implementation details (practical)
1. Build process groups and bind them to physical topology intentionally; avoid accidental rank orders.
2. Choose bucket/chunk sizes so kernels stay large enough for high occupancy while communication can stream.
3. Co-schedule streams: compute stream + comm stream + dependency events to increase overlap safely.
4. Encode layout metadata in checkpoints to permit reshard/restart on changed meshes.
5. Validate numerics under mixed precision with loss-scale or FP8 scale tracking as required.

### Failure modes
- **Bandwidth-bound plateau:** throughput stops scaling as payload term dominates.
- **Latency-bound micro-ops:** too many tiny collectives; startup cost dominates.
- **Imbalance/stragglers:** rank skew from load imbalance or topology asymmetry.
- **Memory spikes:** poor prefetch/reshard boundaries or oversized microbatching.
- **Numerical drift:** precision policy mismatches across shards/collectives.

### Diagnostics and experiments
- Perform ablations: vary one axis degree at a time ($P_{tp}$, $P_{pp}$, etc.).
- Trace per-rank timelines to measure overlap ratio $ho=rac{T_{overlap}}{T_{comm}}$.
- Sweep microbatch count $m$, gradient bucket sizes, and activation checkpoint boundaries.
- Capture p50/p95/p99 iteration time and correlate with collective/kernel breakdowns.

### Study checklist
- Can you state the exact input/output layout for this subsection?
- Can you predict which collective dominates at larger scale?
- Can you explain one condition where this method should *not* be used?

### 4.16 Practical Caveats
### Learning goals
- Understand the exact role of **4.16 Practical Caveats** in the full training/serving stack.
- Identify required tensor layouts before and after this step.
- Map algorithmic choices to communication, memory, and stability consequences.

### Mechanism
- **Inputs/state.** Define local shards, global semantics, and invariants (shape/dtype/order).
- **Transformation.** Execute compute kernels plus communication primitives to move from input layout to output layout.
- **Correctness.** Preserve mathematical equivalence (modulo floating-point ordering) and optimizer semantics.
- **Output contract.** Ensure downstream subsection receives expected placement/state.

### Mathematical view
- Local shard sizing follows $n_{local}=n/p$ along sharded axis, with replicated axes unchanged.
- Runtime contribution is modeled as:
$$
T_{subsec}pprox T_{kernel}+T_{comm}-T_{overlap}+T_{sync}+T_{runtime}.
$$
- If this stage adds collective payload $q$ on group size $p_g$, then increment:
$$
\Delta T_{comm}pprox lpha f(p_g)+eta q g(p_g).
$$
- If this stage changes replication→sharding for tensor $x$, memory delta per rank is approximately:
$$
\Delta M(x)pprox |x|\left(1-rac{1}{p_g}ight).
$$

### Implementation details (practical)
1. Build process groups and bind them to physical topology intentionally; avoid accidental rank orders.
2. Choose bucket/chunk sizes so kernels stay large enough for high occupancy while communication can stream.
3. Co-schedule streams: compute stream + comm stream + dependency events to increase overlap safely.
4. Encode layout metadata in checkpoints to permit reshard/restart on changed meshes.
5. Validate numerics under mixed precision with loss-scale or FP8 scale tracking as required.

### Failure modes
- **Bandwidth-bound plateau:** throughput stops scaling as payload term dominates.
- **Latency-bound micro-ops:** too many tiny collectives; startup cost dominates.
- **Imbalance/stragglers:** rank skew from load imbalance or topology asymmetry.
- **Memory spikes:** poor prefetch/reshard boundaries or oversized microbatching.
- **Numerical drift:** precision policy mismatches across shards/collectives.

### Diagnostics and experiments
- Perform ablations: vary one axis degree at a time ($P_{tp}$, $P_{pp}$, etc.).
- Trace per-rank timelines to measure overlap ratio $ho=rac{T_{overlap}}{T_{comm}}$.
- Sweep microbatch count $m$, gradient bucket sizes, and activation checkpoint boundaries.
- Capture p50/p95/p99 iteration time and correlate with collective/kernel breakdowns.

### Study checklist
- Can you state the exact input/output layout for this subsection?
- Can you predict which collective dominates at larger scale?
- Can you explain one condition where this method should *not* be used?

## 5. Sequence Parallelism and Context Parallelism
This section explains **5. Sequence Parallelism and Context Parallelism** from first principles, implementation mechanics, cost/memory implications, topology interactions, and operational diagnostics.

### 5.1 Notation, Tensor Shapes, and the Attention Operator
### Learning goals
- Understand the exact role of **5.1 Notation, Tensor Shapes, and the Attention Operator** in the full training/serving stack.
- Identify required tensor layouts before and after this step.
- Map algorithmic choices to communication, memory, and stability consequences.

### Mechanism
- **Inputs/state.** Define local shards, global semantics, and invariants (shape/dtype/order).
- **Transformation.** Execute compute kernels plus communication primitives to move from input layout to output layout.
- **Correctness.** Preserve mathematical equivalence (modulo floating-point ordering) and optimizer semantics.
- **Output contract.** Ensure downstream subsection receives expected placement/state.

### Mathematical view
- Local shard sizing follows $n_{local}=n/p$ along sharded axis, with replicated axes unchanged.
- Runtime contribution is modeled as:
$$
T_{subsec}pprox T_{kernel}+T_{comm}-T_{overlap}+T_{sync}+T_{runtime}.
$$
- If this stage adds collective payload $q$ on group size $p_g$, then increment:
$$
\Delta T_{comm}pprox lpha f(p_g)+eta q g(p_g).
$$
- If this stage changes replication→sharding for tensor $x$, memory delta per rank is approximately:
$$
\Delta M(x)pprox |x|\left(1-rac{1}{p_g}ight).
$$

### Implementation details (practical)
1. Build process groups and bind them to physical topology intentionally; avoid accidental rank orders.
2. Choose bucket/chunk sizes so kernels stay large enough for high occupancy while communication can stream.
3. Co-schedule streams: compute stream + comm stream + dependency events to increase overlap safely.
4. Encode layout metadata in checkpoints to permit reshard/restart on changed meshes.
5. Validate numerics under mixed precision with loss-scale or FP8 scale tracking as required.

### Failure modes
- **Bandwidth-bound plateau:** throughput stops scaling as payload term dominates.
- **Latency-bound micro-ops:** too many tiny collectives; startup cost dominates.
- **Imbalance/stragglers:** rank skew from load imbalance or topology asymmetry.
- **Memory spikes:** poor prefetch/reshard boundaries or oversized microbatching.
- **Numerical drift:** precision policy mismatches across shards/collectives.

### Diagnostics and experiments
- Perform ablations: vary one axis degree at a time ($P_{tp}$, $P_{pp}$, etc.).
- Trace per-rank timelines to measure overlap ratio $ho=rac{T_{overlap}}{T_{comm}}$.
- Sweep microbatch count $m$, gradient bucket sizes, and activation checkpoint boundaries.
- Capture p50/p95/p99 iteration time and correlate with collective/kernel breakdowns.

### Study checklist
- Can you state the exact input/output layout for this subsection?
- Can you predict which collective dominates at larger scale?
- Can you explain one condition where this method should *not* be used?

### 5.2 The Primitive Operations and Their Collectives
### Learning goals
- Understand the exact role of **5.2 The Primitive Operations and Their Collectives** in the full training/serving stack.
- Identify required tensor layouts before and after this step.
- Map algorithmic choices to communication, memory, and stability consequences.

### Mechanism
- **Inputs/state.** Define local shards, global semantics, and invariants (shape/dtype/order).
- **Transformation.** Execute compute kernels plus communication primitives to move from input layout to output layout.
- **Correctness.** Preserve mathematical equivalence (modulo floating-point ordering) and optimizer semantics.
- **Output contract.** Ensure downstream subsection receives expected placement/state.

### Mathematical view
- Local shard sizing follows $n_{local}=n/p$ along sharded axis, with replicated axes unchanged.
- Runtime contribution is modeled as:
$$
T_{subsec}pprox T_{kernel}+T_{comm}-T_{overlap}+T_{sync}+T_{runtime}.
$$
- If this stage adds collective payload $q$ on group size $p_g$, then increment:
$$
\Delta T_{comm}pprox lpha f(p_g)+eta q g(p_g).
$$
- If this stage changes replication→sharding for tensor $x$, memory delta per rank is approximately:
$$
\Delta M(x)pprox |x|\left(1-rac{1}{p_g}ight).
$$

### Implementation details (practical)
1. Build process groups and bind them to physical topology intentionally; avoid accidental rank orders.
2. Choose bucket/chunk sizes so kernels stay large enough for high occupancy while communication can stream.
3. Co-schedule streams: compute stream + comm stream + dependency events to increase overlap safely.
4. Encode layout metadata in checkpoints to permit reshard/restart on changed meshes.
5. Validate numerics under mixed precision with loss-scale or FP8 scale tracking as required.

### Failure modes
- **Bandwidth-bound plateau:** throughput stops scaling as payload term dominates.
- **Latency-bound micro-ops:** too many tiny collectives; startup cost dominates.
- **Imbalance/stragglers:** rank skew from load imbalance or topology asymmetry.
- **Memory spikes:** poor prefetch/reshard boundaries or oversized microbatching.
- **Numerical drift:** precision policy mismatches across shards/collectives.

### Diagnostics and experiments
- Perform ablations: vary one axis degree at a time ($P_{tp}$, $P_{pp}$, etc.).
- Trace per-rank timelines to measure overlap ratio $ho=rac{T_{overlap}}{T_{comm}}$.
- Sweep microbatch count $m$, gradient bucket sizes, and activation checkpoint boundaries.
- Capture p50/p95/p99 iteration time and correlate with collective/kernel breakdowns.

### Study checklist
- Can you state the exact input/output layout for this subsection?
- Can you predict which collective dominates at larger scale?
- Can you explain one condition where this method should *not* be used?

### 5.3 Megatron Sequence Parallelism: Activation Sharding, Not Full Long-Context Attention
### Learning goals
- Understand the exact role of **5.3 Megatron Sequence Parallelism: Activation Sharding, Not Full Long-Context Attention** in the full training/serving stack.
- Identify required tensor layouts before and after this step.
- Map algorithmic choices to communication, memory, and stability consequences.

### Mechanism
- **Inputs/state.** Define local shards, global semantics, and invariants (shape/dtype/order).
- **Transformation.** Execute compute kernels plus communication primitives to move from input layout to output layout.
- **Correctness.** Preserve mathematical equivalence (modulo floating-point ordering) and optimizer semantics.
- **Output contract.** Ensure downstream subsection receives expected placement/state.

### Mathematical view
- Local shard sizing follows $n_{local}=n/p$ along sharded axis, with replicated axes unchanged.
- Runtime contribution is modeled as:
$$
T_{subsec}pprox T_{kernel}+T_{comm}-T_{overlap}+T_{sync}+T_{runtime}.
$$
- If this stage adds collective payload $q$ on group size $p_g$, then increment:
$$
\Delta T_{comm}pprox lpha f(p_g)+eta q g(p_g).
$$
- If this stage changes replication→sharding for tensor $x$, memory delta per rank is approximately:
$$
\Delta M(x)pprox |x|\left(1-rac{1}{p_g}ight).
$$

### Implementation details (practical)
1. Build process groups and bind them to physical topology intentionally; avoid accidental rank orders.
2. Choose bucket/chunk sizes so kernels stay large enough for high occupancy while communication can stream.
3. Co-schedule streams: compute stream + comm stream + dependency events to increase overlap safely.
4. Encode layout metadata in checkpoints to permit reshard/restart on changed meshes.
5. Validate numerics under mixed precision with loss-scale or FP8 scale tracking as required.

### Failure modes
- **Bandwidth-bound plateau:** throughput stops scaling as payload term dominates.
- **Latency-bound micro-ops:** too many tiny collectives; startup cost dominates.
- **Imbalance/stragglers:** rank skew from load imbalance or topology asymmetry.
- **Memory spikes:** poor prefetch/reshard boundaries or oversized microbatching.
- **Numerical drift:** precision policy mismatches across shards/collectives.

### Diagnostics and experiments
- Perform ablations: vary one axis degree at a time ($P_{tp}$, $P_{pp}$, etc.).
- Trace per-rank timelines to measure overlap ratio $ho=rac{T_{overlap}}{T_{comm}}$.
- Sweep microbatch count $m$, gradient bucket sizes, and activation checkpoint boundaries.
- Capture p50/p95/p99 iteration time and correlate with collective/kernel breakdowns.

### Study checklist
- Can you state the exact input/output layout for this subsection?
- Can you predict which collective dominates at larger scale?
- Can you explain one condition where this method should *not* be used?

### 5.4 Ulysses: Sequence Shards Become Head Shards Through All-to-All
### Learning goals
- Understand the exact role of **5.4 Ulysses: Sequence Shards Become Head Shards Through All-to-All** in the full training/serving stack.
- Identify required tensor layouts before and after this step.
- Map algorithmic choices to communication, memory, and stability consequences.

### Mechanism
- **Inputs/state.** Define local shards, global semantics, and invariants (shape/dtype/order).
- **Transformation.** Execute compute kernels plus communication primitives to move from input layout to output layout.
- **Correctness.** Preserve mathematical equivalence (modulo floating-point ordering) and optimizer semantics.
- **Output contract.** Ensure downstream subsection receives expected placement/state.

### Mathematical view
- Local shard sizing follows $n_{local}=n/p$ along sharded axis, with replicated axes unchanged.
- Runtime contribution is modeled as:
$$
T_{subsec}pprox T_{kernel}+T_{comm}-T_{overlap}+T_{sync}+T_{runtime}.
$$
- If this stage adds collective payload $q$ on group size $p_g$, then increment:
$$
\Delta T_{comm}pprox lpha f(p_g)+eta q g(p_g).
$$
- If this stage changes replication→sharding for tensor $x$, memory delta per rank is approximately:
$$
\Delta M(x)pprox |x|\left(1-rac{1}{p_g}ight).
$$

### Implementation details (practical)
1. Build process groups and bind them to physical topology intentionally; avoid accidental rank orders.
2. Choose bucket/chunk sizes so kernels stay large enough for high occupancy while communication can stream.
3. Co-schedule streams: compute stream + comm stream + dependency events to increase overlap safely.
4. Encode layout metadata in checkpoints to permit reshard/restart on changed meshes.
5. Validate numerics under mixed precision with loss-scale or FP8 scale tracking as required.

### Failure modes
- **Bandwidth-bound plateau:** throughput stops scaling as payload term dominates.
- **Latency-bound micro-ops:** too many tiny collectives; startup cost dominates.
- **Imbalance/stragglers:** rank skew from load imbalance or topology asymmetry.
- **Memory spikes:** poor prefetch/reshard boundaries or oversized microbatching.
- **Numerical drift:** precision policy mismatches across shards/collectives.

### Diagnostics and experiments
- Perform ablations: vary one axis degree at a time ($P_{tp}$, $P_{pp}$, etc.).
- Trace per-rank timelines to measure overlap ratio $ho=rac{T_{overlap}}{T_{comm}}$.
- Sweep microbatch count $m$, gradient bucket sizes, and activation checkpoint boundaries.
- Capture p50/p95/p99 iteration time and correlate with collective/kernel breakdowns.

### Study checklist
- Can you state the exact input/output layout for this subsection?
- Can you predict which collective dominates at larger scale?
- Can you explain one condition where this method should *not* be used?

### 5.5 Ring Attention: KV Streaming With Point-to-Point Send/Recv
### Learning goals
- Understand the exact role of **5.5 Ring Attention: KV Streaming With Point-to-Point Send/Recv** in the full training/serving stack.
- Identify required tensor layouts before and after this step.
- Map algorithmic choices to communication, memory, and stability consequences.

### Mechanism
- **Inputs/state.** Define local shards, global semantics, and invariants (shape/dtype/order).
- **Transformation.** Execute compute kernels plus communication primitives to move from input layout to output layout.
- **Correctness.** Preserve mathematical equivalence (modulo floating-point ordering) and optimizer semantics.
- **Output contract.** Ensure downstream subsection receives expected placement/state.

### Mathematical view
- Local shard sizing follows $n_{local}=n/p$ along sharded axis, with replicated axes unchanged.
- Runtime contribution is modeled as:
$$
T_{subsec}pprox T_{kernel}+T_{comm}-T_{overlap}+T_{sync}+T_{runtime}.
$$
- If this stage adds collective payload $q$ on group size $p_g$, then increment:
$$
\Delta T_{comm}pprox lpha f(p_g)+eta q g(p_g).
$$
- If this stage changes replication→sharding for tensor $x$, memory delta per rank is approximately:
$$
\Delta M(x)pprox |x|\left(1-rac{1}{p_g}ight).
$$

### Implementation details (practical)
1. Build process groups and bind them to physical topology intentionally; avoid accidental rank orders.
2. Choose bucket/chunk sizes so kernels stay large enough for high occupancy while communication can stream.
3. Co-schedule streams: compute stream + comm stream + dependency events to increase overlap safely.
4. Encode layout metadata in checkpoints to permit reshard/restart on changed meshes.
5. Validate numerics under mixed precision with loss-scale or FP8 scale tracking as required.

### Failure modes
- **Bandwidth-bound plateau:** throughput stops scaling as payload term dominates.
- **Latency-bound micro-ops:** too many tiny collectives; startup cost dominates.
- **Imbalance/stragglers:** rank skew from load imbalance or topology asymmetry.
- **Memory spikes:** poor prefetch/reshard boundaries or oversized microbatching.
- **Numerical drift:** precision policy mismatches across shards/collectives.

### Diagnostics and experiments
- Perform ablations: vary one axis degree at a time ($P_{tp}$, $P_{pp}$, etc.).
- Trace per-rank timelines to measure overlap ratio $ho=rac{T_{overlap}}{T_{comm}}$.
- Sweep microbatch count $m$, gradient bucket sizes, and activation checkpoint boundaries.
- Capture p50/p95/p99 iteration time and correlate with collective/kernel breakdowns.

### Study checklist
- Can you state the exact input/output layout for this subsection?
- Can you predict which collective dominates at larger scale?
- Can you explain one condition where this method should *not* be used?

### 5.6 The Numerical Primitive: Blockwise Online Softmax
### Learning goals
- Understand the exact role of **5.6 The Numerical Primitive: Blockwise Online Softmax** in the full training/serving stack.
- Identify required tensor layouts before and after this step.
- Map algorithmic choices to communication, memory, and stability consequences.

### Mechanism
- **Inputs/state.** Define local shards, global semantics, and invariants (shape/dtype/order).
- **Transformation.** Execute compute kernels plus communication primitives to move from input layout to output layout.
- **Correctness.** Preserve mathematical equivalence (modulo floating-point ordering) and optimizer semantics.
- **Output contract.** Ensure downstream subsection receives expected placement/state.

### Mathematical view
- Local shard sizing follows $n_{local}=n/p$ along sharded axis, with replicated axes unchanged.
- Runtime contribution is modeled as:
$$
T_{subsec}pprox T_{kernel}+T_{comm}-T_{overlap}+T_{sync}+T_{runtime}.
$$
- If this stage adds collective payload $q$ on group size $p_g$, then increment:
$$
\Delta T_{comm}pprox lpha f(p_g)+eta q g(p_g).
$$
- If this stage changes replication→sharding for tensor $x$, memory delta per rank is approximately:
$$
\Delta M(x)pprox |x|\left(1-rac{1}{p_g}ight).
$$

### Implementation details (practical)
1. Build process groups and bind them to physical topology intentionally; avoid accidental rank orders.
2. Choose bucket/chunk sizes so kernels stay large enough for high occupancy while communication can stream.
3. Co-schedule streams: compute stream + comm stream + dependency events to increase overlap safely.
4. Encode layout metadata in checkpoints to permit reshard/restart on changed meshes.
5. Validate numerics under mixed precision with loss-scale or FP8 scale tracking as required.

### Failure modes
- **Bandwidth-bound plateau:** throughput stops scaling as payload term dominates.
- **Latency-bound micro-ops:** too many tiny collectives; startup cost dominates.
- **Imbalance/stragglers:** rank skew from load imbalance or topology asymmetry.
- **Memory spikes:** poor prefetch/reshard boundaries or oversized microbatching.
- **Numerical drift:** precision policy mismatches across shards/collectives.

### Diagnostics and experiments
- Perform ablations: vary one axis degree at a time ($P_{tp}$, $P_{pp}$, etc.).
- Trace per-rank timelines to measure overlap ratio $ho=rac{T_{overlap}}{T_{comm}}$.
- Sweep microbatch count $m$, gradient bucket sizes, and activation checkpoint boundaries.
- Capture p50/p95/p99 iteration time and correlate with collective/kernel breakdowns.

### Study checklist
- Can you state the exact input/output layout for this subsection?
- Can you predict which collective dominates at larger scale?
- Can you explain one condition where this method should *not* be used?

### 5.7 Megatron Context Parallelism: Sequence Sharding for the Whole Network
### Learning goals
- Understand the exact role of **5.7 Megatron Context Parallelism: Sequence Sharding for the Whole Network** in the full training/serving stack.
- Identify required tensor layouts before and after this step.
- Map algorithmic choices to communication, memory, and stability consequences.

### Mechanism
- **Inputs/state.** Define local shards, global semantics, and invariants (shape/dtype/order).
- **Transformation.** Execute compute kernels plus communication primitives to move from input layout to output layout.
- **Correctness.** Preserve mathematical equivalence (modulo floating-point ordering) and optimizer semantics.
- **Output contract.** Ensure downstream subsection receives expected placement/state.

### Mathematical view
- Local shard sizing follows $n_{local}=n/p$ along sharded axis, with replicated axes unchanged.
- Runtime contribution is modeled as:
$$
T_{subsec}pprox T_{kernel}+T_{comm}-T_{overlap}+T_{sync}+T_{runtime}.
$$
- If this stage adds collective payload $q$ on group size $p_g$, then increment:
$$
\Delta T_{comm}pprox lpha f(p_g)+eta q g(p_g).
$$
- If this stage changes replication→sharding for tensor $x$, memory delta per rank is approximately:
$$
\Delta M(x)pprox |x|\left(1-rac{1}{p_g}ight).
$$

### Implementation details (practical)
1. Build process groups and bind them to physical topology intentionally; avoid accidental rank orders.
2. Choose bucket/chunk sizes so kernels stay large enough for high occupancy while communication can stream.
3. Co-schedule streams: compute stream + comm stream + dependency events to increase overlap safely.
4. Encode layout metadata in checkpoints to permit reshard/restart on changed meshes.
5. Validate numerics under mixed precision with loss-scale or FP8 scale tracking as required.

### Failure modes
- **Bandwidth-bound plateau:** throughput stops scaling as payload term dominates.
- **Latency-bound micro-ops:** too many tiny collectives; startup cost dominates.
- **Imbalance/stragglers:** rank skew from load imbalance or topology asymmetry.
- **Memory spikes:** poor prefetch/reshard boundaries or oversized microbatching.
- **Numerical drift:** precision policy mismatches across shards/collectives.

### Diagnostics and experiments
- Perform ablations: vary one axis degree at a time ($P_{tp}$, $P_{pp}$, etc.).
- Trace per-rank timelines to measure overlap ratio $ho=rac{T_{overlap}}{T_{comm}}$.
- Sweep microbatch count $m$, gradient bucket sizes, and activation checkpoint boundaries.
- Capture p50/p95/p99 iteration time and correlate with collective/kernel breakdowns.

### Study checklist
- Can you state the exact input/output layout for this subsection?
- Can you predict which collective dominates at larger scale?
- Can you explain one condition where this method should *not* be used?

### 5.8 Dynamic Context Parallelism: Choose CP Size Per Microbatch
### Learning goals
- Understand the exact role of **5.8 Dynamic Context Parallelism: Choose CP Size Per Microbatch** in the full training/serving stack.
- Identify required tensor layouts before and after this step.
- Map algorithmic choices to communication, memory, and stability consequences.

### Mechanism
- **Inputs/state.** Define local shards, global semantics, and invariants (shape/dtype/order).
- **Transformation.** Execute compute kernels plus communication primitives to move from input layout to output layout.
- **Correctness.** Preserve mathematical equivalence (modulo floating-point ordering) and optimizer semantics.
- **Output contract.** Ensure downstream subsection receives expected placement/state.

### Mathematical view
- Local shard sizing follows $n_{local}=n/p$ along sharded axis, with replicated axes unchanged.
- Runtime contribution is modeled as:
$$
T_{subsec}pprox T_{kernel}+T_{comm}-T_{overlap}+T_{sync}+T_{runtime}.
$$
- If this stage adds collective payload $q$ on group size $p_g$, then increment:
$$
\Delta T_{comm}pprox lpha f(p_g)+eta q g(p_g).
$$
- If this stage changes replication→sharding for tensor $x$, memory delta per rank is approximately:
$$
\Delta M(x)pprox |x|\left(1-rac{1}{p_g}ight).
$$

### Implementation details (practical)
1. Build process groups and bind them to physical topology intentionally; avoid accidental rank orders.
2. Choose bucket/chunk sizes so kernels stay large enough for high occupancy while communication can stream.
3. Co-schedule streams: compute stream + comm stream + dependency events to increase overlap safely.
4. Encode layout metadata in checkpoints to permit reshard/restart on changed meshes.
5. Validate numerics under mixed precision with loss-scale or FP8 scale tracking as required.

### Failure modes
- **Bandwidth-bound plateau:** throughput stops scaling as payload term dominates.
- **Latency-bound micro-ops:** too many tiny collectives; startup cost dominates.
- **Imbalance/stragglers:** rank skew from load imbalance or topology asymmetry.
- **Memory spikes:** poor prefetch/reshard boundaries or oversized microbatching.
- **Numerical drift:** precision policy mismatches across shards/collectives.

### Diagnostics and experiments
- Perform ablations: vary one axis degree at a time ($P_{tp}$, $P_{pp}$, etc.).
- Trace per-rank timelines to measure overlap ratio $ho=rac{T_{overlap}}{T_{comm}}$.
- Sweep microbatch count $m$, gradient bucket sizes, and activation checkpoint boundaries.
- Capture p50/p95/p99 iteration time and correlate with collective/kernel breakdowns.

### Study checklist
- Can you state the exact input/output layout for this subsection?
- Can you predict which collective dominates at larger scale?
- Can you explain one condition where this method should *not* be used?

### 5.9 THD and Ragged Packing: Token-Head-Dimension Layout
### Learning goals
- Understand the exact role of **5.9 THD and Ragged Packing: Token-Head-Dimension Layout** in the full training/serving stack.
- Identify required tensor layouts before and after this step.
- Map algorithmic choices to communication, memory, and stability consequences.

### Mechanism
- **Inputs/state.** Define local shards, global semantics, and invariants (shape/dtype/order).
- **Transformation.** Execute compute kernels plus communication primitives to move from input layout to output layout.
- **Correctness.** Preserve mathematical equivalence (modulo floating-point ordering) and optimizer semantics.
- **Output contract.** Ensure downstream subsection receives expected placement/state.

### Mathematical view
- Local shard sizing follows $n_{local}=n/p$ along sharded axis, with replicated axes unchanged.
- Runtime contribution is modeled as:
$$
T_{subsec}pprox T_{kernel}+T_{comm}-T_{overlap}+T_{sync}+T_{runtime}.
$$
- If this stage adds collective payload $q$ on group size $p_g$, then increment:
$$
\Delta T_{comm}pprox lpha f(p_g)+eta q g(p_g).
$$
- If this stage changes replication→sharding for tensor $x$, memory delta per rank is approximately:
$$
\Delta M(x)pprox |x|\left(1-rac{1}{p_g}ight).
$$

### Implementation details (practical)
1. Build process groups and bind them to physical topology intentionally; avoid accidental rank orders.
2. Choose bucket/chunk sizes so kernels stay large enough for high occupancy while communication can stream.
3. Co-schedule streams: compute stream + comm stream + dependency events to increase overlap safely.
4. Encode layout metadata in checkpoints to permit reshard/restart on changed meshes.
5. Validate numerics under mixed precision with loss-scale or FP8 scale tracking as required.

### Failure modes
- **Bandwidth-bound plateau:** throughput stops scaling as payload term dominates.
- **Latency-bound micro-ops:** too many tiny collectives; startup cost dominates.
- **Imbalance/stragglers:** rank skew from load imbalance or topology asymmetry.
- **Memory spikes:** poor prefetch/reshard boundaries or oversized microbatching.
- **Numerical drift:** precision policy mismatches across shards/collectives.

### Diagnostics and experiments
- Perform ablations: vary one axis degree at a time ($P_{tp}$, $P_{pp}$, etc.).
- Trace per-rank timelines to measure overlap ratio $ho=rac{T_{overlap}}{T_{comm}}$.
- Sweep microbatch count $m$, gradient bucket sizes, and activation checkpoint boundaries.
- Capture p50/p95/p99 iteration time and correlate with collective/kernel breakdowns.

### Study checklist
- Can you state the exact input/output layout for this subsection?
- Can you predict which collective dominates at larger scale?
- Can you explain one condition where this method should *not* be used?

### 5.10 Sparse and Hybrid Attention: Communicate Selected KV, Not Necessarily All KV
### Learning goals
- Understand the exact role of **5.10 Sparse and Hybrid Attention: Communicate Selected KV, Not Necessarily All KV** in the full training/serving stack.
- Identify required tensor layouts before and after this step.
- Map algorithmic choices to communication, memory, and stability consequences.

### Mechanism
- **Inputs/state.** Define local shards, global semantics, and invariants (shape/dtype/order).
- **Transformation.** Execute compute kernels plus communication primitives to move from input layout to output layout.
- **Correctness.** Preserve mathematical equivalence (modulo floating-point ordering) and optimizer semantics.
- **Output contract.** Ensure downstream subsection receives expected placement/state.

### Mathematical view
- Local shard sizing follows $n_{local}=n/p$ along sharded axis, with replicated axes unchanged.
- Runtime contribution is modeled as:
$$
T_{subsec}pprox T_{kernel}+T_{comm}-T_{overlap}+T_{sync}+T_{runtime}.
$$
- If this stage adds collective payload $q$ on group size $p_g$, then increment:
$$
\Delta T_{comm}pprox lpha f(p_g)+eta q g(p_g).
$$
- If this stage changes replication→sharding for tensor $x$, memory delta per rank is approximately:
$$
\Delta M(x)pprox |x|\left(1-rac{1}{p_g}ight).
$$

### Implementation details (practical)
1. Build process groups and bind them to physical topology intentionally; avoid accidental rank orders.
2. Choose bucket/chunk sizes so kernels stay large enough for high occupancy while communication can stream.
3. Co-schedule streams: compute stream + comm stream + dependency events to increase overlap safely.
4. Encode layout metadata in checkpoints to permit reshard/restart on changed meshes.
5. Validate numerics under mixed precision with loss-scale or FP8 scale tracking as required.

### Failure modes
- **Bandwidth-bound plateau:** throughput stops scaling as payload term dominates.
- **Latency-bound micro-ops:** too many tiny collectives; startup cost dominates.
- **Imbalance/stragglers:** rank skew from load imbalance or topology asymmetry.
- **Memory spikes:** poor prefetch/reshard boundaries or oversized microbatching.
- **Numerical drift:** precision policy mismatches across shards/collectives.

### Diagnostics and experiments
- Perform ablations: vary one axis degree at a time ($P_{tp}$, $P_{pp}$, etc.).
- Trace per-rank timelines to measure overlap ratio $ho=rac{T_{overlap}}{T_{comm}}$.
- Sweep microbatch count $m$, gradient bucket sizes, and activation checkpoint boundaries.
- Capture p50/p95/p99 iteration time and correlate with collective/kernel breakdowns.

### Study checklist
- Can you state the exact input/output layout for this subsection?
- Can you predict which collective dominates at larger scale?
- Can you explain one condition where this method should *not* be used?

### 5.11 Inference KV and Context Placement
### Learning goals
- Understand the exact role of **5.11 Inference KV and Context Placement** in the full training/serving stack.
- Identify required tensor layouts before and after this step.
- Map algorithmic choices to communication, memory, and stability consequences.

### Mechanism
- **Inputs/state.** Define local shards, global semantics, and invariants (shape/dtype/order).
- **Transformation.** Execute compute kernels plus communication primitives to move from input layout to output layout.
- **Correctness.** Preserve mathematical equivalence (modulo floating-point ordering) and optimizer semantics.
- **Output contract.** Ensure downstream subsection receives expected placement/state.

### Mathematical view
- Local shard sizing follows $n_{local}=n/p$ along sharded axis, with replicated axes unchanged.
- Runtime contribution is modeled as:
$$
T_{subsec}pprox T_{kernel}+T_{comm}-T_{overlap}+T_{sync}+T_{runtime}.
$$
- If this stage adds collective payload $q$ on group size $p_g$, then increment:
$$
\Delta T_{comm}pprox lpha f(p_g)+eta q g(p_g).
$$
- If this stage changes replication→sharding for tensor $x$, memory delta per rank is approximately:
$$
\Delta M(x)pprox |x|\left(1-rac{1}{p_g}ight).
$$

### Implementation details (practical)
1. Build process groups and bind them to physical topology intentionally; avoid accidental rank orders.
2. Choose bucket/chunk sizes so kernels stay large enough for high occupancy while communication can stream.
3. Co-schedule streams: compute stream + comm stream + dependency events to increase overlap safely.
4. Encode layout metadata in checkpoints to permit reshard/restart on changed meshes.
5. Validate numerics under mixed precision with loss-scale or FP8 scale tracking as required.

### Failure modes
- **Bandwidth-bound plateau:** throughput stops scaling as payload term dominates.
- **Latency-bound micro-ops:** too many tiny collectives; startup cost dominates.
- **Imbalance/stragglers:** rank skew from load imbalance or topology asymmetry.
- **Memory spikes:** poor prefetch/reshard boundaries or oversized microbatching.
- **Numerical drift:** precision policy mismatches across shards/collectives.

### Diagnostics and experiments
- Perform ablations: vary one axis degree at a time ($P_{tp}$, $P_{pp}$, etc.).
- Trace per-rank timelines to measure overlap ratio $ho=rac{T_{overlap}}{T_{comm}}$.
- Sweep microbatch count $m$, gradient bucket sizes, and activation checkpoint boundaries.
- Capture p50/p95/p99 iteration time and correlate with collective/kernel breakdowns.

### Study checklist
- Can you state the exact input/output layout for this subsection?
- Can you predict which collective dominates at larger scale?
- Can you explain one condition where this method should *not* be used?

### 5.12 How to Choose Among SP, Ulysses, Ring, CP, Dynamic-CP, and Sparse Context Placement
### Learning goals
- Understand the exact role of **5.12 How to Choose Among SP, Ulysses, Ring, CP, Dynamic-CP, and Sparse Context Placement** in the full training/serving stack.
- Identify required tensor layouts before and after this step.
- Map algorithmic choices to communication, memory, and stability consequences.

### Mechanism
- **Inputs/state.** Define local shards, global semantics, and invariants (shape/dtype/order).
- **Transformation.** Execute compute kernels plus communication primitives to move from input layout to output layout.
- **Correctness.** Preserve mathematical equivalence (modulo floating-point ordering) and optimizer semantics.
- **Output contract.** Ensure downstream subsection receives expected placement/state.

### Mathematical view
- Local shard sizing follows $n_{local}=n/p$ along sharded axis, with replicated axes unchanged.
- Runtime contribution is modeled as:
$$
T_{subsec}pprox T_{kernel}+T_{comm}-T_{overlap}+T_{sync}+T_{runtime}.
$$
- If this stage adds collective payload $q$ on group size $p_g$, then increment:
$$
\Delta T_{comm}pprox lpha f(p_g)+eta q g(p_g).
$$
- If this stage changes replication→sharding for tensor $x$, memory delta per rank is approximately:
$$
\Delta M(x)pprox |x|\left(1-rac{1}{p_g}ight).
$$

### Implementation details (practical)
1. Build process groups and bind them to physical topology intentionally; avoid accidental rank orders.
2. Choose bucket/chunk sizes so kernels stay large enough for high occupancy while communication can stream.
3. Co-schedule streams: compute stream + comm stream + dependency events to increase overlap safely.
4. Encode layout metadata in checkpoints to permit reshard/restart on changed meshes.
5. Validate numerics under mixed precision with loss-scale or FP8 scale tracking as required.

### Failure modes
- **Bandwidth-bound plateau:** throughput stops scaling as payload term dominates.
- **Latency-bound micro-ops:** too many tiny collectives; startup cost dominates.
- **Imbalance/stragglers:** rank skew from load imbalance or topology asymmetry.
- **Memory spikes:** poor prefetch/reshard boundaries or oversized microbatching.
- **Numerical drift:** precision policy mismatches across shards/collectives.

### Diagnostics and experiments
- Perform ablations: vary one axis degree at a time ($P_{tp}$, $P_{pp}$, etc.).
- Trace per-rank timelines to measure overlap ratio $ho=rac{T_{overlap}}{T_{comm}}$.
- Sweep microbatch count $m$, gradient bucket sizes, and activation checkpoint boundaries.
- Capture p50/p95/p99 iteration time and correlate with collective/kernel breakdowns.

### Study checklist
- Can you state the exact input/output layout for this subsection?
- Can you predict which collective dominates at larger scale?
- Can you explain one condition where this method should *not* be used?

### 5.13 Evidence Boundaries and Terminology Caveats
### Learning goals
- Understand the exact role of **5.13 Evidence Boundaries and Terminology Caveats** in the full training/serving stack.
- Identify required tensor layouts before and after this step.
- Map algorithmic choices to communication, memory, and stability consequences.

### Mechanism
- **Inputs/state.** Define local shards, global semantics, and invariants (shape/dtype/order).
- **Transformation.** Execute compute kernels plus communication primitives to move from input layout to output layout.
- **Correctness.** Preserve mathematical equivalence (modulo floating-point ordering) and optimizer semantics.
- **Output contract.** Ensure downstream subsection receives expected placement/state.

### Mathematical view
- Local shard sizing follows $n_{local}=n/p$ along sharded axis, with replicated axes unchanged.
- Runtime contribution is modeled as:
$$
T_{subsec}pprox T_{kernel}+T_{comm}-T_{overlap}+T_{sync}+T_{runtime}.
$$
- If this stage adds collective payload $q$ on group size $p_g$, then increment:
$$
\Delta T_{comm}pprox lpha f(p_g)+eta q g(p_g).
$$
- If this stage changes replication→sharding for tensor $x$, memory delta per rank is approximately:
$$
\Delta M(x)pprox |x|\left(1-rac{1}{p_g}ight).
$$

### Implementation details (practical)
1. Build process groups and bind them to physical topology intentionally; avoid accidental rank orders.
2. Choose bucket/chunk sizes so kernels stay large enough for high occupancy while communication can stream.
3. Co-schedule streams: compute stream + comm stream + dependency events to increase overlap safely.
4. Encode layout metadata in checkpoints to permit reshard/restart on changed meshes.
5. Validate numerics under mixed precision with loss-scale or FP8 scale tracking as required.

### Failure modes
- **Bandwidth-bound plateau:** throughput stops scaling as payload term dominates.
- **Latency-bound micro-ops:** too many tiny collectives; startup cost dominates.
- **Imbalance/stragglers:** rank skew from load imbalance or topology asymmetry.
- **Memory spikes:** poor prefetch/reshard boundaries or oversized microbatching.
- **Numerical drift:** precision policy mismatches across shards/collectives.

### Diagnostics and experiments
- Perform ablations: vary one axis degree at a time ($P_{tp}$, $P_{pp}$, etc.).
- Trace per-rank timelines to measure overlap ratio $ho=rac{T_{overlap}}{T_{comm}}$.
- Sweep microbatch count $m$, gradient bucket sizes, and activation checkpoint boundaries.
- Capture p50/p95/p99 iteration time and correlate with collective/kernel breakdowns.

### Study checklist
- Can you state the exact input/output layout for this subsection?
- Can you predict which collective dominates at larger scale?
- Can you explain one condition where this method should *not* be used?

## 6. Expert and MoE Parallelism
This section explains **6. Expert and MoE Parallelism** from first principles, implementation mechanics, cost/memory implications, topology interactions, and operational diagnostics.

### 6.1 Symbols, Shapes, and the MoE Layer
### Learning goals
- Understand the exact role of **6.1 Symbols, Shapes, and the MoE Layer** in the full training/serving stack.
- Identify required tensor layouts before and after this step.
- Map algorithmic choices to communication, memory, and stability consequences.

### Mechanism
- **Inputs/state.** Define local shards, global semantics, and invariants (shape/dtype/order).
- **Transformation.** Execute compute kernels plus communication primitives to move from input layout to output layout.
- **Correctness.** Preserve mathematical equivalence (modulo floating-point ordering) and optimizer semantics.
- **Output contract.** Ensure downstream subsection receives expected placement/state.

### Mathematical view
- Local shard sizing follows $n_{local}=n/p$ along sharded axis, with replicated axes unchanged.
- Runtime contribution is modeled as:
$$
T_{subsec}pprox T_{kernel}+T_{comm}-T_{overlap}+T_{sync}+T_{runtime}.
$$
- If this stage adds collective payload $q$ on group size $p_g$, then increment:
$$
\Delta T_{comm}pprox lpha f(p_g)+eta q g(p_g).
$$
- If this stage changes replication→sharding for tensor $x$, memory delta per rank is approximately:
$$
\Delta M(x)pprox |x|\left(1-rac{1}{p_g}ight).
$$

### Implementation details (practical)
1. Build process groups and bind them to physical topology intentionally; avoid accidental rank orders.
2. Choose bucket/chunk sizes so kernels stay large enough for high occupancy while communication can stream.
3. Co-schedule streams: compute stream + comm stream + dependency events to increase overlap safely.
4. Encode layout metadata in checkpoints to permit reshard/restart on changed meshes.
5. Validate numerics under mixed precision with loss-scale or FP8 scale tracking as required.

### Failure modes
- **Bandwidth-bound plateau:** throughput stops scaling as payload term dominates.
- **Latency-bound micro-ops:** too many tiny collectives; startup cost dominates.
- **Imbalance/stragglers:** rank skew from load imbalance or topology asymmetry.
- **Memory spikes:** poor prefetch/reshard boundaries or oversized microbatching.
- **Numerical drift:** precision policy mismatches across shards/collectives.

### Diagnostics and experiments
- Perform ablations: vary one axis degree at a time ($P_{tp}$, $P_{pp}$, etc.).
- Trace per-rank timelines to measure overlap ratio $ho=rac{T_{overlap}}{T_{comm}}$.
- Sweep microbatch count $m$, gradient bucket sizes, and activation checkpoint boundaries.
- Capture p50/p95/p99 iteration time and correlate with collective/kernel breakdowns.

### Study checklist
- Can you state the exact input/output layout for this subsection?
- Can you predict which collective dominates at larger scale?
- Can you explain one condition where this method should *not* be used?

### 6.2 Primitive Technique: Routing and Top-k Assignment
### Learning goals
- Understand the exact role of **6.2 Primitive Technique: Routing and Top-k Assignment** in the full training/serving stack.
- Identify required tensor layouts before and after this step.
- Map algorithmic choices to communication, memory, and stability consequences.

### Mechanism
- **Inputs/state.** Define local shards, global semantics, and invariants (shape/dtype/order).
- **Transformation.** Execute compute kernels plus communication primitives to move from input layout to output layout.
- **Correctness.** Preserve mathematical equivalence (modulo floating-point ordering) and optimizer semantics.
- **Output contract.** Ensure downstream subsection receives expected placement/state.

### Mathematical view
- Local shard sizing follows $n_{local}=n/p$ along sharded axis, with replicated axes unchanged.
- Runtime contribution is modeled as:
$$
T_{subsec}pprox T_{kernel}+T_{comm}-T_{overlap}+T_{sync}+T_{runtime}.
$$
- If this stage adds collective payload $q$ on group size $p_g$, then increment:
$$
\Delta T_{comm}pprox lpha f(p_g)+eta q g(p_g).
$$
- If this stage changes replication→sharding for tensor $x$, memory delta per rank is approximately:
$$
\Delta M(x)pprox |x|\left(1-rac{1}{p_g}ight).
$$

### Implementation details (practical)
1. Build process groups and bind them to physical topology intentionally; avoid accidental rank orders.
2. Choose bucket/chunk sizes so kernels stay large enough for high occupancy while communication can stream.
3. Co-schedule streams: compute stream + comm stream + dependency events to increase overlap safely.
4. Encode layout metadata in checkpoints to permit reshard/restart on changed meshes.
5. Validate numerics under mixed precision with loss-scale or FP8 scale tracking as required.

### Failure modes
- **Bandwidth-bound plateau:** throughput stops scaling as payload term dominates.
- **Latency-bound micro-ops:** too many tiny collectives; startup cost dominates.
- **Imbalance/stragglers:** rank skew from load imbalance or topology asymmetry.
- **Memory spikes:** poor prefetch/reshard boundaries or oversized microbatching.
- **Numerical drift:** precision policy mismatches across shards/collectives.

### Diagnostics and experiments
- Perform ablations: vary one axis degree at a time ($P_{tp}$, $P_{pp}$, etc.).
- Trace per-rank timelines to measure overlap ratio $ho=rac{T_{overlap}}{T_{comm}}$.
- Sweep microbatch count $m$, gradient bucket sizes, and activation checkpoint boundaries.
- Capture p50/p95/p99 iteration time and correlate with collective/kernel breakdowns.

### Study checklist
- Can you state the exact input/output layout for this subsection?
- Can you predict which collective dominates at larger scale?
- Can you explain one condition where this method should *not* be used?

### 6.3 Primitive Technique: Expert Ownership and Local Shard Shapes
### Learning goals
- Understand the exact role of **6.3 Primitive Technique: Expert Ownership and Local Shard Shapes** in the full training/serving stack.
- Identify required tensor layouts before and after this step.
- Map algorithmic choices to communication, memory, and stability consequences.

### Mechanism
- **Inputs/state.** Define local shards, global semantics, and invariants (shape/dtype/order).
- **Transformation.** Execute compute kernels plus communication primitives to move from input layout to output layout.
- **Correctness.** Preserve mathematical equivalence (modulo floating-point ordering) and optimizer semantics.
- **Output contract.** Ensure downstream subsection receives expected placement/state.

### Mathematical view
- Local shard sizing follows $n_{local}=n/p$ along sharded axis, with replicated axes unchanged.
- Runtime contribution is modeled as:
$$
T_{subsec}pprox T_{kernel}+T_{comm}-T_{overlap}+T_{sync}+T_{runtime}.
$$
- If this stage adds collective payload $q$ on group size $p_g$, then increment:
$$
\Delta T_{comm}pprox lpha f(p_g)+eta q g(p_g).
$$
- If this stage changes replication→sharding for tensor $x$, memory delta per rank is approximately:
$$
\Delta M(x)pprox |x|\left(1-rac{1}{p_g}ight).
$$

### Implementation details (practical)
1. Build process groups and bind them to physical topology intentionally; avoid accidental rank orders.
2. Choose bucket/chunk sizes so kernels stay large enough for high occupancy while communication can stream.
3. Co-schedule streams: compute stream + comm stream + dependency events to increase overlap safely.
4. Encode layout metadata in checkpoints to permit reshard/restart on changed meshes.
5. Validate numerics under mixed precision with loss-scale or FP8 scale tracking as required.

### Failure modes
- **Bandwidth-bound plateau:** throughput stops scaling as payload term dominates.
- **Latency-bound micro-ops:** too many tiny collectives; startup cost dominates.
- **Imbalance/stragglers:** rank skew from load imbalance or topology asymmetry.
- **Memory spikes:** poor prefetch/reshard boundaries or oversized microbatching.
- **Numerical drift:** precision policy mismatches across shards/collectives.

### Diagnostics and experiments
- Perform ablations: vary one axis degree at a time ($P_{tp}$, $P_{pp}$, etc.).
- Trace per-rank timelines to measure overlap ratio $ho=rac{T_{overlap}}{T_{comm}}$.
- Sweep microbatch count $m$, gradient bucket sizes, and activation checkpoint boundaries.
- Capture p50/p95/p99 iteration time and correlate with collective/kernel breakdowns.

### Study checklist
- Can you state the exact input/output layout for this subsection?
- Can you predict which collective dominates at larger scale?
- Can you explain one condition where this method should *not* be used?

### 6.4 Primitive Technique: Permutation, Packing, and Inverse Permutation
### Learning goals
- Understand the exact role of **6.4 Primitive Technique: Permutation, Packing, and Inverse Permutation** in the full training/serving stack.
- Identify required tensor layouts before and after this step.
- Map algorithmic choices to communication, memory, and stability consequences.

### Mechanism
- **Inputs/state.** Define local shards, global semantics, and invariants (shape/dtype/order).
- **Transformation.** Execute compute kernels plus communication primitives to move from input layout to output layout.
- **Correctness.** Preserve mathematical equivalence (modulo floating-point ordering) and optimizer semantics.
- **Output contract.** Ensure downstream subsection receives expected placement/state.

### Mathematical view
- Local shard sizing follows $n_{local}=n/p$ along sharded axis, with replicated axes unchanged.
- Runtime contribution is modeled as:
$$
T_{subsec}pprox T_{kernel}+T_{comm}-T_{overlap}+T_{sync}+T_{runtime}.
$$
- If this stage adds collective payload $q$ on group size $p_g$, then increment:
$$
\Delta T_{comm}pprox lpha f(p_g)+eta q g(p_g).
$$
- If this stage changes replication→sharding for tensor $x$, memory delta per rank is approximately:
$$
\Delta M(x)pprox |x|\left(1-rac{1}{p_g}ight).
$$

### Implementation details (practical)
1. Build process groups and bind them to physical topology intentionally; avoid accidental rank orders.
2. Choose bucket/chunk sizes so kernels stay large enough for high occupancy while communication can stream.
3. Co-schedule streams: compute stream + comm stream + dependency events to increase overlap safely.
4. Encode layout metadata in checkpoints to permit reshard/restart on changed meshes.
5. Validate numerics under mixed precision with loss-scale or FP8 scale tracking as required.

### Failure modes
- **Bandwidth-bound plateau:** throughput stops scaling as payload term dominates.
- **Latency-bound micro-ops:** too many tiny collectives; startup cost dominates.
- **Imbalance/stragglers:** rank skew from load imbalance or topology asymmetry.
- **Memory spikes:** poor prefetch/reshard boundaries or oversized microbatching.
- **Numerical drift:** precision policy mismatches across shards/collectives.

### Diagnostics and experiments
- Perform ablations: vary one axis degree at a time ($P_{tp}$, $P_{pp}$, etc.).
- Trace per-rank timelines to measure overlap ratio $ho=rac{T_{overlap}}{T_{comm}}$.
- Sweep microbatch count $m$, gradient bucket sizes, and activation checkpoint boundaries.
- Capture p50/p95/p99 iteration time and correlate with collective/kernel breakdowns.

### Study checklist
- Can you state the exact input/output layout for this subsection?
- Can you predict which collective dominates at larger scale?
- Can you explain one condition where this method should *not* be used?

### 6.5 Primitive Technique: All-to-All Dispatch, Notify, and Combine
### Learning goals
- Understand the exact role of **6.5 Primitive Technique: All-to-All Dispatch, Notify, and Combine** in the full training/serving stack.
- Identify required tensor layouts before and after this step.
- Map algorithmic choices to communication, memory, and stability consequences.

### Mechanism
- **Inputs/state.** Define local shards, global semantics, and invariants (shape/dtype/order).
- **Transformation.** Execute compute kernels plus communication primitives to move from input layout to output layout.
- **Correctness.** Preserve mathematical equivalence (modulo floating-point ordering) and optimizer semantics.
- **Output contract.** Ensure downstream subsection receives expected placement/state.

### Mathematical view
- Local shard sizing follows $n_{local}=n/p$ along sharded axis, with replicated axes unchanged.
- Runtime contribution is modeled as:
$$
T_{subsec}pprox T_{kernel}+T_{comm}-T_{overlap}+T_{sync}+T_{runtime}.
$$
- If this stage adds collective payload $q$ on group size $p_g$, then increment:
$$
\Delta T_{comm}pprox lpha f(p_g)+eta q g(p_g).
$$
- If this stage changes replication→sharding for tensor $x$, memory delta per rank is approximately:
$$
\Delta M(x)pprox |x|\left(1-rac{1}{p_g}ight).
$$

### Implementation details (practical)
1. Build process groups and bind them to physical topology intentionally; avoid accidental rank orders.
2. Choose bucket/chunk sizes so kernels stay large enough for high occupancy while communication can stream.
3. Co-schedule streams: compute stream + comm stream + dependency events to increase overlap safely.
4. Encode layout metadata in checkpoints to permit reshard/restart on changed meshes.
5. Validate numerics under mixed precision with loss-scale or FP8 scale tracking as required.

### Failure modes
- **Bandwidth-bound plateau:** throughput stops scaling as payload term dominates.
- **Latency-bound micro-ops:** too many tiny collectives; startup cost dominates.
- **Imbalance/stragglers:** rank skew from load imbalance or topology asymmetry.
- **Memory spikes:** poor prefetch/reshard boundaries or oversized microbatching.
- **Numerical drift:** precision policy mismatches across shards/collectives.

### Diagnostics and experiments
- Perform ablations: vary one axis degree at a time ($P_{tp}$, $P_{pp}$, etc.).
- Trace per-rank timelines to measure overlap ratio $ho=rac{T_{overlap}}{T_{comm}}$.
- Sweep microbatch count $m$, gradient bucket sizes, and activation checkpoint boundaries.
- Capture p50/p95/p99 iteration time and correlate with collective/kernel breakdowns.

### Study checklist
- Can you state the exact input/output layout for this subsection?
- Can you predict which collective dominates at larger scale?
- Can you explain one condition where this method should *not* be used?

### 6.6 Primitive Technique: Capacity, Dropless Execution, and Padding
### Learning goals
- Understand the exact role of **6.6 Primitive Technique: Capacity, Dropless Execution, and Padding** in the full training/serving stack.
- Identify required tensor layouts before and after this step.
- Map algorithmic choices to communication, memory, and stability consequences.

### Mechanism
- **Inputs/state.** Define local shards, global semantics, and invariants (shape/dtype/order).
- **Transformation.** Execute compute kernels plus communication primitives to move from input layout to output layout.
- **Correctness.** Preserve mathematical equivalence (modulo floating-point ordering) and optimizer semantics.
- **Output contract.** Ensure downstream subsection receives expected placement/state.

### Mathematical view
- Local shard sizing follows $n_{local}=n/p$ along sharded axis, with replicated axes unchanged.
- Runtime contribution is modeled as:
$$
T_{subsec}pprox T_{kernel}+T_{comm}-T_{overlap}+T_{sync}+T_{runtime}.
$$
- If this stage adds collective payload $q$ on group size $p_g$, then increment:
$$
\Delta T_{comm}pprox lpha f(p_g)+eta q g(p_g).
$$
- If this stage changes replication→sharding for tensor $x$, memory delta per rank is approximately:
$$
\Delta M(x)pprox |x|\left(1-rac{1}{p_g}ight).
$$

### Implementation details (practical)
1. Build process groups and bind them to physical topology intentionally; avoid accidental rank orders.
2. Choose bucket/chunk sizes so kernels stay large enough for high occupancy while communication can stream.
3. Co-schedule streams: compute stream + comm stream + dependency events to increase overlap safely.
4. Encode layout metadata in checkpoints to permit reshard/restart on changed meshes.
5. Validate numerics under mixed precision with loss-scale or FP8 scale tracking as required.

### Failure modes
- **Bandwidth-bound plateau:** throughput stops scaling as payload term dominates.
- **Latency-bound micro-ops:** too many tiny collectives; startup cost dominates.
- **Imbalance/stragglers:** rank skew from load imbalance or topology asymmetry.
- **Memory spikes:** poor prefetch/reshard boundaries or oversized microbatching.
- **Numerical drift:** precision policy mismatches across shards/collectives.

### Diagnostics and experiments
- Perform ablations: vary one axis degree at a time ($P_{tp}$, $P_{pp}$, etc.).
- Trace per-rank timelines to measure overlap ratio $ho=rac{T_{overlap}}{T_{comm}}$.
- Sweep microbatch count $m$, gradient bucket sizes, and activation checkpoint boundaries.
- Capture p50/p95/p99 iteration time and correlate with collective/kernel breakdowns.

### Study checklist
- Can you state the exact input/output layout for this subsection?
- Can you predict which collective dominates at larger scale?
- Can you explain one condition where this method should *not* be used?

### 6.7 Primitive Technique: Grouped GEMM, MegaBlocks, DeepGEMM, and Mega MoE
### Learning goals
- Understand the exact role of **6.7 Primitive Technique: Grouped GEMM, MegaBlocks, DeepGEMM, and Mega MoE** in the full training/serving stack.
- Identify required tensor layouts before and after this step.
- Map algorithmic choices to communication, memory, and stability consequences.

### Mechanism
- **Inputs/state.** Define local shards, global semantics, and invariants (shape/dtype/order).
- **Transformation.** Execute compute kernels plus communication primitives to move from input layout to output layout.
- **Correctness.** Preserve mathematical equivalence (modulo floating-point ordering) and optimizer semantics.
- **Output contract.** Ensure downstream subsection receives expected placement/state.

### Mathematical view
- Local shard sizing follows $n_{local}=n/p$ along sharded axis, with replicated axes unchanged.
- Runtime contribution is modeled as:
$$
T_{subsec}pprox T_{kernel}+T_{comm}-T_{overlap}+T_{sync}+T_{runtime}.
$$
- If this stage adds collective payload $q$ on group size $p_g$, then increment:
$$
\Delta T_{comm}pprox lpha f(p_g)+eta q g(p_g).
$$
- If this stage changes replication→sharding for tensor $x$, memory delta per rank is approximately:
$$
\Delta M(x)pprox |x|\left(1-rac{1}{p_g}ight).
$$

### Implementation details (practical)
1. Build process groups and bind them to physical topology intentionally; avoid accidental rank orders.
2. Choose bucket/chunk sizes so kernels stay large enough for high occupancy while communication can stream.
3. Co-schedule streams: compute stream + comm stream + dependency events to increase overlap safely.
4. Encode layout metadata in checkpoints to permit reshard/restart on changed meshes.
5. Validate numerics under mixed precision with loss-scale or FP8 scale tracking as required.

### Failure modes
- **Bandwidth-bound plateau:** throughput stops scaling as payload term dominates.
- **Latency-bound micro-ops:** too many tiny collectives; startup cost dominates.
- **Imbalance/stragglers:** rank skew from load imbalance or topology asymmetry.
- **Memory spikes:** poor prefetch/reshard boundaries or oversized microbatching.
- **Numerical drift:** precision policy mismatches across shards/collectives.

### Diagnostics and experiments
- Perform ablations: vary one axis degree at a time ($P_{tp}$, $P_{pp}$, etc.).
- Trace per-rank timelines to measure overlap ratio $ho=rac{T_{overlap}}{T_{comm}}$.
- Sweep microbatch count $m$, gradient bucket sizes, and activation checkpoint boundaries.
- Capture p50/p95/p99 iteration time and correlate with collective/kernel breakdowns.

### Study checklist
- Can you state the exact input/output layout for this subsection?
- Can you predict which collective dominates at larger scale?
- Can you explain one condition where this method should *not* be used?

### 6.8 Primitive Technique: Backward Pass, Weight Gradients, and Overlap
### Learning goals
- Understand the exact role of **6.8 Primitive Technique: Backward Pass, Weight Gradients, and Overlap** in the full training/serving stack.
- Identify required tensor layouts before and after this step.
- Map algorithmic choices to communication, memory, and stability consequences.

### Mechanism
- **Inputs/state.** Define local shards, global semantics, and invariants (shape/dtype/order).
- **Transformation.** Execute compute kernels plus communication primitives to move from input layout to output layout.
- **Correctness.** Preserve mathematical equivalence (modulo floating-point ordering) and optimizer semantics.
- **Output contract.** Ensure downstream subsection receives expected placement/state.

### Mathematical view
- Local shard sizing follows $n_{local}=n/p$ along sharded axis, with replicated axes unchanged.
- Runtime contribution is modeled as:
$$
T_{subsec}pprox T_{kernel}+T_{comm}-T_{overlap}+T_{sync}+T_{runtime}.
$$
- If this stage adds collective payload $q$ on group size $p_g$, then increment:
$$
\Delta T_{comm}pprox lpha f(p_g)+eta q g(p_g).
$$
- If this stage changes replication→sharding for tensor $x$, memory delta per rank is approximately:
$$
\Delta M(x)pprox |x|\left(1-rac{1}{p_g}ight).
$$

### Implementation details (practical)
1. Build process groups and bind them to physical topology intentionally; avoid accidental rank orders.
2. Choose bucket/chunk sizes so kernels stay large enough for high occupancy while communication can stream.
3. Co-schedule streams: compute stream + comm stream + dependency events to increase overlap safely.
4. Encode layout metadata in checkpoints to permit reshard/restart on changed meshes.
5. Validate numerics under mixed precision with loss-scale or FP8 scale tracking as required.

### Failure modes
- **Bandwidth-bound plateau:** throughput stops scaling as payload term dominates.
- **Latency-bound micro-ops:** too many tiny collectives; startup cost dominates.
- **Imbalance/stragglers:** rank skew from load imbalance or topology asymmetry.
- **Memory spikes:** poor prefetch/reshard boundaries or oversized microbatching.
- **Numerical drift:** precision policy mismatches across shards/collectives.

### Diagnostics and experiments
- Perform ablations: vary one axis degree at a time ($P_{tp}$, $P_{pp}$, etc.).
- Trace per-rank timelines to measure overlap ratio $ho=rac{T_{overlap}}{T_{comm}}$.
- Sweep microbatch count $m$, gradient bucket sizes, and activation checkpoint boundaries.
- Capture p50/p95/p99 iteration time and correlate with collective/kernel breakdowns.

### Study checklist
- Can you state the exact input/output layout for this subsection?
- Can you predict which collective dominates at larger scale?
- Can you explain one condition where this method should *not* be used?

### 6.9 Primitive Technique: DeepEP Modes and Network-Aware EP
### Learning goals
- Understand the exact role of **6.9 Primitive Technique: DeepEP Modes and Network-Aware EP** in the full training/serving stack.
- Identify required tensor layouts before and after this step.
- Map algorithmic choices to communication, memory, and stability consequences.

### Mechanism
- **Inputs/state.** Define local shards, global semantics, and invariants (shape/dtype/order).
- **Transformation.** Execute compute kernels plus communication primitives to move from input layout to output layout.
- **Correctness.** Preserve mathematical equivalence (modulo floating-point ordering) and optimizer semantics.
- **Output contract.** Ensure downstream subsection receives expected placement/state.

### Mathematical view
- Local shard sizing follows $n_{local}=n/p$ along sharded axis, with replicated axes unchanged.
- Runtime contribution is modeled as:
$$
T_{subsec}pprox T_{kernel}+T_{comm}-T_{overlap}+T_{sync}+T_{runtime}.
$$
- If this stage adds collective payload $q$ on group size $p_g$, then increment:
$$
\Delta T_{comm}pprox lpha f(p_g)+eta q g(p_g).
$$
- If this stage changes replication→sharding for tensor $x$, memory delta per rank is approximately:
$$
\Delta M(x)pprox |x|\left(1-rac{1}{p_g}ight).
$$

### Implementation details (practical)
1. Build process groups and bind them to physical topology intentionally; avoid accidental rank orders.
2. Choose bucket/chunk sizes so kernels stay large enough for high occupancy while communication can stream.
3. Co-schedule streams: compute stream + comm stream + dependency events to increase overlap safely.
4. Encode layout metadata in checkpoints to permit reshard/restart on changed meshes.
5. Validate numerics under mixed precision with loss-scale or FP8 scale tracking as required.

### Failure modes
- **Bandwidth-bound plateau:** throughput stops scaling as payload term dominates.
- **Latency-bound micro-ops:** too many tiny collectives; startup cost dominates.
- **Imbalance/stragglers:** rank skew from load imbalance or topology asymmetry.
- **Memory spikes:** poor prefetch/reshard boundaries or oversized microbatching.
- **Numerical drift:** precision policy mismatches across shards/collectives.

### Diagnostics and experiments
- Perform ablations: vary one axis degree at a time ($P_{tp}$, $P_{pp}$, etc.).
- Trace per-rank timelines to measure overlap ratio $ho=rac{T_{overlap}}{T_{comm}}$.
- Sweep microbatch count $m$, gradient bucket sizes, and activation checkpoint boundaries.
- Capture p50/p95/p99 iteration time and correlate with collective/kernel breakdowns.

### Study checklist
- Can you state the exact input/output layout for this subsection?
- Can you predict which collective dominates at larger scale?
- Can you explain one condition where this method should *not* be used?

### 6.10 Primitive Technique: Wide-EP Serving and Hot-Expert Replication
### Learning goals
- Understand the exact role of **6.10 Primitive Technique: Wide-EP Serving and Hot-Expert Replication** in the full training/serving stack.
- Identify required tensor layouts before and after this step.
- Map algorithmic choices to communication, memory, and stability consequences.

### Mechanism
- **Inputs/state.** Define local shards, global semantics, and invariants (shape/dtype/order).
- **Transformation.** Execute compute kernels plus communication primitives to move from input layout to output layout.
- **Correctness.** Preserve mathematical equivalence (modulo floating-point ordering) and optimizer semantics.
- **Output contract.** Ensure downstream subsection receives expected placement/state.

### Mathematical view
- Local shard sizing follows $n_{local}=n/p$ along sharded axis, with replicated axes unchanged.
- Runtime contribution is modeled as:
$$
T_{subsec}pprox T_{kernel}+T_{comm}-T_{overlap}+T_{sync}+T_{runtime}.
$$
- If this stage adds collective payload $q$ on group size $p_g$, then increment:
$$
\Delta T_{comm}pprox lpha f(p_g)+eta q g(p_g).
$$
- If this stage changes replication→sharding for tensor $x$, memory delta per rank is approximately:
$$
\Delta M(x)pprox |x|\left(1-rac{1}{p_g}ight).
$$

### Implementation details (practical)
1. Build process groups and bind them to physical topology intentionally; avoid accidental rank orders.
2. Choose bucket/chunk sizes so kernels stay large enough for high occupancy while communication can stream.
3. Co-schedule streams: compute stream + comm stream + dependency events to increase overlap safely.
4. Encode layout metadata in checkpoints to permit reshard/restart on changed meshes.
5. Validate numerics under mixed precision with loss-scale or FP8 scale tracking as required.

### Failure modes
- **Bandwidth-bound plateau:** throughput stops scaling as payload term dominates.
- **Latency-bound micro-ops:** too many tiny collectives; startup cost dominates.
- **Imbalance/stragglers:** rank skew from load imbalance or topology asymmetry.
- **Memory spikes:** poor prefetch/reshard boundaries or oversized microbatching.
- **Numerical drift:** precision policy mismatches across shards/collectives.

### Diagnostics and experiments
- Perform ablations: vary one axis degree at a time ($P_{tp}$, $P_{pp}$, etc.).
- Trace per-rank timelines to measure overlap ratio $ho=rac{T_{overlap}}{T_{comm}}$.
- Sweep microbatch count $m$, gradient bucket sizes, and activation checkpoint boundaries.
- Capture p50/p95/p99 iteration time and correlate with collective/kernel breakdowns.

### Study checklist
- Can you state the exact input/output layout for this subsection?
- Can you predict which collective dominates at larger scale?
- Can you explain one condition where this method should *not* be used?

### 6.11 Composition With TP, SP, PP, CP, DP, and Precision
### Learning goals
- Understand the exact role of **6.11 Composition With TP, SP, PP, CP, DP, and Precision** in the full training/serving stack.
- Identify required tensor layouts before and after this step.
- Map algorithmic choices to communication, memory, and stability consequences.

### Mechanism
- **Inputs/state.** Define local shards, global semantics, and invariants (shape/dtype/order).
- **Transformation.** Execute compute kernels plus communication primitives to move from input layout to output layout.
- **Correctness.** Preserve mathematical equivalence (modulo floating-point ordering) and optimizer semantics.
- **Output contract.** Ensure downstream subsection receives expected placement/state.

### Mathematical view
- Local shard sizing follows $n_{local}=n/p$ along sharded axis, with replicated axes unchanged.
- Runtime contribution is modeled as:
$$
T_{subsec}pprox T_{kernel}+T_{comm}-T_{overlap}+T_{sync}+T_{runtime}.
$$
- If this stage adds collective payload $q$ on group size $p_g$, then increment:
$$
\Delta T_{comm}pprox lpha f(p_g)+eta q g(p_g).
$$
- If this stage changes replication→sharding for tensor $x$, memory delta per rank is approximately:
$$
\Delta M(x)pprox |x|\left(1-rac{1}{p_g}ight).
$$

### Implementation details (practical)
1. Build process groups and bind them to physical topology intentionally; avoid accidental rank orders.
2. Choose bucket/chunk sizes so kernels stay large enough for high occupancy while communication can stream.
3. Co-schedule streams: compute stream + comm stream + dependency events to increase overlap safely.
4. Encode layout metadata in checkpoints to permit reshard/restart on changed meshes.
5. Validate numerics under mixed precision with loss-scale or FP8 scale tracking as required.

### Failure modes
- **Bandwidth-bound plateau:** throughput stops scaling as payload term dominates.
- **Latency-bound micro-ops:** too many tiny collectives; startup cost dominates.
- **Imbalance/stragglers:** rank skew from load imbalance or topology asymmetry.
- **Memory spikes:** poor prefetch/reshard boundaries or oversized microbatching.
- **Numerical drift:** precision policy mismatches across shards/collectives.

### Diagnostics and experiments
- Perform ablations: vary one axis degree at a time ($P_{tp}$, $P_{pp}$, etc.).
- Trace per-rank timelines to measure overlap ratio $ho=rac{T_{overlap}}{T_{comm}}$.
- Sweep microbatch count $m$, gradient bucket sizes, and activation checkpoint boundaries.
- Capture p50/p95/p99 iteration time and correlate with collective/kernel breakdowns.

### Study checklist
- Can you state the exact input/output layout for this subsection?
- Can you predict which collective dominates at larger scale?
- Can you explain one condition where this method should *not* be used?

### 6.12 End-to-End MoE Execution Recipe
### Learning goals
- Understand the exact role of **6.12 End-to-End MoE Execution Recipe** in the full training/serving stack.
- Identify required tensor layouts before and after this step.
- Map algorithmic choices to communication, memory, and stability consequences.

### Mechanism
- **Inputs/state.** Define local shards, global semantics, and invariants (shape/dtype/order).
- **Transformation.** Execute compute kernels plus communication primitives to move from input layout to output layout.
- **Correctness.** Preserve mathematical equivalence (modulo floating-point ordering) and optimizer semantics.
- **Output contract.** Ensure downstream subsection receives expected placement/state.

### Mathematical view
- Local shard sizing follows $n_{local}=n/p$ along sharded axis, with replicated axes unchanged.
- Runtime contribution is modeled as:
$$
T_{subsec}pprox T_{kernel}+T_{comm}-T_{overlap}+T_{sync}+T_{runtime}.
$$
- If this stage adds collective payload $q$ on group size $p_g$, then increment:
$$
\Delta T_{comm}pprox lpha f(p_g)+eta q g(p_g).
$$
- If this stage changes replication→sharding for tensor $x$, memory delta per rank is approximately:
$$
\Delta M(x)pprox |x|\left(1-rac{1}{p_g}ight).
$$

### Implementation details (practical)
1. Build process groups and bind them to physical topology intentionally; avoid accidental rank orders.
2. Choose bucket/chunk sizes so kernels stay large enough for high occupancy while communication can stream.
3. Co-schedule streams: compute stream + comm stream + dependency events to increase overlap safely.
4. Encode layout metadata in checkpoints to permit reshard/restart on changed meshes.
5. Validate numerics under mixed precision with loss-scale or FP8 scale tracking as required.

### Failure modes
- **Bandwidth-bound plateau:** throughput stops scaling as payload term dominates.
- **Latency-bound micro-ops:** too many tiny collectives; startup cost dominates.
- **Imbalance/stragglers:** rank skew from load imbalance or topology asymmetry.
- **Memory spikes:** poor prefetch/reshard boundaries or oversized microbatching.
- **Numerical drift:** precision policy mismatches across shards/collectives.

### Diagnostics and experiments
- Perform ablations: vary one axis degree at a time ($P_{tp}$, $P_{pp}$, etc.).
- Trace per-rank timelines to measure overlap ratio $ho=rac{T_{overlap}}{T_{comm}}$.
- Sweep microbatch count $m$, gradient bucket sizes, and activation checkpoint boundaries.
- Capture p50/p95/p99 iteration time and correlate with collective/kernel breakdowns.

### Study checklist
- Can you state the exact input/output layout for this subsection?
- Can you predict which collective dominates at larger scale?
- Can you explain one condition where this method should *not* be used?

### 6.13 Evidence Boundaries and Caveats
### Learning goals
- Understand the exact role of **6.13 Evidence Boundaries and Caveats** in the full training/serving stack.
- Identify required tensor layouts before and after this step.
- Map algorithmic choices to communication, memory, and stability consequences.

### Mechanism
- **Inputs/state.** Define local shards, global semantics, and invariants (shape/dtype/order).
- **Transformation.** Execute compute kernels plus communication primitives to move from input layout to output layout.
- **Correctness.** Preserve mathematical equivalence (modulo floating-point ordering) and optimizer semantics.
- **Output contract.** Ensure downstream subsection receives expected placement/state.

### Mathematical view
- Local shard sizing follows $n_{local}=n/p$ along sharded axis, with replicated axes unchanged.
- Runtime contribution is modeled as:
$$
T_{subsec}pprox T_{kernel}+T_{comm}-T_{overlap}+T_{sync}+T_{runtime}.
$$
- If this stage adds collective payload $q$ on group size $p_g$, then increment:
$$
\Delta T_{comm}pprox lpha f(p_g)+eta q g(p_g).
$$
- If this stage changes replication→sharding for tensor $x$, memory delta per rank is approximately:
$$
\Delta M(x)pprox |x|\left(1-rac{1}{p_g}ight).
$$

### Implementation details (practical)
1. Build process groups and bind them to physical topology intentionally; avoid accidental rank orders.
2. Choose bucket/chunk sizes so kernels stay large enough for high occupancy while communication can stream.
3. Co-schedule streams: compute stream + comm stream + dependency events to increase overlap safely.
4. Encode layout metadata in checkpoints to permit reshard/restart on changed meshes.
5. Validate numerics under mixed precision with loss-scale or FP8 scale tracking as required.

### Failure modes
- **Bandwidth-bound plateau:** throughput stops scaling as payload term dominates.
- **Latency-bound micro-ops:** too many tiny collectives; startup cost dominates.
- **Imbalance/stragglers:** rank skew from load imbalance or topology asymmetry.
- **Memory spikes:** poor prefetch/reshard boundaries or oversized microbatching.
- **Numerical drift:** precision policy mismatches across shards/collectives.

### Diagnostics and experiments
- Perform ablations: vary one axis degree at a time ($P_{tp}$, $P_{pp}$, etc.).
- Trace per-rank timelines to measure overlap ratio $ho=rac{T_{overlap}}{T_{comm}}$.
- Sweep microbatch count $m$, gradient bucket sizes, and activation checkpoint boundaries.
- Capture p50/p95/p99 iteration time and correlate with collective/kernel breakdowns.

### Study checklist
- Can you state the exact input/output layout for this subsection?
- Can you predict which collective dominates at larger scale?
- Can you explain one condition where this method should *not* be used?

## 7. Hybrid Mesh Parallelism
This section explains **7. Hybrid Mesh Parallelism** from first principles, implementation mechanics, cost/memory implications, topology interactions, and operational diagnostics.

### 7.1 Rank Mapping and Topology-Aware Placement
### Learning goals
- Understand the exact role of **7.1 Rank Mapping and Topology-Aware Placement** in the full training/serving stack.
- Identify required tensor layouts before and after this step.
- Map algorithmic choices to communication, memory, and stability consequences.

### Mechanism
- **Inputs/state.** Define local shards, global semantics, and invariants (shape/dtype/order).
- **Transformation.** Execute compute kernels plus communication primitives to move from input layout to output layout.
- **Correctness.** Preserve mathematical equivalence (modulo floating-point ordering) and optimizer semantics.
- **Output contract.** Ensure downstream subsection receives expected placement/state.

### Mathematical view
- Local shard sizing follows $n_{local}=n/p$ along sharded axis, with replicated axes unchanged.
- Runtime contribution is modeled as:
$$
T_{subsec}pprox T_{kernel}+T_{comm}-T_{overlap}+T_{sync}+T_{runtime}.
$$
- If this stage adds collective payload $q$ on group size $p_g$, then increment:
$$
\Delta T_{comm}pprox lpha f(p_g)+eta q g(p_g).
$$
- If this stage changes replication→sharding for tensor $x$, memory delta per rank is approximately:
$$
\Delta M(x)pprox |x|\left(1-rac{1}{p_g}ight).
$$

### Implementation details (practical)
1. Build process groups and bind them to physical topology intentionally; avoid accidental rank orders.
2. Choose bucket/chunk sizes so kernels stay large enough for high occupancy while communication can stream.
3. Co-schedule streams: compute stream + comm stream + dependency events to increase overlap safely.
4. Encode layout metadata in checkpoints to permit reshard/restart on changed meshes.
5. Validate numerics under mixed precision with loss-scale or FP8 scale tracking as required.

### Failure modes
- **Bandwidth-bound plateau:** throughput stops scaling as payload term dominates.
- **Latency-bound micro-ops:** too many tiny collectives; startup cost dominates.
- **Imbalance/stragglers:** rank skew from load imbalance or topology asymmetry.
- **Memory spikes:** poor prefetch/reshard boundaries or oversized microbatching.
- **Numerical drift:** precision policy mismatches across shards/collectives.

### Diagnostics and experiments
- Perform ablations: vary one axis degree at a time ($P_{tp}$, $P_{pp}$, etc.).
- Trace per-rank timelines to measure overlap ratio $ho=rac{T_{overlap}}{T_{comm}}$.
- Sweep microbatch count $m$, gradient bucket sizes, and activation checkpoint boundaries.
- Capture p50/p95/p99 iteration time and correlate with collective/kernel breakdowns.

### Study checklist
- Can you state the exact input/output layout for this subsection?
- Can you predict which collective dominates at larger scale?
- Can you explain one condition where this method should *not* be used?

### 7.2 Checkpoint Resharding and Restart Semantics
### Learning goals
- Understand the exact role of **7.2 Checkpoint Resharding and Restart Semantics** in the full training/serving stack.
- Identify required tensor layouts before and after this step.
- Map algorithmic choices to communication, memory, and stability consequences.

### Mechanism
- **Inputs/state.** Define local shards, global semantics, and invariants (shape/dtype/order).
- **Transformation.** Execute compute kernels plus communication primitives to move from input layout to output layout.
- **Correctness.** Preserve mathematical equivalence (modulo floating-point ordering) and optimizer semantics.
- **Output contract.** Ensure downstream subsection receives expected placement/state.

### Mathematical view
- Local shard sizing follows $n_{local}=n/p$ along sharded axis, with replicated axes unchanged.
- Runtime contribution is modeled as:
$$
T_{subsec}pprox T_{kernel}+T_{comm}-T_{overlap}+T_{sync}+T_{runtime}.
$$
- If this stage adds collective payload $q$ on group size $p_g$, then increment:
$$
\Delta T_{comm}pprox lpha f(p_g)+eta q g(p_g).
$$
- If this stage changes replication→sharding for tensor $x$, memory delta per rank is approximately:
$$
\Delta M(x)pprox |x|\left(1-rac{1}{p_g}ight).
$$

### Implementation details (practical)
1. Build process groups and bind them to physical topology intentionally; avoid accidental rank orders.
2. Choose bucket/chunk sizes so kernels stay large enough for high occupancy while communication can stream.
3. Co-schedule streams: compute stream + comm stream + dependency events to increase overlap safely.
4. Encode layout metadata in checkpoints to permit reshard/restart on changed meshes.
5. Validate numerics under mixed precision with loss-scale or FP8 scale tracking as required.

### Failure modes
- **Bandwidth-bound plateau:** throughput stops scaling as payload term dominates.
- **Latency-bound micro-ops:** too many tiny collectives; startup cost dominates.
- **Imbalance/stragglers:** rank skew from load imbalance or topology asymmetry.
- **Memory spikes:** poor prefetch/reshard boundaries or oversized microbatching.
- **Numerical drift:** precision policy mismatches across shards/collectives.

### Diagnostics and experiments
- Perform ablations: vary one axis degree at a time ($P_{tp}$, $P_{pp}$, etc.).
- Trace per-rank timelines to measure overlap ratio $ho=rac{T_{overlap}}{T_{comm}}$.
- Sweep microbatch count $m$, gradient bucket sizes, and activation checkpoint boundaries.
- Capture p50/p95/p99 iteration time and correlate with collective/kernel breakdowns.

### Study checklist
- Can you state the exact input/output layout for this subsection?
- Can you predict which collective dominates at larger scale?
- Can you explain one condition where this method should *not* be used?

### 7.3 Composition Rules
### Learning goals
- Understand the exact role of **7.3 Composition Rules** in the full training/serving stack.
- Identify required tensor layouts before and after this step.
- Map algorithmic choices to communication, memory, and stability consequences.

### Mechanism
- **Inputs/state.** Define local shards, global semantics, and invariants (shape/dtype/order).
- **Transformation.** Execute compute kernels plus communication primitives to move from input layout to output layout.
- **Correctness.** Preserve mathematical equivalence (modulo floating-point ordering) and optimizer semantics.
- **Output contract.** Ensure downstream subsection receives expected placement/state.

### Mathematical view
- Local shard sizing follows $n_{local}=n/p$ along sharded axis, with replicated axes unchanged.
- Runtime contribution is modeled as:
$$
T_{subsec}pprox T_{kernel}+T_{comm}-T_{overlap}+T_{sync}+T_{runtime}.
$$
- If this stage adds collective payload $q$ on group size $p_g$, then increment:
$$
\Delta T_{comm}pprox lpha f(p_g)+eta q g(p_g).
$$
- If this stage changes replication→sharding for tensor $x$, memory delta per rank is approximately:
$$
\Delta M(x)pprox |x|\left(1-rac{1}{p_g}ight).
$$

### Implementation details (practical)
1. Build process groups and bind them to physical topology intentionally; avoid accidental rank orders.
2. Choose bucket/chunk sizes so kernels stay large enough for high occupancy while communication can stream.
3. Co-schedule streams: compute stream + comm stream + dependency events to increase overlap safely.
4. Encode layout metadata in checkpoints to permit reshard/restart on changed meshes.
5. Validate numerics under mixed precision with loss-scale or FP8 scale tracking as required.

### Failure modes
- **Bandwidth-bound plateau:** throughput stops scaling as payload term dominates.
- **Latency-bound micro-ops:** too many tiny collectives; startup cost dominates.
- **Imbalance/stragglers:** rank skew from load imbalance or topology asymmetry.
- **Memory spikes:** poor prefetch/reshard boundaries or oversized microbatching.
- **Numerical drift:** precision policy mismatches across shards/collectives.

### Diagnostics and experiments
- Perform ablations: vary one axis degree at a time ($P_{tp}$, $P_{pp}$, etc.).
- Trace per-rank timelines to measure overlap ratio $ho=rac{T_{overlap}}{T_{comm}}$.
- Sweep microbatch count $m$, gradient bucket sizes, and activation checkpoint boundaries.
- Capture p50/p95/p99 iteration time and correlate with collective/kernel breakdowns.

### Study checklist
- Can you state the exact input/output layout for this subsection?
- Can you predict which collective dominates at larger scale?
- Can you explain one condition where this method should *not* be used?

### 7.4 System Perspectives
### Learning goals
- Understand the exact role of **7.4 System Perspectives** in the full training/serving stack.
- Identify required tensor layouts before and after this step.
- Map algorithmic choices to communication, memory, and stability consequences.

### Mechanism
- **Inputs/state.** Define local shards, global semantics, and invariants (shape/dtype/order).
- **Transformation.** Execute compute kernels plus communication primitives to move from input layout to output layout.
- **Correctness.** Preserve mathematical equivalence (modulo floating-point ordering) and optimizer semantics.
- **Output contract.** Ensure downstream subsection receives expected placement/state.

### Mathematical view
- Local shard sizing follows $n_{local}=n/p$ along sharded axis, with replicated axes unchanged.
- Runtime contribution is modeled as:
$$
T_{subsec}pprox T_{kernel}+T_{comm}-T_{overlap}+T_{sync}+T_{runtime}.
$$
- If this stage adds collective payload $q$ on group size $p_g$, then increment:
$$
\Delta T_{comm}pprox lpha f(p_g)+eta q g(p_g).
$$
- If this stage changes replication→sharding for tensor $x$, memory delta per rank is approximately:
$$
\Delta M(x)pprox |x|\left(1-rac{1}{p_g}ight).
$$

### Implementation details (practical)
1. Build process groups and bind them to physical topology intentionally; avoid accidental rank orders.
2. Choose bucket/chunk sizes so kernels stay large enough for high occupancy while communication can stream.
3. Co-schedule streams: compute stream + comm stream + dependency events to increase overlap safely.
4. Encode layout metadata in checkpoints to permit reshard/restart on changed meshes.
5. Validate numerics under mixed precision with loss-scale or FP8 scale tracking as required.

### Failure modes
- **Bandwidth-bound plateau:** throughput stops scaling as payload term dominates.
- **Latency-bound micro-ops:** too many tiny collectives; startup cost dominates.
- **Imbalance/stragglers:** rank skew from load imbalance or topology asymmetry.
- **Memory spikes:** poor prefetch/reshard boundaries or oversized microbatching.
- **Numerical drift:** precision policy mismatches across shards/collectives.

### Diagnostics and experiments
- Perform ablations: vary one axis degree at a time ($P_{tp}$, $P_{pp}$, etc.).
- Trace per-rank timelines to measure overlap ratio $ho=rac{T_{overlap}}{T_{comm}}$.
- Sweep microbatch count $m$, gradient bucket sizes, and activation checkpoint boundaries.
- Capture p50/p95/p99 iteration time and correlate with collective/kernel breakdowns.

### Study checklist
- Can you state the exact input/output layout for this subsection?
- Can you predict which collective dominates at larger scale?
- Can you explain one condition where this method should *not* be used?

### 7.5 Caveats and Evidence Boundaries
### Learning goals
- Understand the exact role of **7.5 Caveats and Evidence Boundaries** in the full training/serving stack.
- Identify required tensor layouts before and after this step.
- Map algorithmic choices to communication, memory, and stability consequences.

### Mechanism
- **Inputs/state.** Define local shards, global semantics, and invariants (shape/dtype/order).
- **Transformation.** Execute compute kernels plus communication primitives to move from input layout to output layout.
- **Correctness.** Preserve mathematical equivalence (modulo floating-point ordering) and optimizer semantics.
- **Output contract.** Ensure downstream subsection receives expected placement/state.

### Mathematical view
- Local shard sizing follows $n_{local}=n/p$ along sharded axis, with replicated axes unchanged.
- Runtime contribution is modeled as:
$$
T_{subsec}pprox T_{kernel}+T_{comm}-T_{overlap}+T_{sync}+T_{runtime}.
$$
- If this stage adds collective payload $q$ on group size $p_g$, then increment:
$$
\Delta T_{comm}pprox lpha f(p_g)+eta q g(p_g).
$$
- If this stage changes replication→sharding for tensor $x$, memory delta per rank is approximately:
$$
\Delta M(x)pprox |x|\left(1-rac{1}{p_g}ight).
$$

### Implementation details (practical)
1. Build process groups and bind them to physical topology intentionally; avoid accidental rank orders.
2. Choose bucket/chunk sizes so kernels stay large enough for high occupancy while communication can stream.
3. Co-schedule streams: compute stream + comm stream + dependency events to increase overlap safely.
4. Encode layout metadata in checkpoints to permit reshard/restart on changed meshes.
5. Validate numerics under mixed precision with loss-scale or FP8 scale tracking as required.

### Failure modes
- **Bandwidth-bound plateau:** throughput stops scaling as payload term dominates.
- **Latency-bound micro-ops:** too many tiny collectives; startup cost dominates.
- **Imbalance/stragglers:** rank skew from load imbalance or topology asymmetry.
- **Memory spikes:** poor prefetch/reshard boundaries or oversized microbatching.
- **Numerical drift:** precision policy mismatches across shards/collectives.

### Diagnostics and experiments
- Perform ablations: vary one axis degree at a time ($P_{tp}$, $P_{pp}$, etc.).
- Trace per-rank timelines to measure overlap ratio $ho=rac{T_{overlap}}{T_{comm}}$.
- Sweep microbatch count $m$, gradient bucket sizes, and activation checkpoint boundaries.
- Capture p50/p95/p99 iteration time and correlate with collective/kernel breakdowns.

### Study checklist
- Can you state the exact input/output layout for this subsection?
- Can you predict which collective dominates at larger scale?
- Can you explain one condition where this method should *not* be used?

## 8. Inference Parallelism and Serving Systems
This section explains **8. Inference Parallelism and Serving Systems** from first principles, implementation mechanics, cost/memory implications, topology interactions, and operational diagnostics.

### 8.1 Primitive Technique: Replicas and Request Routing
### Learning goals
- Understand the exact role of **8.1 Primitive Technique: Replicas and Request Routing** in the full training/serving stack.
- Identify required tensor layouts before and after this step.
- Map algorithmic choices to communication, memory, and stability consequences.

### Mechanism
- **Inputs/state.** Define local shards, global semantics, and invariants (shape/dtype/order).
- **Transformation.** Execute compute kernels plus communication primitives to move from input layout to output layout.
- **Correctness.** Preserve mathematical equivalence (modulo floating-point ordering) and optimizer semantics.
- **Output contract.** Ensure downstream subsection receives expected placement/state.

### Mathematical view
- Local shard sizing follows $n_{local}=n/p$ along sharded axis, with replicated axes unchanged.
- Runtime contribution is modeled as:
$$
T_{subsec}pprox T_{kernel}+T_{comm}-T_{overlap}+T_{sync}+T_{runtime}.
$$
- If this stage adds collective payload $q$ on group size $p_g$, then increment:
$$
\Delta T_{comm}pprox lpha f(p_g)+eta q g(p_g).
$$
- If this stage changes replication→sharding for tensor $x$, memory delta per rank is approximately:
$$
\Delta M(x)pprox |x|\left(1-rac{1}{p_g}ight).
$$

### Implementation details (practical)
1. Build process groups and bind them to physical topology intentionally; avoid accidental rank orders.
2. Choose bucket/chunk sizes so kernels stay large enough for high occupancy while communication can stream.
3. Co-schedule streams: compute stream + comm stream + dependency events to increase overlap safely.
4. Encode layout metadata in checkpoints to permit reshard/restart on changed meshes.
5. Validate numerics under mixed precision with loss-scale or FP8 scale tracking as required.

### Failure modes
- **Bandwidth-bound plateau:** throughput stops scaling as payload term dominates.
- **Latency-bound micro-ops:** too many tiny collectives; startup cost dominates.
- **Imbalance/stragglers:** rank skew from load imbalance or topology asymmetry.
- **Memory spikes:** poor prefetch/reshard boundaries or oversized microbatching.
- **Numerical drift:** precision policy mismatches across shards/collectives.

### Diagnostics and experiments
- Perform ablations: vary one axis degree at a time ($P_{tp}$, $P_{pp}$, etc.).
- Trace per-rank timelines to measure overlap ratio $ho=rac{T_{overlap}}{T_{comm}}$.
- Sweep microbatch count $m$, gradient bucket sizes, and activation checkpoint boundaries.
- Capture p50/p95/p99 iteration time and correlate with collective/kernel breakdowns.

### Study checklist
- Can you state the exact input/output layout for this subsection?
- Can you predict which collective dominates at larger scale?
- Can you explain one condition where this method should *not* be used?

### 8.2 Primitive Technique: Continuous Batching
### Learning goals
- Understand the exact role of **8.2 Primitive Technique: Continuous Batching** in the full training/serving stack.
- Identify required tensor layouts before and after this step.
- Map algorithmic choices to communication, memory, and stability consequences.

### Mechanism
- **Inputs/state.** Define local shards, global semantics, and invariants (shape/dtype/order).
- **Transformation.** Execute compute kernels plus communication primitives to move from input layout to output layout.
- **Correctness.** Preserve mathematical equivalence (modulo floating-point ordering) and optimizer semantics.
- **Output contract.** Ensure downstream subsection receives expected placement/state.

### Mathematical view
- Local shard sizing follows $n_{local}=n/p$ along sharded axis, with replicated axes unchanged.
- Runtime contribution is modeled as:
$$
T_{subsec}pprox T_{kernel}+T_{comm}-T_{overlap}+T_{sync}+T_{runtime}.
$$
- If this stage adds collective payload $q$ on group size $p_g$, then increment:
$$
\Delta T_{comm}pprox lpha f(p_g)+eta q g(p_g).
$$
- If this stage changes replication→sharding for tensor $x$, memory delta per rank is approximately:
$$
\Delta M(x)pprox |x|\left(1-rac{1}{p_g}ight).
$$

### Implementation details (practical)
1. Build process groups and bind them to physical topology intentionally; avoid accidental rank orders.
2. Choose bucket/chunk sizes so kernels stay large enough for high occupancy while communication can stream.
3. Co-schedule streams: compute stream + comm stream + dependency events to increase overlap safely.
4. Encode layout metadata in checkpoints to permit reshard/restart on changed meshes.
5. Validate numerics under mixed precision with loss-scale or FP8 scale tracking as required.

### Failure modes
- **Bandwidth-bound plateau:** throughput stops scaling as payload term dominates.
- **Latency-bound micro-ops:** too many tiny collectives; startup cost dominates.
- **Imbalance/stragglers:** rank skew from load imbalance or topology asymmetry.
- **Memory spikes:** poor prefetch/reshard boundaries or oversized microbatching.
- **Numerical drift:** precision policy mismatches across shards/collectives.

### Diagnostics and experiments
- Perform ablations: vary one axis degree at a time ($P_{tp}$, $P_{pp}$, etc.).
- Trace per-rank timelines to measure overlap ratio $ho=rac{T_{overlap}}{T_{comm}}$.
- Sweep microbatch count $m$, gradient bucket sizes, and activation checkpoint boundaries.
- Capture p50/p95/p99 iteration time and correlate with collective/kernel breakdowns.

### Study checklist
- Can you state the exact input/output layout for this subsection?
- Can you predict which collective dominates at larger scale?
- Can you explain one condition where this method should *not* be used?

### 8.3 Primitive Technique: Paged KV Cache
### Learning goals
- Understand the exact role of **8.3 Primitive Technique: Paged KV Cache** in the full training/serving stack.
- Identify required tensor layouts before and after this step.
- Map algorithmic choices to communication, memory, and stability consequences.

### Mechanism
- **Inputs/state.** Define local shards, global semantics, and invariants (shape/dtype/order).
- **Transformation.** Execute compute kernels plus communication primitives to move from input layout to output layout.
- **Correctness.** Preserve mathematical equivalence (modulo floating-point ordering) and optimizer semantics.
- **Output contract.** Ensure downstream subsection receives expected placement/state.

### Mathematical view
- Local shard sizing follows $n_{local}=n/p$ along sharded axis, with replicated axes unchanged.
- Runtime contribution is modeled as:
$$
T_{subsec}pprox T_{kernel}+T_{comm}-T_{overlap}+T_{sync}+T_{runtime}.
$$
- If this stage adds collective payload $q$ on group size $p_g$, then increment:
$$
\Delta T_{comm}pprox lpha f(p_g)+eta q g(p_g).
$$
- If this stage changes replication→sharding for tensor $x$, memory delta per rank is approximately:
$$
\Delta M(x)pprox |x|\left(1-rac{1}{p_g}ight).
$$

### Implementation details (practical)
1. Build process groups and bind them to physical topology intentionally; avoid accidental rank orders.
2. Choose bucket/chunk sizes so kernels stay large enough for high occupancy while communication can stream.
3. Co-schedule streams: compute stream + comm stream + dependency events to increase overlap safely.
4. Encode layout metadata in checkpoints to permit reshard/restart on changed meshes.
5. Validate numerics under mixed precision with loss-scale or FP8 scale tracking as required.

### Failure modes
- **Bandwidth-bound plateau:** throughput stops scaling as payload term dominates.
- **Latency-bound micro-ops:** too many tiny collectives; startup cost dominates.
- **Imbalance/stragglers:** rank skew from load imbalance or topology asymmetry.
- **Memory spikes:** poor prefetch/reshard boundaries or oversized microbatching.
- **Numerical drift:** precision policy mismatches across shards/collectives.

### Diagnostics and experiments
- Perform ablations: vary one axis degree at a time ($P_{tp}$, $P_{pp}$, etc.).
- Trace per-rank timelines to measure overlap ratio $ho=rac{T_{overlap}}{T_{comm}}$.
- Sweep microbatch count $m$, gradient bucket sizes, and activation checkpoint boundaries.
- Capture p50/p95/p99 iteration time and correlate with collective/kernel breakdowns.

### Study checklist
- Can you state the exact input/output layout for this subsection?
- Can you predict which collective dominates at larger scale?
- Can you explain one condition where this method should *not* be used?

### 8.4 Primitive Technique: Prefix and Radix Caching
### Learning goals
- Understand the exact role of **8.4 Primitive Technique: Prefix and Radix Caching** in the full training/serving stack.
- Identify required tensor layouts before and after this step.
- Map algorithmic choices to communication, memory, and stability consequences.

### Mechanism
- **Inputs/state.** Define local shards, global semantics, and invariants (shape/dtype/order).
- **Transformation.** Execute compute kernels plus communication primitives to move from input layout to output layout.
- **Correctness.** Preserve mathematical equivalence (modulo floating-point ordering) and optimizer semantics.
- **Output contract.** Ensure downstream subsection receives expected placement/state.

### Mathematical view
- Local shard sizing follows $n_{local}=n/p$ along sharded axis, with replicated axes unchanged.
- Runtime contribution is modeled as:
$$
T_{subsec}pprox T_{kernel}+T_{comm}-T_{overlap}+T_{sync}+T_{runtime}.
$$
- If this stage adds collective payload $q$ on group size $p_g$, then increment:
$$
\Delta T_{comm}pprox lpha f(p_g)+eta q g(p_g).
$$
- If this stage changes replication→sharding for tensor $x$, memory delta per rank is approximately:
$$
\Delta M(x)pprox |x|\left(1-rac{1}{p_g}ight).
$$

### Implementation details (practical)
1. Build process groups and bind them to physical topology intentionally; avoid accidental rank orders.
2. Choose bucket/chunk sizes so kernels stay large enough for high occupancy while communication can stream.
3. Co-schedule streams: compute stream + comm stream + dependency events to increase overlap safely.
4. Encode layout metadata in checkpoints to permit reshard/restart on changed meshes.
5. Validate numerics under mixed precision with loss-scale or FP8 scale tracking as required.

### Failure modes
- **Bandwidth-bound plateau:** throughput stops scaling as payload term dominates.
- **Latency-bound micro-ops:** too many tiny collectives; startup cost dominates.
- **Imbalance/stragglers:** rank skew from load imbalance or topology asymmetry.
- **Memory spikes:** poor prefetch/reshard boundaries or oversized microbatching.
- **Numerical drift:** precision policy mismatches across shards/collectives.

### Diagnostics and experiments
- Perform ablations: vary one axis degree at a time ($P_{tp}$, $P_{pp}$, etc.).
- Trace per-rank timelines to measure overlap ratio $ho=rac{T_{overlap}}{T_{comm}}$.
- Sweep microbatch count $m$, gradient bucket sizes, and activation checkpoint boundaries.
- Capture p50/p95/p99 iteration time and correlate with collective/kernel breakdowns.

### Study checklist
- Can you state the exact input/output layout for this subsection?
- Can you predict which collective dominates at larger scale?
- Can you explain one condition where this method should *not* be used?

### 8.5 Primitive Technique: Prefill/Decode Disaggregation
### Learning goals
- Understand the exact role of **8.5 Primitive Technique: Prefill/Decode Disaggregation** in the full training/serving stack.
- Identify required tensor layouts before and after this step.
- Map algorithmic choices to communication, memory, and stability consequences.

### Mechanism
- **Inputs/state.** Define local shards, global semantics, and invariants (shape/dtype/order).
- **Transformation.** Execute compute kernels plus communication primitives to move from input layout to output layout.
- **Correctness.** Preserve mathematical equivalence (modulo floating-point ordering) and optimizer semantics.
- **Output contract.** Ensure downstream subsection receives expected placement/state.

### Mathematical view
- Local shard sizing follows $n_{local}=n/p$ along sharded axis, with replicated axes unchanged.
- Runtime contribution is modeled as:
$$
T_{subsec}pprox T_{kernel}+T_{comm}-T_{overlap}+T_{sync}+T_{runtime}.
$$
- If this stage adds collective payload $q$ on group size $p_g$, then increment:
$$
\Delta T_{comm}pprox lpha f(p_g)+eta q g(p_g).
$$
- If this stage changes replication→sharding for tensor $x$, memory delta per rank is approximately:
$$
\Delta M(x)pprox |x|\left(1-rac{1}{p_g}ight).
$$

### Implementation details (practical)
1. Build process groups and bind them to physical topology intentionally; avoid accidental rank orders.
2. Choose bucket/chunk sizes so kernels stay large enough for high occupancy while communication can stream.
3. Co-schedule streams: compute stream + comm stream + dependency events to increase overlap safely.
4. Encode layout metadata in checkpoints to permit reshard/restart on changed meshes.
5. Validate numerics under mixed precision with loss-scale or FP8 scale tracking as required.

### Failure modes
- **Bandwidth-bound plateau:** throughput stops scaling as payload term dominates.
- **Latency-bound micro-ops:** too many tiny collectives; startup cost dominates.
- **Imbalance/stragglers:** rank skew from load imbalance or topology asymmetry.
- **Memory spikes:** poor prefetch/reshard boundaries or oversized microbatching.
- **Numerical drift:** precision policy mismatches across shards/collectives.

### Diagnostics and experiments
- Perform ablations: vary one axis degree at a time ($P_{tp}$, $P_{pp}$, etc.).
- Trace per-rank timelines to measure overlap ratio $ho=rac{T_{overlap}}{T_{comm}}$.
- Sweep microbatch count $m$, gradient bucket sizes, and activation checkpoint boundaries.
- Capture p50/p95/p99 iteration time and correlate with collective/kernel breakdowns.

### Study checklist
- Can you state the exact input/output layout for this subsection?
- Can you predict which collective dominates at larger scale?
- Can you explain one condition where this method should *not* be used?

### 8.6 Primitive Technique: Chunked Prefill and SplitFuse
### Learning goals
- Understand the exact role of **8.6 Primitive Technique: Chunked Prefill and SplitFuse** in the full training/serving stack.
- Identify required tensor layouts before and after this step.
- Map algorithmic choices to communication, memory, and stability consequences.

### Mechanism
- **Inputs/state.** Define local shards, global semantics, and invariants (shape/dtype/order).
- **Transformation.** Execute compute kernels plus communication primitives to move from input layout to output layout.
- **Correctness.** Preserve mathematical equivalence (modulo floating-point ordering) and optimizer semantics.
- **Output contract.** Ensure downstream subsection receives expected placement/state.

### Mathematical view
- Local shard sizing follows $n_{local}=n/p$ along sharded axis, with replicated axes unchanged.
- Runtime contribution is modeled as:
$$
T_{subsec}pprox T_{kernel}+T_{comm}-T_{overlap}+T_{sync}+T_{runtime}.
$$
- If this stage adds collective payload $q$ on group size $p_g$, then increment:
$$
\Delta T_{comm}pprox lpha f(p_g)+eta q g(p_g).
$$
- If this stage changes replication→sharding for tensor $x$, memory delta per rank is approximately:
$$
\Delta M(x)pprox |x|\left(1-rac{1}{p_g}ight).
$$

### Implementation details (practical)
1. Build process groups and bind them to physical topology intentionally; avoid accidental rank orders.
2. Choose bucket/chunk sizes so kernels stay large enough for high occupancy while communication can stream.
3. Co-schedule streams: compute stream + comm stream + dependency events to increase overlap safely.
4. Encode layout metadata in checkpoints to permit reshard/restart on changed meshes.
5. Validate numerics under mixed precision with loss-scale or FP8 scale tracking as required.

### Failure modes
- **Bandwidth-bound plateau:** throughput stops scaling as payload term dominates.
- **Latency-bound micro-ops:** too many tiny collectives; startup cost dominates.
- **Imbalance/stragglers:** rank skew from load imbalance or topology asymmetry.
- **Memory spikes:** poor prefetch/reshard boundaries or oversized microbatching.
- **Numerical drift:** precision policy mismatches across shards/collectives.

### Diagnostics and experiments
- Perform ablations: vary one axis degree at a time ($P_{tp}$, $P_{pp}$, etc.).
- Trace per-rank timelines to measure overlap ratio $ho=rac{T_{overlap}}{T_{comm}}$.
- Sweep microbatch count $m$, gradient bucket sizes, and activation checkpoint boundaries.
- Capture p50/p95/p99 iteration time and correlate with collective/kernel breakdowns.

### Study checklist
- Can you state the exact input/output layout for this subsection?
- Can you predict which collective dominates at larger scale?
- Can you explain one condition where this method should *not* be used?

### 8.7 Primitive Technique: Speculative and [MTP](https://arxiv.org/abs/2412.19437) (multi-token prediction) Decoding
### Learning goals
- Understand the exact role of **8.7 Primitive Technique: Speculative and [MTP](https://arxiv.org/abs/2412.19437) (multi-token prediction) Decoding** in the full training/serving stack.
- Identify required tensor layouts before and after this step.
- Map algorithmic choices to communication, memory, and stability consequences.

### Mechanism
- **Inputs/state.** Define local shards, global semantics, and invariants (shape/dtype/order).
- **Transformation.** Execute compute kernels plus communication primitives to move from input layout to output layout.
- **Correctness.** Preserve mathematical equivalence (modulo floating-point ordering) and optimizer semantics.
- **Output contract.** Ensure downstream subsection receives expected placement/state.

### Mathematical view
- Local shard sizing follows $n_{local}=n/p$ along sharded axis, with replicated axes unchanged.
- Runtime contribution is modeled as:
$$
T_{subsec}pprox T_{kernel}+T_{comm}-T_{overlap}+T_{sync}+T_{runtime}.
$$
- If this stage adds collective payload $q$ on group size $p_g$, then increment:
$$
\Delta T_{comm}pprox lpha f(p_g)+eta q g(p_g).
$$
- If this stage changes replication→sharding for tensor $x$, memory delta per rank is approximately:
$$
\Delta M(x)pprox |x|\left(1-rac{1}{p_g}ight).
$$

### Implementation details (practical)
1. Build process groups and bind them to physical topology intentionally; avoid accidental rank orders.
2. Choose bucket/chunk sizes so kernels stay large enough for high occupancy while communication can stream.
3. Co-schedule streams: compute stream + comm stream + dependency events to increase overlap safely.
4. Encode layout metadata in checkpoints to permit reshard/restart on changed meshes.
5. Validate numerics under mixed precision with loss-scale or FP8 scale tracking as required.

### Failure modes
- **Bandwidth-bound plateau:** throughput stops scaling as payload term dominates.
- **Latency-bound micro-ops:** too many tiny collectives; startup cost dominates.
- **Imbalance/stragglers:** rank skew from load imbalance or topology asymmetry.
- **Memory spikes:** poor prefetch/reshard boundaries or oversized microbatching.
- **Numerical drift:** precision policy mismatches across shards/collectives.

### Diagnostics and experiments
- Perform ablations: vary one axis degree at a time ($P_{tp}$, $P_{pp}$, etc.).
- Trace per-rank timelines to measure overlap ratio $ho=rac{T_{overlap}}{T_{comm}}$.
- Sweep microbatch count $m$, gradient bucket sizes, and activation checkpoint boundaries.
- Capture p50/p95/p99 iteration time and correlate with collective/kernel breakdowns.

### Study checklist
- Can you state the exact input/output layout for this subsection?
- Can you predict which collective dominates at larger scale?
- Can you explain one condition where this method should *not* be used?

### 8.8 Primitive Technique: TP, PP, EP, and CP for Serving
### Learning goals
- Understand the exact role of **8.8 Primitive Technique: TP, PP, EP, and CP for Serving** in the full training/serving stack.
- Identify required tensor layouts before and after this step.
- Map algorithmic choices to communication, memory, and stability consequences.

### Mechanism
- **Inputs/state.** Define local shards, global semantics, and invariants (shape/dtype/order).
- **Transformation.** Execute compute kernels plus communication primitives to move from input layout to output layout.
- **Correctness.** Preserve mathematical equivalence (modulo floating-point ordering) and optimizer semantics.
- **Output contract.** Ensure downstream subsection receives expected placement/state.

### Mathematical view
- Local shard sizing follows $n_{local}=n/p$ along sharded axis, with replicated axes unchanged.
- Runtime contribution is modeled as:
$$
T_{subsec}pprox T_{kernel}+T_{comm}-T_{overlap}+T_{sync}+T_{runtime}.
$$
- If this stage adds collective payload $q$ on group size $p_g$, then increment:
$$
\Delta T_{comm}pprox lpha f(p_g)+eta q g(p_g).
$$
- If this stage changes replication→sharding for tensor $x$, memory delta per rank is approximately:
$$
\Delta M(x)pprox |x|\left(1-rac{1}{p_g}ight).
$$

### Implementation details (practical)
1. Build process groups and bind them to physical topology intentionally; avoid accidental rank orders.
2. Choose bucket/chunk sizes so kernels stay large enough for high occupancy while communication can stream.
3. Co-schedule streams: compute stream + comm stream + dependency events to increase overlap safely.
4. Encode layout metadata in checkpoints to permit reshard/restart on changed meshes.
5. Validate numerics under mixed precision with loss-scale or FP8 scale tracking as required.

### Failure modes
- **Bandwidth-bound plateau:** throughput stops scaling as payload term dominates.
- **Latency-bound micro-ops:** too many tiny collectives; startup cost dominates.
- **Imbalance/stragglers:** rank skew from load imbalance or topology asymmetry.
- **Memory spikes:** poor prefetch/reshard boundaries or oversized microbatching.
- **Numerical drift:** precision policy mismatches across shards/collectives.

### Diagnostics and experiments
- Perform ablations: vary one axis degree at a time ($P_{tp}$, $P_{pp}$, etc.).
- Trace per-rank timelines to measure overlap ratio $ho=rac{T_{overlap}}{T_{comm}}$.
- Sweep microbatch count $m$, gradient bucket sizes, and activation checkpoint boundaries.
- Capture p50/p95/p99 iteration time and correlate with collective/kernel breakdowns.

### Study checklist
- Can you state the exact input/output layout for this subsection?
- Can you predict which collective dominates at larger scale?
- Can you explain one condition where this method should *not* be used?

### 8.9 End-to-End Serving Map
### Learning goals
- Understand the exact role of **8.9 End-to-End Serving Map** in the full training/serving stack.
- Identify required tensor layouts before and after this step.
- Map algorithmic choices to communication, memory, and stability consequences.

### Mechanism
- **Inputs/state.** Define local shards, global semantics, and invariants (shape/dtype/order).
- **Transformation.** Execute compute kernels plus communication primitives to move from input layout to output layout.
- **Correctness.** Preserve mathematical equivalence (modulo floating-point ordering) and optimizer semantics.
- **Output contract.** Ensure downstream subsection receives expected placement/state.

### Mathematical view
- Local shard sizing follows $n_{local}=n/p$ along sharded axis, with replicated axes unchanged.
- Runtime contribution is modeled as:
$$
T_{subsec}pprox T_{kernel}+T_{comm}-T_{overlap}+T_{sync}+T_{runtime}.
$$
- If this stage adds collective payload $q$ on group size $p_g$, then increment:
$$
\Delta T_{comm}pprox lpha f(p_g)+eta q g(p_g).
$$
- If this stage changes replication→sharding for tensor $x$, memory delta per rank is approximately:
$$
\Delta M(x)pprox |x|\left(1-rac{1}{p_g}ight).
$$

### Implementation details (practical)
1. Build process groups and bind them to physical topology intentionally; avoid accidental rank orders.
2. Choose bucket/chunk sizes so kernels stay large enough for high occupancy while communication can stream.
3. Co-schedule streams: compute stream + comm stream + dependency events to increase overlap safely.
4. Encode layout metadata in checkpoints to permit reshard/restart on changed meshes.
5. Validate numerics under mixed precision with loss-scale or FP8 scale tracking as required.

### Failure modes
- **Bandwidth-bound plateau:** throughput stops scaling as payload term dominates.
- **Latency-bound micro-ops:** too many tiny collectives; startup cost dominates.
- **Imbalance/stragglers:** rank skew from load imbalance or topology asymmetry.
- **Memory spikes:** poor prefetch/reshard boundaries or oversized microbatching.
- **Numerical drift:** precision policy mismatches across shards/collectives.

### Diagnostics and experiments
- Perform ablations: vary one axis degree at a time ($P_{tp}$, $P_{pp}$, etc.).
- Trace per-rank timelines to measure overlap ratio $ho=rac{T_{overlap}}{T_{comm}}$.
- Sweep microbatch count $m$, gradient bucket sizes, and activation checkpoint boundaries.
- Capture p50/p95/p99 iteration time and correlate with collective/kernel breakdowns.

### Study checklist
- Can you state the exact input/output layout for this subsection?
- Can you predict which collective dominates at larger scale?
- Can you explain one condition where this method should *not* be used?

### 8.10 Sources and Caveats
### Learning goals
- Understand the exact role of **8.10 Sources and Caveats** in the full training/serving stack.
- Identify required tensor layouts before and after this step.
- Map algorithmic choices to communication, memory, and stability consequences.

### Mechanism
- **Inputs/state.** Define local shards, global semantics, and invariants (shape/dtype/order).
- **Transformation.** Execute compute kernels plus communication primitives to move from input layout to output layout.
- **Correctness.** Preserve mathematical equivalence (modulo floating-point ordering) and optimizer semantics.
- **Output contract.** Ensure downstream subsection receives expected placement/state.

### Mathematical view
- Local shard sizing follows $n_{local}=n/p$ along sharded axis, with replicated axes unchanged.
- Runtime contribution is modeled as:
$$
T_{subsec}pprox T_{kernel}+T_{comm}-T_{overlap}+T_{sync}+T_{runtime}.
$$
- If this stage adds collective payload $q$ on group size $p_g$, then increment:
$$
\Delta T_{comm}pprox lpha f(p_g)+eta q g(p_g).
$$
- If this stage changes replication→sharding for tensor $x$, memory delta per rank is approximately:
$$
\Delta M(x)pprox |x|\left(1-rac{1}{p_g}ight).
$$

### Implementation details (practical)
1. Build process groups and bind them to physical topology intentionally; avoid accidental rank orders.
2. Choose bucket/chunk sizes so kernels stay large enough for high occupancy while communication can stream.
3. Co-schedule streams: compute stream + comm stream + dependency events to increase overlap safely.
4. Encode layout metadata in checkpoints to permit reshard/restart on changed meshes.
5. Validate numerics under mixed precision with loss-scale or FP8 scale tracking as required.

### Failure modes
- **Bandwidth-bound plateau:** throughput stops scaling as payload term dominates.
- **Latency-bound micro-ops:** too many tiny collectives; startup cost dominates.
- **Imbalance/stragglers:** rank skew from load imbalance or topology asymmetry.
- **Memory spikes:** poor prefetch/reshard boundaries or oversized microbatching.
- **Numerical drift:** precision policy mismatches across shards/collectives.

### Diagnostics and experiments
- Perform ablations: vary one axis degree at a time ($P_{tp}$, $P_{pp}$, etc.).
- Trace per-rank timelines to measure overlap ratio $ho=rac{T_{overlap}}{T_{comm}}$.
- Sweep microbatch count $m$, gradient bucket sizes, and activation checkpoint boundaries.
- Capture p50/p95/p99 iteration time and correlate with collective/kernel breakdowns.

### Study checklist
- Can you state the exact input/output layout for this subsection?
- Can you predict which collective dominates at larger scale?
- Can you explain one condition where this method should *not* be used?

## 9. Cross-Cutting CUDA and Platform Techniques
This section explains **9. Cross-Cutting CUDA and Platform Techniques** from first principles, implementation mechanics, cost/memory implications, topology interactions, and operational diagnostics.

### 9.1 Overlap: Streams, Events, Buckets, and Progress
### Learning goals
- Understand the exact role of **9.1 Overlap: Streams, Events, Buckets, and Progress** in the full training/serving stack.
- Identify required tensor layouts before and after this step.
- Map algorithmic choices to communication, memory, and stability consequences.

### Mechanism
- **Inputs/state.** Define local shards, global semantics, and invariants (shape/dtype/order).
- **Transformation.** Execute compute kernels plus communication primitives to move from input layout to output layout.
- **Correctness.** Preserve mathematical equivalence (modulo floating-point ordering) and optimizer semantics.
- **Output contract.** Ensure downstream subsection receives expected placement/state.

### Mathematical view
- Local shard sizing follows $n_{local}=n/p$ along sharded axis, with replicated axes unchanged.
- Runtime contribution is modeled as:
$$
T_{subsec}pprox T_{kernel}+T_{comm}-T_{overlap}+T_{sync}+T_{runtime}.
$$
- If this stage adds collective payload $q$ on group size $p_g$, then increment:
$$
\Delta T_{comm}pprox lpha f(p_g)+eta q g(p_g).
$$
- If this stage changes replication→sharding for tensor $x$, memory delta per rank is approximately:
$$
\Delta M(x)pprox |x|\left(1-rac{1}{p_g}ight).
$$

### Implementation details (practical)
1. Build process groups and bind them to physical topology intentionally; avoid accidental rank orders.
2. Choose bucket/chunk sizes so kernels stay large enough for high occupancy while communication can stream.
3. Co-schedule streams: compute stream + comm stream + dependency events to increase overlap safely.
4. Encode layout metadata in checkpoints to permit reshard/restart on changed meshes.
5. Validate numerics under mixed precision with loss-scale or FP8 scale tracking as required.

### Failure modes
- **Bandwidth-bound plateau:** throughput stops scaling as payload term dominates.
- **Latency-bound micro-ops:** too many tiny collectives; startup cost dominates.
- **Imbalance/stragglers:** rank skew from load imbalance or topology asymmetry.
- **Memory spikes:** poor prefetch/reshard boundaries or oversized microbatching.
- **Numerical drift:** precision policy mismatches across shards/collectives.

### Diagnostics and experiments
- Perform ablations: vary one axis degree at a time ($P_{tp}$, $P_{pp}$, etc.).
- Trace per-rank timelines to measure overlap ratio $ho=rac{T_{overlap}}{T_{comm}}$.
- Sweep microbatch count $m$, gradient bucket sizes, and activation checkpoint boundaries.
- Capture p50/p95/p99 iteration time and correlate with collective/kernel breakdowns.

### Study checklist
- Can you state the exact input/output layout for this subsection?
- Can you predict which collective dominates at larger scale?
- Can you explain one condition where this method should *not* be used?

### 9.2 CUDA Graphs and Static Metadata
### Learning goals
- Understand the exact role of **9.2 CUDA Graphs and Static Metadata** in the full training/serving stack.
- Identify required tensor layouts before and after this step.
- Map algorithmic choices to communication, memory, and stability consequences.

### Mechanism
- **Inputs/state.** Define local shards, global semantics, and invariants (shape/dtype/order).
- **Transformation.** Execute compute kernels plus communication primitives to move from input layout to output layout.
- **Correctness.** Preserve mathematical equivalence (modulo floating-point ordering) and optimizer semantics.
- **Output contract.** Ensure downstream subsection receives expected placement/state.

### Mathematical view
- Local shard sizing follows $n_{local}=n/p$ along sharded axis, with replicated axes unchanged.
- Runtime contribution is modeled as:
$$
T_{subsec}pprox T_{kernel}+T_{comm}-T_{overlap}+T_{sync}+T_{runtime}.
$$
- If this stage adds collective payload $q$ on group size $p_g$, then increment:
$$
\Delta T_{comm}pprox lpha f(p_g)+eta q g(p_g).
$$
- If this stage changes replication→sharding for tensor $x$, memory delta per rank is approximately:
$$
\Delta M(x)pprox |x|\left(1-rac{1}{p_g}ight).
$$

### Implementation details (practical)
1. Build process groups and bind them to physical topology intentionally; avoid accidental rank orders.
2. Choose bucket/chunk sizes so kernels stay large enough for high occupancy while communication can stream.
3. Co-schedule streams: compute stream + comm stream + dependency events to increase overlap safely.
4. Encode layout metadata in checkpoints to permit reshard/restart on changed meshes.
5. Validate numerics under mixed precision with loss-scale or FP8 scale tracking as required.

### Failure modes
- **Bandwidth-bound plateau:** throughput stops scaling as payload term dominates.
- **Latency-bound micro-ops:** too many tiny collectives; startup cost dominates.
- **Imbalance/stragglers:** rank skew from load imbalance or topology asymmetry.
- **Memory spikes:** poor prefetch/reshard boundaries or oversized microbatching.
- **Numerical drift:** precision policy mismatches across shards/collectives.

### Diagnostics and experiments
- Perform ablations: vary one axis degree at a time ($P_{tp}$, $P_{pp}$, etc.).
- Trace per-rank timelines to measure overlap ratio $ho=rac{T_{overlap}}{T_{comm}}$.
- Sweep microbatch count $m$, gradient bucket sizes, and activation checkpoint boundaries.
- Capture p50/p95/p99 iteration time and correlate with collective/kernel breakdowns.

### Study checklist
- Can you state the exact input/output layout for this subsection?
- Can you predict which collective dominates at larger scale?
- Can you explain one condition where this method should *not* be used?

### 9.3 Low Precision as Payload Plus Scale Layout
### Learning goals
- Understand the exact role of **9.3 Low Precision as Payload Plus Scale Layout** in the full training/serving stack.
- Identify required tensor layouts before and after this step.
- Map algorithmic choices to communication, memory, and stability consequences.

### Mechanism
- **Inputs/state.** Define local shards, global semantics, and invariants (shape/dtype/order).
- **Transformation.** Execute compute kernels plus communication primitives to move from input layout to output layout.
- **Correctness.** Preserve mathematical equivalence (modulo floating-point ordering) and optimizer semantics.
- **Output contract.** Ensure downstream subsection receives expected placement/state.

### Mathematical view
- Local shard sizing follows $n_{local}=n/p$ along sharded axis, with replicated axes unchanged.
- Runtime contribution is modeled as:
$$
T_{subsec}pprox T_{kernel}+T_{comm}-T_{overlap}+T_{sync}+T_{runtime}.
$$
- If this stage adds collective payload $q$ on group size $p_g$, then increment:
$$
\Delta T_{comm}pprox lpha f(p_g)+eta q g(p_g).
$$
- If this stage changes replication→sharding for tensor $x$, memory delta per rank is approximately:
$$
\Delta M(x)pprox |x|\left(1-rac{1}{p_g}ight).
$$

### Implementation details (practical)
1. Build process groups and bind them to physical topology intentionally; avoid accidental rank orders.
2. Choose bucket/chunk sizes so kernels stay large enough for high occupancy while communication can stream.
3. Co-schedule streams: compute stream + comm stream + dependency events to increase overlap safely.
4. Encode layout metadata in checkpoints to permit reshard/restart on changed meshes.
5. Validate numerics under mixed precision with loss-scale or FP8 scale tracking as required.

### Failure modes
- **Bandwidth-bound plateau:** throughput stops scaling as payload term dominates.
- **Latency-bound micro-ops:** too many tiny collectives; startup cost dominates.
- **Imbalance/stragglers:** rank skew from load imbalance or topology asymmetry.
- **Memory spikes:** poor prefetch/reshard boundaries or oversized microbatching.
- **Numerical drift:** precision policy mismatches across shards/collectives.

### Diagnostics and experiments
- Perform ablations: vary one axis degree at a time ($P_{tp}$, $P_{pp}$, etc.).
- Trace per-rank timelines to measure overlap ratio $ho=rac{T_{overlap}}{T_{comm}}$.
- Sweep microbatch count $m$, gradient bucket sizes, and activation checkpoint boundaries.
- Capture p50/p95/p99 iteration time and correlate with collective/kernel breakdowns.

### Study checklist
- Can you state the exact input/output layout for this subsection?
- Can you predict which collective dominates at larger scale?
- Can you explain one condition where this method should *not* be used?

### 9.4 Attention Kernels: FlashAttention, cuDNN SDPA, NSA, and FlashMLA
### Learning goals
- Understand the exact role of **9.4 Attention Kernels: FlashAttention, cuDNN SDPA, NSA, and FlashMLA** in the full training/serving stack.
- Identify required tensor layouts before and after this step.
- Map algorithmic choices to communication, memory, and stability consequences.

### Mechanism
- **Inputs/state.** Define local shards, global semantics, and invariants (shape/dtype/order).
- **Transformation.** Execute compute kernels plus communication primitives to move from input layout to output layout.
- **Correctness.** Preserve mathematical equivalence (modulo floating-point ordering) and optimizer semantics.
- **Output contract.** Ensure downstream subsection receives expected placement/state.

### Mathematical view
- Local shard sizing follows $n_{local}=n/p$ along sharded axis, with replicated axes unchanged.
- Runtime contribution is modeled as:
$$
T_{subsec}pprox T_{kernel}+T_{comm}-T_{overlap}+T_{sync}+T_{runtime}.
$$
- If this stage adds collective payload $q$ on group size $p_g$, then increment:
$$
\Delta T_{comm}pprox lpha f(p_g)+eta q g(p_g).
$$
- If this stage changes replication→sharding for tensor $x$, memory delta per rank is approximately:
$$
\Delta M(x)pprox |x|\left(1-rac{1}{p_g}ight).
$$

### Implementation details (practical)
1. Build process groups and bind them to physical topology intentionally; avoid accidental rank orders.
2. Choose bucket/chunk sizes so kernels stay large enough for high occupancy while communication can stream.
3. Co-schedule streams: compute stream + comm stream + dependency events to increase overlap safely.
4. Encode layout metadata in checkpoints to permit reshard/restart on changed meshes.
5. Validate numerics under mixed precision with loss-scale or FP8 scale tracking as required.

### Failure modes
- **Bandwidth-bound plateau:** throughput stops scaling as payload term dominates.
- **Latency-bound micro-ops:** too many tiny collectives; startup cost dominates.
- **Imbalance/stragglers:** rank skew from load imbalance or topology asymmetry.
- **Memory spikes:** poor prefetch/reshard boundaries or oversized microbatching.
- **Numerical drift:** precision policy mismatches across shards/collectives.

### Diagnostics and experiments
- Perform ablations: vary one axis degree at a time ($P_{tp}$, $P_{pp}$, etc.).
- Trace per-rank timelines to measure overlap ratio $ho=rac{T_{overlap}}{T_{comm}}$.
- Sweep microbatch count $m$, gradient bucket sizes, and activation checkpoint boundaries.
- Capture p50/p95/p99 iteration time and correlate with collective/kernel breakdowns.

### Study checklist
- Can you state the exact input/output layout for this subsection?
- Can you predict which collective dominates at larger scale?
- Can you explain one condition where this method should *not* be used?

### 9.5 CUTLASS, CuTe, GEMM, and Grouped Expert Compute
### Learning goals
- Understand the exact role of **9.5 CUTLASS, CuTe, GEMM, and Grouped Expert Compute** in the full training/serving stack.
- Identify required tensor layouts before and after this step.
- Map algorithmic choices to communication, memory, and stability consequences.

### Mechanism
- **Inputs/state.** Define local shards, global semantics, and invariants (shape/dtype/order).
- **Transformation.** Execute compute kernels plus communication primitives to move from input layout to output layout.
- **Correctness.** Preserve mathematical equivalence (modulo floating-point ordering) and optimizer semantics.
- **Output contract.** Ensure downstream subsection receives expected placement/state.

### Mathematical view
- Local shard sizing follows $n_{local}=n/p$ along sharded axis, with replicated axes unchanged.
- Runtime contribution is modeled as:
$$
T_{subsec}pprox T_{kernel}+T_{comm}-T_{overlap}+T_{sync}+T_{runtime}.
$$
- If this stage adds collective payload $q$ on group size $p_g$, then increment:
$$
\Delta T_{comm}pprox lpha f(p_g)+eta q g(p_g).
$$
- If this stage changes replication→sharding for tensor $x$, memory delta per rank is approximately:
$$
\Delta M(x)pprox |x|\left(1-rac{1}{p_g}ight).
$$

### Implementation details (practical)
1. Build process groups and bind them to physical topology intentionally; avoid accidental rank orders.
2. Choose bucket/chunk sizes so kernels stay large enough for high occupancy while communication can stream.
3. Co-schedule streams: compute stream + comm stream + dependency events to increase overlap safely.
4. Encode layout metadata in checkpoints to permit reshard/restart on changed meshes.
5. Validate numerics under mixed precision with loss-scale or FP8 scale tracking as required.

### Failure modes
- **Bandwidth-bound plateau:** throughput stops scaling as payload term dominates.
- **Latency-bound micro-ops:** too many tiny collectives; startup cost dominates.
- **Imbalance/stragglers:** rank skew from load imbalance or topology asymmetry.
- **Memory spikes:** poor prefetch/reshard boundaries or oversized microbatching.
- **Numerical drift:** precision policy mismatches across shards/collectives.

### Diagnostics and experiments
- Perform ablations: vary one axis degree at a time ($P_{tp}$, $P_{pp}$, etc.).
- Trace per-rank timelines to measure overlap ratio $ho=rac{T_{overlap}}{T_{comm}}$.
- Sweep microbatch count $m$, gradient bucket sizes, and activation checkpoint boundaries.
- Capture p50/p95/p99 iteration time and correlate with collective/kernel breakdowns.

### Study checklist
- Can you state the exact input/output layout for this subsection?
- Can you predict which collective dominates at larger scale?
- Can you explain one condition where this method should *not* be used?

### 9.6 NVSHMEM, GPU-Initiated Communication, and Persistent Kernels
### Learning goals
- Understand the exact role of **9.6 NVSHMEM, GPU-Initiated Communication, and Persistent Kernels** in the full training/serving stack.
- Identify required tensor layouts before and after this step.
- Map algorithmic choices to communication, memory, and stability consequences.

### Mechanism
- **Inputs/state.** Define local shards, global semantics, and invariants (shape/dtype/order).
- **Transformation.** Execute compute kernels plus communication primitives to move from input layout to output layout.
- **Correctness.** Preserve mathematical equivalence (modulo floating-point ordering) and optimizer semantics.
- **Output contract.** Ensure downstream subsection receives expected placement/state.

### Mathematical view
- Local shard sizing follows $n_{local}=n/p$ along sharded axis, with replicated axes unchanged.
- Runtime contribution is modeled as:
$$
T_{subsec}pprox T_{kernel}+T_{comm}-T_{overlap}+T_{sync}+T_{runtime}.
$$
- If this stage adds collective payload $q$ on group size $p_g$, then increment:
$$
\Delta T_{comm}pprox lpha f(p_g)+eta q g(p_g).
$$
- If this stage changes replication→sharding for tensor $x$, memory delta per rank is approximately:
$$
\Delta M(x)pprox |x|\left(1-rac{1}{p_g}ight).
$$

### Implementation details (practical)
1. Build process groups and bind them to physical topology intentionally; avoid accidental rank orders.
2. Choose bucket/chunk sizes so kernels stay large enough for high occupancy while communication can stream.
3. Co-schedule streams: compute stream + comm stream + dependency events to increase overlap safely.
4. Encode layout metadata in checkpoints to permit reshard/restart on changed meshes.
5. Validate numerics under mixed precision with loss-scale or FP8 scale tracking as required.

### Failure modes
- **Bandwidth-bound plateau:** throughput stops scaling as payload term dominates.
- **Latency-bound micro-ops:** too many tiny collectives; startup cost dominates.
- **Imbalance/stragglers:** rank skew from load imbalance or topology asymmetry.
- **Memory spikes:** poor prefetch/reshard boundaries or oversized microbatching.
- **Numerical drift:** precision policy mismatches across shards/collectives.

### Diagnostics and experiments
- Perform ablations: vary one axis degree at a time ($P_{tp}$, $P_{pp}$, etc.).
- Trace per-rank timelines to measure overlap ratio $ho=rac{T_{overlap}}{T_{comm}}$.
- Sweep microbatch count $m$, gradient bucket sizes, and activation checkpoint boundaries.
- Capture p50/p95/p99 iteration time and correlate with collective/kernel breakdowns.

### Study checklist
- Can you state the exact input/output layout for this subsection?
- Can you predict which collective dominates at larger scale?
- Can you explain one condition where this method should *not* be used?

### 9.7 Kernel Authoring DSLs: Triton, Pallas, TileLang, and ThunderKittens
### Learning goals
- Understand the exact role of **9.7 Kernel Authoring DSLs: Triton, Pallas, TileLang, and ThunderKittens** in the full training/serving stack.
- Identify required tensor layouts before and after this step.
- Map algorithmic choices to communication, memory, and stability consequences.

### Mechanism
- **Inputs/state.** Define local shards, global semantics, and invariants (shape/dtype/order).
- **Transformation.** Execute compute kernels plus communication primitives to move from input layout to output layout.
- **Correctness.** Preserve mathematical equivalence (modulo floating-point ordering) and optimizer semantics.
- **Output contract.** Ensure downstream subsection receives expected placement/state.

### Mathematical view
- Local shard sizing follows $n_{local}=n/p$ along sharded axis, with replicated axes unchanged.
- Runtime contribution is modeled as:
$$
T_{subsec}pprox T_{kernel}+T_{comm}-T_{overlap}+T_{sync}+T_{runtime}.
$$
- If this stage adds collective payload $q$ on group size $p_g$, then increment:
$$
\Delta T_{comm}pprox lpha f(p_g)+eta q g(p_g).
$$
- If this stage changes replication→sharding for tensor $x$, memory delta per rank is approximately:
$$
\Delta M(x)pprox |x|\left(1-rac{1}{p_g}ight).
$$

### Implementation details (practical)
1. Build process groups and bind them to physical topology intentionally; avoid accidental rank orders.
2. Choose bucket/chunk sizes so kernels stay large enough for high occupancy while communication can stream.
3. Co-schedule streams: compute stream + comm stream + dependency events to increase overlap safely.
4. Encode layout metadata in checkpoints to permit reshard/restart on changed meshes.
5. Validate numerics under mixed precision with loss-scale or FP8 scale tracking as required.

### Failure modes
- **Bandwidth-bound plateau:** throughput stops scaling as payload term dominates.
- **Latency-bound micro-ops:** too many tiny collectives; startup cost dominates.
- **Imbalance/stragglers:** rank skew from load imbalance or topology asymmetry.
- **Memory spikes:** poor prefetch/reshard boundaries or oversized microbatching.
- **Numerical drift:** precision policy mismatches across shards/collectives.

### Diagnostics and experiments
- Perform ablations: vary one axis degree at a time ($P_{tp}$, $P_{pp}$, etc.).
- Trace per-rank timelines to measure overlap ratio $ho=rac{T_{overlap}}{T_{comm}}$.
- Sweep microbatch count $m$, gradient bucket sizes, and activation checkpoint boundaries.
- Capture p50/p95/p99 iteration time and correlate with collective/kernel breakdowns.

### Study checklist
- Can you state the exact input/output layout for this subsection?
- Can you predict which collective dominates at larger scale?
- Can you explain one condition where this method should *not* be used?

## 10. Frontier Model and Runtime Case Studies
This section explains **10. Frontier Model and Runtime Case Studies** from first principles, implementation mechanics, cost/memory implications, topology interactions, and operational diagnostics.

### 10.1 DeepSeek-V3: MoE, FP8, DualPipe, and Expert Overlap
### Learning goals
- Understand the exact role of **10.1 DeepSeek-V3: MoE, FP8, DualPipe, and Expert Overlap** in the full training/serving stack.
- Identify required tensor layouts before and after this step.
- Map algorithmic choices to communication, memory, and stability consequences.

### Mechanism
- **Inputs/state.** Define local shards, global semantics, and invariants (shape/dtype/order).
- **Transformation.** Execute compute kernels plus communication primitives to move from input layout to output layout.
- **Correctness.** Preserve mathematical equivalence (modulo floating-point ordering) and optimizer semantics.
- **Output contract.** Ensure downstream subsection receives expected placement/state.

### Mathematical view
- Local shard sizing follows $n_{local}=n/p$ along sharded axis, with replicated axes unchanged.
- Runtime contribution is modeled as:
$$
T_{subsec}pprox T_{kernel}+T_{comm}-T_{overlap}+T_{sync}+T_{runtime}.
$$
- If this stage adds collective payload $q$ on group size $p_g$, then increment:
$$
\Delta T_{comm}pprox lpha f(p_g)+eta q g(p_g).
$$
- If this stage changes replication→sharding for tensor $x$, memory delta per rank is approximately:
$$
\Delta M(x)pprox |x|\left(1-rac{1}{p_g}ight).
$$

### Implementation details (practical)
1. Build process groups and bind them to physical topology intentionally; avoid accidental rank orders.
2. Choose bucket/chunk sizes so kernels stay large enough for high occupancy while communication can stream.
3. Co-schedule streams: compute stream + comm stream + dependency events to increase overlap safely.
4. Encode layout metadata in checkpoints to permit reshard/restart on changed meshes.
5. Validate numerics under mixed precision with loss-scale or FP8 scale tracking as required.

### Failure modes
- **Bandwidth-bound plateau:** throughput stops scaling as payload term dominates.
- **Latency-bound micro-ops:** too many tiny collectives; startup cost dominates.
- **Imbalance/stragglers:** rank skew from load imbalance or topology asymmetry.
- **Memory spikes:** poor prefetch/reshard boundaries or oversized microbatching.
- **Numerical drift:** precision policy mismatches across shards/collectives.

### Diagnostics and experiments
- Perform ablations: vary one axis degree at a time ($P_{tp}$, $P_{pp}$, etc.).
- Trace per-rank timelines to measure overlap ratio $ho=rac{T_{overlap}}{T_{comm}}$.
- Sweep microbatch count $m$, gradient bucket sizes, and activation checkpoint boundaries.
- Capture p50/p95/p99 iteration time and correlate with collective/kernel breakdowns.

### Study checklist
- Can you state the exact input/output layout for this subsection?
- Can you predict which collective dominates at larger scale?
- Can you explain one condition where this method should *not* be used?

### 10.2 DeepSeek-V4: Hybrid Attention, mHC, FP4 Experts, and Serving Layout
### Learning goals
- Understand the exact role of **10.2 DeepSeek-V4: Hybrid Attention, mHC, FP4 Experts, and Serving Layout** in the full training/serving stack.
- Identify required tensor layouts before and after this step.
- Map algorithmic choices to communication, memory, and stability consequences.

### Mechanism
- **Inputs/state.** Define local shards, global semantics, and invariants (shape/dtype/order).
- **Transformation.** Execute compute kernels plus communication primitives to move from input layout to output layout.
- **Correctness.** Preserve mathematical equivalence (modulo floating-point ordering) and optimizer semantics.
- **Output contract.** Ensure downstream subsection receives expected placement/state.

### Mathematical view
- Local shard sizing follows $n_{local}=n/p$ along sharded axis, with replicated axes unchanged.
- Runtime contribution is modeled as:
$$
T_{subsec}pprox T_{kernel}+T_{comm}-T_{overlap}+T_{sync}+T_{runtime}.
$$
- If this stage adds collective payload $q$ on group size $p_g$, then increment:
$$
\Delta T_{comm}pprox lpha f(p_g)+eta q g(p_g).
$$
- If this stage changes replication→sharding for tensor $x$, memory delta per rank is approximately:
$$
\Delta M(x)pprox |x|\left(1-rac{1}{p_g}ight).
$$

### Implementation details (practical)
1. Build process groups and bind them to physical topology intentionally; avoid accidental rank orders.
2. Choose bucket/chunk sizes so kernels stay large enough for high occupancy while communication can stream.
3. Co-schedule streams: compute stream + comm stream + dependency events to increase overlap safely.
4. Encode layout metadata in checkpoints to permit reshard/restart on changed meshes.
5. Validate numerics under mixed precision with loss-scale or FP8 scale tracking as required.

### Failure modes
- **Bandwidth-bound plateau:** throughput stops scaling as payload term dominates.
- **Latency-bound micro-ops:** too many tiny collectives; startup cost dominates.
- **Imbalance/stragglers:** rank skew from load imbalance or topology asymmetry.
- **Memory spikes:** poor prefetch/reshard boundaries or oversized microbatching.
- **Numerical drift:** precision policy mismatches across shards/collectives.

### Diagnostics and experiments
- Perform ablations: vary one axis degree at a time ($P_{tp}$, $P_{pp}$, etc.).
- Trace per-rank timelines to measure overlap ratio $ho=rac{T_{overlap}}{T_{comm}}$.
- Sweep microbatch count $m$, gradient bucket sizes, and activation checkpoint boundaries.
- Capture p50/p95/p99 iteration time and correlate with collective/kernel breakdowns.

### Study checklist
- Can you state the exact input/output layout for this subsection?
- Can you predict which collective dominates at larger scale?
- Can you explain one condition where this method should *not* be used?

### 10.3 Meta Llama 3 and TorchTitan: Dense Hybrid Parallelism
### Learning goals
- Understand the exact role of **10.3 Meta Llama 3 and TorchTitan: Dense Hybrid Parallelism** in the full training/serving stack.
- Identify required tensor layouts before and after this step.
- Map algorithmic choices to communication, memory, and stability consequences.

### Mechanism
- **Inputs/state.** Define local shards, global semantics, and invariants (shape/dtype/order).
- **Transformation.** Execute compute kernels plus communication primitives to move from input layout to output layout.
- **Correctness.** Preserve mathematical equivalence (modulo floating-point ordering) and optimizer semantics.
- **Output contract.** Ensure downstream subsection receives expected placement/state.

### Mathematical view
- Local shard sizing follows $n_{local}=n/p$ along sharded axis, with replicated axes unchanged.
- Runtime contribution is modeled as:
$$
T_{subsec}pprox T_{kernel}+T_{comm}-T_{overlap}+T_{sync}+T_{runtime}.
$$
- If this stage adds collective payload $q$ on group size $p_g$, then increment:
$$
\Delta T_{comm}pprox lpha f(p_g)+eta q g(p_g).
$$
- If this stage changes replication→sharding for tensor $x$, memory delta per rank is approximately:
$$
\Delta M(x)pprox |x|\left(1-rac{1}{p_g}ight).
$$

### Implementation details (practical)
1. Build process groups and bind them to physical topology intentionally; avoid accidental rank orders.
2. Choose bucket/chunk sizes so kernels stay large enough for high occupancy while communication can stream.
3. Co-schedule streams: compute stream + comm stream + dependency events to increase overlap safely.
4. Encode layout metadata in checkpoints to permit reshard/restart on changed meshes.
5. Validate numerics under mixed precision with loss-scale or FP8 scale tracking as required.

### Failure modes
- **Bandwidth-bound plateau:** throughput stops scaling as payload term dominates.
- **Latency-bound micro-ops:** too many tiny collectives; startup cost dominates.
- **Imbalance/stragglers:** rank skew from load imbalance or topology asymmetry.
- **Memory spikes:** poor prefetch/reshard boundaries or oversized microbatching.
- **Numerical drift:** precision policy mismatches across shards/collectives.

### Diagnostics and experiments
- Perform ablations: vary one axis degree at a time ($P_{tp}$, $P_{pp}$, etc.).
- Trace per-rank timelines to measure overlap ratio $ho=rac{T_{overlap}}{T_{comm}}$.
- Sweep microbatch count $m$, gradient bucket sizes, and activation checkpoint boundaries.
- Capture p50/p95/p99 iteration time and correlate with collective/kernel breakdowns.

### Study checklist
- Can you state the exact input/output layout for this subsection?
- Can you predict which collective dominates at larger scale?
- Can you explain one condition where this method should *not* be used?

### 10.4 Qwen, Kimi, and Open MoE Directions
### Learning goals
- Understand the exact role of **10.4 Qwen, Kimi, and Open MoE Directions** in the full training/serving stack.
- Identify required tensor layouts before and after this step.
- Map algorithmic choices to communication, memory, and stability consequences.

### Mechanism
- **Inputs/state.** Define local shards, global semantics, and invariants (shape/dtype/order).
- **Transformation.** Execute compute kernels plus communication primitives to move from input layout to output layout.
- **Correctness.** Preserve mathematical equivalence (modulo floating-point ordering) and optimizer semantics.
- **Output contract.** Ensure downstream subsection receives expected placement/state.

### Mathematical view
- Local shard sizing follows $n_{local}=n/p$ along sharded axis, with replicated axes unchanged.
- Runtime contribution is modeled as:
$$
T_{subsec}pprox T_{kernel}+T_{comm}-T_{overlap}+T_{sync}+T_{runtime}.
$$
- If this stage adds collective payload $q$ on group size $p_g$, then increment:
$$
\Delta T_{comm}pprox lpha f(p_g)+eta q g(p_g).
$$
- If this stage changes replication→sharding for tensor $x$, memory delta per rank is approximately:
$$
\Delta M(x)pprox |x|\left(1-rac{1}{p_g}ight).
$$

### Implementation details (practical)
1. Build process groups and bind them to physical topology intentionally; avoid accidental rank orders.
2. Choose bucket/chunk sizes so kernels stay large enough for high occupancy while communication can stream.
3. Co-schedule streams: compute stream + comm stream + dependency events to increase overlap safely.
4. Encode layout metadata in checkpoints to permit reshard/restart on changed meshes.
5. Validate numerics under mixed precision with loss-scale or FP8 scale tracking as required.

### Failure modes
- **Bandwidth-bound plateau:** throughput stops scaling as payload term dominates.
- **Latency-bound micro-ops:** too many tiny collectives; startup cost dominates.
- **Imbalance/stragglers:** rank skew from load imbalance or topology asymmetry.
- **Memory spikes:** poor prefetch/reshard boundaries or oversized microbatching.
- **Numerical drift:** precision policy mismatches across shards/collectives.

### Diagnostics and experiments
- Perform ablations: vary one axis degree at a time ($P_{tp}$, $P_{pp}$, etc.).
- Trace per-rank timelines to measure overlap ratio $ho=rac{T_{overlap}}{T_{comm}}$.
- Sweep microbatch count $m$, gradient bucket sizes, and activation checkpoint boundaries.
- Capture p50/p95/p99 iteration time and correlate with collective/kernel breakdowns.

### Study checklist
- Can you state the exact input/output layout for this subsection?
- Can you predict which collective dominates at larger scale?
- Can you explain one condition where this method should *not* be used?

### 10.5 Google Pathways, GSPMD, and MaxText: Compiler-Mesh Parallelism
### Learning goals
- Understand the exact role of **10.5 Google Pathways, GSPMD, and MaxText: Compiler-Mesh Parallelism** in the full training/serving stack.
- Identify required tensor layouts before and after this step.
- Map algorithmic choices to communication, memory, and stability consequences.

### Mechanism
- **Inputs/state.** Define local shards, global semantics, and invariants (shape/dtype/order).
- **Transformation.** Execute compute kernels plus communication primitives to move from input layout to output layout.
- **Correctness.** Preserve mathematical equivalence (modulo floating-point ordering) and optimizer semantics.
- **Output contract.** Ensure downstream subsection receives expected placement/state.

### Mathematical view
- Local shard sizing follows $n_{local}=n/p$ along sharded axis, with replicated axes unchanged.
- Runtime contribution is modeled as:
$$
T_{subsec}pprox T_{kernel}+T_{comm}-T_{overlap}+T_{sync}+T_{runtime}.
$$
- If this stage adds collective payload $q$ on group size $p_g$, then increment:
$$
\Delta T_{comm}pprox lpha f(p_g)+eta q g(p_g).
$$
- If this stage changes replication→sharding for tensor $x$, memory delta per rank is approximately:
$$
\Delta M(x)pprox |x|\left(1-rac{1}{p_g}ight).
$$

### Implementation details (practical)
1. Build process groups and bind them to physical topology intentionally; avoid accidental rank orders.
2. Choose bucket/chunk sizes so kernels stay large enough for high occupancy while communication can stream.
3. Co-schedule streams: compute stream + comm stream + dependency events to increase overlap safely.
4. Encode layout metadata in checkpoints to permit reshard/restart on changed meshes.
5. Validate numerics under mixed precision with loss-scale or FP8 scale tracking as required.

### Failure modes
- **Bandwidth-bound plateau:** throughput stops scaling as payload term dominates.
- **Latency-bound micro-ops:** too many tiny collectives; startup cost dominates.
- **Imbalance/stragglers:** rank skew from load imbalance or topology asymmetry.
- **Memory spikes:** poor prefetch/reshard boundaries or oversized microbatching.
- **Numerical drift:** precision policy mismatches across shards/collectives.

### Diagnostics and experiments
- Perform ablations: vary one axis degree at a time ($P_{tp}$, $P_{pp}$, etc.).
- Trace per-rank timelines to measure overlap ratio $ho=rac{T_{overlap}}{T_{comm}}$.
- Sweep microbatch count $m$, gradient bucket sizes, and activation checkpoint boundaries.
- Capture p50/p95/p99 iteration time and correlate with collective/kernel breakdowns.

### Study checklist
- Can you state the exact input/output layout for this subsection?
- Can you predict which collective dominates at larger scale?
- Can you explain one condition where this method should *not* be used?

## 11. Source Carryover From The GPU-Kernel Paper
This section explains **11. Source Carryover From The GPU-Kernel Paper** from first principles, implementation mechanics, cost/memory implications, topology interactions, and operational diagnostics.

## 12. Open Problems and Evidence Boundaries
This section explains **12. Open Problems and Evidence Boundaries** from first principles, implementation mechanics, cost/memory implications, topology interactions, and operational diagnostics.

### 12.1 Unresolved Technical Questions
### Learning goals
- Understand the exact role of **12.1 Unresolved Technical Questions** in the full training/serving stack.
- Identify required tensor layouts before and after this step.
- Map algorithmic choices to communication, memory, and stability consequences.

### Mechanism
- **Inputs/state.** Define local shards, global semantics, and invariants (shape/dtype/order).
- **Transformation.** Execute compute kernels plus communication primitives to move from input layout to output layout.
- **Correctness.** Preserve mathematical equivalence (modulo floating-point ordering) and optimizer semantics.
- **Output contract.** Ensure downstream subsection receives expected placement/state.

### Mathematical view
- Local shard sizing follows $n_{local}=n/p$ along sharded axis, with replicated axes unchanged.
- Runtime contribution is modeled as:
$$
T_{subsec}pprox T_{kernel}+T_{comm}-T_{overlap}+T_{sync}+T_{runtime}.
$$
- If this stage adds collective payload $q$ on group size $p_g$, then increment:
$$
\Delta T_{comm}pprox lpha f(p_g)+eta q g(p_g).
$$
- If this stage changes replication→sharding for tensor $x$, memory delta per rank is approximately:
$$
\Delta M(x)pprox |x|\left(1-rac{1}{p_g}ight).
$$

### Implementation details (practical)
1. Build process groups and bind them to physical topology intentionally; avoid accidental rank orders.
2. Choose bucket/chunk sizes so kernels stay large enough for high occupancy while communication can stream.
3. Co-schedule streams: compute stream + comm stream + dependency events to increase overlap safely.
4. Encode layout metadata in checkpoints to permit reshard/restart on changed meshes.
5. Validate numerics under mixed precision with loss-scale or FP8 scale tracking as required.

### Failure modes
- **Bandwidth-bound plateau:** throughput stops scaling as payload term dominates.
- **Latency-bound micro-ops:** too many tiny collectives; startup cost dominates.
- **Imbalance/stragglers:** rank skew from load imbalance or topology asymmetry.
- **Memory spikes:** poor prefetch/reshard boundaries or oversized microbatching.
- **Numerical drift:** precision policy mismatches across shards/collectives.

### Diagnostics and experiments
- Perform ablations: vary one axis degree at a time ($P_{tp}$, $P_{pp}$, etc.).
- Trace per-rank timelines to measure overlap ratio $ho=rac{T_{overlap}}{T_{comm}}$.
- Sweep microbatch count $m$, gradient bucket sizes, and activation checkpoint boundaries.
- Capture p50/p95/p99 iteration time and correlate with collective/kernel breakdowns.

### Study checklist
- Can you state the exact input/output layout for this subsection?
- Can you predict which collective dominates at larger scale?
- Can you explain one condition where this method should *not* be used?

### 12.2 The Main Boundary
### Learning goals
- Understand the exact role of **12.2 The Main Boundary** in the full training/serving stack.
- Identify required tensor layouts before and after this step.
- Map algorithmic choices to communication, memory, and stability consequences.

### Mechanism
- **Inputs/state.** Define local shards, global semantics, and invariants (shape/dtype/order).
- **Transformation.** Execute compute kernels plus communication primitives to move from input layout to output layout.
- **Correctness.** Preserve mathematical equivalence (modulo floating-point ordering) and optimizer semantics.
- **Output contract.** Ensure downstream subsection receives expected placement/state.

### Mathematical view
- Local shard sizing follows $n_{local}=n/p$ along sharded axis, with replicated axes unchanged.
- Runtime contribution is modeled as:
$$
T_{subsec}pprox T_{kernel}+T_{comm}-T_{overlap}+T_{sync}+T_{runtime}.
$$
- If this stage adds collective payload $q$ on group size $p_g$, then increment:
$$
\Delta T_{comm}pprox lpha f(p_g)+eta q g(p_g).
$$
- If this stage changes replication→sharding for tensor $x$, memory delta per rank is approximately:
$$
\Delta M(x)pprox |x|\left(1-rac{1}{p_g}ight).
$$

### Implementation details (practical)
1. Build process groups and bind them to physical topology intentionally; avoid accidental rank orders.
2. Choose bucket/chunk sizes so kernels stay large enough for high occupancy while communication can stream.
3. Co-schedule streams: compute stream + comm stream + dependency events to increase overlap safely.
4. Encode layout metadata in checkpoints to permit reshard/restart on changed meshes.
5. Validate numerics under mixed precision with loss-scale or FP8 scale tracking as required.

### Failure modes
- **Bandwidth-bound plateau:** throughput stops scaling as payload term dominates.
- **Latency-bound micro-ops:** too many tiny collectives; startup cost dominates.
- **Imbalance/stragglers:** rank skew from load imbalance or topology asymmetry.
- **Memory spikes:** poor prefetch/reshard boundaries or oversized microbatching.
- **Numerical drift:** precision policy mismatches across shards/collectives.

### Diagnostics and experiments
- Perform ablations: vary one axis degree at a time ($P_{tp}$, $P_{pp}$, etc.).
- Trace per-rank timelines to measure overlap ratio $ho=rac{T_{overlap}}{T_{comm}}$.
- Sweep microbatch count $m$, gradient bucket sizes, and activation checkpoint boundaries.
- Capture p50/p95/p99 iteration time and correlate with collective/kernel breakdowns.

### Study checklist
- Can you state the exact input/output layout for this subsection?
- Can you predict which collective dominates at larger scale?
- Can you explain one condition where this method should *not* be used?

## 13. Conclusion
This section explains **13. Conclusion** from first principles, implementation mechanics, cost/memory implications, topology interactions, and operational diagnostics.

## Bibliography and Source Notes
This section explains **Bibliography and Source Notes** from first principles, implementation mechanics, cost/memory implications, topology interactions, and operational diagnostics.

### Recent Training, Parallelism, and Runtime Systems
### Learning goals
- Understand the exact role of **Recent Training, Parallelism, and Runtime Systems** in the full training/serving stack.
- Identify required tensor layouts before and after this step.
- Map algorithmic choices to communication, memory, and stability consequences.

### Mechanism
- **Inputs/state.** Define local shards, global semantics, and invariants (shape/dtype/order).
- **Transformation.** Execute compute kernels plus communication primitives to move from input layout to output layout.
- **Correctness.** Preserve mathematical equivalence (modulo floating-point ordering) and optimizer semantics.
- **Output contract.** Ensure downstream subsection receives expected placement/state.

### Mathematical view
- Local shard sizing follows $n_{local}=n/p$ along sharded axis, with replicated axes unchanged.
- Runtime contribution is modeled as:
$$
T_{subsec}pprox T_{kernel}+T_{comm}-T_{overlap}+T_{sync}+T_{runtime}.
$$
- If this stage adds collective payload $q$ on group size $p_g$, then increment:
$$
\Delta T_{comm}pprox lpha f(p_g)+eta q g(p_g).
$$
- If this stage changes replication→sharding for tensor $x$, memory delta per rank is approximately:
$$
\Delta M(x)pprox |x|\left(1-rac{1}{p_g}ight).
$$

### Implementation details (practical)
1. Build process groups and bind them to physical topology intentionally; avoid accidental rank orders.
2. Choose bucket/chunk sizes so kernels stay large enough for high occupancy while communication can stream.
3. Co-schedule streams: compute stream + comm stream + dependency events to increase overlap safely.
4. Encode layout metadata in checkpoints to permit reshard/restart on changed meshes.
5. Validate numerics under mixed precision with loss-scale or FP8 scale tracking as required.

### Failure modes
- **Bandwidth-bound plateau:** throughput stops scaling as payload term dominates.
- **Latency-bound micro-ops:** too many tiny collectives; startup cost dominates.
- **Imbalance/stragglers:** rank skew from load imbalance or topology asymmetry.
- **Memory spikes:** poor prefetch/reshard boundaries or oversized microbatching.
- **Numerical drift:** precision policy mismatches across shards/collectives.

### Diagnostics and experiments
- Perform ablations: vary one axis degree at a time ($P_{tp}$, $P_{pp}$, etc.).
- Trace per-rank timelines to measure overlap ratio $ho=rac{T_{overlap}}{T_{comm}}$.
- Sweep microbatch count $m$, gradient bucket sizes, and activation checkpoint boundaries.
- Capture p50/p95/p99 iteration time and correlate with collective/kernel breakdowns.

### Study checklist
- Can you state the exact input/output layout for this subsection?
- Can you predict which collective dominates at larger scale?
- Can you explain one condition where this method should *not* be used?

### Pipeline Parallelism
### Learning goals
- Understand the exact role of **Pipeline Parallelism** in the full training/serving stack.
- Identify required tensor layouts before and after this step.
- Map algorithmic choices to communication, memory, and stability consequences.

### Mechanism
- **Inputs/state.** Define local shards, global semantics, and invariants (shape/dtype/order).
- **Transformation.** Execute compute kernels plus communication primitives to move from input layout to output layout.
- **Correctness.** Preserve mathematical equivalence (modulo floating-point ordering) and optimizer semantics.
- **Output contract.** Ensure downstream subsection receives expected placement/state.

### Mathematical view
- Local shard sizing follows $n_{local}=n/p$ along sharded axis, with replicated axes unchanged.
- Runtime contribution is modeled as:
$$
T_{subsec}pprox T_{kernel}+T_{comm}-T_{overlap}+T_{sync}+T_{runtime}.
$$
- If this stage adds collective payload $q$ on group size $p_g$, then increment:
$$
\Delta T_{comm}pprox lpha f(p_g)+eta q g(p_g).
$$
- If this stage changes replication→sharding for tensor $x$, memory delta per rank is approximately:
$$
\Delta M(x)pprox |x|\left(1-rac{1}{p_g}ight).
$$

### Implementation details (practical)
1. Build process groups and bind them to physical topology intentionally; avoid accidental rank orders.
2. Choose bucket/chunk sizes so kernels stay large enough for high occupancy while communication can stream.
3. Co-schedule streams: compute stream + comm stream + dependency events to increase overlap safely.
4. Encode layout metadata in checkpoints to permit reshard/restart on changed meshes.
5. Validate numerics under mixed precision with loss-scale or FP8 scale tracking as required.

### Failure modes
- **Bandwidth-bound plateau:** throughput stops scaling as payload term dominates.
- **Latency-bound micro-ops:** too many tiny collectives; startup cost dominates.
- **Imbalance/stragglers:** rank skew from load imbalance or topology asymmetry.
- **Memory spikes:** poor prefetch/reshard boundaries or oversized microbatching.
- **Numerical drift:** precision policy mismatches across shards/collectives.

### Diagnostics and experiments
- Perform ablations: vary one axis degree at a time ($P_{tp}$, $P_{pp}$, etc.).
- Trace per-rank timelines to measure overlap ratio $ho=rac{T_{overlap}}{T_{comm}}$.
- Sweep microbatch count $m$, gradient bucket sizes, and activation checkpoint boundaries.
- Capture p50/p95/p99 iteration time and correlate with collective/kernel breakdowns.

### Study checklist
- Can you state the exact input/output layout for this subsection?
- Can you predict which collective dominates at larger scale?
- Can you explain one condition where this method should *not* be used?

### Sequence, Context, and Attention Parallelism
### Learning goals
- Understand the exact role of **Sequence, Context, and Attention Parallelism** in the full training/serving stack.
- Identify required tensor layouts before and after this step.
- Map algorithmic choices to communication, memory, and stability consequences.

### Mechanism
- **Inputs/state.** Define local shards, global semantics, and invariants (shape/dtype/order).
- **Transformation.** Execute compute kernels plus communication primitives to move from input layout to output layout.
- **Correctness.** Preserve mathematical equivalence (modulo floating-point ordering) and optimizer semantics.
- **Output contract.** Ensure downstream subsection receives expected placement/state.

### Mathematical view
- Local shard sizing follows $n_{local}=n/p$ along sharded axis, with replicated axes unchanged.
- Runtime contribution is modeled as:
$$
T_{subsec}pprox T_{kernel}+T_{comm}-T_{overlap}+T_{sync}+T_{runtime}.
$$
- If this stage adds collective payload $q$ on group size $p_g$, then increment:
$$
\Delta T_{comm}pprox lpha f(p_g)+eta q g(p_g).
$$
- If this stage changes replication→sharding for tensor $x$, memory delta per rank is approximately:
$$
\Delta M(x)pprox |x|\left(1-rac{1}{p_g}ight).
$$

### Implementation details (practical)
1. Build process groups and bind them to physical topology intentionally; avoid accidental rank orders.
2. Choose bucket/chunk sizes so kernels stay large enough for high occupancy while communication can stream.
3. Co-schedule streams: compute stream + comm stream + dependency events to increase overlap safely.
4. Encode layout metadata in checkpoints to permit reshard/restart on changed meshes.
5. Validate numerics under mixed precision with loss-scale or FP8 scale tracking as required.

### Failure modes
- **Bandwidth-bound plateau:** throughput stops scaling as payload term dominates.
- **Latency-bound micro-ops:** too many tiny collectives; startup cost dominates.
- **Imbalance/stragglers:** rank skew from load imbalance or topology asymmetry.
- **Memory spikes:** poor prefetch/reshard boundaries or oversized microbatching.
- **Numerical drift:** precision policy mismatches across shards/collectives.

### Diagnostics and experiments
- Perform ablations: vary one axis degree at a time ($P_{tp}$, $P_{pp}$, etc.).
- Trace per-rank timelines to measure overlap ratio $ho=rac{T_{overlap}}{T_{comm}}$.
- Sweep microbatch count $m$, gradient bucket sizes, and activation checkpoint boundaries.
- Capture p50/p95/p99 iteration time and correlate with collective/kernel breakdowns.

### Study checklist
- Can you state the exact input/output layout for this subsection?
- Can you predict which collective dominates at larger scale?
- Can you explain one condition where this method should *not* be used?

### MoE and Expert Parallelism
### Learning goals
- Understand the exact role of **MoE and Expert Parallelism** in the full training/serving stack.
- Identify required tensor layouts before and after this step.
- Map algorithmic choices to communication, memory, and stability consequences.

### Mechanism
- **Inputs/state.** Define local shards, global semantics, and invariants (shape/dtype/order).
- **Transformation.** Execute compute kernels plus communication primitives to move from input layout to output layout.
- **Correctness.** Preserve mathematical equivalence (modulo floating-point ordering) and optimizer semantics.
- **Output contract.** Ensure downstream subsection receives expected placement/state.

### Mathematical view
- Local shard sizing follows $n_{local}=n/p$ along sharded axis, with replicated axes unchanged.
- Runtime contribution is modeled as:
$$
T_{subsec}pprox T_{kernel}+T_{comm}-T_{overlap}+T_{sync}+T_{runtime}.
$$
- If this stage adds collective payload $q$ on group size $p_g$, then increment:
$$
\Delta T_{comm}pprox lpha f(p_g)+eta q g(p_g).
$$
- If this stage changes replication→sharding for tensor $x$, memory delta per rank is approximately:
$$
\Delta M(x)pprox |x|\left(1-rac{1}{p_g}ight).
$$

### Implementation details (practical)
1. Build process groups and bind them to physical topology intentionally; avoid accidental rank orders.
2. Choose bucket/chunk sizes so kernels stay large enough for high occupancy while communication can stream.
3. Co-schedule streams: compute stream + comm stream + dependency events to increase overlap safely.
4. Encode layout metadata in checkpoints to permit reshard/restart on changed meshes.
5. Validate numerics under mixed precision with loss-scale or FP8 scale tracking as required.

### Failure modes
- **Bandwidth-bound plateau:** throughput stops scaling as payload term dominates.
- **Latency-bound micro-ops:** too many tiny collectives; startup cost dominates.
- **Imbalance/stragglers:** rank skew from load imbalance or topology asymmetry.
- **Memory spikes:** poor prefetch/reshard boundaries or oversized microbatching.
- **Numerical drift:** precision policy mismatches across shards/collectives.

### Diagnostics and experiments
- Perform ablations: vary one axis degree at a time ($P_{tp}$, $P_{pp}$, etc.).
- Trace per-rank timelines to measure overlap ratio $ho=rac{T_{overlap}}{T_{comm}}$.
- Sweep microbatch count $m$, gradient bucket sizes, and activation checkpoint boundaries.
- Capture p50/p95/p99 iteration time and correlate with collective/kernel breakdowns.

### Study checklist
- Can you state the exact input/output layout for this subsection?
- Can you predict which collective dominates at larger scale?
- Can you explain one condition where this method should *not* be used?

### Inference and Serving
### Learning goals
- Understand the exact role of **Inference and Serving** in the full training/serving stack.
- Identify required tensor layouts before and after this step.
- Map algorithmic choices to communication, memory, and stability consequences.

### Mechanism
- **Inputs/state.** Define local shards, global semantics, and invariants (shape/dtype/order).
- **Transformation.** Execute compute kernels plus communication primitives to move from input layout to output layout.
- **Correctness.** Preserve mathematical equivalence (modulo floating-point ordering) and optimizer semantics.
- **Output contract.** Ensure downstream subsection receives expected placement/state.

### Mathematical view
- Local shard sizing follows $n_{local}=n/p$ along sharded axis, with replicated axes unchanged.
- Runtime contribution is modeled as:
$$
T_{subsec}pprox T_{kernel}+T_{comm}-T_{overlap}+T_{sync}+T_{runtime}.
$$
- If this stage adds collective payload $q$ on group size $p_g$, then increment:
$$
\Delta T_{comm}pprox lpha f(p_g)+eta q g(p_g).
$$
- If this stage changes replication→sharding for tensor $x$, memory delta per rank is approximately:
$$
\Delta M(x)pprox |x|\left(1-rac{1}{p_g}ight).
$$

### Implementation details (practical)
1. Build process groups and bind them to physical topology intentionally; avoid accidental rank orders.
2. Choose bucket/chunk sizes so kernels stay large enough for high occupancy while communication can stream.
3. Co-schedule streams: compute stream + comm stream + dependency events to increase overlap safely.
4. Encode layout metadata in checkpoints to permit reshard/restart on changed meshes.
5. Validate numerics under mixed precision with loss-scale or FP8 scale tracking as required.

### Failure modes
- **Bandwidth-bound plateau:** throughput stops scaling as payload term dominates.
- **Latency-bound micro-ops:** too many tiny collectives; startup cost dominates.
- **Imbalance/stragglers:** rank skew from load imbalance or topology asymmetry.
- **Memory spikes:** poor prefetch/reshard boundaries or oversized microbatching.
- **Numerical drift:** precision policy mismatches across shards/collectives.

### Diagnostics and experiments
- Perform ablations: vary one axis degree at a time ($P_{tp}$, $P_{pp}$, etc.).
- Trace per-rank timelines to measure overlap ratio $ho=rac{T_{overlap}}{T_{comm}}$.
- Sweep microbatch count $m$, gradient bucket sizes, and activation checkpoint boundaries.
- Capture p50/p95/p99 iteration time and correlate with collective/kernel breakdowns.

### Study checklist
- Can you state the exact input/output layout for this subsection?
- Can you predict which collective dominates at larger scale?
- Can you explain one condition where this method should *not* be used?

### Low Precision, Kernels, and Platform
### Learning goals
- Understand the exact role of **Low Precision, Kernels, and Platform** in the full training/serving stack.
- Identify required tensor layouts before and after this step.
- Map algorithmic choices to communication, memory, and stability consequences.

### Mechanism
- **Inputs/state.** Define local shards, global semantics, and invariants (shape/dtype/order).
- **Transformation.** Execute compute kernels plus communication primitives to move from input layout to output layout.
- **Correctness.** Preserve mathematical equivalence (modulo floating-point ordering) and optimizer semantics.
- **Output contract.** Ensure downstream subsection receives expected placement/state.

### Mathematical view
- Local shard sizing follows $n_{local}=n/p$ along sharded axis, with replicated axes unchanged.
- Runtime contribution is modeled as:
$$
T_{subsec}pprox T_{kernel}+T_{comm}-T_{overlap}+T_{sync}+T_{runtime}.
$$
- If this stage adds collective payload $q$ on group size $p_g$, then increment:
$$
\Delta T_{comm}pprox lpha f(p_g)+eta q g(p_g).
$$
- If this stage changes replication→sharding for tensor $x$, memory delta per rank is approximately:
$$
\Delta M(x)pprox |x|\left(1-rac{1}{p_g}ight).
$$

### Implementation details (practical)
1. Build process groups and bind them to physical topology intentionally; avoid accidental rank orders.
2. Choose bucket/chunk sizes so kernels stay large enough for high occupancy while communication can stream.
3. Co-schedule streams: compute stream + comm stream + dependency events to increase overlap safely.
4. Encode layout metadata in checkpoints to permit reshard/restart on changed meshes.
5. Validate numerics under mixed precision with loss-scale or FP8 scale tracking as required.

### Failure modes
- **Bandwidth-bound plateau:** throughput stops scaling as payload term dominates.
- **Latency-bound micro-ops:** too many tiny collectives; startup cost dominates.
- **Imbalance/stragglers:** rank skew from load imbalance or topology asymmetry.
- **Memory spikes:** poor prefetch/reshard boundaries or oversized microbatching.
- **Numerical drift:** precision policy mismatches across shards/collectives.

### Diagnostics and experiments
- Perform ablations: vary one axis degree at a time ($P_{tp}$, $P_{pp}$, etc.).
- Trace per-rank timelines to measure overlap ratio $ho=rac{T_{overlap}}{T_{comm}}$.
- Sweep microbatch count $m$, gradient bucket sizes, and activation checkpoint boundaries.
- Capture p50/p95/p99 iteration time and correlate with collective/kernel breakdowns.

### Study checklist
- Can you state the exact input/output layout for this subsection?
- Can you predict which collective dominates at larger scale?
- Can you explain one condition where this method should *not* be used?

### Compiler and Mesh Systems
### Learning goals
- Understand the exact role of **Compiler and Mesh Systems** in the full training/serving stack.
- Identify required tensor layouts before and after this step.
- Map algorithmic choices to communication, memory, and stability consequences.

### Mechanism
- **Inputs/state.** Define local shards, global semantics, and invariants (shape/dtype/order).
- **Transformation.** Execute compute kernels plus communication primitives to move from input layout to output layout.
- **Correctness.** Preserve mathematical equivalence (modulo floating-point ordering) and optimizer semantics.
- **Output contract.** Ensure downstream subsection receives expected placement/state.

### Mathematical view
- Local shard sizing follows $n_{local}=n/p$ along sharded axis, with replicated axes unchanged.
- Runtime contribution is modeled as:
$$
T_{subsec}pprox T_{kernel}+T_{comm}-T_{overlap}+T_{sync}+T_{runtime}.
$$
- If this stage adds collective payload $q$ on group size $p_g$, then increment:
$$
\Delta T_{comm}pprox lpha f(p_g)+eta q g(p_g).
$$
- If this stage changes replication→sharding for tensor $x$, memory delta per rank is approximately:
$$
\Delta M(x)pprox |x|\left(1-rac{1}{p_g}ight).
$$

### Implementation details (practical)
1. Build process groups and bind them to physical topology intentionally; avoid accidental rank orders.
2. Choose bucket/chunk sizes so kernels stay large enough for high occupancy while communication can stream.
3. Co-schedule streams: compute stream + comm stream + dependency events to increase overlap safely.
4. Encode layout metadata in checkpoints to permit reshard/restart on changed meshes.
5. Validate numerics under mixed precision with loss-scale or FP8 scale tracking as required.

### Failure modes
- **Bandwidth-bound plateau:** throughput stops scaling as payload term dominates.
- **Latency-bound micro-ops:** too many tiny collectives; startup cost dominates.
- **Imbalance/stragglers:** rank skew from load imbalance or topology asymmetry.
- **Memory spikes:** poor prefetch/reshard boundaries or oversized microbatching.
- **Numerical drift:** precision policy mismatches across shards/collectives.

### Diagnostics and experiments
- Perform ablations: vary one axis degree at a time ($P_{tp}$, $P_{pp}$, etc.).
- Trace per-rank timelines to measure overlap ratio $ho=rac{T_{overlap}}{T_{comm}}$.
- Sweep microbatch count $m$, gradient bucket sizes, and activation checkpoint boundaries.
- Capture p50/p95/p99 iteration time and correlate with collective/kernel breakdowns.

### Study checklist
- Can you state the exact input/output layout for this subsection?
- Can you predict which collective dominates at larger scale?
- Can you explain one condition where this method should *not* be used?

### Model Reports Used As Parallelism Context
### Learning goals
- Understand the exact role of **Model Reports Used As Parallelism Context** in the full training/serving stack.
- Identify required tensor layouts before and after this step.
- Map algorithmic choices to communication, memory, and stability consequences.

### Mechanism
- **Inputs/state.** Define local shards, global semantics, and invariants (shape/dtype/order).
- **Transformation.** Execute compute kernels plus communication primitives to move from input layout to output layout.
- **Correctness.** Preserve mathematical equivalence (modulo floating-point ordering) and optimizer semantics.
- **Output contract.** Ensure downstream subsection receives expected placement/state.

### Mathematical view
- Local shard sizing follows $n_{local}=n/p$ along sharded axis, with replicated axes unchanged.
- Runtime contribution is modeled as:
$$
T_{subsec}pprox T_{kernel}+T_{comm}-T_{overlap}+T_{sync}+T_{runtime}.
$$
- If this stage adds collective payload $q$ on group size $p_g$, then increment:
$$
\Delta T_{comm}pprox lpha f(p_g)+eta q g(p_g).
$$
- If this stage changes replication→sharding for tensor $x$, memory delta per rank is approximately:
$$
\Delta M(x)pprox |x|\left(1-rac{1}{p_g}ight).
$$

### Implementation details (practical)
1. Build process groups and bind them to physical topology intentionally; avoid accidental rank orders.
2. Choose bucket/chunk sizes so kernels stay large enough for high occupancy while communication can stream.
3. Co-schedule streams: compute stream + comm stream + dependency events to increase overlap safely.
4. Encode layout metadata in checkpoints to permit reshard/restart on changed meshes.
5. Validate numerics under mixed precision with loss-scale or FP8 scale tracking as required.

### Failure modes
- **Bandwidth-bound plateau:** throughput stops scaling as payload term dominates.
- **Latency-bound micro-ops:** too many tiny collectives; startup cost dominates.
- **Imbalance/stragglers:** rank skew from load imbalance or topology asymmetry.
- **Memory spikes:** poor prefetch/reshard boundaries or oversized microbatching.
- **Numerical drift:** precision policy mismatches across shards/collectives.

### Diagnostics and experiments
- Perform ablations: vary one axis degree at a time ($P_{tp}$, $P_{pp}$, etc.).
- Trace per-rank timelines to measure overlap ratio $ho=rac{T_{overlap}}{T_{comm}}$.
- Sweep microbatch count $m$, gradient bucket sizes, and activation checkpoint boundaries.
- Capture p50/p95/p99 iteration time and correlate with collective/kernel breakdowns.

### Study checklist
- Can you state the exact input/output layout for this subsection?
- Can you predict which collective dominates at larger scale?
- Can you explain one condition where this method should *not* be used?

## Verification Notes
This section explains **Verification Notes** from first principles, implementation mechanics, cost/memory implications, topology interactions, and operational diagnostics.

## Appendix A: End-to-end training step decomposition
For a hybrid configuration, a step can be conceptually decomposed into: data load, pre-processing, forward (with PP schedule), attention/MLP kernels (TP/SP/CP), MoE routing (EP), backward collectives (DP/FSDP/ZeRO), optimizer update, checkpoint/metrics hooks. The critical path is whichever chain has maximal cumulative time after overlap.

## Appendix B: Communication primitive intuition
- AR: everyone contributes and receives full reduced result; useful for consistent replicated states.
- RS: contributes full, receives reduced shard; ideal for sharded gradient paths.
- AG: collects shards into full tensor; common before local compute that needs full view.
- A2A: each rank sends different shard to every other; key for MoE and sequence/head remaps.

## Appendix C: Minimal algorithm pseudocode
```text
for step in training_steps:
  load global batch B and shard by DP
  for microbatch in schedule(PP):
    materialize params if sharded (AG)
    run TP/CP/SP attention + MLP kernels
    if MoE: route -> A2A dispatch -> expert compute -> A2A combine
    emit activations for next PP stage (Send/Recv)
  backward with mirrored schedule
  reduce gradients (AR or RS depending on state sharding)
  optimizer step on local shard
  checkpoint with layout metadata
```

## Appendix D: Decision matrix (condensed)
- If OOM from params/opt states: prioritize ZeRO-3/FSDP-style full sharding.
- If compute kernels too large for single rank: increase TP until comm dominates.
- If depth limits utilization: add PP and tune microbatches to control bubble ratio.
- If long context dominates: adopt CP/ring/Ulysses based on network and attention pattern.
- If model is MoE: optimize A2A + load balancing before chasing kernel micro-optimizations.
- If serving latency is key: prioritize KV/cache/scheduler design over train-time parallel settings.

## Source lineage note
The section structure and source pointers are inherited from `papers/parallelism_reference.md`; this document expands those pointers into self-contained explanatory text for study.