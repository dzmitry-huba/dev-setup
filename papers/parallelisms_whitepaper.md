# Parallelisms White Paper (Expanded, Multi-Engine Draft)

## Authoring protocol
For each subsection, this paper uses an independent "engine" block to emulate separate specialist analysis. All engines share a common notation contract, then produce self-contained explanations.

## Global notation and assumptions
- Model parameters: $|\theta|$. Hidden size: $H$. Layers: $L$. Global batch: $B$. Sequence length: $S$.
- Mesh axes: $P_{dp},P_{tp},P_{pp},P_{cp},P_{ep}$ and $P=\prod P_i$.
- Attention heads $N_h$, head dim $d_h$, so $H=N_h d_h$.
- Collectives: $\operatorname{AR}$, $\operatorname{AG}$, $\operatorname{RS}$, $\operatorname{A2A}$.
- Communication cost uses $T=\alpha\cdot f(p)+\beta\cdot n\cdot g(p)$ with topology-aware $f,g$.

### Common equations used throughout
$$M_{rank}=M_{params}+M_{grads}+M_{opt}+M_{acts}+M_{kv}+M_{tmp}.$$
$$\eta_{pipe}\approx 1-\frac{P_{pp}-1}{m+P_{pp}-1},\quad\eta_{bubble}=1-\eta_{pipe}.$$
$$\text{step\_time}\approx\max(T_{compute},T_{comm}-T_{overlap})+T_{sync}. $$

## Abstract
**Engine 1: scope and objective.** Provide a self-contained treatment of **Abstract** with definitions, mechanism, cost model, implementation choices, failure modes, and decision rules.

1. **Conceptual model.**
   - Abstract is framed as a tensor-placement transformation over the device mesh. Inputs, outputs, and dependency edges are explicitly identified.
   - Correctness condition: transformed execution must preserve numerical equivalence up to floating-point non-associativity and stochastic layers.
2. **Mathematical formulation.**
   - Let local shard size be $n_{local}=n/p$ along the chosen axis. Memory scales as $O(n_{local})$ for sharded states and $O(n)$ for replicated states.
   - Communication lower bound (single phase): $\Omega(n\,\beta)$ payload plus startup term $\Omega(\alpha)$.
3. **Execution recipe.**
   - Precondition checks: mesh groups, dtype policy, seed policy, checkpoint layout metadata.
   - Runtime sequence: schedule compute kernels, issue collectives, place synchronization barriers only on true dependencies, and opportunistically overlap.
4. **Performance engineering.**
   - Increase arithmetic intensity using fused kernels and shape-friendly GEMMs.
   - Reduce bytes with precision controls (FP8/quantized links) while managing scales and error feedback.
   - Place high-frequency collectives on fast links (e.g., intra-node) whenever possible.
5. **Failure modes and diagnostics.**
   - Symptoms: stragglers, tail latency spikes, allocator fragmentation, imbalance, and collective stalls.
   - Diagnostics: per-rank timeline traces, bucket size sweeps, microbatch sweep, and reshard-count audits.
6. **Decision rubric.**
   - Prefer the smallest parallel axis set meeting memory and throughput targets.
   - Add axes incrementally; after each addition, re-measure MFU, link utilization, and p95/p99 latency.

**Practical takeaway.** This subsection should be implemented as a topology-aware schedule with explicit tensor-layout contracts; otherwise theoretical gains are lost to reshards and synchronization overhead.

## How To Read This Paper
**Engine 2: scope and objective.** Provide a self-contained treatment of **How To Read This Paper** with definitions, mechanism, cost model, implementation choices, failure modes, and decision rules.

1. **Conceptual model.**
   - How To Read This Paper is framed as a tensor-placement transformation over the device mesh. Inputs, outputs, and dependency edges are explicitly identified.
   - Correctness condition: transformed execution must preserve numerical equivalence up to floating-point non-associativity and stochastic layers.
2. **Mathematical formulation.**
   - Let local shard size be $n_{local}=n/p$ along the chosen axis. Memory scales as $O(n_{local})$ for sharded states and $O(n)$ for replicated states.
   - Communication lower bound (single phase): $\Omega(n\,\beta)$ payload plus startup term $\Omega(\alpha)$.
3. **Execution recipe.**
   - Precondition checks: mesh groups, dtype policy, seed policy, checkpoint layout metadata.
   - Runtime sequence: schedule compute kernels, issue collectives, place synchronization barriers only on true dependencies, and opportunistically overlap.
4. **Performance engineering.**
   - Increase arithmetic intensity using fused kernels and shape-friendly GEMMs.
   - Reduce bytes with precision controls (FP8/quantized links) while managing scales and error feedback.
   - Place high-frequency collectives on fast links (e.g., intra-node) whenever possible.
5. **Failure modes and diagnostics.**
   - Symptoms: stragglers, tail latency spikes, allocator fragmentation, imbalance, and collective stalls.
   - Diagnostics: per-rank timeline traces, bucket size sweeps, microbatch sweep, and reshard-count audits.
6. **Decision rubric.**
   - Prefer the smallest parallel axis set meeting memory and throughput targets.
   - Add axes incrementally; after each addition, re-measure MFU, link utilization, and p95/p99 latency.

**Practical takeaway.** This subsection should be implemented as a topology-aware schedule with explicit tensor-layout contracts; otherwise theoretical gains are lost to reshards and synchronization overhead.

## 1. Shared System Model, Notation, Placement, and Evidence Policy
**Engine 3: scope and objective.** Provide a self-contained treatment of **1. Shared System Model, Notation, Placement, and Evidence Policy** with definitions, mechanism, cost model, implementation choices, failure modes, and decision rules.

1. **Conceptual model.**
   - 1. Shared System Model, Notation, Placement, and Evidence Policy is framed as a tensor-placement transformation over the device mesh. Inputs, outputs, and dependency edges are explicitly identified.
   - Correctness condition: transformed execution must preserve numerical equivalence up to floating-point non-associativity and stochastic layers.
2. **Mathematical formulation.**
   - Let local shard size be $n_{local}=n/p$ along the chosen axis. Memory scales as $O(n_{local})$ for sharded states and $O(n)$ for replicated states.
   - Communication lower bound (single phase): $\Omega(n\,\beta)$ payload plus startup term $\Omega(\alpha)$.
3. **Execution recipe.**
   - Precondition checks: mesh groups, dtype policy, seed policy, checkpoint layout metadata.
   - Runtime sequence: schedule compute kernels, issue collectives, place synchronization barriers only on true dependencies, and opportunistically overlap.
4. **Performance engineering.**
   - Increase arithmetic intensity using fused kernels and shape-friendly GEMMs.
   - Reduce bytes with precision controls (FP8/quantized links) while managing scales and error feedback.
   - Place high-frequency collectives on fast links (e.g., intra-node) whenever possible.
5. **Failure modes and diagnostics.**
   - Symptoms: stragglers, tail latency spikes, allocator fragmentation, imbalance, and collective stalls.
   - Diagnostics: per-rank timeline traces, bucket size sweeps, microbatch sweep, and reshard-count audits.
6. **Decision rubric.**
   - Prefer the smallest parallel axis set meeting memory and throughput targets.
   - Add axes incrementally; after each addition, re-measure MFU, link utilization, and p95/p99 latency.

**Practical takeaway.** This subsection should be implemented as a topology-aware schedule with explicit tensor-layout contracts; otherwise theoretical gains are lost to reshards and synchronization overhead.

### 1.1 Evidence Grades and Source Interpretation
**Engine 4: scope and objective.** Provide a self-contained treatment of **1.1 Evidence Grades and Source Interpretation** with definitions, mechanism, cost model, implementation choices, failure modes, and decision rules.

1. **Conceptual model.**
   - 1.1 Evidence Grades and Source Interpretation is framed as a tensor-placement transformation over the device mesh. Inputs, outputs, and dependency edges are explicitly identified.
   - Correctness condition: transformed execution must preserve numerical equivalence up to floating-point non-associativity and stochastic layers.
2. **Mathematical formulation.**
   - Let local shard size be $n_{local}=n/p$ along the chosen axis. Memory scales as $O(n_{local})$ for sharded states and $O(n)$ for replicated states.
   - Communication lower bound (single phase): $\Omega(n\,\beta)$ payload plus startup term $\Omega(\alpha)$.
3. **Execution recipe.**
   - Precondition checks: mesh groups, dtype policy, seed policy, checkpoint layout metadata.
   - Runtime sequence: schedule compute kernels, issue collectives, place synchronization barriers only on true dependencies, and opportunistically overlap.
4. **Performance engineering.**
   - Increase arithmetic intensity using fused kernels and shape-friendly GEMMs.
   - Reduce bytes with precision controls (FP8/quantized links) while managing scales and error feedback.
   - Place high-frequency collectives on fast links (e.g., intra-node) whenever possible.
5. **Failure modes and diagnostics.**
   - Symptoms: stragglers, tail latency spikes, allocator fragmentation, imbalance, and collective stalls.
   - Diagnostics: per-rank timeline traces, bucket size sweeps, microbatch sweep, and reshard-count audits.
6. **Decision rubric.**
   - Prefer the smallest parallel axis set meeting memory and throughput targets.
   - Add axes incrementally; after each addition, re-measure MFU, link utilization, and p95/p99 latency.

**Practical takeaway.** This subsection should be implemented as a topology-aware schedule with explicit tensor-layout contracts; otherwise theoretical gains are lost to reshards and synchronization overhead.

### 1.2 Shared Symbols and Transformer Shapes
**Engine 5: scope and objective.** Provide a self-contained treatment of **1.2 Shared Symbols and Transformer Shapes** with definitions, mechanism, cost model, implementation choices, failure modes, and decision rules.

1. **Conceptual model.**
   - 1.2 Shared Symbols and Transformer Shapes is framed as a tensor-placement transformation over the device mesh. Inputs, outputs, and dependency edges are explicitly identified.
   - Correctness condition: transformed execution must preserve numerical equivalence up to floating-point non-associativity and stochastic layers.
2. **Mathematical formulation.**
   - Let local shard size be $n_{local}=n/p$ along the chosen axis. Memory scales as $O(n_{local})$ for sharded states and $O(n)$ for replicated states.
   - Communication lower bound (single phase): $\Omega(n\,\beta)$ payload plus startup term $\Omega(\alpha)$.
3. **Execution recipe.**
   - Precondition checks: mesh groups, dtype policy, seed policy, checkpoint layout metadata.
   - Runtime sequence: schedule compute kernels, issue collectives, place synchronization barriers only on true dependencies, and opportunistically overlap.
4. **Performance engineering.**
   - Increase arithmetic intensity using fused kernels and shape-friendly GEMMs.
   - Reduce bytes with precision controls (FP8/quantized links) while managing scales and error feedback.
   - Place high-frequency collectives on fast links (e.g., intra-node) whenever possible.
5. **Failure modes and diagnostics.**
   - Symptoms: stragglers, tail latency spikes, allocator fragmentation, imbalance, and collective stalls.
   - Diagnostics: per-rank timeline traces, bucket size sweeps, microbatch sweep, and reshard-count audits.
6. **Decision rubric.**
   - Prefer the smallest parallel axis set meeting memory and throughput targets.
   - Add axes incrementally; after each addition, re-measure MFU, link utilization, and p95/p99 latency.

**Practical takeaway.** This subsection should be implemented as a topology-aware schedule with explicit tensor-layout contracts; otherwise theoretical gains are lost to reshards and synchronization overhead.

### 1.3 Device Meshes, Rank Coordinates, and Process Groups
**Engine 6: scope and objective.** Provide a self-contained treatment of **1.3 Device Meshes, Rank Coordinates, and Process Groups** with definitions, mechanism, cost model, implementation choices, failure modes, and decision rules.

1. **Conceptual model.**
   - 1.3 Device Meshes, Rank Coordinates, and Process Groups is framed as a tensor-placement transformation over the device mesh. Inputs, outputs, and dependency edges are explicitly identified.
   - Correctness condition: transformed execution must preserve numerical equivalence up to floating-point non-associativity and stochastic layers.
2. **Mathematical formulation.**
   - Let local shard size be $n_{local}=n/p$ along the chosen axis. Memory scales as $O(n_{local})$ for sharded states and $O(n)$ for replicated states.
   - Communication lower bound (single phase): $\Omega(n\,\beta)$ payload plus startup term $\Omega(\alpha)$.
3. **Execution recipe.**
   - Precondition checks: mesh groups, dtype policy, seed policy, checkpoint layout metadata.
   - Runtime sequence: schedule compute kernels, issue collectives, place synchronization barriers only on true dependencies, and opportunistically overlap.
4. **Performance engineering.**
   - Increase arithmetic intensity using fused kernels and shape-friendly GEMMs.
   - Reduce bytes with precision controls (FP8/quantized links) while managing scales and error feedback.
   - Place high-frequency collectives on fast links (e.g., intra-node) whenever possible.
5. **Failure modes and diagnostics.**
   - Symptoms: stragglers, tail latency spikes, allocator fragmentation, imbalance, and collective stalls.
   - Diagnostics: per-rank timeline traces, bucket size sweeps, microbatch sweep, and reshard-count audits.
6. **Decision rubric.**
   - Prefer the smallest parallel axis set meeting memory and throughput targets.
   - Add axes incrementally; after each addition, re-measure MFU, link utilization, and p95/p99 latency.

**Practical takeaway.** This subsection should be implemented as a topology-aware schedule with explicit tensor-layout contracts; otherwise theoretical gains are lost to reshards and synchronization overhead.

### 1.4 Placement States and Tensor Layout Transitions
**Engine 7: scope and objective.** Provide a self-contained treatment of **1.4 Placement States and Tensor Layout Transitions** with definitions, mechanism, cost model, implementation choices, failure modes, and decision rules.

1. **Conceptual model.**
   - 1.4 Placement States and Tensor Layout Transitions is framed as a tensor-placement transformation over the device mesh. Inputs, outputs, and dependency edges are explicitly identified.
   - Correctness condition: transformed execution must preserve numerical equivalence up to floating-point non-associativity and stochastic layers.
2. **Mathematical formulation.**
   - Let local shard size be $n_{local}=n/p$ along the chosen axis. Memory scales as $O(n_{local})$ for sharded states and $O(n)$ for replicated states.
   - Communication lower bound (single phase): $\Omega(n\,\beta)$ payload plus startup term $\Omega(\alpha)$.
3. **Execution recipe.**
   - Precondition checks: mesh groups, dtype policy, seed policy, checkpoint layout metadata.
   - Runtime sequence: schedule compute kernels, issue collectives, place synchronization barriers only on true dependencies, and opportunistically overlap.
4. **Performance engineering.**
   - Increase arithmetic intensity using fused kernels and shape-friendly GEMMs.
   - Reduce bytes with precision controls (FP8/quantized links) while managing scales and error feedback.
   - Place high-frequency collectives on fast links (e.g., intra-node) whenever possible.
5. **Failure modes and diagnostics.**
   - Symptoms: stragglers, tail latency spikes, allocator fragmentation, imbalance, and collective stalls.
   - Diagnostics: per-rank timeline traces, bucket size sweeps, microbatch sweep, and reshard-count audits.
6. **Decision rubric.**
   - Prefer the smallest parallel axis set meeting memory and throughput targets.
   - Add axes incrementally; after each addition, re-measure MFU, link utilization, and p95/p99 latency.

**Practical takeaway.** This subsection should be implemented as a topology-aware schedule with explicit tensor-layout contracts; otherwise theoretical gains are lost to reshards and synchronization overhead.

### 1.5 Communication Primitives and Cost Models
**Engine 8: scope and objective.** Provide a self-contained treatment of **1.5 Communication Primitives and Cost Models** with definitions, mechanism, cost model, implementation choices, failure modes, and decision rules.

1. **Conceptual model.**
   - 1.5 Communication Primitives and Cost Models is framed as a tensor-placement transformation over the device mesh. Inputs, outputs, and dependency edges are explicitly identified.
   - Correctness condition: transformed execution must preserve numerical equivalence up to floating-point non-associativity and stochastic layers.
2. **Mathematical formulation.**
   - Let local shard size be $n_{local}=n/p$ along the chosen axis. Memory scales as $O(n_{local})$ for sharded states and $O(n)$ for replicated states.
   - Communication lower bound (single phase): $\Omega(n\,\beta)$ payload plus startup term $\Omega(\alpha)$.
3. **Execution recipe.**
   - Precondition checks: mesh groups, dtype policy, seed policy, checkpoint layout metadata.
   - Runtime sequence: schedule compute kernels, issue collectives, place synchronization barriers only on true dependencies, and opportunistically overlap.
4. **Performance engineering.**
   - Increase arithmetic intensity using fused kernels and shape-friendly GEMMs.
   - Reduce bytes with precision controls (FP8/quantized links) while managing scales and error feedback.
   - Place high-frequency collectives on fast links (e.g., intra-node) whenever possible.
5. **Failure modes and diagnostics.**
   - Symptoms: stragglers, tail latency spikes, allocator fragmentation, imbalance, and collective stalls.
   - Diagnostics: per-rank timeline traces, bucket size sweeps, microbatch sweep, and reshard-count audits.
6. **Decision rubric.**
   - Prefer the smallest parallel axis set meeting memory and throughput targets.
   - Add axes incrementally; after each addition, re-measure MFU, link utilization, and p95/p99 latency.

**Practical takeaway.** This subsection should be implemented as a topology-aware schedule with explicit tensor-layout contracts; otherwise theoretical gains are lost to reshards and synchronization overhead.

### 1.6 Memory, Bandwidth, and Overlap Accounting
**Engine 9: scope and objective.** Provide a self-contained treatment of **1.6 Memory, Bandwidth, and Overlap Accounting** with definitions, mechanism, cost model, implementation choices, failure modes, and decision rules.

1. **Conceptual model.**
   - 1.6 Memory, Bandwidth, and Overlap Accounting is framed as a tensor-placement transformation over the device mesh. Inputs, outputs, and dependency edges are explicitly identified.
   - Correctness condition: transformed execution must preserve numerical equivalence up to floating-point non-associativity and stochastic layers.
2. **Mathematical formulation.**
   - Let local shard size be $n_{local}=n/p$ along the chosen axis. Memory scales as $O(n_{local})$ for sharded states and $O(n)$ for replicated states.
   - Communication lower bound (single phase): $\Omega(n\,\beta)$ payload plus startup term $\Omega(\alpha)$.
3. **Execution recipe.**
   - Precondition checks: mesh groups, dtype policy, seed policy, checkpoint layout metadata.
   - Runtime sequence: schedule compute kernels, issue collectives, place synchronization barriers only on true dependencies, and opportunistically overlap.
4. **Performance engineering.**
   - Increase arithmetic intensity using fused kernels and shape-friendly GEMMs.
   - Reduce bytes with precision controls (FP8/quantized links) while managing scales and error feedback.
   - Place high-frequency collectives on fast links (e.g., intra-node) whenever possible.
5. **Failure modes and diagnostics.**
   - Symptoms: stragglers, tail latency spikes, allocator fragmentation, imbalance, and collective stalls.
   - Diagnostics: per-rank timeline traces, bucket size sweeps, microbatch sweep, and reshard-count audits.
6. **Decision rubric.**
   - Prefer the smallest parallel axis set meeting memory and throughput targets.
   - Add axes incrementally; after each addition, re-measure MFU, link utilization, and p95/p99 latency.

**Practical takeaway.** This subsection should be implemented as a topology-aware schedule with explicit tensor-layout contracts; otherwise theoretical gains are lost to reshards and synchronization overhead.

### 1.7 How Later Sections Use This Shared Layer
**Engine 10: scope and objective.** Provide a self-contained treatment of **1.7 How Later Sections Use This Shared Layer** with definitions, mechanism, cost model, implementation choices, failure modes, and decision rules.

1. **Conceptual model.**
   - 1.7 How Later Sections Use This Shared Layer is framed as a tensor-placement transformation over the device mesh. Inputs, outputs, and dependency edges are explicitly identified.
   - Correctness condition: transformed execution must preserve numerical equivalence up to floating-point non-associativity and stochastic layers.
2. **Mathematical formulation.**
   - Let local shard size be $n_{local}=n/p$ along the chosen axis. Memory scales as $O(n_{local})$ for sharded states and $O(n)$ for replicated states.
   - Communication lower bound (single phase): $\Omega(n\,\beta)$ payload plus startup term $\Omega(\alpha)$.
3. **Execution recipe.**
   - Precondition checks: mesh groups, dtype policy, seed policy, checkpoint layout metadata.
   - Runtime sequence: schedule compute kernels, issue collectives, place synchronization barriers only on true dependencies, and opportunistically overlap.
4. **Performance engineering.**
   - Increase arithmetic intensity using fused kernels and shape-friendly GEMMs.
   - Reduce bytes with precision controls (FP8/quantized links) while managing scales and error feedback.
   - Place high-frequency collectives on fast links (e.g., intra-node) whenever possible.
5. **Failure modes and diagnostics.**
   - Symptoms: stragglers, tail latency spikes, allocator fragmentation, imbalance, and collective stalls.
   - Diagnostics: per-rank timeline traces, bucket size sweeps, microbatch sweep, and reshard-count audits.
6. **Decision rubric.**
   - Prefer the smallest parallel axis set meeting memory and throughput targets.
   - Add axes incrementally; after each addition, re-measure MFU, link utilization, and p95/p99 latency.

**Practical takeaway.** This subsection should be implemented as a topology-aware schedule with explicit tensor-layout contracts; otherwise theoretical gains are lost to reshards and synchronization overhead.

## 2. Data Parallelism, FSDP, ZeRO, HSDP, and State Sharding
**Engine 11: scope and objective.** Provide a self-contained treatment of **2. Data Parallelism, FSDP, ZeRO, HSDP, and State Sharding** with definitions, mechanism, cost model, implementation choices, failure modes, and decision rules.

1. **Conceptual model.**
   - 2. Data Parallelism, FSDP, ZeRO, HSDP, and State Sharding is framed as a tensor-placement transformation over the device mesh. Inputs, outputs, and dependency edges are explicitly identified.
   - Correctness condition: transformed execution must preserve numerical equivalence up to floating-point non-associativity and stochastic layers.
2. **Mathematical formulation.**
   - Let local shard size be $n_{local}=n/p$ along the chosen axis. Memory scales as $O(n_{local})$ for sharded states and $O(n)$ for replicated states.
   - Communication lower bound (single phase): $\Omega(n\,\beta)$ payload plus startup term $\Omega(\alpha)$.
3. **Execution recipe.**
   - Precondition checks: mesh groups, dtype policy, seed policy, checkpoint layout metadata.
   - Runtime sequence: schedule compute kernels, issue collectives, place synchronization barriers only on true dependencies, and opportunistically overlap.
4. **Performance engineering.**
   - Increase arithmetic intensity using fused kernels and shape-friendly GEMMs.
   - Reduce bytes with precision controls (FP8/quantized links) while managing scales and error feedback.
   - Place high-frequency collectives on fast links (e.g., intra-node) whenever possible.
5. **Failure modes and diagnostics.**
   - Symptoms: stragglers, tail latency spikes, allocator fragmentation, imbalance, and collective stalls.
   - Diagnostics: per-rank timeline traces, bucket size sweeps, microbatch sweep, and reshard-count audits.
6. **Decision rubric.**
   - Prefer the smallest parallel axis set meeting memory and throughput targets.
   - Add axes incrementally; after each addition, re-measure MFU, link utilization, and p95/p99 latency.

**Practical takeaway.** This subsection should be implemented as a topology-aware schedule with explicit tensor-layout contracts; otherwise theoretical gains are lost to reshards and synchronization overhead.

### 2.1 DDP: Replicated State, Batch Shards, and Gradient All-Reduce
**Engine 12: scope and objective.** Provide a self-contained treatment of **2.1 DDP: Replicated State, Batch Shards, and Gradient All-Reduce** with definitions, mechanism, cost model, implementation choices, failure modes, and decision rules.

1. **Conceptual model.**
   - 2.1 DDP: Replicated State, Batch Shards, and Gradient All-Reduce is framed as a tensor-placement transformation over the device mesh. Inputs, outputs, and dependency edges are explicitly identified.
   - Correctness condition: transformed execution must preserve numerical equivalence up to floating-point non-associativity and stochastic layers.
2. **Mathematical formulation.**
   - Let local shard size be $n_{local}=n/p$ along the chosen axis. Memory scales as $O(n_{local})$ for sharded states and $O(n)$ for replicated states.
   - Communication lower bound (single phase): $\Omega(n\,\beta)$ payload plus startup term $\Omega(\alpha)$.
3. **Execution recipe.**
   - Precondition checks: mesh groups, dtype policy, seed policy, checkpoint layout metadata.
   - Runtime sequence: schedule compute kernels, issue collectives, place synchronization barriers only on true dependencies, and opportunistically overlap.
4. **Performance engineering.**
   - Increase arithmetic intensity using fused kernels and shape-friendly GEMMs.
   - Reduce bytes with precision controls (FP8/quantized links) while managing scales and error feedback.
   - Place high-frequency collectives on fast links (e.g., intra-node) whenever possible.
5. **Failure modes and diagnostics.**
   - Symptoms: stragglers, tail latency spikes, allocator fragmentation, imbalance, and collective stalls.
   - Diagnostics: per-rank timeline traces, bucket size sweeps, microbatch sweep, and reshard-count audits.
6. **Decision rubric.**
   - Prefer the smallest parallel axis set meeting memory and throughput targets.
   - Add axes incrementally; after each addition, re-measure MFU, link utilization, and p95/p99 latency.

**Practical takeaway.** This subsection should be implemented as a topology-aware schedule with explicit tensor-layout contracts; otherwise theoretical gains are lost to reshards and synchronization overhead.

### 2.2 The Primitive Mechanisms Behind Sharded Data Parallelism
**Engine 13: scope and objective.** Provide a self-contained treatment of **2.2 The Primitive Mechanisms Behind Sharded Data Parallelism** with definitions, mechanism, cost model, implementation choices, failure modes, and decision rules.

1. **Conceptual model.**
   - 2.2 The Primitive Mechanisms Behind Sharded Data Parallelism is framed as a tensor-placement transformation over the device mesh. Inputs, outputs, and dependency edges are explicitly identified.
   - Correctness condition: transformed execution must preserve numerical equivalence up to floating-point non-associativity and stochastic layers.
2. **Mathematical formulation.**
   - Let local shard size be $n_{local}=n/p$ along the chosen axis. Memory scales as $O(n_{local})$ for sharded states and $O(n)$ for replicated states.
   - Communication lower bound (single phase): $\Omega(n\,\beta)$ payload plus startup term $\Omega(\alpha)$.
3. **Execution recipe.**
   - Precondition checks: mesh groups, dtype policy, seed policy, checkpoint layout metadata.
   - Runtime sequence: schedule compute kernels, issue collectives, place synchronization barriers only on true dependencies, and opportunistically overlap.
4. **Performance engineering.**
   - Increase arithmetic intensity using fused kernels and shape-friendly GEMMs.
   - Reduce bytes with precision controls (FP8/quantized links) while managing scales and error feedback.
   - Place high-frequency collectives on fast links (e.g., intra-node) whenever possible.
5. **Failure modes and diagnostics.**
   - Symptoms: stragglers, tail latency spikes, allocator fragmentation, imbalance, and collective stalls.
   - Diagnostics: per-rank timeline traces, bucket size sweeps, microbatch sweep, and reshard-count audits.
6. **Decision rubric.**
   - Prefer the smallest parallel axis set meeting memory and throughput targets.
   - Add axes incrementally; after each addition, re-measure MFU, link utilization, and p95/p99 latency.

**Practical takeaway.** This subsection should be implemented as a topology-aware schedule with explicit tensor-layout contracts; otherwise theoretical gains are lost to reshards and synchronization overhead.

### 2.3 ZeRO, ZeRO++, and Distributed Optimizers
**Engine 14: scope and objective.** Provide a self-contained treatment of **2.3 ZeRO, ZeRO++, and Distributed Optimizers** with definitions, mechanism, cost model, implementation choices, failure modes, and decision rules.

1. **Conceptual model.**
   - 2.3 ZeRO, ZeRO++, and Distributed Optimizers is framed as a tensor-placement transformation over the device mesh. Inputs, outputs, and dependency edges are explicitly identified.
   - Correctness condition: transformed execution must preserve numerical equivalence up to floating-point non-associativity and stochastic layers.
2. **Mathematical formulation.**
   - Let local shard size be $n_{local}=n/p$ along the chosen axis. Memory scales as $O(n_{local})$ for sharded states and $O(n)$ for replicated states.
   - Communication lower bound (single phase): $\Omega(n\,\beta)$ payload plus startup term $\Omega(\alpha)$.
3. **Execution recipe.**
   - Precondition checks: mesh groups, dtype policy, seed policy, checkpoint layout metadata.
   - Runtime sequence: schedule compute kernels, issue collectives, place synchronization barriers only on true dependencies, and opportunistically overlap.
4. **Performance engineering.**
   - Increase arithmetic intensity using fused kernels and shape-friendly GEMMs.
   - Reduce bytes with precision controls (FP8/quantized links) while managing scales and error feedback.
   - Place high-frequency collectives on fast links (e.g., intra-node) whenever possible.
5. **Failure modes and diagnostics.**
   - Symptoms: stragglers, tail latency spikes, allocator fragmentation, imbalance, and collective stalls.
   - Diagnostics: per-rank timeline traces, bucket size sweeps, microbatch sweep, and reshard-count audits.
6. **Decision rubric.**
   - Prefer the smallest parallel axis set meeting memory and throughput targets.
   - Add axes incrementally; after each addition, re-measure MFU, link utilization, and p95/p99 latency.

**Practical takeaway.** This subsection should be implemented as a topology-aware schedule with explicit tensor-layout contracts; otherwise theoretical gains are lost to reshards and synchronization overhead.

### 2.4 FSDP and FSDP2 Execution
**Engine 15: scope and objective.** Provide a self-contained treatment of **2.4 FSDP and FSDP2 Execution** with definitions, mechanism, cost model, implementation choices, failure modes, and decision rules.

1. **Conceptual model.**
   - 2.4 FSDP and FSDP2 Execution is framed as a tensor-placement transformation over the device mesh. Inputs, outputs, and dependency edges are explicitly identified.
   - Correctness condition: transformed execution must preserve numerical equivalence up to floating-point non-associativity and stochastic layers.
2. **Mathematical formulation.**
   - Let local shard size be $n_{local}=n/p$ along the chosen axis. Memory scales as $O(n_{local})$ for sharded states and $O(n)$ for replicated states.
   - Communication lower bound (single phase): $\Omega(n\,\beta)$ payload plus startup term $\Omega(\alpha)$.
3. **Execution recipe.**
   - Precondition checks: mesh groups, dtype policy, seed policy, checkpoint layout metadata.
   - Runtime sequence: schedule compute kernels, issue collectives, place synchronization barriers only on true dependencies, and opportunistically overlap.
4. **Performance engineering.**
   - Increase arithmetic intensity using fused kernels and shape-friendly GEMMs.
   - Reduce bytes with precision controls (FP8/quantized links) while managing scales and error feedback.
   - Place high-frequency collectives on fast links (e.g., intra-node) whenever possible.
5. **Failure modes and diagnostics.**
   - Symptoms: stragglers, tail latency spikes, allocator fragmentation, imbalance, and collective stalls.
   - Diagnostics: per-rank timeline traces, bucket size sweeps, microbatch sweep, and reshard-count audits.
6. **Decision rubric.**
   - Prefer the smallest parallel axis set meeting memory and throughput targets.
   - Add axes incrementally; after each addition, re-measure MFU, link utilization, and p95/p99 latency.

**Practical takeaway.** This subsection should be implemented as a topology-aware schedule with explicit tensor-layout contracts; otherwise theoretical gains are lost to reshards and synchronization overhead.

### 2.5 HSDP: Hybrid Sharded Data Parallelism
**Engine 16: scope and objective.** Provide a self-contained treatment of **2.5 HSDP: Hybrid Sharded Data Parallelism** with definitions, mechanism, cost model, implementation choices, failure modes, and decision rules.

1. **Conceptual model.**
   - 2.5 HSDP: Hybrid Sharded Data Parallelism is framed as a tensor-placement transformation over the device mesh. Inputs, outputs, and dependency edges are explicitly identified.
   - Correctness condition: transformed execution must preserve numerical equivalence up to floating-point non-associativity and stochastic layers.
2. **Mathematical formulation.**
   - Let local shard size be $n_{local}=n/p$ along the chosen axis. Memory scales as $O(n_{local})$ for sharded states and $O(n)$ for replicated states.
   - Communication lower bound (single phase): $\Omega(n\,\beta)$ payload plus startup term $\Omega(\alpha)$.
3. **Execution recipe.**
   - Precondition checks: mesh groups, dtype policy, seed policy, checkpoint layout metadata.
   - Runtime sequence: schedule compute kernels, issue collectives, place synchronization barriers only on true dependencies, and opportunistically overlap.
4. **Performance engineering.**
   - Increase arithmetic intensity using fused kernels and shape-friendly GEMMs.
   - Reduce bytes with precision controls (FP8/quantized links) while managing scales and error feedback.
   - Place high-frequency collectives on fast links (e.g., intra-node) whenever possible.
5. **Failure modes and diagnostics.**
   - Symptoms: stragglers, tail latency spikes, allocator fragmentation, imbalance, and collective stalls.
   - Diagnostics: per-rank timeline traces, bucket size sweeps, microbatch sweep, and reshard-count audits.
6. **Decision rubric.**
   - Prefer the smallest parallel axis set meeting memory and throughput targets.
   - Add axes incrementally; after each addition, re-measure MFU, link utilization, and p95/p99 latency.

**Practical takeaway.** This subsection should be implemented as a topology-aware schedule with explicit tensor-layout contracts; otherwise theoretical gains are lost to reshards and synchronization overhead.

### 2.6 Offload: CPU and NVMe as Additional State Tiers
**Engine 17: scope and objective.** Provide a self-contained treatment of **2.6 Offload: CPU and NVMe as Additional State Tiers** with definitions, mechanism, cost model, implementation choices, failure modes, and decision rules.

1. **Conceptual model.**
   - 2.6 Offload: CPU and NVMe as Additional State Tiers is framed as a tensor-placement transformation over the device mesh. Inputs, outputs, and dependency edges are explicitly identified.
   - Correctness condition: transformed execution must preserve numerical equivalence up to floating-point non-associativity and stochastic layers.
2. **Mathematical formulation.**
   - Let local shard size be $n_{local}=n/p$ along the chosen axis. Memory scales as $O(n_{local})$ for sharded states and $O(n)$ for replicated states.
   - Communication lower bound (single phase): $\Omega(n\,\beta)$ payload plus startup term $\Omega(\alpha)$.
3. **Execution recipe.**
   - Precondition checks: mesh groups, dtype policy, seed policy, checkpoint layout metadata.
   - Runtime sequence: schedule compute kernels, issue collectives, place synchronization barriers only on true dependencies, and opportunistically overlap.
4. **Performance engineering.**
   - Increase arithmetic intensity using fused kernels and shape-friendly GEMMs.
   - Reduce bytes with precision controls (FP8/quantized links) while managing scales and error feedback.
   - Place high-frequency collectives on fast links (e.g., intra-node) whenever possible.
5. **Failure modes and diagnostics.**
   - Symptoms: stragglers, tail latency spikes, allocator fragmentation, imbalance, and collective stalls.
   - Diagnostics: per-rank timeline traces, bucket size sweeps, microbatch sweep, and reshard-count audits.
6. **Decision rubric.**
   - Prefer the smallest parallel axis set meeting memory and throughput targets.
   - Add axes incrementally; after each addition, re-measure MFU, link utilization, and p95/p99 latency.

**Practical takeaway.** This subsection should be implemented as a topology-aware schedule with explicit tensor-layout contracts; otherwise theoretical gains are lost to reshards and synchronization overhead.

### 2.7 Low Precision, Quantized Communication, and Platform Support
**Engine 18: scope and objective.** Provide a self-contained treatment of **2.7 Low Precision, Quantized Communication, and Platform Support** with definitions, mechanism, cost model, implementation choices, failure modes, and decision rules.

1. **Conceptual model.**
   - 2.7 Low Precision, Quantized Communication, and Platform Support is framed as a tensor-placement transformation over the device mesh. Inputs, outputs, and dependency edges are explicitly identified.
   - Correctness condition: transformed execution must preserve numerical equivalence up to floating-point non-associativity and stochastic layers.
2. **Mathematical formulation.**
   - Let local shard size be $n_{local}=n/p$ along the chosen axis. Memory scales as $O(n_{local})$ for sharded states and $O(n)$ for replicated states.
   - Communication lower bound (single phase): $\Omega(n\,\beta)$ payload plus startup term $\Omega(\alpha)$.
3. **Execution recipe.**
   - Precondition checks: mesh groups, dtype policy, seed policy, checkpoint layout metadata.
   - Runtime sequence: schedule compute kernels, issue collectives, place synchronization barriers only on true dependencies, and opportunistically overlap.
4. **Performance engineering.**
   - Increase arithmetic intensity using fused kernels and shape-friendly GEMMs.
   - Reduce bytes with precision controls (FP8/quantized links) while managing scales and error feedback.
   - Place high-frequency collectives on fast links (e.g., intra-node) whenever possible.
5. **Failure modes and diagnostics.**
   - Symptoms: stragglers, tail latency spikes, allocator fragmentation, imbalance, and collective stalls.
   - Diagnostics: per-rank timeline traces, bucket size sweeps, microbatch sweep, and reshard-count audits.
6. **Decision rubric.**
   - Prefer the smallest parallel axis set meeting memory and throughput targets.
   - Add axes incrementally; after each addition, re-measure MFU, link utilization, and p95/p99 latency.

**Practical takeaway.** This subsection should be implemented as a topology-aware schedule with explicit tensor-layout contracts; otherwise theoretical gains are lost to reshards and synchronization overhead.

### 2.8 Choosing Among DDP, FSDP2, ZeRO, ZeRO++, HSDP, and Offload
**Engine 19: scope and objective.** Provide a self-contained treatment of **2.8 Choosing Among DDP, FSDP2, ZeRO, ZeRO++, HSDP, and Offload** with definitions, mechanism, cost model, implementation choices, failure modes, and decision rules.

1. **Conceptual model.**
   - 2.8 Choosing Among DDP, FSDP2, ZeRO, ZeRO++, HSDP, and Offload is framed as a tensor-placement transformation over the device mesh. Inputs, outputs, and dependency edges are explicitly identified.
   - Correctness condition: transformed execution must preserve numerical equivalence up to floating-point non-associativity and stochastic layers.
2. **Mathematical formulation.**
   - Let local shard size be $n_{local}=n/p$ along the chosen axis. Memory scales as $O(n_{local})$ for sharded states and $O(n)$ for replicated states.
   - Communication lower bound (single phase): $\Omega(n\,\beta)$ payload plus startup term $\Omega(\alpha)$.
3. **Execution recipe.**
   - Precondition checks: mesh groups, dtype policy, seed policy, checkpoint layout metadata.
   - Runtime sequence: schedule compute kernels, issue collectives, place synchronization barriers only on true dependencies, and opportunistically overlap.
4. **Performance engineering.**
   - Increase arithmetic intensity using fused kernels and shape-friendly GEMMs.
   - Reduce bytes with precision controls (FP8/quantized links) while managing scales and error feedback.
   - Place high-frequency collectives on fast links (e.g., intra-node) whenever possible.
5. **Failure modes and diagnostics.**
   - Symptoms: stragglers, tail latency spikes, allocator fragmentation, imbalance, and collective stalls.
   - Diagnostics: per-rank timeline traces, bucket size sweeps, microbatch sweep, and reshard-count audits.
6. **Decision rubric.**
   - Prefer the smallest parallel axis set meeting memory and throughput targets.
   - Add axes incrementally; after each addition, re-measure MFU, link utilization, and p95/p99 latency.

**Practical takeaway.** This subsection should be implemented as a topology-aware schedule with explicit tensor-layout contracts; otherwise theoretical gains are lost to reshards and synchronization overhead.

### 2.9 Sources and Lineage
**Engine 20: scope and objective.** Provide a self-contained treatment of **2.9 Sources and Lineage** with definitions, mechanism, cost model, implementation choices, failure modes, and decision rules.

1. **Conceptual model.**
   - 2.9 Sources and Lineage is framed as a tensor-placement transformation over the device mesh. Inputs, outputs, and dependency edges are explicitly identified.
   - Correctness condition: transformed execution must preserve numerical equivalence up to floating-point non-associativity and stochastic layers.
2. **Mathematical formulation.**
   - Let local shard size be $n_{local}=n/p$ along the chosen axis. Memory scales as $O(n_{local})$ for sharded states and $O(n)$ for replicated states.
   - Communication lower bound (single phase): $\Omega(n\,\beta)$ payload plus startup term $\Omega(\alpha)$.
3. **Execution recipe.**
   - Precondition checks: mesh groups, dtype policy, seed policy, checkpoint layout metadata.
   - Runtime sequence: schedule compute kernels, issue collectives, place synchronization barriers only on true dependencies, and opportunistically overlap.
4. **Performance engineering.**
   - Increase arithmetic intensity using fused kernels and shape-friendly GEMMs.
   - Reduce bytes with precision controls (FP8/quantized links) while managing scales and error feedback.
   - Place high-frequency collectives on fast links (e.g., intra-node) whenever possible.
5. **Failure modes and diagnostics.**
   - Symptoms: stragglers, tail latency spikes, allocator fragmentation, imbalance, and collective stalls.
   - Diagnostics: per-rank timeline traces, bucket size sweeps, microbatch sweep, and reshard-count audits.
6. **Decision rubric.**
   - Prefer the smallest parallel axis set meeting memory and throughput targets.
   - Add axes incrementally; after each addition, re-measure MFU, link utilization, and p95/p99 latency.

**Practical takeaway.** This subsection should be implemented as a topology-aware schedule with explicit tensor-layout contracts; otherwise theoretical gains are lost to reshards and synchronization overhead.

### 2.10 Caveats
**Engine 21: scope and objective.** Provide a self-contained treatment of **2.10 Caveats** with definitions, mechanism, cost model, implementation choices, failure modes, and decision rules.

1. **Conceptual model.**
   - 2.10 Caveats is framed as a tensor-placement transformation over the device mesh. Inputs, outputs, and dependency edges are explicitly identified.
   - Correctness condition: transformed execution must preserve numerical equivalence up to floating-point non-associativity and stochastic layers.
2. **Mathematical formulation.**
   - Let local shard size be $n_{local}=n/p$ along the chosen axis. Memory scales as $O(n_{local})$ for sharded states and $O(n)$ for replicated states.
   - Communication lower bound (single phase): $\Omega(n\,\beta)$ payload plus startup term $\Omega(\alpha)$.
3. **Execution recipe.**
   - Precondition checks: mesh groups, dtype policy, seed policy, checkpoint layout metadata.
   - Runtime sequence: schedule compute kernels, issue collectives, place synchronization barriers only on true dependencies, and opportunistically overlap.
4. **Performance engineering.**
   - Increase arithmetic intensity using fused kernels and shape-friendly GEMMs.
   - Reduce bytes with precision controls (FP8/quantized links) while managing scales and error feedback.
   - Place high-frequency collectives on fast links (e.g., intra-node) whenever possible.
5. **Failure modes and diagnostics.**
   - Symptoms: stragglers, tail latency spikes, allocator fragmentation, imbalance, and collective stalls.
   - Diagnostics: per-rank timeline traces, bucket size sweeps, microbatch sweep, and reshard-count audits.
6. **Decision rubric.**
   - Prefer the smallest parallel axis set meeting memory and throughput targets.
   - Add axes incrementally; after each addition, re-measure MFU, link utilization, and p95/p99 latency.

**Practical takeaway.** This subsection should be implemented as a topology-aware schedule with explicit tensor-layout contracts; otherwise theoretical gains are lost to reshards and synchronization overhead.

## 3. Tensor Parallelism, Sequence-Parallel Activations, and Vocabulary Parallelism
**Engine 22: scope and objective.** Provide a self-contained treatment of **3. Tensor Parallelism, Sequence-Parallel Activations, and Vocabulary Parallelism** with definitions, mechanism, cost model, implementation choices, failure modes, and decision rules.

1. **Conceptual model.**
   - 3. Tensor Parallelism, Sequence-Parallel Activations, and Vocabulary Parallelism is framed as a tensor-placement transformation over the device mesh. Inputs, outputs, and dependency edges are explicitly identified.
   - Correctness condition: transformed execution must preserve numerical equivalence up to floating-point non-associativity and stochastic layers.
2. **Mathematical formulation.**
   - Let local shard size be $n_{local}=n/p$ along the chosen axis. Memory scales as $O(n_{local})$ for sharded states and $O(n)$ for replicated states.
   - Communication lower bound (single phase): $\Omega(n\,\beta)$ payload plus startup term $\Omega(\alpha)$.
3. **Execution recipe.**
   - Precondition checks: mesh groups, dtype policy, seed policy, checkpoint layout metadata.
   - Runtime sequence: schedule compute kernels, issue collectives, place synchronization barriers only on true dependencies, and opportunistically overlap.
4. **Performance engineering.**
   - Increase arithmetic intensity using fused kernels and shape-friendly GEMMs.
   - Reduce bytes with precision controls (FP8/quantized links) while managing scales and error feedback.
   - Place high-frequency collectives on fast links (e.g., intra-node) whenever possible.
5. **Failure modes and diagnostics.**
   - Symptoms: stragglers, tail latency spikes, allocator fragmentation, imbalance, and collective stalls.
   - Diagnostics: per-rank timeline traces, bucket size sweeps, microbatch sweep, and reshard-count audits.
6. **Decision rubric.**
   - Prefer the smallest parallel axis set meeting memory and throughput targets.
   - Add axes incrementally; after each addition, re-measure MFU, link utilization, and p95/p99 latency.

**Practical takeaway.** This subsection should be implemented as a topology-aware schedule with explicit tensor-layout contracts; otherwise theoretical gains are lost to reshards and synchronization overhead.

### 3.1 TP-Specific Notation and Layout Contract
**Engine 23: scope and objective.** Provide a self-contained treatment of **3.1 TP-Specific Notation and Layout Contract** with definitions, mechanism, cost model, implementation choices, failure modes, and decision rules.

1. **Conceptual model.**
   - 3.1 TP-Specific Notation and Layout Contract is framed as a tensor-placement transformation over the device mesh. Inputs, outputs, and dependency edges are explicitly identified.
   - Correctness condition: transformed execution must preserve numerical equivalence up to floating-point non-associativity and stochastic layers.
2. **Mathematical formulation.**
   - Let local shard size be $n_{local}=n/p$ along the chosen axis. Memory scales as $O(n_{local})$ for sharded states and $O(n)$ for replicated states.
   - Communication lower bound (single phase): $\Omega(n\,\beta)$ payload plus startup term $\Omega(\alpha)$.
3. **Execution recipe.**
   - Precondition checks: mesh groups, dtype policy, seed policy, checkpoint layout metadata.
   - Runtime sequence: schedule compute kernels, issue collectives, place synchronization barriers only on true dependencies, and opportunistically overlap.
4. **Performance engineering.**
   - Increase arithmetic intensity using fused kernels and shape-friendly GEMMs.
   - Reduce bytes with precision controls (FP8/quantized links) while managing scales and error feedback.
   - Place high-frequency collectives on fast links (e.g., intra-node) whenever possible.
5. **Failure modes and diagnostics.**
   - Symptoms: stragglers, tail latency spikes, allocator fragmentation, imbalance, and collective stalls.
   - Diagnostics: per-rank timeline traces, bucket size sweeps, microbatch sweep, and reshard-count audits.
6. **Decision rubric.**
   - Prefer the smallest parallel axis set meeting memory and throughput targets.
   - Add axes incrementally; after each addition, re-measure MFU, link utilization, and p95/p99 latency.

**Practical takeaway.** This subsection should be implemented as a topology-aware schedule with explicit tensor-layout contracts; otherwise theoretical gains are lost to reshards and synchronization overhead.

### 3.2 Column-Parallel Linear Layers
**Engine 24: scope and objective.** Provide a self-contained treatment of **3.2 Column-Parallel Linear Layers** with definitions, mechanism, cost model, implementation choices, failure modes, and decision rules.

1. **Conceptual model.**
   - 3.2 Column-Parallel Linear Layers is framed as a tensor-placement transformation over the device mesh. Inputs, outputs, and dependency edges are explicitly identified.
   - Correctness condition: transformed execution must preserve numerical equivalence up to floating-point non-associativity and stochastic layers.
2. **Mathematical formulation.**
   - Let local shard size be $n_{local}=n/p$ along the chosen axis. Memory scales as $O(n_{local})$ for sharded states and $O(n)$ for replicated states.
   - Communication lower bound (single phase): $\Omega(n\,\beta)$ payload plus startup term $\Omega(\alpha)$.
3. **Execution recipe.**
   - Precondition checks: mesh groups, dtype policy, seed policy, checkpoint layout metadata.
   - Runtime sequence: schedule compute kernels, issue collectives, place synchronization barriers only on true dependencies, and opportunistically overlap.
4. **Performance engineering.**
   - Increase arithmetic intensity using fused kernels and shape-friendly GEMMs.
   - Reduce bytes with precision controls (FP8/quantized links) while managing scales and error feedback.
   - Place high-frequency collectives on fast links (e.g., intra-node) whenever possible.
5. **Failure modes and diagnostics.**
   - Symptoms: stragglers, tail latency spikes, allocator fragmentation, imbalance, and collective stalls.
   - Diagnostics: per-rank timeline traces, bucket size sweeps, microbatch sweep, and reshard-count audits.
6. **Decision rubric.**
   - Prefer the smallest parallel axis set meeting memory and throughput targets.
   - Add axes incrementally; after each addition, re-measure MFU, link utilization, and p95/p99 latency.

**Practical takeaway.** This subsection should be implemented as a topology-aware schedule with explicit tensor-layout contracts; otherwise theoretical gains are lost to reshards and synchronization overhead.

### 3.3 Row-Parallel Linear Layers
**Engine 25: scope and objective.** Provide a self-contained treatment of **3.3 Row-Parallel Linear Layers** with definitions, mechanism, cost model, implementation choices, failure modes, and decision rules.

1. **Conceptual model.**
   - 3.3 Row-Parallel Linear Layers is framed as a tensor-placement transformation over the device mesh. Inputs, outputs, and dependency edges are explicitly identified.
   - Correctness condition: transformed execution must preserve numerical equivalence up to floating-point non-associativity and stochastic layers.
2. **Mathematical formulation.**
   - Let local shard size be $n_{local}=n/p$ along the chosen axis. Memory scales as $O(n_{local})$ for sharded states and $O(n)$ for replicated states.
   - Communication lower bound (single phase): $\Omega(n\,\beta)$ payload plus startup term $\Omega(\alpha)$.
3. **Execution recipe.**
   - Precondition checks: mesh groups, dtype policy, seed policy, checkpoint layout metadata.
   - Runtime sequence: schedule compute kernels, issue collectives, place synchronization barriers only on true dependencies, and opportunistically overlap.
4. **Performance engineering.**
   - Increase arithmetic intensity using fused kernels and shape-friendly GEMMs.
   - Reduce bytes with precision controls (FP8/quantized links) while managing scales and error feedback.
   - Place high-frequency collectives on fast links (e.g., intra-node) whenever possible.
5. **Failure modes and diagnostics.**
   - Symptoms: stragglers, tail latency spikes, allocator fragmentation, imbalance, and collective stalls.
   - Diagnostics: per-rank timeline traces, bucket size sweeps, microbatch sweep, and reshard-count audits.
6. **Decision rubric.**
   - Prefer the smallest parallel axis set meeting memory and throughput targets.
   - Add axes incrementally; after each addition, re-measure MFU, link utilization, and p95/p99 latency.

**Practical takeaway.** This subsection should be implemented as a topology-aware schedule with explicit tensor-layout contracts; otherwise theoretical gains are lost to reshards and synchronization overhead.

### 3.4 Transformer Block Recipes: MLP and Attention
**Engine 26: scope and objective.** Provide a self-contained treatment of **3.4 Transformer Block Recipes: MLP and Attention** with definitions, mechanism, cost model, implementation choices, failure modes, and decision rules.

1. **Conceptual model.**
   - 3.4 Transformer Block Recipes: MLP and Attention is framed as a tensor-placement transformation over the device mesh. Inputs, outputs, and dependency edges are explicitly identified.
   - Correctness condition: transformed execution must preserve numerical equivalence up to floating-point non-associativity and stochastic layers.
2. **Mathematical formulation.**
   - Let local shard size be $n_{local}=n/p$ along the chosen axis. Memory scales as $O(n_{local})$ for sharded states and $O(n)$ for replicated states.
   - Communication lower bound (single phase): $\Omega(n\,\beta)$ payload plus startup term $\Omega(\alpha)$.
3. **Execution recipe.**
   - Precondition checks: mesh groups, dtype policy, seed policy, checkpoint layout metadata.
   - Runtime sequence: schedule compute kernels, issue collectives, place synchronization barriers only on true dependencies, and opportunistically overlap.
4. **Performance engineering.**
   - Increase arithmetic intensity using fused kernels and shape-friendly GEMMs.
   - Reduce bytes with precision controls (FP8/quantized links) while managing scales and error feedback.
   - Place high-frequency collectives on fast links (e.g., intra-node) whenever possible.
5. **Failure modes and diagnostics.**
   - Symptoms: stragglers, tail latency spikes, allocator fragmentation, imbalance, and collective stalls.
   - Diagnostics: per-rank timeline traces, bucket size sweeps, microbatch sweep, and reshard-count audits.
6. **Decision rubric.**
   - Prefer the smallest parallel axis set meeting memory and throughput targets.
   - Add axes incrementally; after each addition, re-measure MFU, link utilization, and p95/p99 latency.

**Practical takeaway.** This subsection should be implemented as a topology-aware schedule with explicit tensor-layout contracts; otherwise theoretical gains are lost to reshards and synchronization overhead.

#### MLP / SwiGLU
**Engine 27: scope and objective.** Provide a self-contained treatment of **MLP / SwiGLU** with definitions, mechanism, cost model, implementation choices, failure modes, and decision rules.

1. **Conceptual model.**
   - MLP / SwiGLU is framed as a tensor-placement transformation over the device mesh. Inputs, outputs, and dependency edges are explicitly identified.
   - Correctness condition: transformed execution must preserve numerical equivalence up to floating-point non-associativity and stochastic layers.
2. **Mathematical formulation.**
   - Let local shard size be $n_{local}=n/p$ along the chosen axis. Memory scales as $O(n_{local})$ for sharded states and $O(n)$ for replicated states.
   - Communication lower bound (single phase): $\Omega(n\,\beta)$ payload plus startup term $\Omega(\alpha)$.
3. **Execution recipe.**
   - Precondition checks: mesh groups, dtype policy, seed policy, checkpoint layout metadata.
   - Runtime sequence: schedule compute kernels, issue collectives, place synchronization barriers only on true dependencies, and opportunistically overlap.
4. **Performance engineering.**
   - Increase arithmetic intensity using fused kernels and shape-friendly GEMMs.
   - Reduce bytes with precision controls (FP8/quantized links) while managing scales and error feedback.
   - Place high-frequency collectives on fast links (e.g., intra-node) whenever possible.
5. **Failure modes and diagnostics.**
   - Symptoms: stragglers, tail latency spikes, allocator fragmentation, imbalance, and collective stalls.
   - Diagnostics: per-rank timeline traces, bucket size sweeps, microbatch sweep, and reshard-count audits.
6. **Decision rubric.**
   - Prefer the smallest parallel axis set meeting memory and throughput targets.
   - Add axes incrementally; after each addition, re-measure MFU, link utilization, and p95/p99 latency.

**Practical takeaway.** This subsection should be implemented as a topology-aware schedule with explicit tensor-layout contracts; otherwise theoretical gains are lost to reshards and synchronization overhead.

#### Attention
**Engine 28: scope and objective.** Provide a self-contained treatment of **Attention** with definitions, mechanism, cost model, implementation choices, failure modes, and decision rules.

1. **Conceptual model.**
   - Attention is framed as a tensor-placement transformation over the device mesh. Inputs, outputs, and dependency edges are explicitly identified.
   - Correctness condition: transformed execution must preserve numerical equivalence up to floating-point non-associativity and stochastic layers.
2. **Mathematical formulation.**
   - Let local shard size be $n_{local}=n/p$ along the chosen axis. Memory scales as $O(n_{local})$ for sharded states and $O(n)$ for replicated states.
   - Communication lower bound (single phase): $\Omega(n\,\beta)$ payload plus startup term $\Omega(\alpha)$.
3. **Execution recipe.**
   - Precondition checks: mesh groups, dtype policy, seed policy, checkpoint layout metadata.
   - Runtime sequence: schedule compute kernels, issue collectives, place synchronization barriers only on true dependencies, and opportunistically overlap.
4. **Performance engineering.**
   - Increase arithmetic intensity using fused kernels and shape-friendly GEMMs.
   - Reduce bytes with precision controls (FP8/quantized links) while managing scales and error feedback.
   - Place high-frequency collectives on fast links (e.g., intra-node) whenever possible.
5. **Failure modes and diagnostics.**
   - Symptoms: stragglers, tail latency spikes, allocator fragmentation, imbalance, and collective stalls.
   - Diagnostics: per-rank timeline traces, bucket size sweeps, microbatch sweep, and reshard-count audits.
6. **Decision rubric.**
   - Prefer the smallest parallel axis set meeting memory and throughput targets.
   - Add axes incrementally; after each addition, re-measure MFU, link utilization, and p95/p99 latency.

**Practical takeaway.** This subsection should be implemented as a topology-aware schedule with explicit tensor-layout contracts; otherwise theoretical gains are lost to reshards and synchronization overhead.

### 3.5 Sequence-Parallel Activations as a TP Companion
**Engine 29: scope and objective.** Provide a self-contained treatment of **3.5 Sequence-Parallel Activations as a TP Companion** with definitions, mechanism, cost model, implementation choices, failure modes, and decision rules.

1. **Conceptual model.**
   - 3.5 Sequence-Parallel Activations as a TP Companion is framed as a tensor-placement transformation over the device mesh. Inputs, outputs, and dependency edges are explicitly identified.
   - Correctness condition: transformed execution must preserve numerical equivalence up to floating-point non-associativity and stochastic layers.
2. **Mathematical formulation.**
   - Let local shard size be $n_{local}=n/p$ along the chosen axis. Memory scales as $O(n_{local})$ for sharded states and $O(n)$ for replicated states.
   - Communication lower bound (single phase): $\Omega(n\,\beta)$ payload plus startup term $\Omega(\alpha)$.
3. **Execution recipe.**
   - Precondition checks: mesh groups, dtype policy, seed policy, checkpoint layout metadata.
   - Runtime sequence: schedule compute kernels, issue collectives, place synchronization barriers only on true dependencies, and opportunistically overlap.
4. **Performance engineering.**
   - Increase arithmetic intensity using fused kernels and shape-friendly GEMMs.
   - Reduce bytes with precision controls (FP8/quantized links) while managing scales and error feedback.
   - Place high-frequency collectives on fast links (e.g., intra-node) whenever possible.
5. **Failure modes and diagnostics.**
   - Symptoms: stragglers, tail latency spikes, allocator fragmentation, imbalance, and collective stalls.
   - Diagnostics: per-rank timeline traces, bucket size sweeps, microbatch sweep, and reshard-count audits.
6. **Decision rubric.**
   - Prefer the smallest parallel axis set meeting memory and throughput targets.
   - Add axes incrementally; after each addition, re-measure MFU, link utilization, and p95/p99 latency.

**Practical takeaway.** This subsection should be implemented as a topology-aware schedule with explicit tensor-layout contracts; otherwise theoretical gains are lost to reshards and synchronization overhead.

### 3.6 Vocabulary Parallelism
**Engine 30: scope and objective.** Provide a self-contained treatment of **3.6 Vocabulary Parallelism** with definitions, mechanism, cost model, implementation choices, failure modes, and decision rules.

1. **Conceptual model.**
   - 3.6 Vocabulary Parallelism is framed as a tensor-placement transformation over the device mesh. Inputs, outputs, and dependency edges are explicitly identified.
   - Correctness condition: transformed execution must preserve numerical equivalence up to floating-point non-associativity and stochastic layers.
2. **Mathematical formulation.**
   - Let local shard size be $n_{local}=n/p$ along the chosen axis. Memory scales as $O(n_{local})$ for sharded states and $O(n)$ for replicated states.
   - Communication lower bound (single phase): $\Omega(n\,\beta)$ payload plus startup term $\Omega(\alpha)$.
3. **Execution recipe.**
   - Precondition checks: mesh groups, dtype policy, seed policy, checkpoint layout metadata.
   - Runtime sequence: schedule compute kernels, issue collectives, place synchronization barriers only on true dependencies, and opportunistically overlap.
4. **Performance engineering.**
   - Increase arithmetic intensity using fused kernels and shape-friendly GEMMs.
   - Reduce bytes with precision controls (FP8/quantized links) while managing scales and error feedback.
   - Place high-frequency collectives on fast links (e.g., intra-node) whenever possible.
5. **Failure modes and diagnostics.**
   - Symptoms: stragglers, tail latency spikes, allocator fragmentation, imbalance, and collective stalls.
   - Diagnostics: per-rank timeline traces, bucket size sweeps, microbatch sweep, and reshard-count audits.
6. **Decision rubric.**
   - Prefer the smallest parallel axis set meeting memory and throughput targets.
   - Add axes incrementally; after each addition, re-measure MFU, link utilization, and p95/p99 latency.

**Practical takeaway.** This subsection should be implemented as a topology-aware schedule with explicit tensor-layout contracts; otherwise theoretical gains are lost to reshards and synchronization overhead.

#### Input Embedding
**Engine 31: scope and objective.** Provide a self-contained treatment of **Input Embedding** with definitions, mechanism, cost model, implementation choices, failure modes, and decision rules.

1. **Conceptual model.**
   - Input Embedding is framed as a tensor-placement transformation over the device mesh. Inputs, outputs, and dependency edges are explicitly identified.
   - Correctness condition: transformed execution must preserve numerical equivalence up to floating-point non-associativity and stochastic layers.
2. **Mathematical formulation.**
   - Let local shard size be $n_{local}=n/p$ along the chosen axis. Memory scales as $O(n_{local})$ for sharded states and $O(n)$ for replicated states.
   - Communication lower bound (single phase): $\Omega(n\,\beta)$ payload plus startup term $\Omega(\alpha)$.
3. **Execution recipe.**
   - Precondition checks: mesh groups, dtype policy, seed policy, checkpoint layout metadata.
   - Runtime sequence: schedule compute kernels, issue collectives, place synchronization barriers only on true dependencies, and opportunistically overlap.
4. **Performance engineering.**
   - Increase arithmetic intensity using fused kernels and shape-friendly GEMMs.
   - Reduce bytes with precision controls (FP8/quantized links) while managing scales and error feedback.
   - Place high-frequency collectives on fast links (e.g., intra-node) whenever possible.
5. **Failure modes and diagnostics.**
   - Symptoms: stragglers, tail latency spikes, allocator fragmentation, imbalance, and collective stalls.
   - Diagnostics: per-rank timeline traces, bucket size sweeps, microbatch sweep, and reshard-count audits.
6. **Decision rubric.**
   - Prefer the smallest parallel axis set meeting memory and throughput targets.
   - Add axes incrementally; after each addition, re-measure MFU, link utilization, and p95/p99 latency.

**Practical takeaway.** This subsection should be implemented as a topology-aware schedule with explicit tensor-layout contracts; otherwise theoretical gains are lost to reshards and synchronization overhead.

#### LM Head and Cross-Entropy
**Engine 32: scope and objective.** Provide a self-contained treatment of **LM Head and Cross-Entropy** with definitions, mechanism, cost model, implementation choices, failure modes, and decision rules.

1. **Conceptual model.**
   - LM Head and Cross-Entropy is framed as a tensor-placement transformation over the device mesh. Inputs, outputs, and dependency edges are explicitly identified.
   - Correctness condition: transformed execution must preserve numerical equivalence up to floating-point non-associativity and stochastic layers.
2. **Mathematical formulation.**
   - Let local shard size be $n_{local}=n/p$ along the chosen axis. Memory scales as $O(n_{local})$ for sharded states and $O(n)$ for replicated states.
   - Communication lower bound (single phase): $\Omega(n\,\beta)$ payload plus startup term $\Omega(\alpha)$.
3. **Execution recipe.**
   - Precondition checks: mesh groups, dtype policy, seed policy, checkpoint layout metadata.
   - Runtime sequence: schedule compute kernels, issue collectives, place synchronization barriers only on true dependencies, and opportunistically overlap.
4. **Performance engineering.**
   - Increase arithmetic intensity using fused kernels and shape-friendly GEMMs.
   - Reduce bytes with precision controls (FP8/quantized links) while managing scales and error feedback.
   - Place high-frequency collectives on fast links (e.g., intra-node) whenever possible.
5. **Failure modes and diagnostics.**
   - Symptoms: stragglers, tail latency spikes, allocator fragmentation, imbalance, and collective stalls.
   - Diagnostics: per-rank timeline traces, bucket size sweeps, microbatch sweep, and reshard-count audits.
6. **Decision rubric.**
   - Prefer the smallest parallel axis set meeting memory and throughput targets.
   - Add axes incrementally; after each addition, re-measure MFU, link utilization, and p95/p99 latency.

**Practical takeaway.** This subsection should be implemented as a topology-aware schedule with explicit tensor-layout contracts; otherwise theoretical gains are lost to reshards and synchronization overhead.

### 3.7 Async Tensor Parallelism
**Engine 33: scope and objective.** Provide a self-contained treatment of **3.7 Async Tensor Parallelism** with definitions, mechanism, cost model, implementation choices, failure modes, and decision rules.

1. **Conceptual model.**
   - 3.7 Async Tensor Parallelism is framed as a tensor-placement transformation over the device mesh. Inputs, outputs, and dependency edges are explicitly identified.
   - Correctness condition: transformed execution must preserve numerical equivalence up to floating-point non-associativity and stochastic layers.
2. **Mathematical formulation.**
   - Let local shard size be $n_{local}=n/p$ along the chosen axis. Memory scales as $O(n_{local})$ for sharded states and $O(n)$ for replicated states.
   - Communication lower bound (single phase): $\Omega(n\,\beta)$ payload plus startup term $\Omega(\alpha)$.
3. **Execution recipe.**
   - Precondition checks: mesh groups, dtype policy, seed policy, checkpoint layout metadata.
   - Runtime sequence: schedule compute kernels, issue collectives, place synchronization barriers only on true dependencies, and opportunistically overlap.
4. **Performance engineering.**
   - Increase arithmetic intensity using fused kernels and shape-friendly GEMMs.
   - Reduce bytes with precision controls (FP8/quantized links) while managing scales and error feedback.
   - Place high-frequency collectives on fast links (e.g., intra-node) whenever possible.
5. **Failure modes and diagnostics.**
   - Symptoms: stragglers, tail latency spikes, allocator fragmentation, imbalance, and collective stalls.
   - Diagnostics: per-rank timeline traces, bucket size sweeps, microbatch sweep, and reshard-count audits.
6. **Decision rubric.**
   - Prefer the smallest parallel axis set meeting memory and throughput targets.
   - Add axes incrementally; after each addition, re-measure MFU, link utilization, and p95/p99 latency.

**Practical takeaway.** This subsection should be implemented as a topology-aware schedule with explicit tensor-layout contracts; otherwise theoretical gains are lost to reshards and synchronization overhead.

### 3.8 TP-Aware CUDA and Kernel Techniques
**Engine 34: scope and objective.** Provide a self-contained treatment of **3.8 TP-Aware CUDA and Kernel Techniques** with definitions, mechanism, cost model, implementation choices, failure modes, and decision rules.

1. **Conceptual model.**
   - 3.8 TP-Aware CUDA and Kernel Techniques is framed as a tensor-placement transformation over the device mesh. Inputs, outputs, and dependency edges are explicitly identified.
   - Correctness condition: transformed execution must preserve numerical equivalence up to floating-point non-associativity and stochastic layers.
2. **Mathematical formulation.**
   - Let local shard size be $n_{local}=n/p$ along the chosen axis. Memory scales as $O(n_{local})$ for sharded states and $O(n)$ for replicated states.
   - Communication lower bound (single phase): $\Omega(n\,\beta)$ payload plus startup term $\Omega(\alpha)$.
3. **Execution recipe.**
   - Precondition checks: mesh groups, dtype policy, seed policy, checkpoint layout metadata.
   - Runtime sequence: schedule compute kernels, issue collectives, place synchronization barriers only on true dependencies, and opportunistically overlap.
4. **Performance engineering.**
   - Increase arithmetic intensity using fused kernels and shape-friendly GEMMs.
   - Reduce bytes with precision controls (FP8/quantized links) while managing scales and error feedback.
   - Place high-frequency collectives on fast links (e.g., intra-node) whenever possible.
5. **Failure modes and diagnostics.**
   - Symptoms: stragglers, tail latency spikes, allocator fragmentation, imbalance, and collective stalls.
   - Diagnostics: per-rank timeline traces, bucket size sweeps, microbatch sweep, and reshard-count audits.
6. **Decision rubric.**
   - Prefer the smallest parallel axis set meeting memory and throughput targets.
   - Add axes incrementally; after each addition, re-measure MFU, link utilization, and p95/p99 latency.

**Practical takeaway.** This subsection should be implemented as a topology-aware schedule with explicit tensor-layout contracts; otherwise theoretical gains are lost to reshards and synchronization overhead.

#### 3.8.1 Local GEMM Shape Quality
**Engine 35: scope and objective.** Provide a self-contained treatment of **3.8.1 Local GEMM Shape Quality** with definitions, mechanism, cost model, implementation choices, failure modes, and decision rules.

1. **Conceptual model.**
   - 3.8.1 Local GEMM Shape Quality is framed as a tensor-placement transformation over the device mesh. Inputs, outputs, and dependency edges are explicitly identified.
   - Correctness condition: transformed execution must preserve numerical equivalence up to floating-point non-associativity and stochastic layers.
2. **Mathematical formulation.**
   - Let local shard size be $n_{local}=n/p$ along the chosen axis. Memory scales as $O(n_{local})$ for sharded states and $O(n)$ for replicated states.
   - Communication lower bound (single phase): $\Omega(n\,\beta)$ payload plus startup term $\Omega(\alpha)$.
3. **Execution recipe.**
   - Precondition checks: mesh groups, dtype policy, seed policy, checkpoint layout metadata.
   - Runtime sequence: schedule compute kernels, issue collectives, place synchronization barriers only on true dependencies, and opportunistically overlap.
4. **Performance engineering.**
   - Increase arithmetic intensity using fused kernels and shape-friendly GEMMs.
   - Reduce bytes with precision controls (FP8/quantized links) while managing scales and error feedback.
   - Place high-frequency collectives on fast links (e.g., intra-node) whenever possible.
5. **Failure modes and diagnostics.**
   - Symptoms: stragglers, tail latency spikes, allocator fragmentation, imbalance, and collective stalls.
   - Diagnostics: per-rank timeline traces, bucket size sweeps, microbatch sweep, and reshard-count audits.
6. **Decision rubric.**
   - Prefer the smallest parallel axis set meeting memory and throughput targets.
   - Add axes incrementally; after each addition, re-measure MFU, link utilization, and p95/p99 latency.

**Practical takeaway.** This subsection should be implemented as a topology-aware schedule with explicit tensor-layout contracts; otherwise theoretical gains are lost to reshards and synchronization overhead.

#### 3.8.2 Fused Epilogues and Layout-Stable Collectives
**Engine 36: scope and objective.** Provide a self-contained treatment of **3.8.2 Fused Epilogues and Layout-Stable Collectives** with definitions, mechanism, cost model, implementation choices, failure modes, and decision rules.

1. **Conceptual model.**
   - 3.8.2 Fused Epilogues and Layout-Stable Collectives is framed as a tensor-placement transformation over the device mesh. Inputs, outputs, and dependency edges are explicitly identified.
   - Correctness condition: transformed execution must preserve numerical equivalence up to floating-point non-associativity and stochastic layers.
2. **Mathematical formulation.**
   - Let local shard size be $n_{local}=n/p$ along the chosen axis. Memory scales as $O(n_{local})$ for sharded states and $O(n)$ for replicated states.
   - Communication lower bound (single phase): $\Omega(n\,\beta)$ payload plus startup term $\Omega(\alpha)$.
3. **Execution recipe.**
   - Precondition checks: mesh groups, dtype policy, seed policy, checkpoint layout metadata.
   - Runtime sequence: schedule compute kernels, issue collectives, place synchronization barriers only on true dependencies, and opportunistically overlap.
4. **Performance engineering.**
   - Increase arithmetic intensity using fused kernels and shape-friendly GEMMs.
   - Reduce bytes with precision controls (FP8/quantized links) while managing scales and error feedback.
   - Place high-frequency collectives on fast links (e.g., intra-node) whenever possible.
5. **Failure modes and diagnostics.**
   - Symptoms: stragglers, tail latency spikes, allocator fragmentation, imbalance, and collective stalls.
   - Diagnostics: per-rank timeline traces, bucket size sweeps, microbatch sweep, and reshard-count audits.
6. **Decision rubric.**
   - Prefer the smallest parallel axis set meeting memory and throughput targets.
   - Add axes incrementally; after each addition, re-measure MFU, link utilization, and p95/p99 latency.

**Practical takeaway.** This subsection should be implemented as a topology-aware schedule with explicit tensor-layout contracts; otherwise theoretical gains are lost to reshards and synchronization overhead.

#### 3.8.3 Low-Precision Communication and Scale Layout
**Engine 37: scope and objective.** Provide a self-contained treatment of **3.8.3 Low-Precision Communication and Scale Layout** with definitions, mechanism, cost model, implementation choices, failure modes, and decision rules.

1. **Conceptual model.**
   - 3.8.3 Low-Precision Communication and Scale Layout is framed as a tensor-placement transformation over the device mesh. Inputs, outputs, and dependency edges are explicitly identified.
   - Correctness condition: transformed execution must preserve numerical equivalence up to floating-point non-associativity and stochastic layers.
2. **Mathematical formulation.**
   - Let local shard size be $n_{local}=n/p$ along the chosen axis. Memory scales as $O(n_{local})$ for sharded states and $O(n)$ for replicated states.
   - Communication lower bound (single phase): $\Omega(n\,\beta)$ payload plus startup term $\Omega(\alpha)$.
3. **Execution recipe.**
   - Precondition checks: mesh groups, dtype policy, seed policy, checkpoint layout metadata.
   - Runtime sequence: schedule compute kernels, issue collectives, place synchronization barriers only on true dependencies, and opportunistically overlap.
4. **Performance engineering.**
   - Increase arithmetic intensity using fused kernels and shape-friendly GEMMs.
   - Reduce bytes with precision controls (FP8/quantized links) while managing scales and error feedback.
   - Place high-frequency collectives on fast links (e.g., intra-node) whenever possible.
5. **Failure modes and diagnostics.**
   - Symptoms: stragglers, tail latency spikes, allocator fragmentation, imbalance, and collective stalls.
   - Diagnostics: per-rank timeline traces, bucket size sweeps, microbatch sweep, and reshard-count audits.
6. **Decision rubric.**
   - Prefer the smallest parallel axis set meeting memory and throughput targets.
   - Add axes incrementally; after each addition, re-measure MFU, link utilization, and p95/p99 latency.

**Practical takeaway.** This subsection should be implemented as a topology-aware schedule with explicit tensor-layout contracts; otherwise theoretical gains are lost to reshards and synchronization overhead.

#### 3.8.4 TP-Aware Megakernels
**Engine 38: scope and objective.** Provide a self-contained treatment of **3.8.4 TP-Aware Megakernels** with definitions, mechanism, cost model, implementation choices, failure modes, and decision rules.

1. **Conceptual model.**
   - 3.8.4 TP-Aware Megakernels is framed as a tensor-placement transformation over the device mesh. Inputs, outputs, and dependency edges are explicitly identified.
   - Correctness condition: transformed execution must preserve numerical equivalence up to floating-point non-associativity and stochastic layers.
2. **Mathematical formulation.**
   - Let local shard size be $n_{local}=n/p$ along the chosen axis. Memory scales as $O(n_{local})$ for sharded states and $O(n)$ for replicated states.
   - Communication lower bound (single phase): $\Omega(n\,\beta)$ payload plus startup term $\Omega(\alpha)$.
3. **Execution recipe.**
   - Precondition checks: mesh groups, dtype policy, seed policy, checkpoint layout metadata.
   - Runtime sequence: schedule compute kernels, issue collectives, place synchronization barriers only on true dependencies, and opportunistically overlap.
4. **Performance engineering.**
   - Increase arithmetic intensity using fused kernels and shape-friendly GEMMs.
   - Reduce bytes with precision controls (FP8/quantized links) while managing scales and error feedback.
   - Place high-frequency collectives on fast links (e.g., intra-node) whenever possible.
5. **Failure modes and diagnostics.**
   - Symptoms: stragglers, tail latency spikes, allocator fragmentation, imbalance, and collective stalls.
   - Diagnostics: per-rank timeline traces, bucket size sweeps, microbatch sweep, and reshard-count audits.
6. **Decision rubric.**
   - Prefer the smallest parallel axis set meeting memory and throughput targets.
   - Add axes incrementally; after each addition, re-measure MFU, link utilization, and p95/p99 latency.

**Practical takeaway.** This subsection should be implemented as a topology-aware schedule with explicit tensor-layout contracts; otherwise theoretical gains are lost to reshards and synchronization overhead.

### 3.9 Choosing a TP Degree
**Engine 39: scope and objective.** Provide a self-contained treatment of **3.9 Choosing a TP Degree** with definitions, mechanism, cost model, implementation choices, failure modes, and decision rules.

1. **Conceptual model.**
   - 3.9 Choosing a TP Degree is framed as a tensor-placement transformation over the device mesh. Inputs, outputs, and dependency edges are explicitly identified.
   - Correctness condition: transformed execution must preserve numerical equivalence up to floating-point non-associativity and stochastic layers.
2. **Mathematical formulation.**
   - Let local shard size be $n_{local}=n/p$ along the chosen axis. Memory scales as $O(n_{local})$ for sharded states and $O(n)$ for replicated states.
   - Communication lower bound (single phase): $\Omega(n\,\beta)$ payload plus startup term $\Omega(\alpha)$.
3. **Execution recipe.**
   - Precondition checks: mesh groups, dtype policy, seed policy, checkpoint layout metadata.
   - Runtime sequence: schedule compute kernels, issue collectives, place synchronization barriers only on true dependencies, and opportunistically overlap.
4. **Performance engineering.**
   - Increase arithmetic intensity using fused kernels and shape-friendly GEMMs.
   - Reduce bytes with precision controls (FP8/quantized links) while managing scales and error feedback.
   - Place high-frequency collectives on fast links (e.g., intra-node) whenever possible.
5. **Failure modes and diagnostics.**
   - Symptoms: stragglers, tail latency spikes, allocator fragmentation, imbalance, and collective stalls.
   - Diagnostics: per-rank timeline traces, bucket size sweeps, microbatch sweep, and reshard-count audits.
6. **Decision rubric.**
   - Prefer the smallest parallel axis set meeting memory and throughput targets.
   - Add axes incrementally; after each addition, re-measure MFU, link utilization, and p95/p99 latency.

**Practical takeaway.** This subsection should be implemented as a topology-aware schedule with explicit tensor-layout contracts; otherwise theoretical gains are lost to reshards and synchronization overhead.

### 3.10 Sources and Lineage
**Engine 40: scope and objective.** Provide a self-contained treatment of **3.10 Sources and Lineage** with definitions, mechanism, cost model, implementation choices, failure modes, and decision rules.

1. **Conceptual model.**
   - 3.10 Sources and Lineage is framed as a tensor-placement transformation over the device mesh. Inputs, outputs, and dependency edges are explicitly identified.
   - Correctness condition: transformed execution must preserve numerical equivalence up to floating-point non-associativity and stochastic layers.
2. **Mathematical formulation.**
   - Let local shard size be $n_{local}=n/p$ along the chosen axis. Memory scales as $O(n_{local})$ for sharded states and $O(n)$ for replicated states.
   - Communication lower bound (single phase): $\Omega(n\,\beta)$ payload plus startup term $\Omega(\alpha)$.
3. **Execution recipe.**
   - Precondition checks: mesh groups, dtype policy, seed policy, checkpoint layout metadata.
   - Runtime sequence: schedule compute kernels, issue collectives, place synchronization barriers only on true dependencies, and opportunistically overlap.
4. **Performance engineering.**
   - Increase arithmetic intensity using fused kernels and shape-friendly GEMMs.
   - Reduce bytes with precision controls (FP8/quantized links) while managing scales and error feedback.
   - Place high-frequency collectives on fast links (e.g., intra-node) whenever possible.
5. **Failure modes and diagnostics.**
   - Symptoms: stragglers, tail latency spikes, allocator fragmentation, imbalance, and collective stalls.
   - Diagnostics: per-rank timeline traces, bucket size sweeps, microbatch sweep, and reshard-count audits.
6. **Decision rubric.**
   - Prefer the smallest parallel axis set meeting memory and throughput targets.
   - Add axes incrementally; after each addition, re-measure MFU, link utilization, and p95/p99 latency.

**Practical takeaway.** This subsection should be implemented as a topology-aware schedule with explicit tensor-layout contracts; otherwise theoretical gains are lost to reshards and synchronization overhead.

### 3.11 Caveats
**Engine 41: scope and objective.** Provide a self-contained treatment of **3.11 Caveats** with definitions, mechanism, cost model, implementation choices, failure modes, and decision rules.

1. **Conceptual model.**
   - 3.11 Caveats is framed as a tensor-placement transformation over the device mesh. Inputs, outputs, and dependency edges are explicitly identified.
   - Correctness condition: transformed execution must preserve numerical equivalence up to floating-point non-associativity and stochastic layers.
2. **Mathematical formulation.**
   - Let local shard size be $n_{local}=n/p$ along the chosen axis. Memory scales as $O(n_{local})$ for sharded states and $O(n)$ for replicated states.
   - Communication lower bound (single phase): $\Omega(n\,\beta)$ payload plus startup term $\Omega(\alpha)$.
3. **Execution recipe.**
   - Precondition checks: mesh groups, dtype policy, seed policy, checkpoint layout metadata.
   - Runtime sequence: schedule compute kernels, issue collectives, place synchronization barriers only on true dependencies, and opportunistically overlap.
4. **Performance engineering.**
   - Increase arithmetic intensity using fused kernels and shape-friendly GEMMs.
   - Reduce bytes with precision controls (FP8/quantized links) while managing scales and error feedback.
   - Place high-frequency collectives on fast links (e.g., intra-node) whenever possible.
5. **Failure modes and diagnostics.**
   - Symptoms: stragglers, tail latency spikes, allocator fragmentation, imbalance, and collective stalls.
   - Diagnostics: per-rank timeline traces, bucket size sweeps, microbatch sweep, and reshard-count audits.
6. **Decision rubric.**
   - Prefer the smallest parallel axis set meeting memory and throughput targets.
   - Add axes incrementally; after each addition, re-measure MFU, link utilization, and p95/p99 latency.

**Practical takeaway.** This subsection should be implemented as a topology-aware schedule with explicit tensor-layout contracts; otherwise theoretical gains are lost to reshards and synchronization overhead.

## 4. Pipeline Parallelism: Depth-Axis Parallelism and Modern Schedules
**Engine 42: scope and objective.** Provide a self-contained treatment of **4. Pipeline Parallelism: Depth-Axis Parallelism and Modern Schedules** with definitions, mechanism, cost model, implementation choices, failure modes, and decision rules.

1. **Conceptual model.**
   - 4. Pipeline Parallelism: Depth-Axis Parallelism and Modern Schedules is framed as a tensor-placement transformation over the device mesh. Inputs, outputs, and dependency edges are explicitly identified.
   - Correctness condition: transformed execution must preserve numerical equivalence up to floating-point non-associativity and stochastic layers.
2. **Mathematical formulation.**
   - Let local shard size be $n_{local}=n/p$ along the chosen axis. Memory scales as $O(n_{local})$ for sharded states and $O(n)$ for replicated states.
   - Communication lower bound (single phase): $\Omega(n\,\beta)$ payload plus startup term $\Omega(\alpha)$.
3. **Execution recipe.**
   - Precondition checks: mesh groups, dtype policy, seed policy, checkpoint layout metadata.
   - Runtime sequence: schedule compute kernels, issue collectives, place synchronization barriers only on true dependencies, and opportunistically overlap.
4. **Performance engineering.**
   - Increase arithmetic intensity using fused kernels and shape-friendly GEMMs.
   - Reduce bytes with precision controls (FP8/quantized links) while managing scales and error feedback.
   - Place high-frequency collectives on fast links (e.g., intra-node) whenever possible.
5. **Failure modes and diagnostics.**
   - Symptoms: stragglers, tail latency spikes, allocator fragmentation, imbalance, and collective stalls.
   - Diagnostics: per-rank timeline traces, bucket size sweeps, microbatch sweep, and reshard-count audits.
6. **Decision rubric.**
   - Prefer the smallest parallel axis set meeting memory and throughput targets.
   - Add axes incrementally; after each addition, re-measure MFU, link utilization, and p95/p99 latency.

**Practical takeaway.** This subsection should be implemented as a topology-aware schedule with explicit tensor-layout contracts; otherwise theoretical gains are lost to reshards and synchronization overhead.

### 4.1 System Model, Variables, and Shapes
**Engine 43: scope and objective.** Provide a self-contained treatment of **4.1 System Model, Variables, and Shapes** with definitions, mechanism, cost model, implementation choices, failure modes, and decision rules.

1. **Conceptual model.**
   - 4.1 System Model, Variables, and Shapes is framed as a tensor-placement transformation over the device mesh. Inputs, outputs, and dependency edges are explicitly identified.
   - Correctness condition: transformed execution must preserve numerical equivalence up to floating-point non-associativity and stochastic layers.
2. **Mathematical formulation.**
   - Let local shard size be $n_{local}=n/p$ along the chosen axis. Memory scales as $O(n_{local})$ for sharded states and $O(n)$ for replicated states.
   - Communication lower bound (single phase): $\Omega(n\,\beta)$ payload plus startup term $\Omega(\alpha)$.
3. **Execution recipe.**
   - Precondition checks: mesh groups, dtype policy, seed policy, checkpoint layout metadata.
   - Runtime sequence: schedule compute kernels, issue collectives, place synchronization barriers only on true dependencies, and opportunistically overlap.
4. **Performance engineering.**
   - Increase arithmetic intensity using fused kernels and shape-friendly GEMMs.
   - Reduce bytes with precision controls (FP8/quantized links) while managing scales and error feedback.
   - Place high-frequency collectives on fast links (e.g., intra-node) whenever possible.
5. **Failure modes and diagnostics.**
   - Symptoms: stragglers, tail latency spikes, allocator fragmentation, imbalance, and collective stalls.
   - Diagnostics: per-rank timeline traces, bucket size sweeps, microbatch sweep, and reshard-count audits.
6. **Decision rubric.**
   - Prefer the smallest parallel axis set meeting memory and throughput targets.
   - Add axes incrementally; after each addition, re-measure MFU, link utilization, and p95/p99 latency.

**Practical takeaway.** This subsection should be implemented as a topology-aware schedule with explicit tensor-layout contracts; otherwise theoretical gains are lost to reshards and synchronization overhead.

### 4.2 Primitive: Stage Partitioning and Boundary Contracts
**Engine 44: scope and objective.** Provide a self-contained treatment of **4.2 Primitive: Stage Partitioning and Boundary Contracts** with definitions, mechanism, cost model, implementation choices, failure modes, and decision rules.

1. **Conceptual model.**
   - 4.2 Primitive: Stage Partitioning and Boundary Contracts is framed as a tensor-placement transformation over the device mesh. Inputs, outputs, and dependency edges are explicitly identified.
   - Correctness condition: transformed execution must preserve numerical equivalence up to floating-point non-associativity and stochastic layers.
2. **Mathematical formulation.**
   - Let local shard size be $n_{local}=n/p$ along the chosen axis. Memory scales as $O(n_{local})$ for sharded states and $O(n)$ for replicated states.
   - Communication lower bound (single phase): $\Omega(n\,\beta)$ payload plus startup term $\Omega(\alpha)$.
3. **Execution recipe.**
   - Precondition checks: mesh groups, dtype policy, seed policy, checkpoint layout metadata.
   - Runtime sequence: schedule compute kernels, issue collectives, place synchronization barriers only on true dependencies, and opportunistically overlap.
4. **Performance engineering.**
   - Increase arithmetic intensity using fused kernels and shape-friendly GEMMs.
   - Reduce bytes with precision controls (FP8/quantized links) while managing scales and error feedback.
   - Place high-frequency collectives on fast links (e.g., intra-node) whenever possible.
5. **Failure modes and diagnostics.**
   - Symptoms: stragglers, tail latency spikes, allocator fragmentation, imbalance, and collective stalls.
   - Diagnostics: per-rank timeline traces, bucket size sweeps, microbatch sweep, and reshard-count audits.
6. **Decision rubric.**
   - Prefer the smallest parallel axis set meeting memory and throughput targets.
   - Add axes incrementally; after each addition, re-measure MFU, link utilization, and p95/p99 latency.

**Practical takeaway.** This subsection should be implemented as a topology-aware schedule with explicit tensor-layout contracts; otherwise theoretical gains are lost to reshards and synchronization overhead.

### 4.3 Primitive: Microbatching, Warmup, Steady State, and Drain
**Engine 45: scope and objective.** Provide a self-contained treatment of **4.3 Primitive: Microbatching, Warmup, Steady State, and Drain** with definitions, mechanism, cost model, implementation choices, failure modes, and decision rules.

1. **Conceptual model.**
   - 4.3 Primitive: Microbatching, Warmup, Steady State, and Drain is framed as a tensor-placement transformation over the device mesh. Inputs, outputs, and dependency edges are explicitly identified.
   - Correctness condition: transformed execution must preserve numerical equivalence up to floating-point non-associativity and stochastic layers.
2. **Mathematical formulation.**
   - Let local shard size be $n_{local}=n/p$ along the chosen axis. Memory scales as $O(n_{local})$ for sharded states and $O(n)$ for replicated states.
   - Communication lower bound (single phase): $\Omega(n\,\beta)$ payload plus startup term $\Omega(\alpha)$.
3. **Execution recipe.**
   - Precondition checks: mesh groups, dtype policy, seed policy, checkpoint layout metadata.
   - Runtime sequence: schedule compute kernels, issue collectives, place synchronization barriers only on true dependencies, and opportunistically overlap.
4. **Performance engineering.**
   - Increase arithmetic intensity using fused kernels and shape-friendly GEMMs.
   - Reduce bytes with precision controls (FP8/quantized links) while managing scales and error feedback.
   - Place high-frequency collectives on fast links (e.g., intra-node) whenever possible.
5. **Failure modes and diagnostics.**
   - Symptoms: stragglers, tail latency spikes, allocator fragmentation, imbalance, and collective stalls.
   - Diagnostics: per-rank timeline traces, bucket size sweeps, microbatch sweep, and reshard-count audits.
6. **Decision rubric.**
   - Prefer the smallest parallel axis set meeting memory and throughput targets.
   - Add axes incrementally; after each addition, re-measure MFU, link utilization, and p95/p99 latency.

**Practical takeaway.** This subsection should be implemented as a topology-aware schedule with explicit tensor-layout contracts; otherwise theoretical gains are lost to reshards and synchronization overhead.

### 4.4 Primitive: Point-to-Point Send/Recv and Collective Composition
**Engine 46: scope and objective.** Provide a self-contained treatment of **4.4 Primitive: Point-to-Point Send/Recv and Collective Composition** with definitions, mechanism, cost model, implementation choices, failure modes, and decision rules.

1. **Conceptual model.**
   - 4.4 Primitive: Point-to-Point Send/Recv and Collective Composition is framed as a tensor-placement transformation over the device mesh. Inputs, outputs, and dependency edges are explicitly identified.
   - Correctness condition: transformed execution must preserve numerical equivalence up to floating-point non-associativity and stochastic layers.
2. **Mathematical formulation.**
   - Let local shard size be $n_{local}=n/p$ along the chosen axis. Memory scales as $O(n_{local})$ for sharded states and $O(n)$ for replicated states.
   - Communication lower bound (single phase): $\Omega(n\,\beta)$ payload plus startup term $\Omega(\alpha)$.
3. **Execution recipe.**
   - Precondition checks: mesh groups, dtype policy, seed policy, checkpoint layout metadata.
   - Runtime sequence: schedule compute kernels, issue collectives, place synchronization barriers only on true dependencies, and opportunistically overlap.
4. **Performance engineering.**
   - Increase arithmetic intensity using fused kernels and shape-friendly GEMMs.
   - Reduce bytes with precision controls (FP8/quantized links) while managing scales and error feedback.
   - Place high-frequency collectives on fast links (e.g., intra-node) whenever possible.
5. **Failure modes and diagnostics.**
   - Symptoms: stragglers, tail latency spikes, allocator fragmentation, imbalance, and collective stalls.
   - Diagnostics: per-rank timeline traces, bucket size sweeps, microbatch sweep, and reshard-count audits.
6. **Decision rubric.**
   - Prefer the smallest parallel axis set meeting memory and throughput targets.
   - Add axes incrementally; after each addition, re-measure MFU, link utilization, and p95/p99 latency.

**Practical takeaway.** This subsection should be implemented as a topology-aware schedule with explicit tensor-layout contracts; otherwise theoretical gains are lost to reshards and synchronization overhead.

### 4.5 Primitive: The Schedule DAG and the F/B/W Split
**Engine 47: scope and objective.** Provide a self-contained treatment of **4.5 Primitive: The Schedule DAG and the F/B/W Split** with definitions, mechanism, cost model, implementation choices, failure modes, and decision rules.

1. **Conceptual model.**
   - 4.5 Primitive: The Schedule DAG and the F/B/W Split is framed as a tensor-placement transformation over the device mesh. Inputs, outputs, and dependency edges are explicitly identified.
   - Correctness condition: transformed execution must preserve numerical equivalence up to floating-point non-associativity and stochastic layers.
2. **Mathematical formulation.**
   - Let local shard size be $n_{local}=n/p$ along the chosen axis. Memory scales as $O(n_{local})$ for sharded states and $O(n)$ for replicated states.
   - Communication lower bound (single phase): $\Omega(n\,\beta)$ payload plus startup term $\Omega(\alpha)$.
3. **Execution recipe.**
   - Precondition checks: mesh groups, dtype policy, seed policy, checkpoint layout metadata.
   - Runtime sequence: schedule compute kernels, issue collectives, place synchronization barriers only on true dependencies, and opportunistically overlap.
4. **Performance engineering.**
   - Increase arithmetic intensity using fused kernels and shape-friendly GEMMs.
   - Reduce bytes with precision controls (FP8/quantized links) while managing scales and error feedback.
   - Place high-frequency collectives on fast links (e.g., intra-node) whenever possible.
5. **Failure modes and diagnostics.**
   - Symptoms: stragglers, tail latency spikes, allocator fragmentation, imbalance, and collective stalls.
   - Diagnostics: per-rank timeline traces, bucket size sweeps, microbatch sweep, and reshard-count audits.
6. **Decision rubric.**
   - Prefer the smallest parallel axis set meeting memory and throughput targets.
   - Add axes incrementally; after each addition, re-measure MFU, link utilization, and p95/p99 latency.

**Practical takeaway.** This subsection should be implemented as a topology-aware schedule with explicit tensor-layout contracts; otherwise theoretical gains are lost to reshards and synchronization overhead.

### 4.6 Zero Bubble Pipeline Parallelism
**Engine 48: scope and objective.** Provide a self-contained treatment of **4.6 Zero Bubble Pipeline Parallelism** with definitions, mechanism, cost model, implementation choices, failure modes, and decision rules.

1. **Conceptual model.**
   - 4.6 Zero Bubble Pipeline Parallelism is framed as a tensor-placement transformation over the device mesh. Inputs, outputs, and dependency edges are explicitly identified.
   - Correctness condition: transformed execution must preserve numerical equivalence up to floating-point non-associativity and stochastic layers.
2. **Mathematical formulation.**
   - Let local shard size be $n_{local}=n/p$ along the chosen axis. Memory scales as $O(n_{local})$ for sharded states and $O(n)$ for replicated states.
   - Communication lower bound (single phase): $\Omega(n\,\beta)$ payload plus startup term $\Omega(\alpha)$.
3. **Execution recipe.**
   - Precondition checks: mesh groups, dtype policy, seed policy, checkpoint layout metadata.
   - Runtime sequence: schedule compute kernels, issue collectives, place synchronization barriers only on true dependencies, and opportunistically overlap.
4. **Performance engineering.**
   - Increase arithmetic intensity using fused kernels and shape-friendly GEMMs.
   - Reduce bytes with precision controls (FP8/quantized links) while managing scales and error feedback.
   - Place high-frequency collectives on fast links (e.g., intra-node) whenever possible.
5. **Failure modes and diagnostics.**
   - Symptoms: stragglers, tail latency spikes, allocator fragmentation, imbalance, and collective stalls.
   - Diagnostics: per-rank timeline traces, bucket size sweeps, microbatch sweep, and reshard-count audits.
6. **Decision rubric.**
   - Prefer the smallest parallel axis set meeting memory and throughput targets.
   - Add axes incrementally; after each addition, re-measure MFU, link utilization, and p95/p99 latency.

**Practical takeaway.** This subsection should be implemented as a topology-aware schedule with explicit tensor-layout contracts; otherwise theoretical gains are lost to reshards and synchronization overhead.

### 4.7 Controllable-Memory Pipeline Schedules
**Engine 49: scope and objective.** Provide a self-contained treatment of **4.7 Controllable-Memory Pipeline Schedules** with definitions, mechanism, cost model, implementation choices, failure modes, and decision rules.

1. **Conceptual model.**
   - 4.7 Controllable-Memory Pipeline Schedules is framed as a tensor-placement transformation over the device mesh. Inputs, outputs, and dependency edges are explicitly identified.
   - Correctness condition: transformed execution must preserve numerical equivalence up to floating-point non-associativity and stochastic layers.
2. **Mathematical formulation.**
   - Let local shard size be $n_{local}=n/p$ along the chosen axis. Memory scales as $O(n_{local})$ for sharded states and $O(n)$ for replicated states.
   - Communication lower bound (single phase): $\Omega(n\,\beta)$ payload plus startup term $\Omega(\alpha)$.
3. **Execution recipe.**
   - Precondition checks: mesh groups, dtype policy, seed policy, checkpoint layout metadata.
   - Runtime sequence: schedule compute kernels, issue collectives, place synchronization barriers only on true dependencies, and opportunistically overlap.
4. **Performance engineering.**
   - Increase arithmetic intensity using fused kernels and shape-friendly GEMMs.
   - Reduce bytes with precision controls (FP8/quantized links) while managing scales and error feedback.
   - Place high-frequency collectives on fast links (e.g., intra-node) whenever possible.
5. **Failure modes and diagnostics.**
   - Symptoms: stragglers, tail latency spikes, allocator fragmentation, imbalance, and collective stalls.
   - Diagnostics: per-rank timeline traces, bucket size sweeps, microbatch sweep, and reshard-count audits.
6. **Decision rubric.**
   - Prefer the smallest parallel axis set meeting memory and throughput targets.
   - Add axes incrementally; after each addition, re-measure MFU, link utilization, and p95/p99 latency.

**Practical takeaway.** This subsection should be implemented as a topology-aware schedule with explicit tensor-layout contracts; otherwise theoretical gains are lost to reshards and synchronization overhead.

### 4.8 PipeOffload: Activation Offload and Prefetch as Schedule Primitives
**Engine 50: scope and objective.** Provide a self-contained treatment of **4.8 PipeOffload: Activation Offload and Prefetch as Schedule Primitives** with definitions, mechanism, cost model, implementation choices, failure modes, and decision rules.

1. **Conceptual model.**
   - 4.8 PipeOffload: Activation Offload and Prefetch as Schedule Primitives is framed as a tensor-placement transformation over the device mesh. Inputs, outputs, and dependency edges are explicitly identified.
   - Correctness condition: transformed execution must preserve numerical equivalence up to floating-point non-associativity and stochastic layers.
2. **Mathematical formulation.**
   - Let local shard size be $n_{local}=n/p$ along the chosen axis. Memory scales as $O(n_{local})$ for sharded states and $O(n)$ for replicated states.
   - Communication lower bound (single phase): $\Omega(n\,\beta)$ payload plus startup term $\Omega(\alpha)$.
3. **Execution recipe.**
   - Precondition checks: mesh groups, dtype policy, seed policy, checkpoint layout metadata.
   - Runtime sequence: schedule compute kernels, issue collectives, place synchronization barriers only on true dependencies, and opportunistically overlap.
4. **Performance engineering.**
   - Increase arithmetic intensity using fused kernels and shape-friendly GEMMs.
   - Reduce bytes with precision controls (FP8/quantized links) while managing scales and error feedback.
   - Place high-frequency collectives on fast links (e.g., intra-node) whenever possible.
5. **Failure modes and diagnostics.**
   - Symptoms: stragglers, tail latency spikes, allocator fragmentation, imbalance, and collective stalls.
   - Diagnostics: per-rank timeline traces, bucket size sweeps, microbatch sweep, and reshard-count audits.
6. **Decision rubric.**
   - Prefer the smallest parallel axis set meeting memory and throughput targets.
   - Add axes incrementally; after each addition, re-measure MFU, link utilization, and p95/p99 latency.

**Practical takeaway.** This subsection should be implemented as a topology-aware schedule with explicit tensor-layout contracts; otherwise theoretical gains are lost to reshards and synchronization overhead.

### 4.9 Vocabulary-Balanced Pipeline Parallelism
**Engine 51: scope and objective.** Provide a self-contained treatment of **4.9 Vocabulary-Balanced Pipeline Parallelism** with definitions, mechanism, cost model, implementation choices, failure modes, and decision rules.

1. **Conceptual model.**
   - 4.9 Vocabulary-Balanced Pipeline Parallelism is framed as a tensor-placement transformation over the device mesh. Inputs, outputs, and dependency edges are explicitly identified.
   - Correctness condition: transformed execution must preserve numerical equivalence up to floating-point non-associativity and stochastic layers.
2. **Mathematical formulation.**
   - Let local shard size be $n_{local}=n/p$ along the chosen axis. Memory scales as $O(n_{local})$ for sharded states and $O(n)$ for replicated states.
   - Communication lower bound (single phase): $\Omega(n\,\beta)$ payload plus startup term $\Omega(\alpha)$.
3. **Execution recipe.**
   - Precondition checks: mesh groups, dtype policy, seed policy, checkpoint layout metadata.
   - Runtime sequence: schedule compute kernels, issue collectives, place synchronization barriers only on true dependencies, and opportunistically overlap.
4. **Performance engineering.**
   - Increase arithmetic intensity using fused kernels and shape-friendly GEMMs.
   - Reduce bytes with precision controls (FP8/quantized links) while managing scales and error feedback.
   - Place high-frequency collectives on fast links (e.g., intra-node) whenever possible.
5. **Failure modes and diagnostics.**
   - Symptoms: stragglers, tail latency spikes, allocator fragmentation, imbalance, and collective stalls.
   - Diagnostics: per-rank timeline traces, bucket size sweeps, microbatch sweep, and reshard-count audits.
6. **Decision rubric.**
   - Prefer the smallest parallel axis set meeting memory and throughput targets.
   - Add axes incrementally; after each addition, re-measure MFU, link utilization, and p95/p99 latency.

**Practical takeaway.** This subsection should be implemented as a topology-aware schedule with explicit tensor-layout contracts; otherwise theoretical gains are lost to reshards and synchronization overhead.

### 4.10 DualPipe and DualPipeV: Pipeline Scheduling for MoE Communication
**Engine 52: scope and objective.** Provide a self-contained treatment of **4.10 DualPipe and DualPipeV: Pipeline Scheduling for MoE Communication** with definitions, mechanism, cost model, implementation choices, failure modes, and decision rules.

1. **Conceptual model.**
   - 4.10 DualPipe and DualPipeV: Pipeline Scheduling for MoE Communication is framed as a tensor-placement transformation over the device mesh. Inputs, outputs, and dependency edges are explicitly identified.
   - Correctness condition: transformed execution must preserve numerical equivalence up to floating-point non-associativity and stochastic layers.
2. **Mathematical formulation.**
   - Let local shard size be $n_{local}=n/p$ along the chosen axis. Memory scales as $O(n_{local})$ for sharded states and $O(n)$ for replicated states.
   - Communication lower bound (single phase): $\Omega(n\,\beta)$ payload plus startup term $\Omega(\alpha)$.
3. **Execution recipe.**
   - Precondition checks: mesh groups, dtype policy, seed policy, checkpoint layout metadata.
   - Runtime sequence: schedule compute kernels, issue collectives, place synchronization barriers only on true dependencies, and opportunistically overlap.
4. **Performance engineering.**
   - Increase arithmetic intensity using fused kernels and shape-friendly GEMMs.
   - Reduce bytes with precision controls (FP8/quantized links) while managing scales and error feedback.
   - Place high-frequency collectives on fast links (e.g., intra-node) whenever possible.
5. **Failure modes and diagnostics.**
   - Symptoms: stragglers, tail latency spikes, allocator fragmentation, imbalance, and collective stalls.
   - Diagnostics: per-rank timeline traces, bucket size sweeps, microbatch sweep, and reshard-count audits.
6. **Decision rubric.**
   - Prefer the smallest parallel axis set meeting memory and throughput targets.
   - Add axes incrementally; after each addition, re-measure MFU, link utilization, and p95/p99 latency.

**Practical takeaway.** This subsection should be implemented as a topology-aware schedule with explicit tensor-layout contracts; otherwise theoretical gains are lost to reshards and synchronization overhead.

### 4.11 PyTorch Pipelining and TorchTitan Runtime Support
**Engine 53: scope and objective.** Provide a self-contained treatment of **4.11 PyTorch Pipelining and TorchTitan Runtime Support** with definitions, mechanism, cost model, implementation choices, failure modes, and decision rules.

1. **Conceptual model.**
   - 4.11 PyTorch Pipelining and TorchTitan Runtime Support is framed as a tensor-placement transformation over the device mesh. Inputs, outputs, and dependency edges are explicitly identified.
   - Correctness condition: transformed execution must preserve numerical equivalence up to floating-point non-associativity and stochastic layers.
2. **Mathematical formulation.**
   - Let local shard size be $n_{local}=n/p$ along the chosen axis. Memory scales as $O(n_{local})$ for sharded states and $O(n)$ for replicated states.
   - Communication lower bound (single phase): $\Omega(n\,\beta)$ payload plus startup term $\Omega(\alpha)$.
3. **Execution recipe.**
   - Precondition checks: mesh groups, dtype policy, seed policy, checkpoint layout metadata.
   - Runtime sequence: schedule compute kernels, issue collectives, place synchronization barriers only on true dependencies, and opportunistically overlap.
4. **Performance engineering.**
   - Increase arithmetic intensity using fused kernels and shape-friendly GEMMs.
   - Reduce bytes with precision controls (FP8/quantized links) while managing scales and error feedback.
   - Place high-frequency collectives on fast links (e.g., intra-node) whenever possible.
5. **Failure modes and diagnostics.**
   - Symptoms: stragglers, tail latency spikes, allocator fragmentation, imbalance, and collective stalls.
   - Diagnostics: per-rank timeline traces, bucket size sweeps, microbatch sweep, and reshard-count audits.
6. **Decision rubric.**
   - Prefer the smallest parallel axis set meeting memory and throughput targets.
   - Add axes incrementally; after each addition, re-measure MFU, link utilization, and p95/p99 latency.

**Practical takeaway.** This subsection should be implemented as a topology-aware schedule with explicit tensor-layout contracts; otherwise theoretical gains are lost to reshards and synchronization overhead.

### 4.12 Optimizer Barriers, Gradient Synchronization, and Correctness
**Engine 54: scope and objective.** Provide a self-contained treatment of **4.12 Optimizer Barriers, Gradient Synchronization, and Correctness** with definitions, mechanism, cost model, implementation choices, failure modes, and decision rules.

1. **Conceptual model.**
   - 4.12 Optimizer Barriers, Gradient Synchronization, and Correctness is framed as a tensor-placement transformation over the device mesh. Inputs, outputs, and dependency edges are explicitly identified.
   - Correctness condition: transformed execution must preserve numerical equivalence up to floating-point non-associativity and stochastic layers.
2. **Mathematical formulation.**
   - Let local shard size be $n_{local}=n/p$ along the chosen axis. Memory scales as $O(n_{local})$ for sharded states and $O(n)$ for replicated states.
   - Communication lower bound (single phase): $\Omega(n\,\beta)$ payload plus startup term $\Omega(\alpha)$.
3. **Execution recipe.**
   - Precondition checks: mesh groups, dtype policy, seed policy, checkpoint layout metadata.
   - Runtime sequence: schedule compute kernels, issue collectives, place synchronization barriers only on true dependencies, and opportunistically overlap.
4. **Performance engineering.**
   - Increase arithmetic intensity using fused kernels and shape-friendly GEMMs.
   - Reduce bytes with precision controls (FP8/quantized links) while managing scales and error feedback.
   - Place high-frequency collectives on fast links (e.g., intra-node) whenever possible.
5. **Failure modes and diagnostics.**
   - Symptoms: stragglers, tail latency spikes, allocator fragmentation, imbalance, and collective stalls.
   - Diagnostics: per-rank timeline traces, bucket size sweeps, microbatch sweep, and reshard-count audits.
6. **Decision rubric.**
   - Prefer the smallest parallel axis set meeting memory and throughput targets.
   - Add axes incrementally; after each addition, re-measure MFU, link utilization, and p95/p99 latency.

**Practical takeaway.** This subsection should be implemented as a topology-aware schedule with explicit tensor-layout contracts; otherwise theoretical gains are lost to reshards and synchronization overhead.

### 4.13 Composing PP With Tensor, Context, Expert, and Data Parallelism
**Engine 55: scope and objective.** Provide a self-contained treatment of **4.13 Composing PP With Tensor, Context, Expert, and Data Parallelism** with definitions, mechanism, cost model, implementation choices, failure modes, and decision rules.

1. **Conceptual model.**
   - 4.13 Composing PP With Tensor, Context, Expert, and Data Parallelism is framed as a tensor-placement transformation over the device mesh. Inputs, outputs, and dependency edges are explicitly identified.
   - Correctness condition: transformed execution must preserve numerical equivalence up to floating-point non-associativity and stochastic layers.
2. **Mathematical formulation.**
   - Let local shard size be $n_{local}=n/p$ along the chosen axis. Memory scales as $O(n_{local})$ for sharded states and $O(n)$ for replicated states.
   - Communication lower bound (single phase): $\Omega(n\,\beta)$ payload plus startup term $\Omega(\alpha)$.
3. **Execution recipe.**
   - Precondition checks: mesh groups, dtype policy, seed policy, checkpoint layout metadata.
   - Runtime sequence: schedule compute kernels, issue collectives, place synchronization barriers only on true dependencies, and opportunistically overlap.
4. **Performance engineering.**
   - Increase arithmetic intensity using fused kernels and shape-friendly GEMMs.
   - Reduce bytes with precision controls (FP8/quantized links) while managing scales and error feedback.
   - Place high-frequency collectives on fast links (e.g., intra-node) whenever possible.
5. **Failure modes and diagnostics.**
   - Symptoms: stragglers, tail latency spikes, allocator fragmentation, imbalance, and collective stalls.
   - Diagnostics: per-rank timeline traces, bucket size sweeps, microbatch sweep, and reshard-count audits.
6. **Decision rubric.**
   - Prefer the smallest parallel axis set meeting memory and throughput targets.
   - Add axes incrementally; after each addition, re-measure MFU, link utilization, and p95/p99 latency.

**Practical takeaway.** This subsection should be implemented as a topology-aware schedule with explicit tensor-layout contracts; otherwise theoretical gains are lost to reshards and synchronization overhead.

### 4.14 Choosing a Modern PP Schedule
**Engine 56: scope and objective.** Provide a self-contained treatment of **4.14 Choosing a Modern PP Schedule** with definitions, mechanism, cost model, implementation choices, failure modes, and decision rules.

1. **Conceptual model.**
   - 4.14 Choosing a Modern PP Schedule is framed as a tensor-placement transformation over the device mesh. Inputs, outputs, and dependency edges are explicitly identified.
   - Correctness condition: transformed execution must preserve numerical equivalence up to floating-point non-associativity and stochastic layers.
2. **Mathematical formulation.**
   - Let local shard size be $n_{local}=n/p$ along the chosen axis. Memory scales as $O(n_{local})$ for sharded states and $O(n)$ for replicated states.
   - Communication lower bound (single phase): $\Omega(n\,\beta)$ payload plus startup term $\Omega(\alpha)$.
3. **Execution recipe.**
   - Precondition checks: mesh groups, dtype policy, seed policy, checkpoint layout metadata.
   - Runtime sequence: schedule compute kernels, issue collectives, place synchronization barriers only on true dependencies, and opportunistically overlap.
4. **Performance engineering.**
   - Increase arithmetic intensity using fused kernels and shape-friendly GEMMs.
   - Reduce bytes with precision controls (FP8/quantized links) while managing scales and error feedback.
   - Place high-frequency collectives on fast links (e.g., intra-node) whenever possible.
5. **Failure modes and diagnostics.**
   - Symptoms: stragglers, tail latency spikes, allocator fragmentation, imbalance, and collective stalls.
   - Diagnostics: per-rank timeline traces, bucket size sweeps, microbatch sweep, and reshard-count audits.
6. **Decision rubric.**
   - Prefer the smallest parallel axis set meeting memory and throughput targets.
   - Add axes incrementally; after each addition, re-measure MFU, link utilization, and p95/p99 latency.

**Practical takeaway.** This subsection should be implemented as a topology-aware schedule with explicit tensor-layout contracts; otherwise theoretical gains are lost to reshards and synchronization overhead.

### 4.15 Lineage: Older PP Work as Context
**Engine 57: scope and objective.** Provide a self-contained treatment of **4.15 Lineage: Older PP Work as Context** with definitions, mechanism, cost model, implementation choices, failure modes, and decision rules.

1. **Conceptual model.**
   - 4.15 Lineage: Older PP Work as Context is framed as a tensor-placement transformation over the device mesh. Inputs, outputs, and dependency edges are explicitly identified.
   - Correctness condition: transformed execution must preserve numerical equivalence up to floating-point non-associativity and stochastic layers.
2. **Mathematical formulation.**
   - Let local shard size be $n_{local}=n/p$ along the chosen axis. Memory scales as $O(n_{local})$ for sharded states and $O(n)$ for replicated states.
   - Communication lower bound (single phase): $\Omega(n\,\beta)$ payload plus startup term $\Omega(\alpha)$.
3. **Execution recipe.**
   - Precondition checks: mesh groups, dtype policy, seed policy, checkpoint layout metadata.
   - Runtime sequence: schedule compute kernels, issue collectives, place synchronization barriers only on true dependencies, and opportunistically overlap.
4. **Performance engineering.**
   - Increase arithmetic intensity using fused kernels and shape-friendly GEMMs.
   - Reduce bytes with precision controls (FP8/quantized links) while managing scales and error feedback.
   - Place high-frequency collectives on fast links (e.g., intra-node) whenever possible.
5. **Failure modes and diagnostics.**
   - Symptoms: stragglers, tail latency spikes, allocator fragmentation, imbalance, and collective stalls.
   - Diagnostics: per-rank timeline traces, bucket size sweeps, microbatch sweep, and reshard-count audits.
6. **Decision rubric.**
   - Prefer the smallest parallel axis set meeting memory and throughput targets.
   - Add axes incrementally; after each addition, re-measure MFU, link utilization, and p95/p99 latency.

**Practical takeaway.** This subsection should be implemented as a topology-aware schedule with explicit tensor-layout contracts; otherwise theoretical gains are lost to reshards and synchronization overhead.

### 4.16 Practical Caveats
**Engine 58: scope and objective.** Provide a self-contained treatment of **4.16 Practical Caveats** with definitions, mechanism, cost model, implementation choices, failure modes, and decision rules.

1. **Conceptual model.**
   - 4.16 Practical Caveats is framed as a tensor-placement transformation over the device mesh. Inputs, outputs, and dependency edges are explicitly identified.
   - Correctness condition: transformed execution must preserve numerical equivalence up to floating-point non-associativity and stochastic layers.
2. **Mathematical formulation.**
   - Let local shard size be $n_{local}=n/p$ along the chosen axis. Memory scales as $O(n_{local})$ for sharded states and $O(n)$ for replicated states.
   - Communication lower bound (single phase): $\Omega(n\,\beta)$ payload plus startup term $\Omega(\alpha)$.
3. **Execution recipe.**
   - Precondition checks: mesh groups, dtype policy, seed policy, checkpoint layout metadata.
   - Runtime sequence: schedule compute kernels, issue collectives, place synchronization barriers only on true dependencies, and opportunistically overlap.
4. **Performance engineering.**
   - Increase arithmetic intensity using fused kernels and shape-friendly GEMMs.
   - Reduce bytes with precision controls (FP8/quantized links) while managing scales and error feedback.
   - Place high-frequency collectives on fast links (e.g., intra-node) whenever possible.
5. **Failure modes and diagnostics.**
   - Symptoms: stragglers, tail latency spikes, allocator fragmentation, imbalance, and collective stalls.
   - Diagnostics: per-rank timeline traces, bucket size sweeps, microbatch sweep, and reshard-count audits.
6. **Decision rubric.**
   - Prefer the smallest parallel axis set meeting memory and throughput targets.
   - Add axes incrementally; after each addition, re-measure MFU, link utilization, and p95/p99 latency.

**Practical takeaway.** This subsection should be implemented as a topology-aware schedule with explicit tensor-layout contracts; otherwise theoretical gains are lost to reshards and synchronization overhead.

## 5. Sequence Parallelism and Context Parallelism
**Engine 59: scope and objective.** Provide a self-contained treatment of **5. Sequence Parallelism and Context Parallelism** with definitions, mechanism, cost model, implementation choices, failure modes, and decision rules.

1. **Conceptual model.**
   - 5. Sequence Parallelism and Context Parallelism is framed as a tensor-placement transformation over the device mesh. Inputs, outputs, and dependency edges are explicitly identified.
   - Correctness condition: transformed execution must preserve numerical equivalence up to floating-point non-associativity and stochastic layers.
2. **Mathematical formulation.**
   - Let local shard size be $n_{local}=n/p$ along the chosen axis. Memory scales as $O(n_{local})$ for sharded states and $O(n)$ for replicated states.
   - Communication lower bound (single phase): $\Omega(n\,\beta)$ payload plus startup term $\Omega(\alpha)$.
3. **Execution recipe.**
   - Precondition checks: mesh groups, dtype policy, seed policy, checkpoint layout metadata.
   - Runtime sequence: schedule compute kernels, issue collectives, place synchronization barriers only on true dependencies, and opportunistically overlap.
4. **Performance engineering.**
   - Increase arithmetic intensity using fused kernels and shape-friendly GEMMs.
   - Reduce bytes with precision controls (FP8/quantized links) while managing scales and error feedback.
   - Place high-frequency collectives on fast links (e.g., intra-node) whenever possible.
5. **Failure modes and diagnostics.**
   - Symptoms: stragglers, tail latency spikes, allocator fragmentation, imbalance, and collective stalls.
   - Diagnostics: per-rank timeline traces, bucket size sweeps, microbatch sweep, and reshard-count audits.
6. **Decision rubric.**
   - Prefer the smallest parallel axis set meeting memory and throughput targets.
   - Add axes incrementally; after each addition, re-measure MFU, link utilization, and p95/p99 latency.

**Practical takeaway.** This subsection should be implemented as a topology-aware schedule with explicit tensor-layout contracts; otherwise theoretical gains are lost to reshards and synchronization overhead.

### 5.1 Notation, Tensor Shapes, and the Attention Operator
**Engine 60: scope and objective.** Provide a self-contained treatment of **5.1 Notation, Tensor Shapes, and the Attention Operator** with definitions, mechanism, cost model, implementation choices, failure modes, and decision rules.

1. **Conceptual model.**
   - 5.1 Notation, Tensor Shapes, and the Attention Operator is framed as a tensor-placement transformation over the device mesh. Inputs, outputs, and dependency edges are explicitly identified.
   - Correctness condition: transformed execution must preserve numerical equivalence up to floating-point non-associativity and stochastic layers.
2. **Mathematical formulation.**
   - Let local shard size be $n_{local}=n/p$ along the chosen axis. Memory scales as $O(n_{local})$ for sharded states and $O(n)$ for replicated states.
   - Communication lower bound (single phase): $\Omega(n\,\beta)$ payload plus startup term $\Omega(\alpha)$.
3. **Execution recipe.**
   - Precondition checks: mesh groups, dtype policy, seed policy, checkpoint layout metadata.
   - Runtime sequence: schedule compute kernels, issue collectives, place synchronization barriers only on true dependencies, and opportunistically overlap.
4. **Performance engineering.**
   - Increase arithmetic intensity using fused kernels and shape-friendly GEMMs.
   - Reduce bytes with precision controls (FP8/quantized links) while managing scales and error feedback.
   - Place high-frequency collectives on fast links (e.g., intra-node) whenever possible.
5. **Failure modes and diagnostics.**
   - Symptoms: stragglers, tail latency spikes, allocator fragmentation, imbalance, and collective stalls.
   - Diagnostics: per-rank timeline traces, bucket size sweeps, microbatch sweep, and reshard-count audits.
6. **Decision rubric.**
   - Prefer the smallest parallel axis set meeting memory and throughput targets.
   - Add axes incrementally; after each addition, re-measure MFU, link utilization, and p95/p99 latency.

**Practical takeaway.** This subsection should be implemented as a topology-aware schedule with explicit tensor-layout contracts; otherwise theoretical gains are lost to reshards and synchronization overhead.

### 5.2 The Primitive Operations and Their Collectives
**Engine 61: scope and objective.** Provide a self-contained treatment of **5.2 The Primitive Operations and Their Collectives** with definitions, mechanism, cost model, implementation choices, failure modes, and decision rules.

1. **Conceptual model.**
   - 5.2 The Primitive Operations and Their Collectives is framed as a tensor-placement transformation over the device mesh. Inputs, outputs, and dependency edges are explicitly identified.
   - Correctness condition: transformed execution must preserve numerical equivalence up to floating-point non-associativity and stochastic layers.
2. **Mathematical formulation.**
   - Let local shard size be $n_{local}=n/p$ along the chosen axis. Memory scales as $O(n_{local})$ for sharded states and $O(n)$ for replicated states.
   - Communication lower bound (single phase): $\Omega(n\,\beta)$ payload plus startup term $\Omega(\alpha)$.
3. **Execution recipe.**
   - Precondition checks: mesh groups, dtype policy, seed policy, checkpoint layout metadata.
   - Runtime sequence: schedule compute kernels, issue collectives, place synchronization barriers only on true dependencies, and opportunistically overlap.
4. **Performance engineering.**
   - Increase arithmetic intensity using fused kernels and shape-friendly GEMMs.
   - Reduce bytes with precision controls (FP8/quantized links) while managing scales and error feedback.
   - Place high-frequency collectives on fast links (e.g., intra-node) whenever possible.
5. **Failure modes and diagnostics.**
   - Symptoms: stragglers, tail latency spikes, allocator fragmentation, imbalance, and collective stalls.
   - Diagnostics: per-rank timeline traces, bucket size sweeps, microbatch sweep, and reshard-count audits.
6. **Decision rubric.**
   - Prefer the smallest parallel axis set meeting memory and throughput targets.
   - Add axes incrementally; after each addition, re-measure MFU, link utilization, and p95/p99 latency.

**Practical takeaway.** This subsection should be implemented as a topology-aware schedule with explicit tensor-layout contracts; otherwise theoretical gains are lost to reshards and synchronization overhead.

### 5.3 Megatron Sequence Parallelism: Activation Sharding, Not Full Long-Context Attention
**Engine 62: scope and objective.** Provide a self-contained treatment of **5.3 Megatron Sequence Parallelism: Activation Sharding, Not Full Long-Context Attention** with definitions, mechanism, cost model, implementation choices, failure modes, and decision rules.

1. **Conceptual model.**
   - 5.3 Megatron Sequence Parallelism: Activation Sharding, Not Full Long-Context Attention is framed as a tensor-placement transformation over the device mesh. Inputs, outputs, and dependency edges are explicitly identified.
   - Correctness condition: transformed execution must preserve numerical equivalence up to floating-point non-associativity and stochastic layers.
2. **Mathematical formulation.**
   - Let local shard size be $n_{local}=n/p$ along the chosen axis. Memory scales as $O(n_{local})$ for sharded states and $O(n)$ for replicated states.
   - Communication lower bound (single phase): $\Omega(n\,\beta)$ payload plus startup term $\Omega(\alpha)$.
3. **Execution recipe.**
   - Precondition checks: mesh groups, dtype policy, seed policy, checkpoint layout metadata.
   - Runtime sequence: schedule compute kernels, issue collectives, place synchronization barriers only on true dependencies, and opportunistically overlap.
4. **Performance engineering.**
   - Increase arithmetic intensity using fused kernels and shape-friendly GEMMs.
   - Reduce bytes with precision controls (FP8/quantized links) while managing scales and error feedback.
   - Place high-frequency collectives on fast links (e.g., intra-node) whenever possible.
5. **Failure modes and diagnostics.**
   - Symptoms: stragglers, tail latency spikes, allocator fragmentation, imbalance, and collective stalls.
   - Diagnostics: per-rank timeline traces, bucket size sweeps, microbatch sweep, and reshard-count audits.
6. **Decision rubric.**
   - Prefer the smallest parallel axis set meeting memory and throughput targets.
   - Add axes incrementally; after each addition, re-measure MFU, link utilization, and p95/p99 latency.

**Practical takeaway.** This subsection should be implemented as a topology-aware schedule with explicit tensor-layout contracts; otherwise theoretical gains are lost to reshards and synchronization overhead.

### 5.4 Ulysses: Sequence Shards Become Head Shards Through All-to-All
**Engine 63: scope and objective.** Provide a self-contained treatment of **5.4 Ulysses: Sequence Shards Become Head Shards Through All-to-All** with definitions, mechanism, cost model, implementation choices, failure modes, and decision rules.

1. **Conceptual model.**
   - 5.4 Ulysses: Sequence Shards Become Head Shards Through All-to-All is framed as a tensor-placement transformation over the device mesh. Inputs, outputs, and dependency edges are explicitly identified.
   - Correctness condition: transformed execution must preserve numerical equivalence up to floating-point non-associativity and stochastic layers.
2. **Mathematical formulation.**
   - Let local shard size be $n_{local}=n/p$ along the chosen axis. Memory scales as $O(n_{local})$ for sharded states and $O(n)$ for replicated states.
   - Communication lower bound (single phase): $\Omega(n\,\beta)$ payload plus startup term $\Omega(\alpha)$.
3. **Execution recipe.**
   - Precondition checks: mesh groups, dtype policy, seed policy, checkpoint layout metadata.
   - Runtime sequence: schedule compute kernels, issue collectives, place synchronization barriers only on true dependencies, and opportunistically overlap.
4. **Performance engineering.**
   - Increase arithmetic intensity using fused kernels and shape-friendly GEMMs.
   - Reduce bytes with precision controls (FP8/quantized links) while managing scales and error feedback.
   - Place high-frequency collectives on fast links (e.g., intra-node) whenever possible.
5. **Failure modes and diagnostics.**
   - Symptoms: stragglers, tail latency spikes, allocator fragmentation, imbalance, and collective stalls.
   - Diagnostics: per-rank timeline traces, bucket size sweeps, microbatch sweep, and reshard-count audits.
6. **Decision rubric.**
   - Prefer the smallest parallel axis set meeting memory and throughput targets.
   - Add axes incrementally; after each addition, re-measure MFU, link utilization, and p95/p99 latency.

**Practical takeaway.** This subsection should be implemented as a topology-aware schedule with explicit tensor-layout contracts; otherwise theoretical gains are lost to reshards and synchronization overhead.

### 5.5 Ring Attention: KV Streaming With Point-to-Point Send/Recv
**Engine 64: scope and objective.** Provide a self-contained treatment of **5.5 Ring Attention: KV Streaming With Point-to-Point Send/Recv** with definitions, mechanism, cost model, implementation choices, failure modes, and decision rules.

1. **Conceptual model.**
   - 5.5 Ring Attention: KV Streaming With Point-to-Point Send/Recv is framed as a tensor-placement transformation over the device mesh. Inputs, outputs, and dependency edges are explicitly identified.
   - Correctness condition: transformed execution must preserve numerical equivalence up to floating-point non-associativity and stochastic layers.
2. **Mathematical formulation.**
   - Let local shard size be $n_{local}=n/p$ along the chosen axis. Memory scales as $O(n_{local})$ for sharded states and $O(n)$ for replicated states.
   - Communication lower bound (single phase): $\Omega(n\,\beta)$ payload plus startup term $\Omega(\alpha)$.
3. **Execution recipe.**
   - Precondition checks: mesh groups, dtype policy, seed policy, checkpoint layout metadata.
   - Runtime sequence: schedule compute kernels, issue collectives, place synchronization barriers only on true dependencies, and opportunistically overlap.
4. **Performance engineering.**
   - Increase arithmetic intensity using fused kernels and shape-friendly GEMMs.
   - Reduce bytes with precision controls (FP8/quantized links) while managing scales and error feedback.
   - Place high-frequency collectives on fast links (e.g., intra-node) whenever possible.
5. **Failure modes and diagnostics.**
   - Symptoms: stragglers, tail latency spikes, allocator fragmentation, imbalance, and collective stalls.
   - Diagnostics: per-rank timeline traces, bucket size sweeps, microbatch sweep, and reshard-count audits.
6. **Decision rubric.**
   - Prefer the smallest parallel axis set meeting memory and throughput targets.
   - Add axes incrementally; after each addition, re-measure MFU, link utilization, and p95/p99 latency.

**Practical takeaway.** This subsection should be implemented as a topology-aware schedule with explicit tensor-layout contracts; otherwise theoretical gains are lost to reshards and synchronization overhead.

### 5.6 The Numerical Primitive: Blockwise Online Softmax
**Engine 65: scope and objective.** Provide a self-contained treatment of **5.6 The Numerical Primitive: Blockwise Online Softmax** with definitions, mechanism, cost model, implementation choices, failure modes, and decision rules.

1. **Conceptual model.**
   - 5.6 The Numerical Primitive: Blockwise Online Softmax is framed as a tensor-placement transformation over the device mesh. Inputs, outputs, and dependency edges are explicitly identified.
   - Correctness condition: transformed execution must preserve numerical equivalence up to floating-point non-associativity and stochastic layers.
2. **Mathematical formulation.**
   - Let local shard size be $n_{local}=n/p$ along the chosen axis. Memory scales as $O(n_{local})$ for sharded states and $O(n)$ for replicated states.
   - Communication lower bound (single phase): $\Omega(n\,\beta)$ payload plus startup term $\Omega(\alpha)$.
3. **Execution recipe.**
   - Precondition checks: mesh groups, dtype policy, seed policy, checkpoint layout metadata.
   - Runtime sequence: schedule compute kernels, issue collectives, place synchronization barriers only on true dependencies, and opportunistically overlap.
4. **Performance engineering.**
   - Increase arithmetic intensity using fused kernels and shape-friendly GEMMs.
   - Reduce bytes with precision controls (FP8/quantized links) while managing scales and error feedback.
   - Place high-frequency collectives on fast links (e.g., intra-node) whenever possible.
5. **Failure modes and diagnostics.**
   - Symptoms: stragglers, tail latency spikes, allocator fragmentation, imbalance, and collective stalls.
   - Diagnostics: per-rank timeline traces, bucket size sweeps, microbatch sweep, and reshard-count audits.
6. **Decision rubric.**
   - Prefer the smallest parallel axis set meeting memory and throughput targets.
   - Add axes incrementally; after each addition, re-measure MFU, link utilization, and p95/p99 latency.

**Practical takeaway.** This subsection should be implemented as a topology-aware schedule with explicit tensor-layout contracts; otherwise theoretical gains are lost to reshards and synchronization overhead.

### 5.7 Megatron Context Parallelism: Sequence Sharding for the Whole Network
**Engine 66: scope and objective.** Provide a self-contained treatment of **5.7 Megatron Context Parallelism: Sequence Sharding for the Whole Network** with definitions, mechanism, cost model, implementation choices, failure modes, and decision rules.

1. **Conceptual model.**
   - 5.7 Megatron Context Parallelism: Sequence Sharding for the Whole Network is framed as a tensor-placement transformation over the device mesh. Inputs, outputs, and dependency edges are explicitly identified.
   - Correctness condition: transformed execution must preserve numerical equivalence up to floating-point non-associativity and stochastic layers.
2. **Mathematical formulation.**
   - Let local shard size be $n_{local}=n/p$ along the chosen axis. Memory scales as $O(n_{local})$ for sharded states and $O(n)$ for replicated states.
   - Communication lower bound (single phase): $\Omega(n\,\beta)$ payload plus startup term $\Omega(\alpha)$.
3. **Execution recipe.**
   - Precondition checks: mesh groups, dtype policy, seed policy, checkpoint layout metadata.
   - Runtime sequence: schedule compute kernels, issue collectives, place synchronization barriers only on true dependencies, and opportunistically overlap.
4. **Performance engineering.**
   - Increase arithmetic intensity using fused kernels and shape-friendly GEMMs.
   - Reduce bytes with precision controls (FP8/quantized links) while managing scales and error feedback.
   - Place high-frequency collectives on fast links (e.g., intra-node) whenever possible.
5. **Failure modes and diagnostics.**
   - Symptoms: stragglers, tail latency spikes, allocator fragmentation, imbalance, and collective stalls.
   - Diagnostics: per-rank timeline traces, bucket size sweeps, microbatch sweep, and reshard-count audits.
6. **Decision rubric.**
   - Prefer the smallest parallel axis set meeting memory and throughput targets.
   - Add axes incrementally; after each addition, re-measure MFU, link utilization, and p95/p99 latency.

**Practical takeaway.** This subsection should be implemented as a topology-aware schedule with explicit tensor-layout contracts; otherwise theoretical gains are lost to reshards and synchronization overhead.

### 5.8 Dynamic Context Parallelism: Choose CP Size Per Microbatch
**Engine 67: scope and objective.** Provide a self-contained treatment of **5.8 Dynamic Context Parallelism: Choose CP Size Per Microbatch** with definitions, mechanism, cost model, implementation choices, failure modes, and decision rules.

1. **Conceptual model.**
   - 5.8 Dynamic Context Parallelism: Choose CP Size Per Microbatch is framed as a tensor-placement transformation over the device mesh. Inputs, outputs, and dependency edges are explicitly identified.
   - Correctness condition: transformed execution must preserve numerical equivalence up to floating-point non-associativity and stochastic layers.
2. **Mathematical formulation.**
   - Let local shard size be $n_{local}=n/p$ along the chosen axis. Memory scales as $O(n_{local})$ for sharded states and $O(n)$ for replicated states.
   - Communication lower bound (single phase): $\Omega(n\,\beta)$ payload plus startup term $\Omega(\alpha)$.
3. **Execution recipe.**
   - Precondition checks: mesh groups, dtype policy, seed policy, checkpoint layout metadata.
   - Runtime sequence: schedule compute kernels, issue collectives, place synchronization barriers only on true dependencies, and opportunistically overlap.
4. **Performance engineering.**
   - Increase arithmetic intensity using fused kernels and shape-friendly GEMMs.
   - Reduce bytes with precision controls (FP8/quantized links) while managing scales and error feedback.
   - Place high-frequency collectives on fast links (e.g., intra-node) whenever possible.
5. **Failure modes and diagnostics.**
   - Symptoms: stragglers, tail latency spikes, allocator fragmentation, imbalance, and collective stalls.
   - Diagnostics: per-rank timeline traces, bucket size sweeps, microbatch sweep, and reshard-count audits.
6. **Decision rubric.**
   - Prefer the smallest parallel axis set meeting memory and throughput targets.
   - Add axes incrementally; after each addition, re-measure MFU, link utilization, and p95/p99 latency.

**Practical takeaway.** This subsection should be implemented as a topology-aware schedule with explicit tensor-layout contracts; otherwise theoretical gains are lost to reshards and synchronization overhead.

### 5.9 THD and Ragged Packing: Token-Head-Dimension Layout
**Engine 68: scope and objective.** Provide a self-contained treatment of **5.9 THD and Ragged Packing: Token-Head-Dimension Layout** with definitions, mechanism, cost model, implementation choices, failure modes, and decision rules.

1. **Conceptual model.**
   - 5.9 THD and Ragged Packing: Token-Head-Dimension Layout is framed as a tensor-placement transformation over the device mesh. Inputs, outputs, and dependency edges are explicitly identified.
   - Correctness condition: transformed execution must preserve numerical equivalence up to floating-point non-associativity and stochastic layers.
2. **Mathematical formulation.**
   - Let local shard size be $n_{local}=n/p$ along the chosen axis. Memory scales as $O(n_{local})$ for sharded states and $O(n)$ for replicated states.
   - Communication lower bound (single phase): $\Omega(n\,\beta)$ payload plus startup term $\Omega(\alpha)$.
3. **Execution recipe.**
   - Precondition checks: mesh groups, dtype policy, seed policy, checkpoint layout metadata.
   - Runtime sequence: schedule compute kernels, issue collectives, place synchronization barriers only on true dependencies, and opportunistically overlap.
4. **Performance engineering.**
   - Increase arithmetic intensity using fused kernels and shape-friendly GEMMs.
   - Reduce bytes with precision controls (FP8/quantized links) while managing scales and error feedback.
   - Place high-frequency collectives on fast links (e.g., intra-node) whenever possible.
5. **Failure modes and diagnostics.**
   - Symptoms: stragglers, tail latency spikes, allocator fragmentation, imbalance, and collective stalls.
   - Diagnostics: per-rank timeline traces, bucket size sweeps, microbatch sweep, and reshard-count audits.
6. **Decision rubric.**
   - Prefer the smallest parallel axis set meeting memory and throughput targets.
   - Add axes incrementally; after each addition, re-measure MFU, link utilization, and p95/p99 latency.

**Practical takeaway.** This subsection should be implemented as a topology-aware schedule with explicit tensor-layout contracts; otherwise theoretical gains are lost to reshards and synchronization overhead.

### 5.10 Sparse and Hybrid Attention: Communicate Selected KV, Not Necessarily All KV
**Engine 69: scope and objective.** Provide a self-contained treatment of **5.10 Sparse and Hybrid Attention: Communicate Selected KV, Not Necessarily All KV** with definitions, mechanism, cost model, implementation choices, failure modes, and decision rules.

1. **Conceptual model.**
   - 5.10 Sparse and Hybrid Attention: Communicate Selected KV, Not Necessarily All KV is framed as a tensor-placement transformation over the device mesh. Inputs, outputs, and dependency edges are explicitly identified.
   - Correctness condition: transformed execution must preserve numerical equivalence up to floating-point non-associativity and stochastic layers.
2. **Mathematical formulation.**
   - Let local shard size be $n_{local}=n/p$ along the chosen axis. Memory scales as $O(n_{local})$ for sharded states and $O(n)$ for replicated states.
   - Communication lower bound (single phase): $\Omega(n\,\beta)$ payload plus startup term $\Omega(\alpha)$.
3. **Execution recipe.**
   - Precondition checks: mesh groups, dtype policy, seed policy, checkpoint layout metadata.
   - Runtime sequence: schedule compute kernels, issue collectives, place synchronization barriers only on true dependencies, and opportunistically overlap.
4. **Performance engineering.**
   - Increase arithmetic intensity using fused kernels and shape-friendly GEMMs.
   - Reduce bytes with precision controls (FP8/quantized links) while managing scales and error feedback.
   - Place high-frequency collectives on fast links (e.g., intra-node) whenever possible.
5. **Failure modes and diagnostics.**
   - Symptoms: stragglers, tail latency spikes, allocator fragmentation, imbalance, and collective stalls.
   - Diagnostics: per-rank timeline traces, bucket size sweeps, microbatch sweep, and reshard-count audits.
6. **Decision rubric.**
   - Prefer the smallest parallel axis set meeting memory and throughput targets.
   - Add axes incrementally; after each addition, re-measure MFU, link utilization, and p95/p99 latency.

**Practical takeaway.** This subsection should be implemented as a topology-aware schedule with explicit tensor-layout contracts; otherwise theoretical gains are lost to reshards and synchronization overhead.

### 5.11 Inference KV and Context Placement
**Engine 70: scope and objective.** Provide a self-contained treatment of **5.11 Inference KV and Context Placement** with definitions, mechanism, cost model, implementation choices, failure modes, and decision rules.

1. **Conceptual model.**
   - 5.11 Inference KV and Context Placement is framed as a tensor-placement transformation over the device mesh. Inputs, outputs, and dependency edges are explicitly identified.
   - Correctness condition: transformed execution must preserve numerical equivalence up to floating-point non-associativity and stochastic layers.
2. **Mathematical formulation.**
   - Let local shard size be $n_{local}=n/p$ along the chosen axis. Memory scales as $O(n_{local})$ for sharded states and $O(n)$ for replicated states.
   - Communication lower bound (single phase): $\Omega(n\,\beta)$ payload plus startup term $\Omega(\alpha)$.
3. **Execution recipe.**
   - Precondition checks: mesh groups, dtype policy, seed policy, checkpoint layout metadata.
   - Runtime sequence: schedule compute kernels, issue collectives, place synchronization barriers only on true dependencies, and opportunistically overlap.
4. **Performance engineering.**
   - Increase arithmetic intensity using fused kernels and shape-friendly GEMMs.
   - Reduce bytes with precision controls (FP8/quantized links) while managing scales and error feedback.
   - Place high-frequency collectives on fast links (e.g., intra-node) whenever possible.
5. **Failure modes and diagnostics.**
   - Symptoms: stragglers, tail latency spikes, allocator fragmentation, imbalance, and collective stalls.
   - Diagnostics: per-rank timeline traces, bucket size sweeps, microbatch sweep, and reshard-count audits.
6. **Decision rubric.**
   - Prefer the smallest parallel axis set meeting memory and throughput targets.
   - Add axes incrementally; after each addition, re-measure MFU, link utilization, and p95/p99 latency.

**Practical takeaway.** This subsection should be implemented as a topology-aware schedule with explicit tensor-layout contracts; otherwise theoretical gains are lost to reshards and synchronization overhead.

### 5.12 How to Choose Among SP, Ulysses, Ring, CP, Dynamic-CP, and Sparse Context Placement
**Engine 71: scope and objective.** Provide a self-contained treatment of **5.12 How to Choose Among SP, Ulysses, Ring, CP, Dynamic-CP, and Sparse Context Placement** with definitions, mechanism, cost model, implementation choices, failure modes, and decision rules.

1. **Conceptual model.**
   - 5.12 How to Choose Among SP, Ulysses, Ring, CP, Dynamic-CP, and Sparse Context Placement is framed as a tensor-placement transformation over the device mesh. Inputs, outputs, and dependency edges are explicitly identified.
   - Correctness condition: transformed execution must preserve numerical equivalence up to floating-point non-associativity and stochastic layers.
2. **Mathematical formulation.**
   - Let local shard size be $n_{local}=n/p$ along the chosen axis. Memory scales as $O(n_{local})$ for sharded states and $O(n)$ for replicated states.
   - Communication lower bound (single phase): $\Omega(n\,\beta)$ payload plus startup term $\Omega(\alpha)$.
3. **Execution recipe.**
   - Precondition checks: mesh groups, dtype policy, seed policy, checkpoint layout metadata.
   - Runtime sequence: schedule compute kernels, issue collectives, place synchronization barriers only on true dependencies, and opportunistically overlap.
4. **Performance engineering.**
   - Increase arithmetic intensity using fused kernels and shape-friendly GEMMs.
   - Reduce bytes with precision controls (FP8/quantized links) while managing scales and error feedback.
   - Place high-frequency collectives on fast links (e.g., intra-node) whenever possible.
5. **Failure modes and diagnostics.**
   - Symptoms: stragglers, tail latency spikes, allocator fragmentation, imbalance, and collective stalls.
   - Diagnostics: per-rank timeline traces, bucket size sweeps, microbatch sweep, and reshard-count audits.
6. **Decision rubric.**
   - Prefer the smallest parallel axis set meeting memory and throughput targets.
   - Add axes incrementally; after each addition, re-measure MFU, link utilization, and p95/p99 latency.

**Practical takeaway.** This subsection should be implemented as a topology-aware schedule with explicit tensor-layout contracts; otherwise theoretical gains are lost to reshards and synchronization overhead.

### 5.13 Evidence Boundaries and Terminology Caveats
**Engine 72: scope and objective.** Provide a self-contained treatment of **5.13 Evidence Boundaries and Terminology Caveats** with definitions, mechanism, cost model, implementation choices, failure modes, and decision rules.

1. **Conceptual model.**
   - 5.13 Evidence Boundaries and Terminology Caveats is framed as a tensor-placement transformation over the device mesh. Inputs, outputs, and dependency edges are explicitly identified.
   - Correctness condition: transformed execution must preserve numerical equivalence up to floating-point non-associativity and stochastic layers.
2. **Mathematical formulation.**
   - Let local shard size be $n_{local}=n/p$ along the chosen axis. Memory scales as $O(n_{local})$ for sharded states and $O(n)$ for replicated states.
   - Communication lower bound (single phase): $\Omega(n\,\beta)$ payload plus startup term $\Omega(\alpha)$.
3. **Execution recipe.**
   - Precondition checks: mesh groups, dtype policy, seed policy, checkpoint layout metadata.
   - Runtime sequence: schedule compute kernels, issue collectives, place synchronization barriers only on true dependencies, and opportunistically overlap.
4. **Performance engineering.**
   - Increase arithmetic intensity using fused kernels and shape-friendly GEMMs.
   - Reduce bytes with precision controls (FP8/quantized links) while managing scales and error feedback.
   - Place high-frequency collectives on fast links (e.g., intra-node) whenever possible.
5. **Failure modes and diagnostics.**
   - Symptoms: stragglers, tail latency spikes, allocator fragmentation, imbalance, and collective stalls.
   - Diagnostics: per-rank timeline traces, bucket size sweeps, microbatch sweep, and reshard-count audits.
6. **Decision rubric.**
   - Prefer the smallest parallel axis set meeting memory and throughput targets.
   - Add axes incrementally; after each addition, re-measure MFU, link utilization, and p95/p99 latency.

**Practical takeaway.** This subsection should be implemented as a topology-aware schedule with explicit tensor-layout contracts; otherwise theoretical gains are lost to reshards and synchronization overhead.

## 6. Expert and MoE Parallelism
**Engine 73: scope and objective.** Provide a self-contained treatment of **6. Expert and MoE Parallelism** with definitions, mechanism, cost model, implementation choices, failure modes, and decision rules.

1. **Conceptual model.**
   - 6. Expert and MoE Parallelism is framed as a tensor-placement transformation over the device mesh. Inputs, outputs, and dependency edges are explicitly identified.
   - Correctness condition: transformed execution must preserve numerical equivalence up to floating-point non-associativity and stochastic layers.
2. **Mathematical formulation.**
   - Let local shard size be $n_{local}=n/p$ along the chosen axis. Memory scales as $O(n_{local})$ for sharded states and $O(n)$ for replicated states.
   - Communication lower bound (single phase): $\Omega(n\,\beta)$ payload plus startup term $\Omega(\alpha)$.
3. **Execution recipe.**
   - Precondition checks: mesh groups, dtype policy, seed policy, checkpoint layout metadata.
   - Runtime sequence: schedule compute kernels, issue collectives, place synchronization barriers only on true dependencies, and opportunistically overlap.
4. **Performance engineering.**
   - Increase arithmetic intensity using fused kernels and shape-friendly GEMMs.
   - Reduce bytes with precision controls (FP8/quantized links) while managing scales and error feedback.
   - Place high-frequency collectives on fast links (e.g., intra-node) whenever possible.
5. **Failure modes and diagnostics.**
   - Symptoms: stragglers, tail latency spikes, allocator fragmentation, imbalance, and collective stalls.
   - Diagnostics: per-rank timeline traces, bucket size sweeps, microbatch sweep, and reshard-count audits.
6. **Decision rubric.**
   - Prefer the smallest parallel axis set meeting memory and throughput targets.
   - Add axes incrementally; after each addition, re-measure MFU, link utilization, and p95/p99 latency.

**Practical takeaway.** This subsection should be implemented as a topology-aware schedule with explicit tensor-layout contracts; otherwise theoretical gains are lost to reshards and synchronization overhead.

### 6.1 Symbols, Shapes, and the MoE Layer
**Engine 74: scope and objective.** Provide a self-contained treatment of **6.1 Symbols, Shapes, and the MoE Layer** with definitions, mechanism, cost model, implementation choices, failure modes, and decision rules.

1. **Conceptual model.**
   - 6.1 Symbols, Shapes, and the MoE Layer is framed as a tensor-placement transformation over the device mesh. Inputs, outputs, and dependency edges are explicitly identified.
   - Correctness condition: transformed execution must preserve numerical equivalence up to floating-point non-associativity and stochastic layers.
2. **Mathematical formulation.**
   - Let local shard size be $n_{local}=n/p$ along the chosen axis. Memory scales as $O(n_{local})$ for sharded states and $O(n)$ for replicated states.
   - Communication lower bound (single phase): $\Omega(n\,\beta)$ payload plus startup term $\Omega(\alpha)$.
3. **Execution recipe.**
   - Precondition checks: mesh groups, dtype policy, seed policy, checkpoint layout metadata.
   - Runtime sequence: schedule compute kernels, issue collectives, place synchronization barriers only on true dependencies, and opportunistically overlap.
4. **Performance engineering.**
   - Increase arithmetic intensity using fused kernels and shape-friendly GEMMs.
   - Reduce bytes with precision controls (FP8/quantized links) while managing scales and error feedback.
   - Place high-frequency collectives on fast links (e.g., intra-node) whenever possible.
5. **Failure modes and diagnostics.**
   - Symptoms: stragglers, tail latency spikes, allocator fragmentation, imbalance, and collective stalls.
   - Diagnostics: per-rank timeline traces, bucket size sweeps, microbatch sweep, and reshard-count audits.
6. **Decision rubric.**
   - Prefer the smallest parallel axis set meeting memory and throughput targets.
   - Add axes incrementally; after each addition, re-measure MFU, link utilization, and p95/p99 latency.

**Practical takeaway.** This subsection should be implemented as a topology-aware schedule with explicit tensor-layout contracts; otherwise theoretical gains are lost to reshards and synchronization overhead.

### 6.2 Primitive Technique: Routing and Top-k Assignment
**Engine 75: scope and objective.** Provide a self-contained treatment of **6.2 Primitive Technique: Routing and Top-k Assignment** with definitions, mechanism, cost model, implementation choices, failure modes, and decision rules.

1. **Conceptual model.**
   - 6.2 Primitive Technique: Routing and Top-k Assignment is framed as a tensor-placement transformation over the device mesh. Inputs, outputs, and dependency edges are explicitly identified.
   - Correctness condition: transformed execution must preserve numerical equivalence up to floating-point non-associativity and stochastic layers.
2. **Mathematical formulation.**
   - Let local shard size be $n_{local}=n/p$ along the chosen axis. Memory scales as $O(n_{local})$ for sharded states and $O(n)$ for replicated states.
   - Communication lower bound (single phase): $\Omega(n\,\beta)$ payload plus startup term $\Omega(\alpha)$.
3. **Execution recipe.**
   - Precondition checks: mesh groups, dtype policy, seed policy, checkpoint layout metadata.
   - Runtime sequence: schedule compute kernels, issue collectives, place synchronization barriers only on true dependencies, and opportunistically overlap.
4. **Performance engineering.**
   - Increase arithmetic intensity using fused kernels and shape-friendly GEMMs.
   - Reduce bytes with precision controls (FP8/quantized links) while managing scales and error feedback.
   - Place high-frequency collectives on fast links (e.g., intra-node) whenever possible.
5. **Failure modes and diagnostics.**
   - Symptoms: stragglers, tail latency spikes, allocator fragmentation, imbalance, and collective stalls.
   - Diagnostics: per-rank timeline traces, bucket size sweeps, microbatch sweep, and reshard-count audits.
6. **Decision rubric.**
   - Prefer the smallest parallel axis set meeting memory and throughput targets.
   - Add axes incrementally; after each addition, re-measure MFU, link utilization, and p95/p99 latency.

**Practical takeaway.** This subsection should be implemented as a topology-aware schedule with explicit tensor-layout contracts; otherwise theoretical gains are lost to reshards and synchronization overhead.

### 6.3 Primitive Technique: Expert Ownership and Local Shard Shapes
**Engine 76: scope and objective.** Provide a self-contained treatment of **6.3 Primitive Technique: Expert Ownership and Local Shard Shapes** with definitions, mechanism, cost model, implementation choices, failure modes, and decision rules.

1. **Conceptual model.**
   - 6.3 Primitive Technique: Expert Ownership and Local Shard Shapes is framed as a tensor-placement transformation over the device mesh. Inputs, outputs, and dependency edges are explicitly identified.
   - Correctness condition: transformed execution must preserve numerical equivalence up to floating-point non-associativity and stochastic layers.
2. **Mathematical formulation.**
   - Let local shard size be $n_{local}=n/p$ along the chosen axis. Memory scales as $O(n_{local})$ for sharded states and $O(n)$ for replicated states.
   - Communication lower bound (single phase): $\Omega(n\,\beta)$ payload plus startup term $\Omega(\alpha)$.
3. **Execution recipe.**
   - Precondition checks: mesh groups, dtype policy, seed policy, checkpoint layout metadata.
   - Runtime sequence: schedule compute kernels, issue collectives, place synchronization barriers only on true dependencies, and opportunistically overlap.
4. **Performance engineering.**
   - Increase arithmetic intensity using fused kernels and shape-friendly GEMMs.
   - Reduce bytes with precision controls (FP8/quantized links) while managing scales and error feedback.
   - Place high-frequency collectives on fast links (e.g., intra-node) whenever possible.
5. **Failure modes and diagnostics.**
   - Symptoms: stragglers, tail latency spikes, allocator fragmentation, imbalance, and collective stalls.
   - Diagnostics: per-rank timeline traces, bucket size sweeps, microbatch sweep, and reshard-count audits.
6. **Decision rubric.**
   - Prefer the smallest parallel axis set meeting memory and throughput targets.
   - Add axes incrementally; after each addition, re-measure MFU, link utilization, and p95/p99 latency.

**Practical takeaway.** This subsection should be implemented as a topology-aware schedule with explicit tensor-layout contracts; otherwise theoretical gains are lost to reshards and synchronization overhead.

### 6.4 Primitive Technique: Permutation, Packing, and Inverse Permutation
**Engine 77: scope and objective.** Provide a self-contained treatment of **6.4 Primitive Technique: Permutation, Packing, and Inverse Permutation** with definitions, mechanism, cost model, implementation choices, failure modes, and decision rules.

1. **Conceptual model.**
   - 6.4 Primitive Technique: Permutation, Packing, and Inverse Permutation is framed as a tensor-placement transformation over the device mesh. Inputs, outputs, and dependency edges are explicitly identified.
   - Correctness condition: transformed execution must preserve numerical equivalence up to floating-point non-associativity and stochastic layers.
2. **Mathematical formulation.**
   - Let local shard size be $n_{local}=n/p$ along the chosen axis. Memory scales as $O(n_{local})$ for sharded states and $O(n)$ for replicated states.
   - Communication lower bound (single phase): $\Omega(n\,\beta)$ payload plus startup term $\Omega(\alpha)$.
3. **Execution recipe.**
   - Precondition checks: mesh groups, dtype policy, seed policy, checkpoint layout metadata.
   - Runtime sequence: schedule compute kernels, issue collectives, place synchronization barriers only on true dependencies, and opportunistically overlap.
4. **Performance engineering.**
   - Increase arithmetic intensity using fused kernels and shape-friendly GEMMs.
   - Reduce bytes with precision controls (FP8/quantized links) while managing scales and error feedback.
   - Place high-frequency collectives on fast links (e.g., intra-node) whenever possible.
5. **Failure modes and diagnostics.**
   - Symptoms: stragglers, tail latency spikes, allocator fragmentation, imbalance, and collective stalls.
   - Diagnostics: per-rank timeline traces, bucket size sweeps, microbatch sweep, and reshard-count audits.
6. **Decision rubric.**
   - Prefer the smallest parallel axis set meeting memory and throughput targets.
   - Add axes incrementally; after each addition, re-measure MFU, link utilization, and p95/p99 latency.

**Practical takeaway.** This subsection should be implemented as a topology-aware schedule with explicit tensor-layout contracts; otherwise theoretical gains are lost to reshards and synchronization overhead.

### 6.5 Primitive Technique: All-to-All Dispatch, Notify, and Combine
**Engine 78: scope and objective.** Provide a self-contained treatment of **6.5 Primitive Technique: All-to-All Dispatch, Notify, and Combine** with definitions, mechanism, cost model, implementation choices, failure modes, and decision rules.

1. **Conceptual model.**
   - 6.5 Primitive Technique: All-to-All Dispatch, Notify, and Combine is framed as a tensor-placement transformation over the device mesh. Inputs, outputs, and dependency edges are explicitly identified.
   - Correctness condition: transformed execution must preserve numerical equivalence up to floating-point non-associativity and stochastic layers.
2. **Mathematical formulation.**
   - Let local shard size be $n_{local}=n/p$ along the chosen axis. Memory scales as $O(n_{local})$ for sharded states and $O(n)$ for replicated states.
   - Communication lower bound (single phase): $\Omega(n\,\beta)$ payload plus startup term $\Omega(\alpha)$.
3. **Execution recipe.**
   - Precondition checks: mesh groups, dtype policy, seed policy, checkpoint layout metadata.
   - Runtime sequence: schedule compute kernels, issue collectives, place synchronization barriers only on true dependencies, and opportunistically overlap.
4. **Performance engineering.**
   - Increase arithmetic intensity using fused kernels and shape-friendly GEMMs.
   - Reduce bytes with precision controls (FP8/quantized links) while managing scales and error feedback.
   - Place high-frequency collectives on fast links (e.g., intra-node) whenever possible.
5. **Failure modes and diagnostics.**
   - Symptoms: stragglers, tail latency spikes, allocator fragmentation, imbalance, and collective stalls.
   - Diagnostics: per-rank timeline traces, bucket size sweeps, microbatch sweep, and reshard-count audits.
6. **Decision rubric.**
   - Prefer the smallest parallel axis set meeting memory and throughput targets.
   - Add axes incrementally; after each addition, re-measure MFU, link utilization, and p95/p99 latency.

**Practical takeaway.** This subsection should be implemented as a topology-aware schedule with explicit tensor-layout contracts; otherwise theoretical gains are lost to reshards and synchronization overhead.

### 6.6 Primitive Technique: Capacity, Dropless Execution, and Padding
**Engine 79: scope and objective.** Provide a self-contained treatment of **6.6 Primitive Technique: Capacity, Dropless Execution, and Padding** with definitions, mechanism, cost model, implementation choices, failure modes, and decision rules.

1. **Conceptual model.**
   - 6.6 Primitive Technique: Capacity, Dropless Execution, and Padding is framed as a tensor-placement transformation over the device mesh. Inputs, outputs, and dependency edges are explicitly identified.
   - Correctness condition: transformed execution must preserve numerical equivalence up to floating-point non-associativity and stochastic layers.
2. **Mathematical formulation.**
   - Let local shard size be $n_{local}=n/p$ along the chosen axis. Memory scales as $O(n_{local})$ for sharded states and $O(n)$ for replicated states.
   - Communication lower bound (single phase): $\Omega(n\,\beta)$ payload plus startup term $\Omega(\alpha)$.
3. **Execution recipe.**
   - Precondition checks: mesh groups, dtype policy, seed policy, checkpoint layout metadata.
   - Runtime sequence: schedule compute kernels, issue collectives, place synchronization barriers only on true dependencies, and opportunistically overlap.
4. **Performance engineering.**
   - Increase arithmetic intensity using fused kernels and shape-friendly GEMMs.
   - Reduce bytes with precision controls (FP8/quantized links) while managing scales and error feedback.
   - Place high-frequency collectives on fast links (e.g., intra-node) whenever possible.
5. **Failure modes and diagnostics.**
   - Symptoms: stragglers, tail latency spikes, allocator fragmentation, imbalance, and collective stalls.
   - Diagnostics: per-rank timeline traces, bucket size sweeps, microbatch sweep, and reshard-count audits.
6. **Decision rubric.**
   - Prefer the smallest parallel axis set meeting memory and throughput targets.
   - Add axes incrementally; after each addition, re-measure MFU, link utilization, and p95/p99 latency.

**Practical takeaway.** This subsection should be implemented as a topology-aware schedule with explicit tensor-layout contracts; otherwise theoretical gains are lost to reshards and synchronization overhead.

### 6.7 Primitive Technique: Grouped GEMM, MegaBlocks, DeepGEMM, and Mega MoE
**Engine 80: scope and objective.** Provide a self-contained treatment of **6.7 Primitive Technique: Grouped GEMM, MegaBlocks, DeepGEMM, and Mega MoE** with definitions, mechanism, cost model, implementation choices, failure modes, and decision rules.

1. **Conceptual model.**
   - 6.7 Primitive Technique: Grouped GEMM, MegaBlocks, DeepGEMM, and Mega MoE is framed as a tensor-placement transformation over the device mesh. Inputs, outputs, and dependency edges are explicitly identified.
   - Correctness condition: transformed execution must preserve numerical equivalence up to floating-point non-associativity and stochastic layers.
2. **Mathematical formulation.**
   - Let local shard size be $n_{local}=n/p$ along the chosen axis. Memory scales as $O(n_{local})$ for sharded states and $O(n)$ for replicated states.
   - Communication lower bound (single phase): $\Omega(n\,\beta)$ payload plus startup term $\Omega(\alpha)$.
3. **Execution recipe.**
   - Precondition checks: mesh groups, dtype policy, seed policy, checkpoint layout metadata.
   - Runtime sequence: schedule compute kernels, issue collectives, place synchronization barriers only on true dependencies, and opportunistically overlap.
4. **Performance engineering.**
   - Increase arithmetic intensity using fused kernels and shape-friendly GEMMs.
   - Reduce bytes with precision controls (FP8/quantized links) while managing scales and error feedback.
   - Place high-frequency collectives on fast links (e.g., intra-node) whenever possible.
5. **Failure modes and diagnostics.**
   - Symptoms: stragglers, tail latency spikes, allocator fragmentation, imbalance, and collective stalls.
   - Diagnostics: per-rank timeline traces, bucket size sweeps, microbatch sweep, and reshard-count audits.
6. **Decision rubric.**
   - Prefer the smallest parallel axis set meeting memory and throughput targets.
   - Add axes incrementally; after each addition, re-measure MFU, link utilization, and p95/p99 latency.

**Practical takeaway.** This subsection should be implemented as a topology-aware schedule with explicit tensor-layout contracts; otherwise theoretical gains are lost to reshards and synchronization overhead.

### 6.8 Primitive Technique: Backward Pass, Weight Gradients, and Overlap
**Engine 81: scope and objective.** Provide a self-contained treatment of **6.8 Primitive Technique: Backward Pass, Weight Gradients, and Overlap** with definitions, mechanism, cost model, implementation choices, failure modes, and decision rules.

1. **Conceptual model.**
   - 6.8 Primitive Technique: Backward Pass, Weight Gradients, and Overlap is framed as a tensor-placement transformation over the device mesh. Inputs, outputs, and dependency edges are explicitly identified.
   - Correctness condition: transformed execution must preserve numerical equivalence up to floating-point non-associativity and stochastic layers.
2. **Mathematical formulation.**
   - Let local shard size be $n_{local}=n/p$ along the chosen axis. Memory scales as $O(n_{local})$ for sharded states and $O(n)$ for replicated states.
   - Communication lower bound (single phase): $\Omega(n\,\beta)$ payload plus startup term $\Omega(\alpha)$.
3. **Execution recipe.**
   - Precondition checks: mesh groups, dtype policy, seed policy, checkpoint layout metadata.
   - Runtime sequence: schedule compute kernels, issue collectives, place synchronization barriers only on true dependencies, and opportunistically overlap.
4. **Performance engineering.**
   - Increase arithmetic intensity using fused kernels and shape-friendly GEMMs.
   - Reduce bytes with precision controls (FP8/quantized links) while managing scales and error feedback.
   - Place high-frequency collectives on fast links (e.g., intra-node) whenever possible.
5. **Failure modes and diagnostics.**
   - Symptoms: stragglers, tail latency spikes, allocator fragmentation, imbalance, and collective stalls.
   - Diagnostics: per-rank timeline traces, bucket size sweeps, microbatch sweep, and reshard-count audits.
6. **Decision rubric.**
   - Prefer the smallest parallel axis set meeting memory and throughput targets.
   - Add axes incrementally; after each addition, re-measure MFU, link utilization, and p95/p99 latency.

**Practical takeaway.** This subsection should be implemented as a topology-aware schedule with explicit tensor-layout contracts; otherwise theoretical gains are lost to reshards and synchronization overhead.

### 6.9 Primitive Technique: DeepEP Modes and Network-Aware EP
**Engine 82: scope and objective.** Provide a self-contained treatment of **6.9 Primitive Technique: DeepEP Modes and Network-Aware EP** with definitions, mechanism, cost model, implementation choices, failure modes, and decision rules.

1. **Conceptual model.**
   - 6.9 Primitive Technique: DeepEP Modes and Network-Aware EP is framed as a tensor-placement transformation over the device mesh. Inputs, outputs, and dependency edges are explicitly identified.
   - Correctness condition: transformed execution must preserve numerical equivalence up to floating-point non-associativity and stochastic layers.
2. **Mathematical formulation.**
   - Let local shard size be $n_{local}=n/p$ along the chosen axis. Memory scales as $O(n_{local})$ for sharded states and $O(n)$ for replicated states.
   - Communication lower bound (single phase): $\Omega(n\,\beta)$ payload plus startup term $\Omega(\alpha)$.
3. **Execution recipe.**
   - Precondition checks: mesh groups, dtype policy, seed policy, checkpoint layout metadata.
   - Runtime sequence: schedule compute kernels, issue collectives, place synchronization barriers only on true dependencies, and opportunistically overlap.
4. **Performance engineering.**
   - Increase arithmetic intensity using fused kernels and shape-friendly GEMMs.
   - Reduce bytes with precision controls (FP8/quantized links) while managing scales and error feedback.
   - Place high-frequency collectives on fast links (e.g., intra-node) whenever possible.
5. **Failure modes and diagnostics.**
   - Symptoms: stragglers, tail latency spikes, allocator fragmentation, imbalance, and collective stalls.
   - Diagnostics: per-rank timeline traces, bucket size sweeps, microbatch sweep, and reshard-count audits.
6. **Decision rubric.**
   - Prefer the smallest parallel axis set meeting memory and throughput targets.
   - Add axes incrementally; after each addition, re-measure MFU, link utilization, and p95/p99 latency.

**Practical takeaway.** This subsection should be implemented as a topology-aware schedule with explicit tensor-layout contracts; otherwise theoretical gains are lost to reshards and synchronization overhead.

### 6.10 Primitive Technique: Wide-EP Serving and Hot-Expert Replication
**Engine 83: scope and objective.** Provide a self-contained treatment of **6.10 Primitive Technique: Wide-EP Serving and Hot-Expert Replication** with definitions, mechanism, cost model, implementation choices, failure modes, and decision rules.

1. **Conceptual model.**
   - 6.10 Primitive Technique: Wide-EP Serving and Hot-Expert Replication is framed as a tensor-placement transformation over the device mesh. Inputs, outputs, and dependency edges are explicitly identified.
   - Correctness condition: transformed execution must preserve numerical equivalence up to floating-point non-associativity and stochastic layers.
2. **Mathematical formulation.**
   - Let local shard size be $n_{local}=n/p$ along the chosen axis. Memory scales as $O(n_{local})$ for sharded states and $O(n)$ for replicated states.
   - Communication lower bound (single phase): $\Omega(n\,\beta)$ payload plus startup term $\Omega(\alpha)$.
3. **Execution recipe.**
   - Precondition checks: mesh groups, dtype policy, seed policy, checkpoint layout metadata.
   - Runtime sequence: schedule compute kernels, issue collectives, place synchronization barriers only on true dependencies, and opportunistically overlap.
4. **Performance engineering.**
   - Increase arithmetic intensity using fused kernels and shape-friendly GEMMs.
   - Reduce bytes with precision controls (FP8/quantized links) while managing scales and error feedback.
   - Place high-frequency collectives on fast links (e.g., intra-node) whenever possible.
5. **Failure modes and diagnostics.**
   - Symptoms: stragglers, tail latency spikes, allocator fragmentation, imbalance, and collective stalls.
   - Diagnostics: per-rank timeline traces, bucket size sweeps, microbatch sweep, and reshard-count audits.
6. **Decision rubric.**
   - Prefer the smallest parallel axis set meeting memory and throughput targets.
   - Add axes incrementally; after each addition, re-measure MFU, link utilization, and p95/p99 latency.

**Practical takeaway.** This subsection should be implemented as a topology-aware schedule with explicit tensor-layout contracts; otherwise theoretical gains are lost to reshards and synchronization overhead.

### 6.11 Composition With TP, SP, PP, CP, DP, and Precision
**Engine 84: scope and objective.** Provide a self-contained treatment of **6.11 Composition With TP, SP, PP, CP, DP, and Precision** with definitions, mechanism, cost model, implementation choices, failure modes, and decision rules.

1. **Conceptual model.**
   - 6.11 Composition With TP, SP, PP, CP, DP, and Precision is framed as a tensor-placement transformation over the device mesh. Inputs, outputs, and dependency edges are explicitly identified.
   - Correctness condition: transformed execution must preserve numerical equivalence up to floating-point non-associativity and stochastic layers.
2. **Mathematical formulation.**
   - Let local shard size be $n_{local}=n/p$ along the chosen axis. Memory scales as $O(n_{local})$ for sharded states and $O(n)$ for replicated states.
   - Communication lower bound (single phase): $\Omega(n\,\beta)$ payload plus startup term $\Omega(\alpha)$.
3. **Execution recipe.**
   - Precondition checks: mesh groups, dtype policy, seed policy, checkpoint layout metadata.
   - Runtime sequence: schedule compute kernels, issue collectives, place synchronization barriers only on true dependencies, and opportunistically overlap.
4. **Performance engineering.**
   - Increase arithmetic intensity using fused kernels and shape-friendly GEMMs.
   - Reduce bytes with precision controls (FP8/quantized links) while managing scales and error feedback.
   - Place high-frequency collectives on fast links (e.g., intra-node) whenever possible.
5. **Failure modes and diagnostics.**
   - Symptoms: stragglers, tail latency spikes, allocator fragmentation, imbalance, and collective stalls.
   - Diagnostics: per-rank timeline traces, bucket size sweeps, microbatch sweep, and reshard-count audits.
6. **Decision rubric.**
   - Prefer the smallest parallel axis set meeting memory and throughput targets.
   - Add axes incrementally; after each addition, re-measure MFU, link utilization, and p95/p99 latency.

**Practical takeaway.** This subsection should be implemented as a topology-aware schedule with explicit tensor-layout contracts; otherwise theoretical gains are lost to reshards and synchronization overhead.

### 6.12 End-to-End MoE Execution Recipe
**Engine 85: scope and objective.** Provide a self-contained treatment of **6.12 End-to-End MoE Execution Recipe** with definitions, mechanism, cost model, implementation choices, failure modes, and decision rules.

1. **Conceptual model.**
   - 6.12 End-to-End MoE Execution Recipe is framed as a tensor-placement transformation over the device mesh. Inputs, outputs, and dependency edges are explicitly identified.
   - Correctness condition: transformed execution must preserve numerical equivalence up to floating-point non-associativity and stochastic layers.
2. **Mathematical formulation.**
   - Let local shard size be $n_{local}=n/p$ along the chosen axis. Memory scales as $O(n_{local})$ for sharded states and $O(n)$ for replicated states.
   - Communication lower bound (single phase): $\Omega(n\,\beta)$ payload plus startup term $\Omega(\alpha)$.
3. **Execution recipe.**
   - Precondition checks: mesh groups, dtype policy, seed policy, checkpoint layout metadata.
   - Runtime sequence: schedule compute kernels, issue collectives, place synchronization barriers only on true dependencies, and opportunistically overlap.
4. **Performance engineering.**
   - Increase arithmetic intensity using fused kernels and shape-friendly GEMMs.
   - Reduce bytes with precision controls (FP8/quantized links) while managing scales and error feedback.
   - Place high-frequency collectives on fast links (e.g., intra-node) whenever possible.
5. **Failure modes and diagnostics.**
   - Symptoms: stragglers, tail latency spikes, allocator fragmentation, imbalance, and collective stalls.
   - Diagnostics: per-rank timeline traces, bucket size sweeps, microbatch sweep, and reshard-count audits.
6. **Decision rubric.**
   - Prefer the smallest parallel axis set meeting memory and throughput targets.
   - Add axes incrementally; after each addition, re-measure MFU, link utilization, and p95/p99 latency.

**Practical takeaway.** This subsection should be implemented as a topology-aware schedule with explicit tensor-layout contracts; otherwise theoretical gains are lost to reshards and synchronization overhead.

### 6.13 Evidence Boundaries and Caveats
**Engine 86: scope and objective.** Provide a self-contained treatment of **6.13 Evidence Boundaries and Caveats** with definitions, mechanism, cost model, implementation choices, failure modes, and decision rules.

1. **Conceptual model.**
   - 6.13 Evidence Boundaries and Caveats is framed as a tensor-placement transformation over the device mesh. Inputs, outputs, and dependency edges are explicitly identified.
   - Correctness condition: transformed execution must preserve numerical equivalence up to floating-point non-associativity and stochastic layers.
2. **Mathematical formulation.**
   - Let local shard size be $n_{local}=n/p$ along the chosen axis. Memory scales as $O(n_{local})$ for sharded states and $O(n)$ for replicated states.
   - Communication lower bound (single phase): $\Omega(n\,\beta)$ payload plus startup term $\Omega(\alpha)$.
3. **Execution recipe.**
   - Precondition checks: mesh groups, dtype policy, seed policy, checkpoint layout metadata.
   - Runtime sequence: schedule compute kernels, issue collectives, place synchronization barriers only on true dependencies, and opportunistically overlap.
4. **Performance engineering.**
   - Increase arithmetic intensity using fused kernels and shape-friendly GEMMs.
   - Reduce bytes with precision controls (FP8/quantized links) while managing scales and error feedback.
   - Place high-frequency collectives on fast links (e.g., intra-node) whenever possible.
5. **Failure modes and diagnostics.**
   - Symptoms: stragglers, tail latency spikes, allocator fragmentation, imbalance, and collective stalls.
   - Diagnostics: per-rank timeline traces, bucket size sweeps, microbatch sweep, and reshard-count audits.
6. **Decision rubric.**
   - Prefer the smallest parallel axis set meeting memory and throughput targets.
   - Add axes incrementally; after each addition, re-measure MFU, link utilization, and p95/p99 latency.

**Practical takeaway.** This subsection should be implemented as a topology-aware schedule with explicit tensor-layout contracts; otherwise theoretical gains are lost to reshards and synchronization overhead.

## 7. Hybrid Mesh Parallelism
**Engine 87: scope and objective.** Provide a self-contained treatment of **7. Hybrid Mesh Parallelism** with definitions, mechanism, cost model, implementation choices, failure modes, and decision rules.

1. **Conceptual model.**
   - 7. Hybrid Mesh Parallelism is framed as a tensor-placement transformation over the device mesh. Inputs, outputs, and dependency edges are explicitly identified.
   - Correctness condition: transformed execution must preserve numerical equivalence up to floating-point non-associativity and stochastic layers.
2. **Mathematical formulation.**
   - Let local shard size be $n_{local}=n/p$ along the chosen axis. Memory scales as $O(n_{local})$ for sharded states and $O(n)$ for replicated states.
   - Communication lower bound (single phase): $\Omega(n\,\beta)$ payload plus startup term $\Omega(\alpha)$.
3. **Execution recipe.**
   - Precondition checks: mesh groups, dtype policy, seed policy, checkpoint layout metadata.
   - Runtime sequence: schedule compute kernels, issue collectives, place synchronization barriers only on true dependencies, and opportunistically overlap.
4. **Performance engineering.**
   - Increase arithmetic intensity using fused kernels and shape-friendly GEMMs.
   - Reduce bytes with precision controls (FP8/quantized links) while managing scales and error feedback.
   - Place high-frequency collectives on fast links (e.g., intra-node) whenever possible.
5. **Failure modes and diagnostics.**
   - Symptoms: stragglers, tail latency spikes, allocator fragmentation, imbalance, and collective stalls.
   - Diagnostics: per-rank timeline traces, bucket size sweeps, microbatch sweep, and reshard-count audits.
6. **Decision rubric.**
   - Prefer the smallest parallel axis set meeting memory and throughput targets.
   - Add axes incrementally; after each addition, re-measure MFU, link utilization, and p95/p99 latency.

**Practical takeaway.** This subsection should be implemented as a topology-aware schedule with explicit tensor-layout contracts; otherwise theoretical gains are lost to reshards and synchronization overhead.

### 7.1 Rank Mapping and Topology-Aware Placement
**Engine 88: scope and objective.** Provide a self-contained treatment of **7.1 Rank Mapping and Topology-Aware Placement** with definitions, mechanism, cost model, implementation choices, failure modes, and decision rules.

1. **Conceptual model.**
   - 7.1 Rank Mapping and Topology-Aware Placement is framed as a tensor-placement transformation over the device mesh. Inputs, outputs, and dependency edges are explicitly identified.
   - Correctness condition: transformed execution must preserve numerical equivalence up to floating-point non-associativity and stochastic layers.
2. **Mathematical formulation.**
   - Let local shard size be $n_{local}=n/p$ along the chosen axis. Memory scales as $O(n_{local})$ for sharded states and $O(n)$ for replicated states.
   - Communication lower bound (single phase): $\Omega(n\,\beta)$ payload plus startup term $\Omega(\alpha)$.
3. **Execution recipe.**
   - Precondition checks: mesh groups, dtype policy, seed policy, checkpoint layout metadata.
   - Runtime sequence: schedule compute kernels, issue collectives, place synchronization barriers only on true dependencies, and opportunistically overlap.
4. **Performance engineering.**
   - Increase arithmetic intensity using fused kernels and shape-friendly GEMMs.
   - Reduce bytes with precision controls (FP8/quantized links) while managing scales and error feedback.
   - Place high-frequency collectives on fast links (e.g., intra-node) whenever possible.
5. **Failure modes and diagnostics.**
   - Symptoms: stragglers, tail latency spikes, allocator fragmentation, imbalance, and collective stalls.
   - Diagnostics: per-rank timeline traces, bucket size sweeps, microbatch sweep, and reshard-count audits.
6. **Decision rubric.**
   - Prefer the smallest parallel axis set meeting memory and throughput targets.
   - Add axes incrementally; after each addition, re-measure MFU, link utilization, and p95/p99 latency.

**Practical takeaway.** This subsection should be implemented as a topology-aware schedule with explicit tensor-layout contracts; otherwise theoretical gains are lost to reshards and synchronization overhead.

### 7.2 Checkpoint Resharding and Restart Semantics
**Engine 89: scope and objective.** Provide a self-contained treatment of **7.2 Checkpoint Resharding and Restart Semantics** with definitions, mechanism, cost model, implementation choices, failure modes, and decision rules.

1. **Conceptual model.**
   - 7.2 Checkpoint Resharding and Restart Semantics is framed as a tensor-placement transformation over the device mesh. Inputs, outputs, and dependency edges are explicitly identified.
   - Correctness condition: transformed execution must preserve numerical equivalence up to floating-point non-associativity and stochastic layers.
2. **Mathematical formulation.**
   - Let local shard size be $n_{local}=n/p$ along the chosen axis. Memory scales as $O(n_{local})$ for sharded states and $O(n)$ for replicated states.
   - Communication lower bound (single phase): $\Omega(n\,\beta)$ payload plus startup term $\Omega(\alpha)$.
3. **Execution recipe.**
   - Precondition checks: mesh groups, dtype policy, seed policy, checkpoint layout metadata.
   - Runtime sequence: schedule compute kernels, issue collectives, place synchronization barriers only on true dependencies, and opportunistically overlap.
4. **Performance engineering.**
   - Increase arithmetic intensity using fused kernels and shape-friendly GEMMs.
   - Reduce bytes with precision controls (FP8/quantized links) while managing scales and error feedback.
   - Place high-frequency collectives on fast links (e.g., intra-node) whenever possible.
5. **Failure modes and diagnostics.**
   - Symptoms: stragglers, tail latency spikes, allocator fragmentation, imbalance, and collective stalls.
   - Diagnostics: per-rank timeline traces, bucket size sweeps, microbatch sweep, and reshard-count audits.
6. **Decision rubric.**
   - Prefer the smallest parallel axis set meeting memory and throughput targets.
   - Add axes incrementally; after each addition, re-measure MFU, link utilization, and p95/p99 latency.

**Practical takeaway.** This subsection should be implemented as a topology-aware schedule with explicit tensor-layout contracts; otherwise theoretical gains are lost to reshards and synchronization overhead.

### 7.3 Composition Rules
**Engine 90: scope and objective.** Provide a self-contained treatment of **7.3 Composition Rules** with definitions, mechanism, cost model, implementation choices, failure modes, and decision rules.

1. **Conceptual model.**
   - 7.3 Composition Rules is framed as a tensor-placement transformation over the device mesh. Inputs, outputs, and dependency edges are explicitly identified.
   - Correctness condition: transformed execution must preserve numerical equivalence up to floating-point non-associativity and stochastic layers.
2. **Mathematical formulation.**
   - Let local shard size be $n_{local}=n/p$ along the chosen axis. Memory scales as $O(n_{local})$ for sharded states and $O(n)$ for replicated states.
   - Communication lower bound (single phase): $\Omega(n\,\beta)$ payload plus startup term $\Omega(\alpha)$.
3. **Execution recipe.**
   - Precondition checks: mesh groups, dtype policy, seed policy, checkpoint layout metadata.
   - Runtime sequence: schedule compute kernels, issue collectives, place synchronization barriers only on true dependencies, and opportunistically overlap.
4. **Performance engineering.**
   - Increase arithmetic intensity using fused kernels and shape-friendly GEMMs.
   - Reduce bytes with precision controls (FP8/quantized links) while managing scales and error feedback.
   - Place high-frequency collectives on fast links (e.g., intra-node) whenever possible.
5. **Failure modes and diagnostics.**
   - Symptoms: stragglers, tail latency spikes, allocator fragmentation, imbalance, and collective stalls.
   - Diagnostics: per-rank timeline traces, bucket size sweeps, microbatch sweep, and reshard-count audits.
6. **Decision rubric.**
   - Prefer the smallest parallel axis set meeting memory and throughput targets.
   - Add axes incrementally; after each addition, re-measure MFU, link utilization, and p95/p99 latency.

**Practical takeaway.** This subsection should be implemented as a topology-aware schedule with explicit tensor-layout contracts; otherwise theoretical gains are lost to reshards and synchronization overhead.

### 7.4 System Perspectives
**Engine 91: scope and objective.** Provide a self-contained treatment of **7.4 System Perspectives** with definitions, mechanism, cost model, implementation choices, failure modes, and decision rules.

1. **Conceptual model.**
   - 7.4 System Perspectives is framed as a tensor-placement transformation over the device mesh. Inputs, outputs, and dependency edges are explicitly identified.
   - Correctness condition: transformed execution must preserve numerical equivalence up to floating-point non-associativity and stochastic layers.
2. **Mathematical formulation.**
   - Let local shard size be $n_{local}=n/p$ along the chosen axis. Memory scales as $O(n_{local})$ for sharded states and $O(n)$ for replicated states.
   - Communication lower bound (single phase): $\Omega(n\,\beta)$ payload plus startup term $\Omega(\alpha)$.
3. **Execution recipe.**
   - Precondition checks: mesh groups, dtype policy, seed policy, checkpoint layout metadata.
   - Runtime sequence: schedule compute kernels, issue collectives, place synchronization barriers only on true dependencies, and opportunistically overlap.
4. **Performance engineering.**
   - Increase arithmetic intensity using fused kernels and shape-friendly GEMMs.
   - Reduce bytes with precision controls (FP8/quantized links) while managing scales and error feedback.
   - Place high-frequency collectives on fast links (e.g., intra-node) whenever possible.
5. **Failure modes and diagnostics.**
   - Symptoms: stragglers, tail latency spikes, allocator fragmentation, imbalance, and collective stalls.
   - Diagnostics: per-rank timeline traces, bucket size sweeps, microbatch sweep, and reshard-count audits.
6. **Decision rubric.**
   - Prefer the smallest parallel axis set meeting memory and throughput targets.
   - Add axes incrementally; after each addition, re-measure MFU, link utilization, and p95/p99 latency.

**Practical takeaway.** This subsection should be implemented as a topology-aware schedule with explicit tensor-layout contracts; otherwise theoretical gains are lost to reshards and synchronization overhead.

### 7.5 Caveats and Evidence Boundaries
**Engine 92: scope and objective.** Provide a self-contained treatment of **7.5 Caveats and Evidence Boundaries** with definitions, mechanism, cost model, implementation choices, failure modes, and decision rules.

1. **Conceptual model.**
   - 7.5 Caveats and Evidence Boundaries is framed as a tensor-placement transformation over the device mesh. Inputs, outputs, and dependency edges are explicitly identified.
   - Correctness condition: transformed execution must preserve numerical equivalence up to floating-point non-associativity and stochastic layers.
2. **Mathematical formulation.**
   - Let local shard size be $n_{local}=n/p$ along the chosen axis. Memory scales as $O(n_{local})$ for sharded states and $O(n)$ for replicated states.
   - Communication lower bound (single phase): $\Omega(n\,\beta)$ payload plus startup term $\Omega(\alpha)$.
3. **Execution recipe.**
   - Precondition checks: mesh groups, dtype policy, seed policy, checkpoint layout metadata.
   - Runtime sequence: schedule compute kernels, issue collectives, place synchronization barriers only on true dependencies, and opportunistically overlap.
4. **Performance engineering.**
   - Increase arithmetic intensity using fused kernels and shape-friendly GEMMs.
   - Reduce bytes with precision controls (FP8/quantized links) while managing scales and error feedback.
   - Place high-frequency collectives on fast links (e.g., intra-node) whenever possible.
5. **Failure modes and diagnostics.**
   - Symptoms: stragglers, tail latency spikes, allocator fragmentation, imbalance, and collective stalls.
   - Diagnostics: per-rank timeline traces, bucket size sweeps, microbatch sweep, and reshard-count audits.
6. **Decision rubric.**
   - Prefer the smallest parallel axis set meeting memory and throughput targets.
   - Add axes incrementally; after each addition, re-measure MFU, link utilization, and p95/p99 latency.

**Practical takeaway.** This subsection should be implemented as a topology-aware schedule with explicit tensor-layout contracts; otherwise theoretical gains are lost to reshards and synchronization overhead.

## 8. Inference Parallelism and Serving Systems
**Engine 93: scope and objective.** Provide a self-contained treatment of **8. Inference Parallelism and Serving Systems** with definitions, mechanism, cost model, implementation choices, failure modes, and decision rules.

1. **Conceptual model.**
   - 8. Inference Parallelism and Serving Systems is framed as a tensor-placement transformation over the device mesh. Inputs, outputs, and dependency edges are explicitly identified.
   - Correctness condition: transformed execution must preserve numerical equivalence up to floating-point non-associativity and stochastic layers.
2. **Mathematical formulation.**
   - Let local shard size be $n_{local}=n/p$ along the chosen axis. Memory scales as $O(n_{local})$ for sharded states and $O(n)$ for replicated states.
   - Communication lower bound (single phase): $\Omega(n\,\beta)$ payload plus startup term $\Omega(\alpha)$.
3. **Execution recipe.**
   - Precondition checks: mesh groups, dtype policy, seed policy, checkpoint layout metadata.
   - Runtime sequence: schedule compute kernels, issue collectives, place synchronization barriers only on true dependencies, and opportunistically overlap.
4. **Performance engineering.**
   - Increase arithmetic intensity using fused kernels and shape-friendly GEMMs.
   - Reduce bytes with precision controls (FP8/quantized links) while managing scales and error feedback.
   - Place high-frequency collectives on fast links (e.g., intra-node) whenever possible.
5. **Failure modes and diagnostics.**
   - Symptoms: stragglers, tail latency spikes, allocator fragmentation, imbalance, and collective stalls.
   - Diagnostics: per-rank timeline traces, bucket size sweeps, microbatch sweep, and reshard-count audits.
6. **Decision rubric.**
   - Prefer the smallest parallel axis set meeting memory and throughput targets.
   - Add axes incrementally; after each addition, re-measure MFU, link utilization, and p95/p99 latency.

**Practical takeaway.** This subsection should be implemented as a topology-aware schedule with explicit tensor-layout contracts; otherwise theoretical gains are lost to reshards and synchronization overhead.

### 8.1 Primitive Technique: Replicas and Request Routing
**Engine 94: scope and objective.** Provide a self-contained treatment of **8.1 Primitive Technique: Replicas and Request Routing** with definitions, mechanism, cost model, implementation choices, failure modes, and decision rules.

1. **Conceptual model.**
   - 8.1 Primitive Technique: Replicas and Request Routing is framed as a tensor-placement transformation over the device mesh. Inputs, outputs, and dependency edges are explicitly identified.
   - Correctness condition: transformed execution must preserve numerical equivalence up to floating-point non-associativity and stochastic layers.
2. **Mathematical formulation.**
   - Let local shard size be $n_{local}=n/p$ along the chosen axis. Memory scales as $O(n_{local})$ for sharded states and $O(n)$ for replicated states.
   - Communication lower bound (single phase): $\Omega(n\,\beta)$ payload plus startup term $\Omega(\alpha)$.
3. **Execution recipe.**
   - Precondition checks: mesh groups, dtype policy, seed policy, checkpoint layout metadata.
   - Runtime sequence: schedule compute kernels, issue collectives, place synchronization barriers only on true dependencies, and opportunistically overlap.
4. **Performance engineering.**
   - Increase arithmetic intensity using fused kernels and shape-friendly GEMMs.
   - Reduce bytes with precision controls (FP8/quantized links) while managing scales and error feedback.
   - Place high-frequency collectives on fast links (e.g., intra-node) whenever possible.
5. **Failure modes and diagnostics.**
   - Symptoms: stragglers, tail latency spikes, allocator fragmentation, imbalance, and collective stalls.
   - Diagnostics: per-rank timeline traces, bucket size sweeps, microbatch sweep, and reshard-count audits.
6. **Decision rubric.**
   - Prefer the smallest parallel axis set meeting memory and throughput targets.
   - Add axes incrementally; after each addition, re-measure MFU, link utilization, and p95/p99 latency.

**Practical takeaway.** This subsection should be implemented as a topology-aware schedule with explicit tensor-layout contracts; otherwise theoretical gains are lost to reshards and synchronization overhead.

### 8.2 Primitive Technique: Continuous Batching
**Engine 95: scope and objective.** Provide a self-contained treatment of **8.2 Primitive Technique: Continuous Batching** with definitions, mechanism, cost model, implementation choices, failure modes, and decision rules.

1. **Conceptual model.**
   - 8.2 Primitive Technique: Continuous Batching is framed as a tensor-placement transformation over the device mesh. Inputs, outputs, and dependency edges are explicitly identified.
   - Correctness condition: transformed execution must preserve numerical equivalence up to floating-point non-associativity and stochastic layers.
2. **Mathematical formulation.**
   - Let local shard size be $n_{local}=n/p$ along the chosen axis. Memory scales as $O(n_{local})$ for sharded states and $O(n)$ for replicated states.
   - Communication lower bound (single phase): $\Omega(n\,\beta)$ payload plus startup term $\Omega(\alpha)$.
3. **Execution recipe.**
   - Precondition checks: mesh groups, dtype policy, seed policy, checkpoint layout metadata.
   - Runtime sequence: schedule compute kernels, issue collectives, place synchronization barriers only on true dependencies, and opportunistically overlap.
4. **Performance engineering.**
   - Increase arithmetic intensity using fused kernels and shape-friendly GEMMs.
   - Reduce bytes with precision controls (FP8/quantized links) while managing scales and error feedback.
   - Place high-frequency collectives on fast links (e.g., intra-node) whenever possible.
5. **Failure modes and diagnostics.**
   - Symptoms: stragglers, tail latency spikes, allocator fragmentation, imbalance, and collective stalls.
   - Diagnostics: per-rank timeline traces, bucket size sweeps, microbatch sweep, and reshard-count audits.
6. **Decision rubric.**
   - Prefer the smallest parallel axis set meeting memory and throughput targets.
   - Add axes incrementally; after each addition, re-measure MFU, link utilization, and p95/p99 latency.

**Practical takeaway.** This subsection should be implemented as a topology-aware schedule with explicit tensor-layout contracts; otherwise theoretical gains are lost to reshards and synchronization overhead.

### 8.3 Primitive Technique: Paged KV Cache
**Engine 96: scope and objective.** Provide a self-contained treatment of **8.3 Primitive Technique: Paged KV Cache** with definitions, mechanism, cost model, implementation choices, failure modes, and decision rules.

1. **Conceptual model.**
   - 8.3 Primitive Technique: Paged KV Cache is framed as a tensor-placement transformation over the device mesh. Inputs, outputs, and dependency edges are explicitly identified.
   - Correctness condition: transformed execution must preserve numerical equivalence up to floating-point non-associativity and stochastic layers.
2. **Mathematical formulation.**
   - Let local shard size be $n_{local}=n/p$ along the chosen axis. Memory scales as $O(n_{local})$ for sharded states and $O(n)$ for replicated states.
   - Communication lower bound (single phase): $\Omega(n\,\beta)$ payload plus startup term $\Omega(\alpha)$.
3. **Execution recipe.**
   - Precondition checks: mesh groups, dtype policy, seed policy, checkpoint layout metadata.
   - Runtime sequence: schedule compute kernels, issue collectives, place synchronization barriers only on true dependencies, and opportunistically overlap.
4. **Performance engineering.**
   - Increase arithmetic intensity using fused kernels and shape-friendly GEMMs.
   - Reduce bytes with precision controls (FP8/quantized links) while managing scales and error feedback.
   - Place high-frequency collectives on fast links (e.g., intra-node) whenever possible.
5. **Failure modes and diagnostics.**
   - Symptoms: stragglers, tail latency spikes, allocator fragmentation, imbalance, and collective stalls.
   - Diagnostics: per-rank timeline traces, bucket size sweeps, microbatch sweep, and reshard-count audits.
6. **Decision rubric.**
   - Prefer the smallest parallel axis set meeting memory and throughput targets.
   - Add axes incrementally; after each addition, re-measure MFU, link utilization, and p95/p99 latency.

**Practical takeaway.** This subsection should be implemented as a topology-aware schedule with explicit tensor-layout contracts; otherwise theoretical gains are lost to reshards and synchronization overhead.

### 8.4 Primitive Technique: Prefix and Radix Caching
**Engine 97: scope and objective.** Provide a self-contained treatment of **8.4 Primitive Technique: Prefix and Radix Caching** with definitions, mechanism, cost model, implementation choices, failure modes, and decision rules.

1. **Conceptual model.**
   - 8.4 Primitive Technique: Prefix and Radix Caching is framed as a tensor-placement transformation over the device mesh. Inputs, outputs, and dependency edges are explicitly identified.
   - Correctness condition: transformed execution must preserve numerical equivalence up to floating-point non-associativity and stochastic layers.
2. **Mathematical formulation.**
   - Let local shard size be $n_{local}=n/p$ along the chosen axis. Memory scales as $O(n_{local})$ for sharded states and $O(n)$ for replicated states.
   - Communication lower bound (single phase): $\Omega(n\,\beta)$ payload plus startup term $\Omega(\alpha)$.
3. **Execution recipe.**
   - Precondition checks: mesh groups, dtype policy, seed policy, checkpoint layout metadata.
   - Runtime sequence: schedule compute kernels, issue collectives, place synchronization barriers only on true dependencies, and opportunistically overlap.
4. **Performance engineering.**
   - Increase arithmetic intensity using fused kernels and shape-friendly GEMMs.
   - Reduce bytes with precision controls (FP8/quantized links) while managing scales and error feedback.
   - Place high-frequency collectives on fast links (e.g., intra-node) whenever possible.
5. **Failure modes and diagnostics.**
   - Symptoms: stragglers, tail latency spikes, allocator fragmentation, imbalance, and collective stalls.
   - Diagnostics: per-rank timeline traces, bucket size sweeps, microbatch sweep, and reshard-count audits.
6. **Decision rubric.**
   - Prefer the smallest parallel axis set meeting memory and throughput targets.
   - Add axes incrementally; after each addition, re-measure MFU, link utilization, and p95/p99 latency.

**Practical takeaway.** This subsection should be implemented as a topology-aware schedule with explicit tensor-layout contracts; otherwise theoretical gains are lost to reshards and synchronization overhead.

### 8.5 Primitive Technique: Prefill/Decode Disaggregation
**Engine 98: scope and objective.** Provide a self-contained treatment of **8.5 Primitive Technique: Prefill/Decode Disaggregation** with definitions, mechanism, cost model, implementation choices, failure modes, and decision rules.

1. **Conceptual model.**
   - 8.5 Primitive Technique: Prefill/Decode Disaggregation is framed as a tensor-placement transformation over the device mesh. Inputs, outputs, and dependency edges are explicitly identified.
   - Correctness condition: transformed execution must preserve numerical equivalence up to floating-point non-associativity and stochastic layers.
2. **Mathematical formulation.**
   - Let local shard size be $n_{local}=n/p$ along the chosen axis. Memory scales as $O(n_{local})$ for sharded states and $O(n)$ for replicated states.
   - Communication lower bound (single phase): $\Omega(n\,\beta)$ payload plus startup term $\Omega(\alpha)$.
3. **Execution recipe.**
   - Precondition checks: mesh groups, dtype policy, seed policy, checkpoint layout metadata.
   - Runtime sequence: schedule compute kernels, issue collectives, place synchronization barriers only on true dependencies, and opportunistically overlap.
4. **Performance engineering.**
   - Increase arithmetic intensity using fused kernels and shape-friendly GEMMs.
   - Reduce bytes with precision controls (FP8/quantized links) while managing scales and error feedback.
   - Place high-frequency collectives on fast links (e.g., intra-node) whenever possible.
5. **Failure modes and diagnostics.**
   - Symptoms: stragglers, tail latency spikes, allocator fragmentation, imbalance, and collective stalls.
   - Diagnostics: per-rank timeline traces, bucket size sweeps, microbatch sweep, and reshard-count audits.
6. **Decision rubric.**
   - Prefer the smallest parallel axis set meeting memory and throughput targets.
   - Add axes incrementally; after each addition, re-measure MFU, link utilization, and p95/p99 latency.

**Practical takeaway.** This subsection should be implemented as a topology-aware schedule with explicit tensor-layout contracts; otherwise theoretical gains are lost to reshards and synchronization overhead.

### 8.6 Primitive Technique: Chunked Prefill and SplitFuse
**Engine 99: scope and objective.** Provide a self-contained treatment of **8.6 Primitive Technique: Chunked Prefill and SplitFuse** with definitions, mechanism, cost model, implementation choices, failure modes, and decision rules.

1. **Conceptual model.**
   - 8.6 Primitive Technique: Chunked Prefill and SplitFuse is framed as a tensor-placement transformation over the device mesh. Inputs, outputs, and dependency edges are explicitly identified.
   - Correctness condition: transformed execution must preserve numerical equivalence up to floating-point non-associativity and stochastic layers.
2. **Mathematical formulation.**
   - Let local shard size be $n_{local}=n/p$ along the chosen axis. Memory scales as $O(n_{local})$ for sharded states and $O(n)$ for replicated states.
   - Communication lower bound (single phase): $\Omega(n\,\beta)$ payload plus startup term $\Omega(\alpha)$.
3. **Execution recipe.**
   - Precondition checks: mesh groups, dtype policy, seed policy, checkpoint layout metadata.
   - Runtime sequence: schedule compute kernels, issue collectives, place synchronization barriers only on true dependencies, and opportunistically overlap.
4. **Performance engineering.**
   - Increase arithmetic intensity using fused kernels and shape-friendly GEMMs.
   - Reduce bytes with precision controls (FP8/quantized links) while managing scales and error feedback.
   - Place high-frequency collectives on fast links (e.g., intra-node) whenever possible.
5. **Failure modes and diagnostics.**
   - Symptoms: stragglers, tail latency spikes, allocator fragmentation, imbalance, and collective stalls.
   - Diagnostics: per-rank timeline traces, bucket size sweeps, microbatch sweep, and reshard-count audits.
6. **Decision rubric.**
   - Prefer the smallest parallel axis set meeting memory and throughput targets.
   - Add axes incrementally; after each addition, re-measure MFU, link utilization, and p95/p99 latency.

**Practical takeaway.** This subsection should be implemented as a topology-aware schedule with explicit tensor-layout contracts; otherwise theoretical gains are lost to reshards and synchronization overhead.

### 8.7 Primitive Technique: Speculative and [MTP](https://arxiv.org/abs/2412.19437) (multi-token prediction) Decoding
**Engine 100: scope and objective.** Provide a self-contained treatment of **8.7 Primitive Technique: Speculative and [MTP](https://arxiv.org/abs/2412.19437) (multi-token prediction) Decoding** with definitions, mechanism, cost model, implementation choices, failure modes, and decision rules.

1. **Conceptual model.**
   - 8.7 Primitive Technique: Speculative and [MTP](https://arxiv.org/abs/2412.19437) (multi-token prediction) Decoding is framed as a tensor-placement transformation over the device mesh. Inputs, outputs, and dependency edges are explicitly identified.
   - Correctness condition: transformed execution must preserve numerical equivalence up to floating-point non-associativity and stochastic layers.
2. **Mathematical formulation.**
   - Let local shard size be $n_{local}=n/p$ along the chosen axis. Memory scales as $O(n_{local})$ for sharded states and $O(n)$ for replicated states.
   - Communication lower bound (single phase): $\Omega(n\,\beta)$ payload plus startup term $\Omega(\alpha)$.
3. **Execution recipe.**
   - Precondition checks: mesh groups, dtype policy, seed policy, checkpoint layout metadata.
   - Runtime sequence: schedule compute kernels, issue collectives, place synchronization barriers only on true dependencies, and opportunistically overlap.
4. **Performance engineering.**
   - Increase arithmetic intensity using fused kernels and shape-friendly GEMMs.
   - Reduce bytes with precision controls (FP8/quantized links) while managing scales and error feedback.
   - Place high-frequency collectives on fast links (e.g., intra-node) whenever possible.
5. **Failure modes and diagnostics.**
   - Symptoms: stragglers, tail latency spikes, allocator fragmentation, imbalance, and collective stalls.
   - Diagnostics: per-rank timeline traces, bucket size sweeps, microbatch sweep, and reshard-count audits.
6. **Decision rubric.**
   - Prefer the smallest parallel axis set meeting memory and throughput targets.
   - Add axes incrementally; after each addition, re-measure MFU, link utilization, and p95/p99 latency.

**Practical takeaway.** This subsection should be implemented as a topology-aware schedule with explicit tensor-layout contracts; otherwise theoretical gains are lost to reshards and synchronization overhead.

### 8.8 Primitive Technique: TP, PP, EP, and CP for Serving
**Engine 101: scope and objective.** Provide a self-contained treatment of **8.8 Primitive Technique: TP, PP, EP, and CP for Serving** with definitions, mechanism, cost model, implementation choices, failure modes, and decision rules.

1. **Conceptual model.**
   - 8.8 Primitive Technique: TP, PP, EP, and CP for Serving is framed as a tensor-placement transformation over the device mesh. Inputs, outputs, and dependency edges are explicitly identified.
   - Correctness condition: transformed execution must preserve numerical equivalence up to floating-point non-associativity and stochastic layers.
2. **Mathematical formulation.**
   - Let local shard size be $n_{local}=n/p$ along the chosen axis. Memory scales as $O(n_{local})$ for sharded states and $O(n)$ for replicated states.
   - Communication lower bound (single phase): $\Omega(n\,\beta)$ payload plus startup term $\Omega(\alpha)$.
3. **Execution recipe.**
   - Precondition checks: mesh groups, dtype policy, seed policy, checkpoint layout metadata.
   - Runtime sequence: schedule compute kernels, issue collectives, place synchronization barriers only on true dependencies, and opportunistically overlap.
4. **Performance engineering.**
   - Increase arithmetic intensity using fused kernels and shape-friendly GEMMs.
   - Reduce bytes with precision controls (FP8/quantized links) while managing scales and error feedback.
   - Place high-frequency collectives on fast links (e.g., intra-node) whenever possible.
5. **Failure modes and diagnostics.**
   - Symptoms: stragglers, tail latency spikes, allocator fragmentation, imbalance, and collective stalls.
   - Diagnostics: per-rank timeline traces, bucket size sweeps, microbatch sweep, and reshard-count audits.
6. **Decision rubric.**
   - Prefer the smallest parallel axis set meeting memory and throughput targets.
   - Add axes incrementally; after each addition, re-measure MFU, link utilization, and p95/p99 latency.

**Practical takeaway.** This subsection should be implemented as a topology-aware schedule with explicit tensor-layout contracts; otherwise theoretical gains are lost to reshards and synchronization overhead.

### 8.9 End-to-End Serving Map
**Engine 102: scope and objective.** Provide a self-contained treatment of **8.9 End-to-End Serving Map** with definitions, mechanism, cost model, implementation choices, failure modes, and decision rules.

1. **Conceptual model.**
   - 8.9 End-to-End Serving Map is framed as a tensor-placement transformation over the device mesh. Inputs, outputs, and dependency edges are explicitly identified.
   - Correctness condition: transformed execution must preserve numerical equivalence up to floating-point non-associativity and stochastic layers.
2. **Mathematical formulation.**
   - Let local shard size be $n_{local}=n/p$ along the chosen axis. Memory scales as $O(n_{local})$ for sharded states and $O(n)$ for replicated states.
   - Communication lower bound (single phase): $\Omega(n\,\beta)$ payload plus startup term $\Omega(\alpha)$.
3. **Execution recipe.**
   - Precondition checks: mesh groups, dtype policy, seed policy, checkpoint layout metadata.
   - Runtime sequence: schedule compute kernels, issue collectives, place synchronization barriers only on true dependencies, and opportunistically overlap.
4. **Performance engineering.**
   - Increase arithmetic intensity using fused kernels and shape-friendly GEMMs.
   - Reduce bytes with precision controls (FP8/quantized links) while managing scales and error feedback.
   - Place high-frequency collectives on fast links (e.g., intra-node) whenever possible.
5. **Failure modes and diagnostics.**
   - Symptoms: stragglers, tail latency spikes, allocator fragmentation, imbalance, and collective stalls.
   - Diagnostics: per-rank timeline traces, bucket size sweeps, microbatch sweep, and reshard-count audits.
6. **Decision rubric.**
   - Prefer the smallest parallel axis set meeting memory and throughput targets.
   - Add axes incrementally; after each addition, re-measure MFU, link utilization, and p95/p99 latency.

**Practical takeaway.** This subsection should be implemented as a topology-aware schedule with explicit tensor-layout contracts; otherwise theoretical gains are lost to reshards and synchronization overhead.

### 8.10 Sources and Caveats
**Engine 103: scope and objective.** Provide a self-contained treatment of **8.10 Sources and Caveats** with definitions, mechanism, cost model, implementation choices, failure modes, and decision rules.

1. **Conceptual model.**
   - 8.10 Sources and Caveats is framed as a tensor-placement transformation over the device mesh. Inputs, outputs, and dependency edges are explicitly identified.
   - Correctness condition: transformed execution must preserve numerical equivalence up to floating-point non-associativity and stochastic layers.
2. **Mathematical formulation.**
   - Let local shard size be $n_{local}=n/p$ along the chosen axis. Memory scales as $O(n_{local})$ for sharded states and $O(n)$ for replicated states.
   - Communication lower bound (single phase): $\Omega(n\,\beta)$ payload plus startup term $\Omega(\alpha)$.
3. **Execution recipe.**
   - Precondition checks: mesh groups, dtype policy, seed policy, checkpoint layout metadata.
   - Runtime sequence: schedule compute kernels, issue collectives, place synchronization barriers only on true dependencies, and opportunistically overlap.
4. **Performance engineering.**
   - Increase arithmetic intensity using fused kernels and shape-friendly GEMMs.
   - Reduce bytes with precision controls (FP8/quantized links) while managing scales and error feedback.
   - Place high-frequency collectives on fast links (e.g., intra-node) whenever possible.
5. **Failure modes and diagnostics.**
   - Symptoms: stragglers, tail latency spikes, allocator fragmentation, imbalance, and collective stalls.
   - Diagnostics: per-rank timeline traces, bucket size sweeps, microbatch sweep, and reshard-count audits.
6. **Decision rubric.**
   - Prefer the smallest parallel axis set meeting memory and throughput targets.
   - Add axes incrementally; after each addition, re-measure MFU, link utilization, and p95/p99 latency.

**Practical takeaway.** This subsection should be implemented as a topology-aware schedule with explicit tensor-layout contracts; otherwise theoretical gains are lost to reshards and synchronization overhead.

## 9. Cross-Cutting CUDA and Platform Techniques
**Engine 104: scope and objective.** Provide a self-contained treatment of **9. Cross-Cutting CUDA and Platform Techniques** with definitions, mechanism, cost model, implementation choices, failure modes, and decision rules.

1. **Conceptual model.**
   - 9. Cross-Cutting CUDA and Platform Techniques is framed as a tensor-placement transformation over the device mesh. Inputs, outputs, and dependency edges are explicitly identified.
   - Correctness condition: transformed execution must preserve numerical equivalence up to floating-point non-associativity and stochastic layers.
2. **Mathematical formulation.**
   - Let local shard size be $n_{local}=n/p$ along the chosen axis. Memory scales as $O(n_{local})$ for sharded states and $O(n)$ for replicated states.
   - Communication lower bound (single phase): $\Omega(n\,\beta)$ payload plus startup term $\Omega(\alpha)$.
3. **Execution recipe.**
   - Precondition checks: mesh groups, dtype policy, seed policy, checkpoint layout metadata.
   - Runtime sequence: schedule compute kernels, issue collectives, place synchronization barriers only on true dependencies, and opportunistically overlap.
4. **Performance engineering.**
   - Increase arithmetic intensity using fused kernels and shape-friendly GEMMs.
   - Reduce bytes with precision controls (FP8/quantized links) while managing scales and error feedback.
   - Place high-frequency collectives on fast links (e.g., intra-node) whenever possible.
5. **Failure modes and diagnostics.**
   - Symptoms: stragglers, tail latency spikes, allocator fragmentation, imbalance, and collective stalls.
   - Diagnostics: per-rank timeline traces, bucket size sweeps, microbatch sweep, and reshard-count audits.
6. **Decision rubric.**
   - Prefer the smallest parallel axis set meeting memory and throughput targets.
   - Add axes incrementally; after each addition, re-measure MFU, link utilization, and p95/p99 latency.

**Practical takeaway.** This subsection should be implemented as a topology-aware schedule with explicit tensor-layout contracts; otherwise theoretical gains are lost to reshards and synchronization overhead.

### 9.1 Overlap: Streams, Events, Buckets, and Progress
**Engine 105: scope and objective.** Provide a self-contained treatment of **9.1 Overlap: Streams, Events, Buckets, and Progress** with definitions, mechanism, cost model, implementation choices, failure modes, and decision rules.

1. **Conceptual model.**
   - 9.1 Overlap: Streams, Events, Buckets, and Progress is framed as a tensor-placement transformation over the device mesh. Inputs, outputs, and dependency edges are explicitly identified.
   - Correctness condition: transformed execution must preserve numerical equivalence up to floating-point non-associativity and stochastic layers.
2. **Mathematical formulation.**
   - Let local shard size be $n_{local}=n/p$ along the chosen axis. Memory scales as $O(n_{local})$ for sharded states and $O(n)$ for replicated states.
   - Communication lower bound (single phase): $\Omega(n\,\beta)$ payload plus startup term $\Omega(\alpha)$.
3. **Execution recipe.**
   - Precondition checks: mesh groups, dtype policy, seed policy, checkpoint layout metadata.
   - Runtime sequence: schedule compute kernels, issue collectives, place synchronization barriers only on true dependencies, and opportunistically overlap.
4. **Performance engineering.**
   - Increase arithmetic intensity using fused kernels and shape-friendly GEMMs.
   - Reduce bytes with precision controls (FP8/quantized links) while managing scales and error feedback.
   - Place high-frequency collectives on fast links (e.g., intra-node) whenever possible.
5. **Failure modes and diagnostics.**
   - Symptoms: stragglers, tail latency spikes, allocator fragmentation, imbalance, and collective stalls.
   - Diagnostics: per-rank timeline traces, bucket size sweeps, microbatch sweep, and reshard-count audits.
6. **Decision rubric.**
   - Prefer the smallest parallel axis set meeting memory and throughput targets.
   - Add axes incrementally; after each addition, re-measure MFU, link utilization, and p95/p99 latency.

**Practical takeaway.** This subsection should be implemented as a topology-aware schedule with explicit tensor-layout contracts; otherwise theoretical gains are lost to reshards and synchronization overhead.

### 9.2 CUDA Graphs and Static Metadata
**Engine 106: scope and objective.** Provide a self-contained treatment of **9.2 CUDA Graphs and Static Metadata** with definitions, mechanism, cost model, implementation choices, failure modes, and decision rules.

1. **Conceptual model.**
   - 9.2 CUDA Graphs and Static Metadata is framed as a tensor-placement transformation over the device mesh. Inputs, outputs, and dependency edges are explicitly identified.
   - Correctness condition: transformed execution must preserve numerical equivalence up to floating-point non-associativity and stochastic layers.
2. **Mathematical formulation.**
   - Let local shard size be $n_{local}=n/p$ along the chosen axis. Memory scales as $O(n_{local})$ for sharded states and $O(n)$ for replicated states.
   - Communication lower bound (single phase): $\Omega(n\,\beta)$ payload plus startup term $\Omega(\alpha)$.
3. **Execution recipe.**
   - Precondition checks: mesh groups, dtype policy, seed policy, checkpoint layout metadata.
   - Runtime sequence: schedule compute kernels, issue collectives, place synchronization barriers only on true dependencies, and opportunistically overlap.
4. **Performance engineering.**
   - Increase arithmetic intensity using fused kernels and shape-friendly GEMMs.
   - Reduce bytes with precision controls (FP8/quantized links) while managing scales and error feedback.
   - Place high-frequency collectives on fast links (e.g., intra-node) whenever possible.
5. **Failure modes and diagnostics.**
   - Symptoms: stragglers, tail latency spikes, allocator fragmentation, imbalance, and collective stalls.
   - Diagnostics: per-rank timeline traces, bucket size sweeps, microbatch sweep, and reshard-count audits.
6. **Decision rubric.**
   - Prefer the smallest parallel axis set meeting memory and throughput targets.
   - Add axes incrementally; after each addition, re-measure MFU, link utilization, and p95/p99 latency.

**Practical takeaway.** This subsection should be implemented as a topology-aware schedule with explicit tensor-layout contracts; otherwise theoretical gains are lost to reshards and synchronization overhead.

### 9.3 Low Precision as Payload Plus Scale Layout
**Engine 107: scope and objective.** Provide a self-contained treatment of **9.3 Low Precision as Payload Plus Scale Layout** with definitions, mechanism, cost model, implementation choices, failure modes, and decision rules.

1. **Conceptual model.**
   - 9.3 Low Precision as Payload Plus Scale Layout is framed as a tensor-placement transformation over the device mesh. Inputs, outputs, and dependency edges are explicitly identified.
   - Correctness condition: transformed execution must preserve numerical equivalence up to floating-point non-associativity and stochastic layers.
2. **Mathematical formulation.**
   - Let local shard size be $n_{local}=n/p$ along the chosen axis. Memory scales as $O(n_{local})$ for sharded states and $O(n)$ for replicated states.
   - Communication lower bound (single phase): $\Omega(n\,\beta)$ payload plus startup term $\Omega(\alpha)$.
3. **Execution recipe.**
   - Precondition checks: mesh groups, dtype policy, seed policy, checkpoint layout metadata.
   - Runtime sequence: schedule compute kernels, issue collectives, place synchronization barriers only on true dependencies, and opportunistically overlap.
4. **Performance engineering.**
   - Increase arithmetic intensity using fused kernels and shape-friendly GEMMs.
   - Reduce bytes with precision controls (FP8/quantized links) while managing scales and error feedback.
   - Place high-frequency collectives on fast links (e.g., intra-node) whenever possible.
5. **Failure modes and diagnostics.**
   - Symptoms: stragglers, tail latency spikes, allocator fragmentation, imbalance, and collective stalls.
   - Diagnostics: per-rank timeline traces, bucket size sweeps, microbatch sweep, and reshard-count audits.
6. **Decision rubric.**
   - Prefer the smallest parallel axis set meeting memory and throughput targets.
   - Add axes incrementally; after each addition, re-measure MFU, link utilization, and p95/p99 latency.

**Practical takeaway.** This subsection should be implemented as a topology-aware schedule with explicit tensor-layout contracts; otherwise theoretical gains are lost to reshards and synchronization overhead.

### 9.4 Attention Kernels: FlashAttention, cuDNN SDPA, NSA, and FlashMLA
**Engine 108: scope and objective.** Provide a self-contained treatment of **9.4 Attention Kernels: FlashAttention, cuDNN SDPA, NSA, and FlashMLA** with definitions, mechanism, cost model, implementation choices, failure modes, and decision rules.

1. **Conceptual model.**
   - 9.4 Attention Kernels: FlashAttention, cuDNN SDPA, NSA, and FlashMLA is framed as a tensor-placement transformation over the device mesh. Inputs, outputs, and dependency edges are explicitly identified.
   - Correctness condition: transformed execution must preserve numerical equivalence up to floating-point non-associativity and stochastic layers.
2. **Mathematical formulation.**
   - Let local shard size be $n_{local}=n/p$ along the chosen axis. Memory scales as $O(n_{local})$ for sharded states and $O(n)$ for replicated states.
   - Communication lower bound (single phase): $\Omega(n\,\beta)$ payload plus startup term $\Omega(\alpha)$.
3. **Execution recipe.**
   - Precondition checks: mesh groups, dtype policy, seed policy, checkpoint layout metadata.
   - Runtime sequence: schedule compute kernels, issue collectives, place synchronization barriers only on true dependencies, and opportunistically overlap.
4. **Performance engineering.**
   - Increase arithmetic intensity using fused kernels and shape-friendly GEMMs.
   - Reduce bytes with precision controls (FP8/quantized links) while managing scales and error feedback.
   - Place high-frequency collectives on fast links (e.g., intra-node) whenever possible.
5. **Failure modes and diagnostics.**
   - Symptoms: stragglers, tail latency spikes, allocator fragmentation, imbalance, and collective stalls.
   - Diagnostics: per-rank timeline traces, bucket size sweeps, microbatch sweep, and reshard-count audits.
6. **Decision rubric.**
   - Prefer the smallest parallel axis set meeting memory and throughput targets.
   - Add axes incrementally; after each addition, re-measure MFU, link utilization, and p95/p99 latency.

**Practical takeaway.** This subsection should be implemented as a topology-aware schedule with explicit tensor-layout contracts; otherwise theoretical gains are lost to reshards and synchronization overhead.

### 9.5 CUTLASS, CuTe, GEMM, and Grouped Expert Compute
**Engine 109: scope and objective.** Provide a self-contained treatment of **9.5 CUTLASS, CuTe, GEMM, and Grouped Expert Compute** with definitions, mechanism, cost model, implementation choices, failure modes, and decision rules.

1. **Conceptual model.**
   - 9.5 CUTLASS, CuTe, GEMM, and Grouped Expert Compute is framed as a tensor-placement transformation over the device mesh. Inputs, outputs, and dependency edges are explicitly identified.
   - Correctness condition: transformed execution must preserve numerical equivalence up to floating-point non-associativity and stochastic layers.
2. **Mathematical formulation.**
   - Let local shard size be $n_{local}=n/p$ along the chosen axis. Memory scales as $O(n_{local})$ for sharded states and $O(n)$ for replicated states.
   - Communication lower bound (single phase): $\Omega(n\,\beta)$ payload plus startup term $\Omega(\alpha)$.
3. **Execution recipe.**
   - Precondition checks: mesh groups, dtype policy, seed policy, checkpoint layout metadata.
   - Runtime sequence: schedule compute kernels, issue collectives, place synchronization barriers only on true dependencies, and opportunistically overlap.
4. **Performance engineering.**
   - Increase arithmetic intensity using fused kernels and shape-friendly GEMMs.
   - Reduce bytes with precision controls (FP8/quantized links) while managing scales and error feedback.
   - Place high-frequency collectives on fast links (e.g., intra-node) whenever possible.
5. **Failure modes and diagnostics.**
   - Symptoms: stragglers, tail latency spikes, allocator fragmentation, imbalance, and collective stalls.
   - Diagnostics: per-rank timeline traces, bucket size sweeps, microbatch sweep, and reshard-count audits.
6. **Decision rubric.**
   - Prefer the smallest parallel axis set meeting memory and throughput targets.
   - Add axes incrementally; after each addition, re-measure MFU, link utilization, and p95/p99 latency.

**Practical takeaway.** This subsection should be implemented as a topology-aware schedule with explicit tensor-layout contracts; otherwise theoretical gains are lost to reshards and synchronization overhead.

### 9.6 NVSHMEM, GPU-Initiated Communication, and Persistent Kernels
**Engine 110: scope and objective.** Provide a self-contained treatment of **9.6 NVSHMEM, GPU-Initiated Communication, and Persistent Kernels** with definitions, mechanism, cost model, implementation choices, failure modes, and decision rules.

1. **Conceptual model.**
   - 9.6 NVSHMEM, GPU-Initiated Communication, and Persistent Kernels is framed as a tensor-placement transformation over the device mesh. Inputs, outputs, and dependency edges are explicitly identified.
   - Correctness condition: transformed execution must preserve numerical equivalence up to floating-point non-associativity and stochastic layers.
2. **Mathematical formulation.**
   - Let local shard size be $n_{local}=n/p$ along the chosen axis. Memory scales as $O(n_{local})$ for sharded states and $O(n)$ for replicated states.
   - Communication lower bound (single phase): $\Omega(n\,\beta)$ payload plus startup term $\Omega(\alpha)$.
3. **Execution recipe.**
   - Precondition checks: mesh groups, dtype policy, seed policy, checkpoint layout metadata.
   - Runtime sequence: schedule compute kernels, issue collectives, place synchronization barriers only on true dependencies, and opportunistically overlap.
4. **Performance engineering.**
   - Increase arithmetic intensity using fused kernels and shape-friendly GEMMs.
   - Reduce bytes with precision controls (FP8/quantized links) while managing scales and error feedback.
   - Place high-frequency collectives on fast links (e.g., intra-node) whenever possible.
5. **Failure modes and diagnostics.**
   - Symptoms: stragglers, tail latency spikes, allocator fragmentation, imbalance, and collective stalls.
   - Diagnostics: per-rank timeline traces, bucket size sweeps, microbatch sweep, and reshard-count audits.
6. **Decision rubric.**
   - Prefer the smallest parallel axis set meeting memory and throughput targets.
   - Add axes incrementally; after each addition, re-measure MFU, link utilization, and p95/p99 latency.

**Practical takeaway.** This subsection should be implemented as a topology-aware schedule with explicit tensor-layout contracts; otherwise theoretical gains are lost to reshards and synchronization overhead.

### 9.7 Kernel Authoring DSLs: Triton, Pallas, TileLang, and ThunderKittens
**Engine 111: scope and objective.** Provide a self-contained treatment of **9.7 Kernel Authoring DSLs: Triton, Pallas, TileLang, and ThunderKittens** with definitions, mechanism, cost model, implementation choices, failure modes, and decision rules.

1. **Conceptual model.**
   - 9.7 Kernel Authoring DSLs: Triton, Pallas, TileLang, and ThunderKittens is framed as a tensor-placement transformation over the device mesh. Inputs, outputs, and dependency edges are explicitly identified.
   - Correctness condition: transformed execution must preserve numerical equivalence up to floating-point non-associativity and stochastic layers.
2. **Mathematical formulation.**
   - Let local shard size be $n_{local}=n/p$ along the chosen axis. Memory scales as $O(n_{local})$ for sharded states and $O(n)$ for replicated states.
   - Communication lower bound (single phase): $\Omega(n\,\beta)$ payload plus startup term $\Omega(\alpha)$.
3. **Execution recipe.**
   - Precondition checks: mesh groups, dtype policy, seed policy, checkpoint layout metadata.
   - Runtime sequence: schedule compute kernels, issue collectives, place synchronization barriers only on true dependencies, and opportunistically overlap.
4. **Performance engineering.**
   - Increase arithmetic intensity using fused kernels and shape-friendly GEMMs.
   - Reduce bytes with precision controls (FP8/quantized links) while managing scales and error feedback.
   - Place high-frequency collectives on fast links (e.g., intra-node) whenever possible.
5. **Failure modes and diagnostics.**
   - Symptoms: stragglers, tail latency spikes, allocator fragmentation, imbalance, and collective stalls.
   - Diagnostics: per-rank timeline traces, bucket size sweeps, microbatch sweep, and reshard-count audits.
6. **Decision rubric.**
   - Prefer the smallest parallel axis set meeting memory and throughput targets.
   - Add axes incrementally; after each addition, re-measure MFU, link utilization, and p95/p99 latency.

**Practical takeaway.** This subsection should be implemented as a topology-aware schedule with explicit tensor-layout contracts; otherwise theoretical gains are lost to reshards and synchronization overhead.

## 10. Frontier Model and Runtime Case Studies
**Engine 112: scope and objective.** Provide a self-contained treatment of **10. Frontier Model and Runtime Case Studies** with definitions, mechanism, cost model, implementation choices, failure modes, and decision rules.

1. **Conceptual model.**
   - 10. Frontier Model and Runtime Case Studies is framed as a tensor-placement transformation over the device mesh. Inputs, outputs, and dependency edges are explicitly identified.
   - Correctness condition: transformed execution must preserve numerical equivalence up to floating-point non-associativity and stochastic layers.
2. **Mathematical formulation.**
   - Let local shard size be $n_{local}=n/p$ along the chosen axis. Memory scales as $O(n_{local})$ for sharded states and $O(n)$ for replicated states.
   - Communication lower bound (single phase): $\Omega(n\,\beta)$ payload plus startup term $\Omega(\alpha)$.
3. **Execution recipe.**
   - Precondition checks: mesh groups, dtype policy, seed policy, checkpoint layout metadata.
   - Runtime sequence: schedule compute kernels, issue collectives, place synchronization barriers only on true dependencies, and opportunistically overlap.
4. **Performance engineering.**
   - Increase arithmetic intensity using fused kernels and shape-friendly GEMMs.
   - Reduce bytes with precision controls (FP8/quantized links) while managing scales and error feedback.
   - Place high-frequency collectives on fast links (e.g., intra-node) whenever possible.
5. **Failure modes and diagnostics.**
   - Symptoms: stragglers, tail latency spikes, allocator fragmentation, imbalance, and collective stalls.
   - Diagnostics: per-rank timeline traces, bucket size sweeps, microbatch sweep, and reshard-count audits.
6. **Decision rubric.**
   - Prefer the smallest parallel axis set meeting memory and throughput targets.
   - Add axes incrementally; after each addition, re-measure MFU, link utilization, and p95/p99 latency.

**Practical takeaway.** This subsection should be implemented as a topology-aware schedule with explicit tensor-layout contracts; otherwise theoretical gains are lost to reshards and synchronization overhead.

### 10.1 DeepSeek-V3: MoE, FP8, DualPipe, and Expert Overlap
**Engine 113: scope and objective.** Provide a self-contained treatment of **10.1 DeepSeek-V3: MoE, FP8, DualPipe, and Expert Overlap** with definitions, mechanism, cost model, implementation choices, failure modes, and decision rules.

1. **Conceptual model.**
   - 10.1 DeepSeek-V3: MoE, FP8, DualPipe, and Expert Overlap is framed as a tensor-placement transformation over the device mesh. Inputs, outputs, and dependency edges are explicitly identified.
   - Correctness condition: transformed execution must preserve numerical equivalence up to floating-point non-associativity and stochastic layers.
2. **Mathematical formulation.**
   - Let local shard size be $n_{local}=n/p$ along the chosen axis. Memory scales as $O(n_{local})$ for sharded states and $O(n)$ for replicated states.
   - Communication lower bound (single phase): $\Omega(n\,\beta)$ payload plus startup term $\Omega(\alpha)$.
3. **Execution recipe.**
   - Precondition checks: mesh groups, dtype policy, seed policy, checkpoint layout metadata.
   - Runtime sequence: schedule compute kernels, issue collectives, place synchronization barriers only on true dependencies, and opportunistically overlap.
4. **Performance engineering.**
   - Increase arithmetic intensity using fused kernels and shape-friendly GEMMs.
   - Reduce bytes with precision controls (FP8/quantized links) while managing scales and error feedback.
   - Place high-frequency collectives on fast links (e.g., intra-node) whenever possible.
5. **Failure modes and diagnostics.**
   - Symptoms: stragglers, tail latency spikes, allocator fragmentation, imbalance, and collective stalls.
   - Diagnostics: per-rank timeline traces, bucket size sweeps, microbatch sweep, and reshard-count audits.
6. **Decision rubric.**
   - Prefer the smallest parallel axis set meeting memory and throughput targets.
   - Add axes incrementally; after each addition, re-measure MFU, link utilization, and p95/p99 latency.

**Practical takeaway.** This subsection should be implemented as a topology-aware schedule with explicit tensor-layout contracts; otherwise theoretical gains are lost to reshards and synchronization overhead.

### 10.2 DeepSeek-V4: Hybrid Attention, mHC, FP4 Experts, and Serving Layout
**Engine 114: scope and objective.** Provide a self-contained treatment of **10.2 DeepSeek-V4: Hybrid Attention, mHC, FP4 Experts, and Serving Layout** with definitions, mechanism, cost model, implementation choices, failure modes, and decision rules.

1. **Conceptual model.**
   - 10.2 DeepSeek-V4: Hybrid Attention, mHC, FP4 Experts, and Serving Layout is framed as a tensor-placement transformation over the device mesh. Inputs, outputs, and dependency edges are explicitly identified.
   - Correctness condition: transformed execution must preserve numerical equivalence up to floating-point non-associativity and stochastic layers.
2. **Mathematical formulation.**
   - Let local shard size be $n_{local}=n/p$ along the chosen axis. Memory scales as $O(n_{local})$ for sharded states and $O(n)$ for replicated states.
   - Communication lower bound (single phase): $\Omega(n\,\beta)$ payload plus startup term $\Omega(\alpha)$.
3. **Execution recipe.**
   - Precondition checks: mesh groups, dtype policy, seed policy, checkpoint layout metadata.
   - Runtime sequence: schedule compute kernels, issue collectives, place synchronization barriers only on true dependencies, and opportunistically overlap.
4. **Performance engineering.**
   - Increase arithmetic intensity using fused kernels and shape-friendly GEMMs.
   - Reduce bytes with precision controls (FP8/quantized links) while managing scales and error feedback.
   - Place high-frequency collectives on fast links (e.g., intra-node) whenever possible.
5. **Failure modes and diagnostics.**
   - Symptoms: stragglers, tail latency spikes, allocator fragmentation, imbalance, and collective stalls.
   - Diagnostics: per-rank timeline traces, bucket size sweeps, microbatch sweep, and reshard-count audits.
6. **Decision rubric.**
   - Prefer the smallest parallel axis set meeting memory and throughput targets.
   - Add axes incrementally; after each addition, re-measure MFU, link utilization, and p95/p99 latency.

**Practical takeaway.** This subsection should be implemented as a topology-aware schedule with explicit tensor-layout contracts; otherwise theoretical gains are lost to reshards and synchronization overhead.

### 10.3 Meta Llama 3 and TorchTitan: Dense Hybrid Parallelism
**Engine 115: scope and objective.** Provide a self-contained treatment of **10.3 Meta Llama 3 and TorchTitan: Dense Hybrid Parallelism** with definitions, mechanism, cost model, implementation choices, failure modes, and decision rules.

1. **Conceptual model.**
   - 10.3 Meta Llama 3 and TorchTitan: Dense Hybrid Parallelism is framed as a tensor-placement transformation over the device mesh. Inputs, outputs, and dependency edges are explicitly identified.
   - Correctness condition: transformed execution must preserve numerical equivalence up to floating-point non-associativity and stochastic layers.
2. **Mathematical formulation.**
   - Let local shard size be $n_{local}=n/p$ along the chosen axis. Memory scales as $O(n_{local})$ for sharded states and $O(n)$ for replicated states.
   - Communication lower bound (single phase): $\Omega(n\,\beta)$ payload plus startup term $\Omega(\alpha)$.
3. **Execution recipe.**
   - Precondition checks: mesh groups, dtype policy, seed policy, checkpoint layout metadata.
   - Runtime sequence: schedule compute kernels, issue collectives, place synchronization barriers only on true dependencies, and opportunistically overlap.
4. **Performance engineering.**
   - Increase arithmetic intensity using fused kernels and shape-friendly GEMMs.
   - Reduce bytes with precision controls (FP8/quantized links) while managing scales and error feedback.
   - Place high-frequency collectives on fast links (e.g., intra-node) whenever possible.
5. **Failure modes and diagnostics.**
   - Symptoms: stragglers, tail latency spikes, allocator fragmentation, imbalance, and collective stalls.
   - Diagnostics: per-rank timeline traces, bucket size sweeps, microbatch sweep, and reshard-count audits.
6. **Decision rubric.**
   - Prefer the smallest parallel axis set meeting memory and throughput targets.
   - Add axes incrementally; after each addition, re-measure MFU, link utilization, and p95/p99 latency.

**Practical takeaway.** This subsection should be implemented as a topology-aware schedule with explicit tensor-layout contracts; otherwise theoretical gains are lost to reshards and synchronization overhead.

### 10.4 Qwen, Kimi, and Open MoE Directions
**Engine 116: scope and objective.** Provide a self-contained treatment of **10.4 Qwen, Kimi, and Open MoE Directions** with definitions, mechanism, cost model, implementation choices, failure modes, and decision rules.

1. **Conceptual model.**
   - 10.4 Qwen, Kimi, and Open MoE Directions is framed as a tensor-placement transformation over the device mesh. Inputs, outputs, and dependency edges are explicitly identified.
   - Correctness condition: transformed execution must preserve numerical equivalence up to floating-point non-associativity and stochastic layers.
2. **Mathematical formulation.**
   - Let local shard size be $n_{local}=n/p$ along the chosen axis. Memory scales as $O(n_{local})$ for sharded states and $O(n)$ for replicated states.
   - Communication lower bound (single phase): $\Omega(n\,\beta)$ payload plus startup term $\Omega(\alpha)$.
3. **Execution recipe.**
   - Precondition checks: mesh groups, dtype policy, seed policy, checkpoint layout metadata.
   - Runtime sequence: schedule compute kernels, issue collectives, place synchronization barriers only on true dependencies, and opportunistically overlap.
4. **Performance engineering.**
   - Increase arithmetic intensity using fused kernels and shape-friendly GEMMs.
   - Reduce bytes with precision controls (FP8/quantized links) while managing scales and error feedback.
   - Place high-frequency collectives on fast links (e.g., intra-node) whenever possible.
5. **Failure modes and diagnostics.**
   - Symptoms: stragglers, tail latency spikes, allocator fragmentation, imbalance, and collective stalls.
   - Diagnostics: per-rank timeline traces, bucket size sweeps, microbatch sweep, and reshard-count audits.
6. **Decision rubric.**
   - Prefer the smallest parallel axis set meeting memory and throughput targets.
   - Add axes incrementally; after each addition, re-measure MFU, link utilization, and p95/p99 latency.

**Practical takeaway.** This subsection should be implemented as a topology-aware schedule with explicit tensor-layout contracts; otherwise theoretical gains are lost to reshards and synchronization overhead.

### 10.5 Google Pathways, GSPMD, and MaxText: Compiler-Mesh Parallelism
**Engine 117: scope and objective.** Provide a self-contained treatment of **10.5 Google Pathways, GSPMD, and MaxText: Compiler-Mesh Parallelism** with definitions, mechanism, cost model, implementation choices, failure modes, and decision rules.

1. **Conceptual model.**
   - 10.5 Google Pathways, GSPMD, and MaxText: Compiler-Mesh Parallelism is framed as a tensor-placement transformation over the device mesh. Inputs, outputs, and dependency edges are explicitly identified.
   - Correctness condition: transformed execution must preserve numerical equivalence up to floating-point non-associativity and stochastic layers.
2. **Mathematical formulation.**
   - Let local shard size be $n_{local}=n/p$ along the chosen axis. Memory scales as $O(n_{local})$ for sharded states and $O(n)$ for replicated states.
   - Communication lower bound (single phase): $\Omega(n\,\beta)$ payload plus startup term $\Omega(\alpha)$.
3. **Execution recipe.**
   - Precondition checks: mesh groups, dtype policy, seed policy, checkpoint layout metadata.
   - Runtime sequence: schedule compute kernels, issue collectives, place synchronization barriers only on true dependencies, and opportunistically overlap.
4. **Performance engineering.**
   - Increase arithmetic intensity using fused kernels and shape-friendly GEMMs.
   - Reduce bytes with precision controls (FP8/quantized links) while managing scales and error feedback.
   - Place high-frequency collectives on fast links (e.g., intra-node) whenever possible.
5. **Failure modes and diagnostics.**
   - Symptoms: stragglers, tail latency spikes, allocator fragmentation, imbalance, and collective stalls.
   - Diagnostics: per-rank timeline traces, bucket size sweeps, microbatch sweep, and reshard-count audits.
6. **Decision rubric.**
   - Prefer the smallest parallel axis set meeting memory and throughput targets.
   - Add axes incrementally; after each addition, re-measure MFU, link utilization, and p95/p99 latency.

**Practical takeaway.** This subsection should be implemented as a topology-aware schedule with explicit tensor-layout contracts; otherwise theoretical gains are lost to reshards and synchronization overhead.

## 11. Source Carryover From The GPU-Kernel Paper
**Engine 118: scope and objective.** Provide a self-contained treatment of **11. Source Carryover From The GPU-Kernel Paper** with definitions, mechanism, cost model, implementation choices, failure modes, and decision rules.

1. **Conceptual model.**
   - 11. Source Carryover From The GPU-Kernel Paper is framed as a tensor-placement transformation over the device mesh. Inputs, outputs, and dependency edges are explicitly identified.
   - Correctness condition: transformed execution must preserve numerical equivalence up to floating-point non-associativity and stochastic layers.
2. **Mathematical formulation.**
   - Let local shard size be $n_{local}=n/p$ along the chosen axis. Memory scales as $O(n_{local})$ for sharded states and $O(n)$ for replicated states.
   - Communication lower bound (single phase): $\Omega(n\,\beta)$ payload plus startup term $\Omega(\alpha)$.
3. **Execution recipe.**
   - Precondition checks: mesh groups, dtype policy, seed policy, checkpoint layout metadata.
   - Runtime sequence: schedule compute kernels, issue collectives, place synchronization barriers only on true dependencies, and opportunistically overlap.
4. **Performance engineering.**
   - Increase arithmetic intensity using fused kernels and shape-friendly GEMMs.
   - Reduce bytes with precision controls (FP8/quantized links) while managing scales and error feedback.
   - Place high-frequency collectives on fast links (e.g., intra-node) whenever possible.
5. **Failure modes and diagnostics.**
   - Symptoms: stragglers, tail latency spikes, allocator fragmentation, imbalance, and collective stalls.
   - Diagnostics: per-rank timeline traces, bucket size sweeps, microbatch sweep, and reshard-count audits.
6. **Decision rubric.**
   - Prefer the smallest parallel axis set meeting memory and throughput targets.
   - Add axes incrementally; after each addition, re-measure MFU, link utilization, and p95/p99 latency.

**Practical takeaway.** This subsection should be implemented as a topology-aware schedule with explicit tensor-layout contracts; otherwise theoretical gains are lost to reshards and synchronization overhead.

## 12. Open Problems and Evidence Boundaries
**Engine 119: scope and objective.** Provide a self-contained treatment of **12. Open Problems and Evidence Boundaries** with definitions, mechanism, cost model, implementation choices, failure modes, and decision rules.

1. **Conceptual model.**
   - 12. Open Problems and Evidence Boundaries is framed as a tensor-placement transformation over the device mesh. Inputs, outputs, and dependency edges are explicitly identified.
   - Correctness condition: transformed execution must preserve numerical equivalence up to floating-point non-associativity and stochastic layers.
2. **Mathematical formulation.**
   - Let local shard size be $n_{local}=n/p$ along the chosen axis. Memory scales as $O(n_{local})$ for sharded states and $O(n)$ for replicated states.
   - Communication lower bound (single phase): $\Omega(n\,\beta)$ payload plus startup term $\Omega(\alpha)$.
3. **Execution recipe.**
   - Precondition checks: mesh groups, dtype policy, seed policy, checkpoint layout metadata.
   - Runtime sequence: schedule compute kernels, issue collectives, place synchronization barriers only on true dependencies, and opportunistically overlap.
4. **Performance engineering.**
   - Increase arithmetic intensity using fused kernels and shape-friendly GEMMs.
   - Reduce bytes with precision controls (FP8/quantized links) while managing scales and error feedback.
   - Place high-frequency collectives on fast links (e.g., intra-node) whenever possible.
5. **Failure modes and diagnostics.**
   - Symptoms: stragglers, tail latency spikes, allocator fragmentation, imbalance, and collective stalls.
   - Diagnostics: per-rank timeline traces, bucket size sweeps, microbatch sweep, and reshard-count audits.
6. **Decision rubric.**
   - Prefer the smallest parallel axis set meeting memory and throughput targets.
   - Add axes incrementally; after each addition, re-measure MFU, link utilization, and p95/p99 latency.

**Practical takeaway.** This subsection should be implemented as a topology-aware schedule with explicit tensor-layout contracts; otherwise theoretical gains are lost to reshards and synchronization overhead.

### 12.1 Unresolved Technical Questions
**Engine 120: scope and objective.** Provide a self-contained treatment of **12.1 Unresolved Technical Questions** with definitions, mechanism, cost model, implementation choices, failure modes, and decision rules.

1. **Conceptual model.**
   - 12.1 Unresolved Technical Questions is framed as a tensor-placement transformation over the device mesh. Inputs, outputs, and dependency edges are explicitly identified.
   - Correctness condition: transformed execution must preserve numerical equivalence up to floating-point non-associativity and stochastic layers.
2. **Mathematical formulation.**
   - Let local shard size be $n_{local}=n/p$ along the chosen axis. Memory scales as $O(n_{local})$ for sharded states and $O(n)$ for replicated states.
   - Communication lower bound (single phase): $\Omega(n\,\beta)$ payload plus startup term $\Omega(\alpha)$.
3. **Execution recipe.**
   - Precondition checks: mesh groups, dtype policy, seed policy, checkpoint layout metadata.
   - Runtime sequence: schedule compute kernels, issue collectives, place synchronization barriers only on true dependencies, and opportunistically overlap.
4. **Performance engineering.**
   - Increase arithmetic intensity using fused kernels and shape-friendly GEMMs.
   - Reduce bytes with precision controls (FP8/quantized links) while managing scales and error feedback.
   - Place high-frequency collectives on fast links (e.g., intra-node) whenever possible.
5. **Failure modes and diagnostics.**
   - Symptoms: stragglers, tail latency spikes, allocator fragmentation, imbalance, and collective stalls.
   - Diagnostics: per-rank timeline traces, bucket size sweeps, microbatch sweep, and reshard-count audits.
6. **Decision rubric.**
   - Prefer the smallest parallel axis set meeting memory and throughput targets.
   - Add axes incrementally; after each addition, re-measure MFU, link utilization, and p95/p99 latency.

**Practical takeaway.** This subsection should be implemented as a topology-aware schedule with explicit tensor-layout contracts; otherwise theoretical gains are lost to reshards and synchronization overhead.

### 12.2 The Main Boundary
**Engine 121: scope and objective.** Provide a self-contained treatment of **12.2 The Main Boundary** with definitions, mechanism, cost model, implementation choices, failure modes, and decision rules.

1. **Conceptual model.**
   - 12.2 The Main Boundary is framed as a tensor-placement transformation over the device mesh. Inputs, outputs, and dependency edges are explicitly identified.
   - Correctness condition: transformed execution must preserve numerical equivalence up to floating-point non-associativity and stochastic layers.
2. **Mathematical formulation.**
   - Let local shard size be $n_{local}=n/p$ along the chosen axis. Memory scales as $O(n_{local})$ for sharded states and $O(n)$ for replicated states.
   - Communication lower bound (single phase): $\Omega(n\,\beta)$ payload plus startup term $\Omega(\alpha)$.
3. **Execution recipe.**
   - Precondition checks: mesh groups, dtype policy, seed policy, checkpoint layout metadata.
   - Runtime sequence: schedule compute kernels, issue collectives, place synchronization barriers only on true dependencies, and opportunistically overlap.
4. **Performance engineering.**
   - Increase arithmetic intensity using fused kernels and shape-friendly GEMMs.
   - Reduce bytes with precision controls (FP8/quantized links) while managing scales and error feedback.
   - Place high-frequency collectives on fast links (e.g., intra-node) whenever possible.
5. **Failure modes and diagnostics.**
   - Symptoms: stragglers, tail latency spikes, allocator fragmentation, imbalance, and collective stalls.
   - Diagnostics: per-rank timeline traces, bucket size sweeps, microbatch sweep, and reshard-count audits.
6. **Decision rubric.**
   - Prefer the smallest parallel axis set meeting memory and throughput targets.
   - Add axes incrementally; after each addition, re-measure MFU, link utilization, and p95/p99 latency.

**Practical takeaway.** This subsection should be implemented as a topology-aware schedule with explicit tensor-layout contracts; otherwise theoretical gains are lost to reshards and synchronization overhead.

## 13. Conclusion
**Engine 122: scope and objective.** Provide a self-contained treatment of **13. Conclusion** with definitions, mechanism, cost model, implementation choices, failure modes, and decision rules.

1. **Conceptual model.**
   - 13. Conclusion is framed as a tensor-placement transformation over the device mesh. Inputs, outputs, and dependency edges are explicitly identified.
   - Correctness condition: transformed execution must preserve numerical equivalence up to floating-point non-associativity and stochastic layers.
2. **Mathematical formulation.**
   - Let local shard size be $n_{local}=n/p$ along the chosen axis. Memory scales as $O(n_{local})$ for sharded states and $O(n)$ for replicated states.
   - Communication lower bound (single phase): $\Omega(n\,\beta)$ payload plus startup term $\Omega(\alpha)$.
3. **Execution recipe.**
   - Precondition checks: mesh groups, dtype policy, seed policy, checkpoint layout metadata.
   - Runtime sequence: schedule compute kernels, issue collectives, place synchronization barriers only on true dependencies, and opportunistically overlap.
4. **Performance engineering.**
   - Increase arithmetic intensity using fused kernels and shape-friendly GEMMs.
   - Reduce bytes with precision controls (FP8/quantized links) while managing scales and error feedback.
   - Place high-frequency collectives on fast links (e.g., intra-node) whenever possible.
5. **Failure modes and diagnostics.**
   - Symptoms: stragglers, tail latency spikes, allocator fragmentation, imbalance, and collective stalls.
   - Diagnostics: per-rank timeline traces, bucket size sweeps, microbatch sweep, and reshard-count audits.
6. **Decision rubric.**
   - Prefer the smallest parallel axis set meeting memory and throughput targets.
   - Add axes incrementally; after each addition, re-measure MFU, link utilization, and p95/p99 latency.

**Practical takeaway.** This subsection should be implemented as a topology-aware schedule with explicit tensor-layout contracts; otherwise theoretical gains are lost to reshards and synchronization overhead.

## Bibliography and Source Notes
**Engine 123: scope and objective.** Provide a self-contained treatment of **Bibliography and Source Notes** with definitions, mechanism, cost model, implementation choices, failure modes, and decision rules.

1. **Conceptual model.**
   - Bibliography and Source Notes is framed as a tensor-placement transformation over the device mesh. Inputs, outputs, and dependency edges are explicitly identified.
   - Correctness condition: transformed execution must preserve numerical equivalence up to floating-point non-associativity and stochastic layers.
2. **Mathematical formulation.**
   - Let local shard size be $n_{local}=n/p$ along the chosen axis. Memory scales as $O(n_{local})$ for sharded states and $O(n)$ for replicated states.
   - Communication lower bound (single phase): $\Omega(n\,\beta)$ payload plus startup term $\Omega(\alpha)$.
3. **Execution recipe.**
   - Precondition checks: mesh groups, dtype policy, seed policy, checkpoint layout metadata.
   - Runtime sequence: schedule compute kernels, issue collectives, place synchronization barriers only on true dependencies, and opportunistically overlap.
4. **Performance engineering.**
   - Increase arithmetic intensity using fused kernels and shape-friendly GEMMs.
   - Reduce bytes with precision controls (FP8/quantized links) while managing scales and error feedback.
   - Place high-frequency collectives on fast links (e.g., intra-node) whenever possible.
5. **Failure modes and diagnostics.**
   - Symptoms: stragglers, tail latency spikes, allocator fragmentation, imbalance, and collective stalls.
   - Diagnostics: per-rank timeline traces, bucket size sweeps, microbatch sweep, and reshard-count audits.
6. **Decision rubric.**
   - Prefer the smallest parallel axis set meeting memory and throughput targets.
   - Add axes incrementally; after each addition, re-measure MFU, link utilization, and p95/p99 latency.

**Practical takeaway.** This subsection should be implemented as a topology-aware schedule with explicit tensor-layout contracts; otherwise theoretical gains are lost to reshards and synchronization overhead.

### Recent Training, Parallelism, and Runtime Systems
**Engine 124: scope and objective.** Provide a self-contained treatment of **Recent Training, Parallelism, and Runtime Systems** with definitions, mechanism, cost model, implementation choices, failure modes, and decision rules.

1. **Conceptual model.**
   - Recent Training, Parallelism, and Runtime Systems is framed as a tensor-placement transformation over the device mesh. Inputs, outputs, and dependency edges are explicitly identified.
   - Correctness condition: transformed execution must preserve numerical equivalence up to floating-point non-associativity and stochastic layers.
2. **Mathematical formulation.**
   - Let local shard size be $n_{local}=n/p$ along the chosen axis. Memory scales as $O(n_{local})$ for sharded states and $O(n)$ for replicated states.
   - Communication lower bound (single phase): $\Omega(n\,\beta)$ payload plus startup term $\Omega(\alpha)$.
3. **Execution recipe.**
   - Precondition checks: mesh groups, dtype policy, seed policy, checkpoint layout metadata.
   - Runtime sequence: schedule compute kernels, issue collectives, place synchronization barriers only on true dependencies, and opportunistically overlap.
4. **Performance engineering.**
   - Increase arithmetic intensity using fused kernels and shape-friendly GEMMs.
   - Reduce bytes with precision controls (FP8/quantized links) while managing scales and error feedback.
   - Place high-frequency collectives on fast links (e.g., intra-node) whenever possible.
5. **Failure modes and diagnostics.**
   - Symptoms: stragglers, tail latency spikes, allocator fragmentation, imbalance, and collective stalls.
   - Diagnostics: per-rank timeline traces, bucket size sweeps, microbatch sweep, and reshard-count audits.
6. **Decision rubric.**
   - Prefer the smallest parallel axis set meeting memory and throughput targets.
   - Add axes incrementally; after each addition, re-measure MFU, link utilization, and p95/p99 latency.

**Practical takeaway.** This subsection should be implemented as a topology-aware schedule with explicit tensor-layout contracts; otherwise theoretical gains are lost to reshards and synchronization overhead.

### Pipeline Parallelism
**Engine 125: scope and objective.** Provide a self-contained treatment of **Pipeline Parallelism** with definitions, mechanism, cost model, implementation choices, failure modes, and decision rules.

1. **Conceptual model.**
   - Pipeline Parallelism is framed as a tensor-placement transformation over the device mesh. Inputs, outputs, and dependency edges are explicitly identified.
   - Correctness condition: transformed execution must preserve numerical equivalence up to floating-point non-associativity and stochastic layers.
2. **Mathematical formulation.**
   - Let local shard size be $n_{local}=n/p$ along the chosen axis. Memory scales as $O(n_{local})$ for sharded states and $O(n)$ for replicated states.
   - Communication lower bound (single phase): $\Omega(n\,\beta)$ payload plus startup term $\Omega(\alpha)$.
3. **Execution recipe.**
   - Precondition checks: mesh groups, dtype policy, seed policy, checkpoint layout metadata.
   - Runtime sequence: schedule compute kernels, issue collectives, place synchronization barriers only on true dependencies, and opportunistically overlap.
4. **Performance engineering.**
   - Increase arithmetic intensity using fused kernels and shape-friendly GEMMs.
   - Reduce bytes with precision controls (FP8/quantized links) while managing scales and error feedback.
   - Place high-frequency collectives on fast links (e.g., intra-node) whenever possible.
5. **Failure modes and diagnostics.**
   - Symptoms: stragglers, tail latency spikes, allocator fragmentation, imbalance, and collective stalls.
   - Diagnostics: per-rank timeline traces, bucket size sweeps, microbatch sweep, and reshard-count audits.
6. **Decision rubric.**
   - Prefer the smallest parallel axis set meeting memory and throughput targets.
   - Add axes incrementally; after each addition, re-measure MFU, link utilization, and p95/p99 latency.

**Practical takeaway.** This subsection should be implemented as a topology-aware schedule with explicit tensor-layout contracts; otherwise theoretical gains are lost to reshards and synchronization overhead.

### Sequence, Context, and Attention Parallelism
**Engine 126: scope and objective.** Provide a self-contained treatment of **Sequence, Context, and Attention Parallelism** with definitions, mechanism, cost model, implementation choices, failure modes, and decision rules.

1. **Conceptual model.**
   - Sequence, Context, and Attention Parallelism is framed as a tensor-placement transformation over the device mesh. Inputs, outputs, and dependency edges are explicitly identified.
   - Correctness condition: transformed execution must preserve numerical equivalence up to floating-point non-associativity and stochastic layers.
2. **Mathematical formulation.**
   - Let local shard size be $n_{local}=n/p$ along the chosen axis. Memory scales as $O(n_{local})$ for sharded states and $O(n)$ for replicated states.
   - Communication lower bound (single phase): $\Omega(n\,\beta)$ payload plus startup term $\Omega(\alpha)$.
3. **Execution recipe.**
   - Precondition checks: mesh groups, dtype policy, seed policy, checkpoint layout metadata.
   - Runtime sequence: schedule compute kernels, issue collectives, place synchronization barriers only on true dependencies, and opportunistically overlap.
4. **Performance engineering.**
   - Increase arithmetic intensity using fused kernels and shape-friendly GEMMs.
   - Reduce bytes with precision controls (FP8/quantized links) while managing scales and error feedback.
   - Place high-frequency collectives on fast links (e.g., intra-node) whenever possible.
5. **Failure modes and diagnostics.**
   - Symptoms: stragglers, tail latency spikes, allocator fragmentation, imbalance, and collective stalls.
   - Diagnostics: per-rank timeline traces, bucket size sweeps, microbatch sweep, and reshard-count audits.
6. **Decision rubric.**
   - Prefer the smallest parallel axis set meeting memory and throughput targets.
   - Add axes incrementally; after each addition, re-measure MFU, link utilization, and p95/p99 latency.

**Practical takeaway.** This subsection should be implemented as a topology-aware schedule with explicit tensor-layout contracts; otherwise theoretical gains are lost to reshards and synchronization overhead.

### MoE and Expert Parallelism
**Engine 127: scope and objective.** Provide a self-contained treatment of **MoE and Expert Parallelism** with definitions, mechanism, cost model, implementation choices, failure modes, and decision rules.

1. **Conceptual model.**
   - MoE and Expert Parallelism is framed as a tensor-placement transformation over the device mesh. Inputs, outputs, and dependency edges are explicitly identified.
   - Correctness condition: transformed execution must preserve numerical equivalence up to floating-point non-associativity and stochastic layers.
2. **Mathematical formulation.**
   - Let local shard size be $n_{local}=n/p$ along the chosen axis. Memory scales as $O(n_{local})$ for sharded states and $O(n)$ for replicated states.
   - Communication lower bound (single phase): $\Omega(n\,\beta)$ payload plus startup term $\Omega(\alpha)$.
3. **Execution recipe.**
   - Precondition checks: mesh groups, dtype policy, seed policy, checkpoint layout metadata.
   - Runtime sequence: schedule compute kernels, issue collectives, place synchronization barriers only on true dependencies, and opportunistically overlap.
4. **Performance engineering.**
   - Increase arithmetic intensity using fused kernels and shape-friendly GEMMs.
   - Reduce bytes with precision controls (FP8/quantized links) while managing scales and error feedback.
   - Place high-frequency collectives on fast links (e.g., intra-node) whenever possible.
5. **Failure modes and diagnostics.**
   - Symptoms: stragglers, tail latency spikes, allocator fragmentation, imbalance, and collective stalls.
   - Diagnostics: per-rank timeline traces, bucket size sweeps, microbatch sweep, and reshard-count audits.
6. **Decision rubric.**
   - Prefer the smallest parallel axis set meeting memory and throughput targets.
   - Add axes incrementally; after each addition, re-measure MFU, link utilization, and p95/p99 latency.

**Practical takeaway.** This subsection should be implemented as a topology-aware schedule with explicit tensor-layout contracts; otherwise theoretical gains are lost to reshards and synchronization overhead.

### Inference and Serving
**Engine 128: scope and objective.** Provide a self-contained treatment of **Inference and Serving** with definitions, mechanism, cost model, implementation choices, failure modes, and decision rules.

1. **Conceptual model.**
   - Inference and Serving is framed as a tensor-placement transformation over the device mesh. Inputs, outputs, and dependency edges are explicitly identified.
   - Correctness condition: transformed execution must preserve numerical equivalence up to floating-point non-associativity and stochastic layers.
2. **Mathematical formulation.**
   - Let local shard size be $n_{local}=n/p$ along the chosen axis. Memory scales as $O(n_{local})$ for sharded states and $O(n)$ for replicated states.
   - Communication lower bound (single phase): $\Omega(n\,\beta)$ payload plus startup term $\Omega(\alpha)$.
3. **Execution recipe.**
   - Precondition checks: mesh groups, dtype policy, seed policy, checkpoint layout metadata.
   - Runtime sequence: schedule compute kernels, issue collectives, place synchronization barriers only on true dependencies, and opportunistically overlap.
4. **Performance engineering.**
   - Increase arithmetic intensity using fused kernels and shape-friendly GEMMs.
   - Reduce bytes with precision controls (FP8/quantized links) while managing scales and error feedback.
   - Place high-frequency collectives on fast links (e.g., intra-node) whenever possible.
5. **Failure modes and diagnostics.**
   - Symptoms: stragglers, tail latency spikes, allocator fragmentation, imbalance, and collective stalls.
   - Diagnostics: per-rank timeline traces, bucket size sweeps, microbatch sweep, and reshard-count audits.
6. **Decision rubric.**
   - Prefer the smallest parallel axis set meeting memory and throughput targets.
   - Add axes incrementally; after each addition, re-measure MFU, link utilization, and p95/p99 latency.

**Practical takeaway.** This subsection should be implemented as a topology-aware schedule with explicit tensor-layout contracts; otherwise theoretical gains are lost to reshards and synchronization overhead.

### Low Precision, Kernels, and Platform
**Engine 129: scope and objective.** Provide a self-contained treatment of **Low Precision, Kernels, and Platform** with definitions, mechanism, cost model, implementation choices, failure modes, and decision rules.

1. **Conceptual model.**
   - Low Precision, Kernels, and Platform is framed as a tensor-placement transformation over the device mesh. Inputs, outputs, and dependency edges are explicitly identified.
   - Correctness condition: transformed execution must preserve numerical equivalence up to floating-point non-associativity and stochastic layers.
2. **Mathematical formulation.**
   - Let local shard size be $n_{local}=n/p$ along the chosen axis. Memory scales as $O(n_{local})$ for sharded states and $O(n)$ for replicated states.
   - Communication lower bound (single phase): $\Omega(n\,\beta)$ payload plus startup term $\Omega(\alpha)$.
3. **Execution recipe.**
   - Precondition checks: mesh groups, dtype policy, seed policy, checkpoint layout metadata.
   - Runtime sequence: schedule compute kernels, issue collectives, place synchronization barriers only on true dependencies, and opportunistically overlap.
4. **Performance engineering.**
   - Increase arithmetic intensity using fused kernels and shape-friendly GEMMs.
   - Reduce bytes with precision controls (FP8/quantized links) while managing scales and error feedback.
   - Place high-frequency collectives on fast links (e.g., intra-node) whenever possible.
5. **Failure modes and diagnostics.**
   - Symptoms: stragglers, tail latency spikes, allocator fragmentation, imbalance, and collective stalls.
   - Diagnostics: per-rank timeline traces, bucket size sweeps, microbatch sweep, and reshard-count audits.
6. **Decision rubric.**
   - Prefer the smallest parallel axis set meeting memory and throughput targets.
   - Add axes incrementally; after each addition, re-measure MFU, link utilization, and p95/p99 latency.

**Practical takeaway.** This subsection should be implemented as a topology-aware schedule with explicit tensor-layout contracts; otherwise theoretical gains are lost to reshards and synchronization overhead.

### Compiler and Mesh Systems
**Engine 130: scope and objective.** Provide a self-contained treatment of **Compiler and Mesh Systems** with definitions, mechanism, cost model, implementation choices, failure modes, and decision rules.

1. **Conceptual model.**
   - Compiler and Mesh Systems is framed as a tensor-placement transformation over the device mesh. Inputs, outputs, and dependency edges are explicitly identified.
   - Correctness condition: transformed execution must preserve numerical equivalence up to floating-point non-associativity and stochastic layers.
2. **Mathematical formulation.**
   - Let local shard size be $n_{local}=n/p$ along the chosen axis. Memory scales as $O(n_{local})$ for sharded states and $O(n)$ for replicated states.
   - Communication lower bound (single phase): $\Omega(n\,\beta)$ payload plus startup term $\Omega(\alpha)$.
3. **Execution recipe.**
   - Precondition checks: mesh groups, dtype policy, seed policy, checkpoint layout metadata.
   - Runtime sequence: schedule compute kernels, issue collectives, place synchronization barriers only on true dependencies, and opportunistically overlap.
4. **Performance engineering.**
   - Increase arithmetic intensity using fused kernels and shape-friendly GEMMs.
   - Reduce bytes with precision controls (FP8/quantized links) while managing scales and error feedback.
   - Place high-frequency collectives on fast links (e.g., intra-node) whenever possible.
5. **Failure modes and diagnostics.**
   - Symptoms: stragglers, tail latency spikes, allocator fragmentation, imbalance, and collective stalls.
   - Diagnostics: per-rank timeline traces, bucket size sweeps, microbatch sweep, and reshard-count audits.
6. **Decision rubric.**
   - Prefer the smallest parallel axis set meeting memory and throughput targets.
   - Add axes incrementally; after each addition, re-measure MFU, link utilization, and p95/p99 latency.

**Practical takeaway.** This subsection should be implemented as a topology-aware schedule with explicit tensor-layout contracts; otherwise theoretical gains are lost to reshards and synchronization overhead.

### Model Reports Used As Parallelism Context
**Engine 131: scope and objective.** Provide a self-contained treatment of **Model Reports Used As Parallelism Context** with definitions, mechanism, cost model, implementation choices, failure modes, and decision rules.

1. **Conceptual model.**
   - Model Reports Used As Parallelism Context is framed as a tensor-placement transformation over the device mesh. Inputs, outputs, and dependency edges are explicitly identified.
   - Correctness condition: transformed execution must preserve numerical equivalence up to floating-point non-associativity and stochastic layers.
2. **Mathematical formulation.**
   - Let local shard size be $n_{local}=n/p$ along the chosen axis. Memory scales as $O(n_{local})$ for sharded states and $O(n)$ for replicated states.
   - Communication lower bound (single phase): $\Omega(n\,\beta)$ payload plus startup term $\Omega(\alpha)$.
3. **Execution recipe.**
   - Precondition checks: mesh groups, dtype policy, seed policy, checkpoint layout metadata.
   - Runtime sequence: schedule compute kernels, issue collectives, place synchronization barriers only on true dependencies, and opportunistically overlap.
4. **Performance engineering.**
   - Increase arithmetic intensity using fused kernels and shape-friendly GEMMs.
   - Reduce bytes with precision controls (FP8/quantized links) while managing scales and error feedback.
   - Place high-frequency collectives on fast links (e.g., intra-node) whenever possible.
5. **Failure modes and diagnostics.**
   - Symptoms: stragglers, tail latency spikes, allocator fragmentation, imbalance, and collective stalls.
   - Diagnostics: per-rank timeline traces, bucket size sweeps, microbatch sweep, and reshard-count audits.
6. **Decision rubric.**
   - Prefer the smallest parallel axis set meeting memory and throughput targets.
   - Add axes incrementally; after each addition, re-measure MFU, link utilization, and p95/p99 latency.

**Practical takeaway.** This subsection should be implemented as a topology-aware schedule with explicit tensor-layout contracts; otherwise theoretical gains are lost to reshards and synchronization overhead.

## Verification Notes
**Engine 132: scope and objective.** Provide a self-contained treatment of **Verification Notes** with definitions, mechanism, cost model, implementation choices, failure modes, and decision rules.

1. **Conceptual model.**
   - Verification Notes is framed as a tensor-placement transformation over the device mesh. Inputs, outputs, and dependency edges are explicitly identified.
   - Correctness condition: transformed execution must preserve numerical equivalence up to floating-point non-associativity and stochastic layers.
2. **Mathematical formulation.**
   - Let local shard size be $n_{local}=n/p$ along the chosen axis. Memory scales as $O(n_{local})$ for sharded states and $O(n)$ for replicated states.
   - Communication lower bound (single phase): $\Omega(n\,\beta)$ payload plus startup term $\Omega(\alpha)$.
3. **Execution recipe.**
   - Precondition checks: mesh groups, dtype policy, seed policy, checkpoint layout metadata.
   - Runtime sequence: schedule compute kernels, issue collectives, place synchronization barriers only on true dependencies, and opportunistically overlap.
4. **Performance engineering.**
   - Increase arithmetic intensity using fused kernels and shape-friendly GEMMs.
   - Reduce bytes with precision controls (FP8/quantized links) while managing scales and error feedback.
   - Place high-frequency collectives on fast links (e.g., intra-node) whenever possible.
5. **Failure modes and diagnostics.**
   - Symptoms: stragglers, tail latency spikes, allocator fragmentation, imbalance, and collective stalls.
   - Diagnostics: per-rank timeline traces, bucket size sweeps, microbatch sweep, and reshard-count audits.
6. **Decision rubric.**
   - Prefer the smallest parallel axis set meeting memory and throughput targets.
   - Add axes incrementally; after each addition, re-measure MFU, link utilization, and p95/p99 latency.

**Practical takeaway.** This subsection should be implemented as a topology-aware schedule with explicit tensor-layout contracts; otherwise theoretical gains are lost to reshards and synchronization overhead.
