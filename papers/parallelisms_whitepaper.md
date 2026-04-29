# End-to-End Parallelisms for Large Language Models

## Abstract
This white paper explains, from first principles to production practice, how modern LLM training and serving systems compose parallelism strategies. It synthesizes the source list in `papers/parallelism_reference.md` into one coherent narrative: what each strategy does, which collectives it uses, where the memory and bandwidth go, how to reason about correctness, and how to choose trade-offs on real clusters.

---

## 1) System model and shared notation
We assume a transformer with:
- batch size $B$
- sequence length $S$
- hidden size $H$
- attention heads $N_h$, head width $d_h$ so $H=N_h d_h$
- layers $L$
- vocabulary size $V$

A global mesh is factored into axes:
$$
(P_{dp},P_{tp},P_{pp},P_{cp},P_{ep}),\qquad P=\prod_i P_i.
$$

- **DP** data-parallel group
- **TP** tensor-parallel group
- **PP** pipeline stage group
- **CP** context/sequence-parallel group
- **EP** expert-parallel group (MoE)

Collectives:
- $\operatorname{AR}$ all-reduce
- $\operatorname{AG}$ all-gather
- $\operatorname{RS}$ reduce-scatter
- $\operatorname{A2A}$ all-to-all

Communication model:
$$
T_{comm}(n,p)\approx \alpha\,f(p)+\beta\,n\,g(p),
$$
where $n$ is payload bytes, $p$ is group size, $\alpha$ captures startup/latency, $\beta$ inverse bandwidth.

Per-rank memory budget:
$$
M_{rank}=M_{params}+M_{grads}+M_{opt}+M_{acts}+M_{kv}+M_{tmp}.
$$

This single model is enough to explain all major LLM parallelisms as **layout transformations + collectives + schedule constraints**.

---

## 2) Data parallelism family: DDP, ZeRO, FSDP, HSDP

## 2.1 DDP baseline
In DDP, every rank has a full copy of model parameters and optimizer states. Each rank processes a different mini-batch shard and computes gradients locally.

- Backward pass produces per-rank gradient tensor $g^{(r)}$.
- Synchronization uses all-reduce:
$$
\bar g = \frac{1}{P_{dp}}\sum_{r=1}^{P_{dp}} g^{(r)}.
$$
- All ranks then apply the same optimizer step.

**Strengths:** simple, stable, excellent for small-to-mid models.
**Bottlenecks:** memory scales poorly because params+grads+optimizer are replicated.

## 2.2 Why ZeRO/FSDP exist
Memory replication becomes dominant at scale. ZeRO and FSDP replace replication with sharding:
- shard optimizer states
- shard gradients
- shard parameters (full sharding)

A useful identity:
$$
\operatorname{AR}(x)=\operatorname{AG}(\operatorname{RS}(x)).
$$
This is why many sharded systems use RS+AG rather than monolithic AR.

## 2.3 ZeRO stages (conceptual)
- **Stage 1:** optimizer states sharded.
- **Stage 2:** optimizer + gradients sharded.
- **Stage 3:** optimizer + gradients + parameters sharded.

At stage 3, parameter materialization is on-demand around each layer compute. The trade-off is memory reduction vs added communication frequency.

## 2.4 FSDP/FSDP2 execution pattern
For each wrapped module:
1. **Pre-forward**: all-gather parameter shards to assemble full weights.
2. Compute forward.
3. Optionally reshard after forward.
4. Backward compute.
5. Reduce-scatter gradients.

This turns “always-resident full parameters” into a **just-in-time weight residency model**.

Practical tuning levers:
- wrap granularity (too fine => overhead; too coarse => memory spikes)
- prefetch strategy
- bucket sizes
- mixed precision + accumulation strategy

## 2.5 HSDP: hierarchical sharding
HSDP combines sharding and replication across hierarchy levels (e.g., shard within node, replicate across nodes, or vice versa).

Goal: align expensive collectives with fast links (NVLink/NVSwitch) and reduce slow-link traffic (IB/Ethernet). It is topology-aware ZeRO/FSDP behavior.

## 2.6 Offload tiers (CPU/NVMe)
When GPU memory is insufficient, offload states to CPU RAM or NVMe.

Benefit:
- enables larger models / batches than VRAM allows.

Cost:
- introduces transfer latency and bandwidth constraints.
- requires strict overlap of transfer with compute to avoid throughput collapse.

Rule of thumb: offload is a **capacity unlock**, not a free speed win.

---

## 3) Tensor parallelism (TP), sequence-parallel companions, vocabulary parallelism

## 3.1 TP core idea
Split matrix multiplications inside a layer across devices.

For $Y=XW$:
- **Column parallel**: split columns of $W$ across ranks.
- **Row parallel**: split rows of $W$ across ranks.

This lowers per-rank compute and memory for very wide layers.

## 3.2 Column-parallel linear
If $W=[W_1,\dots,W_{P_{tp}}]$ by columns, each rank computes
$$
Y_i = XW_i.
$$
Output may stay sharded or be gathered depending on next op.

## 3.3 Row-parallel linear
If $W^T=[W_1^T,\dots,W_{P_{tp}}^T]$ by rows, each rank computes a partial output and results are summed (AR-like semantic) to form full output.

## 3.4 TP in MLP and attention
Typical transformer block pattern:
- MLP up-projection: column parallel
- activation/gating local
- down-projection: row parallel

Attention often uses TP for QKV projections and output projection; placement choices determine where collectives happen.

## 3.5 Sequence parallel (SP) as TP companion
SP shards activations along sequence dimension for selected ops (norm/dropout/residual), reducing activation memory pressure that TP alone doesn’t fix.

## 3.6 Vocabulary parallelism
Embedding and LM head are costly for large $V$. Partition vocabulary dimension so each rank stores a slice of embedding/output weights; softmax/logits then use distributed reduction/gather patterns.

## 3.7 Choosing TP degree
Increasing $P_{tp}$:
- helps when layers are very wide and single-GPU GEMMs are too large
- hurts if interconnect is weak or collective frequency becomes dominant

You want the point where local GEMMs stay efficient while communication does not dominate the roofline.

---

## 4) Pipeline parallelism (PP)

PP partitions layers across depth-axis stages.

With $P_{pp}$ stages and microbatch count $m$, bubble fraction is approximately:
$$
\eta_{bubble}\approx\frac{P_{pp}-1}{m+P_{pp}-1}.
$$
So higher $m$ reduces bubbles but raises activation memory.

## 4.1 Schedule anatomy
A full iteration has:
- warmup
- steady state
- drain

Classic schedules include GPipe-like and 1F1B variants. Modern schedules reduce bubbles and manage memory explicitly.

## 4.2 Correctness constraints
Because gradients are delayed across stages, update ordering and barriers matter:
- stage-consistent microbatch ordering
- synchronized optimizer step boundary
- deterministic accumulation semantics

## 4.3 PP interactions
PP composes naturally with TP and DP but introduces additional point-to-point traffic and schedule complexity.

---

## 5) Context/sequence parallelism for long context

Long-context attention is often the dominant memory and bandwidth bottleneck.

## 5.1 Ring and blockwise attention ideas
Many CP/ring methods avoid materializing full $S\times S$ attention globally by streaming KV blocks and using numerically stable online softmax.

Given running stats $(m,l)$ and block stats $(m_b,l_b)$:
$$
m' = \max(m,m_b),\qquad l' = l\,e^{m-m'} + l_b\,e^{m_b-m'}.
$$
This preserves softmax normalization while processing blocks incrementally.

## 5.2 Ulysses-style remapping
Ulysses-like approaches transform sequence sharding into head-local work via all-to-all, then reverse map. This changes communication shape and can improve compute locality depending on topology.

## 5.3 Megatron context parallelism
CP shards context across ranks for broader network coverage than classic SP. It can unlock larger sequence lengths while balancing communication and memory.

## 5.4 Dynamic CP
Adaptive CP degree by microbatch/context length can reduce wasted work on heterogeneous request lengths.

---

## 6) Expert parallelism (MoE)

MoE replaces one dense FFN with many experts and a router.

For token representation $x$, router produces top-$k$ experts $\mathcal{E}_k(x)$ and weights $w_e(x)$.
Output:
$$
y = \sum_{e\in\mathcal{E}_k(x)} w_e(x)\,\mathrm{Expert}_e(x).
$$

## 6.1 Distributed MoE flow
1. Router picks experts.
2. Tokens are permuted/packed by destination expert.
3. A2A dispatch sends token batches to owning ranks.
4. Local expert GEMMs run (often grouped GEMM).
5. A2A combine returns outputs.
6. Inverse permutation restores token order.

## 6.2 Capacity and load balancing
Capacity factor limits tokens per expert; overflow may be dropped or rerouted depending on policy. Load imbalance is the central systems challenge.

## 6.3 Performance drivers
- efficient token packing
- A2A bandwidth and overlap
- grouped/fused expert kernels
- routing stability and auxiliary balancing losses

MoE wins when sparse activation savings exceed dispatch overhead.

---

## 7) Hybrid mesh composition: how real systems are built

Production training usually combines multiple axes:
- DP for batch scaling
- TP for width scaling
- PP for depth scaling
- CP/SP for long context and activation control
- EP for sparse capacity

### 7.1 Composition heuristic
1. Fit model memory target first (choose sharding/offload).
2. Reach per-step throughput target (TP/PP tuning).
3. Add long-context strategy (SP/CP/ring).
4. Add MoE/EP if model architecture requires it.
5. Re-map axes to physical topology.

### 7.2 Topology-aware rank mapping
Keep high-frequency, low-payload-sensitive collectives on fastest links.
- intra-node: TP, many EP patterns
- inter-node: DP replicas or less frequent sync paths

### 7.3 Checkpointing and resharding
Checkpoint format must encode layout metadata to support restart on different meshes. Without robust reshard tooling, operational flexibility suffers.

---

## 8) Inference parallelism and serving systems

Training-optimal choices are not always serving-optimal.

## 8.1 Continuous batching
Scheduler merges arriving requests into rolling decode batches. Gains throughput but needs careful fairness and tail-latency control.

## 8.2 Paged KV cache
KV stored in paged blocks to reduce fragmentation and enable reuse/eviction policies for long-running services.

## 8.3 Prefix/radix caching
Shared prompt prefixes are reused across requests to amortize prefill.

## 8.4 Prefill/decode disaggregation
Separate hardware pools for prefill and decode can improve utilization because these phases have different compute/memory profiles.

## 8.5 Speculative / multi-token decoding
Use draft proposals and verification to reduce average latency per accepted token.

## 8.6 Serving-time TP/PP/EP/CP
Serving parallelism must optimize p50 and p99, not just raw throughput. KV movement and scheduler behavior often dominate.

---

## 9) Cross-cutting kernel/runtime techniques

These techniques amplify all parallel axes:
- CUDA graphs reduce launch overhead and jitter.
- Fused kernels raise arithmetic intensity.
- Low precision (FP8 and related) reduces bandwidth and memory, if scale handling is robust.
- Advanced attention kernels reduce IO and improve long-context efficiency.
- Overlap orchestration (streams/events) determines realized vs theoretical gains.

A practical model:
$$
T_{step}=\max(T_{critical\_compute},T_{critical\_comm})+T_{runtime\_overheads}.
$$
Most engineering work reduces the critical path and overhead terms.

---

## 10) Failure modes, diagnostics, and validation workflow

Common failure modes:
- collective stalls from stragglers
- MoE hot experts
- PP bubbles larger than expected
- activation memory spikes from poor wrap/schedule
- long-tail latency in serving due to batching policy

Diagnostics checklist:
1. per-rank timeline trace
2. link utilization and collective time breakdown
3. microbatch/bucket sweep
4. precision sweep with convergence checks
5. rank-map A/B test (topology sensitivity)

Validation principles:
- verify numerical parity against simpler baseline (small scale)
- scale gradually and re-measure convergence + throughput + stability
- distinguish algorithmic improvements from kernel/runtime artifacts

---

## 11) End-to-end decision framework

Given model + hardware + objective (train speed, cost, serving latency), choose as follows:

1. **Memory fit**
   - If model fits comfortably: DDP baseline.
   - If not: FSDP/ZeRO, then offload if still necessary.

2. **Compute scaling**
   - If layers too wide/slow per device: add TP.
   - If depth too large for single rank efficiency: add PP.

3. **Context scaling**
   - For long contexts: add SP/CP/ring depending on attention strategy and interconnect.

4. **Architecture scaling**
   - For MoE models: add EP and prioritize A2A efficiency + load balance.

5. **Topology optimization**
   - Map axes to physical hierarchy, then retune bucket sizes, microbatch count, overlap.

6. **Serving-specific rewrite**
   - Re-optimize around KV cache, batching, and latency SLOs.

---

## 12) What “good” looks like in production

A strong production parallel stack has:
- explicit layout contracts per layer
- minimal unnecessary reshards
- stable overlap schedule
- predictable checkpoint/resume semantics
- robust observability on collectives, memory, and latency percentiles
- tunable policies for workload heterogeneity

In practice, the best system is not one axis; it is a **co-designed hybrid** where algorithms, collectives, kernels, scheduler, and topology are optimized together.

---

## 13) Source mapping
This paper is derived from the sectioned source map in `papers/parallelism_reference.md`, including PyTorch distributed/FSDP docs, Megatron-Core parallelism guidance, DeepSpeed ZeRO/ZeRO++ material, NCCL collective definitions, context/ring/Ulysses references, MoE/EP sources, and serving/runtime references listed there.
