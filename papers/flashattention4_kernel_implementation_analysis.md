# FlashAttention-4 Kernel Implementation: Forward First, Backward Second

Author: Codex research note  
Date: 2026-05-04  

Primary artifacts inspected:

- Paper: [FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling](https://arxiv.org/abs/2603.05451), arXiv v1, submitted 2026-03-05.
- Open-source implementation: [Dao-AILab/flash-attention CuTe FA4 code](https://github.com/Dao-AILab/flash-attention/tree/cb213fce11c3baf9168f7fa607bc7f22e3323554/flash_attn/cute), inspected at commit `cb213fce11c3baf9168f7fa607bc7f22e3323554`.
- NVIDIA context: [CuTe DSL tcgen05 API](https://docs.nvidia.com/cutlass/4.4.1/media/docs/pythonDSL/cute_dsl_api/cute_nvgpu_tcgen05.html), [CUTLASS Blackwell docs](https://docs.nvidia.com/cutlass/4.3.2/media/docs/cpp/blackwell.html), and [CUDA Programming Guide: Distributed Shared Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#distributed-shared-memory).

## How To Read This

This version is deliberately structured as two passes:

1. Forward pass.
2. Backward pass.

Each pass uses the same four-step format:

1. Math algorithm.
2. Scheduler and pipeline pseudocode.
3. Specialized warp pseudocode.
4. Data placement and synchronization summary.

Location tags:

- `[GMEM]`: global memory.
- `[SMEM]`: shared memory inside the CTA.
- `[TMEM]`: Blackwell tensor memory, used for tensor-core accumulators and producer/consumer handoff.
- `[RMEM]`: per-thread registers.
- `[DSMEM]`: distributed shared memory, meaning another CTA's shared memory in the same cluster.

CUDA feature tags:

- `[TMA]`: Tensor Memory Accelerator bulk tensor load/store.
- `[cp.async]`: asynchronous copy path used for fallback or bulk copies.
- `[tcgen05.mma]`: Blackwell fifth-generation tensor-core MMA.
- `[TMEM copy]`: `tcgen05.copy.Ld32x32bOp` or `tcgen05.copy.St32x32bOp`.
- `[mbarrier]`: memory barrier used by TMA, TMEM split arrival, or DSMEM transfer.
- `[NamedBarrier]`: CTA named barrier.
- `[Pipeline*]`: CUTLASS/CuTe pipeline object.

Shape symbols:

| Symbol | Meaning |
|---|---|
| `BM` | Query block height, 128 in the main SM100 paths. |
| `BN` | Key/value block height, 128 in the main SM100 paths. |
| `D` | Rounded Q/K head dimension. |
| `DV` | Rounded V/O head dimension. |
| `QS` | Number of forward Q slots (`q_slot`), usually 2. |
| `CG` | CTA group size, 1 or 2. In 2-CTA mode one MMA spans a CTA pair. |

Source map with evidence grades:

| Topic | Source | Evidence |
|---|---|---|
| FA4 package and entrypoint | [FA4 CuTe README](https://github.com/Dao-AILab/flash-attention/blob/cb213fce11c3baf9168f7fa607bc7f22e3323554/flash_attn/cute/README.md#L1-L33) | official repo |
| Forward setup, tiles, warp roles, TMEM offsets | [forward constructor](https://github.com/Dao-AILab/flash-attention/blob/cb213fce11c3baf9168f7fa607bc7f22e3323554/flash_attn/cute/flash_fwd_sm100.py#L126-L313) | official repo |
| Forward pipelines | [forward pipeline construction](https://github.com/Dao-AILab/flash-attention/blob/cb213fce11c3baf9168f7fa607bc7f22e3323554/flash_attn/cute/flash_fwd_sm100.py#L892-L990) | official repo |
| Forward load/MMA/softmax/correction | [load](https://github.com/Dao-AILab/flash-attention/blob/cb213fce11c3baf9168f7fa607bc7f22e3323554/flash_attn/cute/flash_fwd_sm100.py#L1328-L1510), [MMA](https://github.com/Dao-AILab/flash-attention/blob/cb213fce11c3baf9168f7fa607bc7f22e3323554/flash_attn/cute/flash_fwd_sm100.py#L1535-L1789), [softmax](https://github.com/Dao-AILab/flash-attention/blob/cb213fce11c3baf9168f7fa607bc7f22e3323554/flash_attn/cute/flash_fwd_sm100.py#L2187-L2321), [correction](https://github.com/Dao-AILab/flash-attention/blob/cb213fce11c3baf9168f7fa607bc7f22e3323554/flash_attn/cute/flash_fwd_sm100.py#L2360-L2528) | official repo |
| Backward setup, tilers, warp roles, TMEM offsets | [backward constructor](https://github.com/Dao-AILab/flash-attention/blob/cb213fce11c3baf9168f7fa607bc7f22e3323554/flash_attn/cute/flash_bwd_sm100.py#L81-L233) | official repo |
| Backward pipelines | [backward pipeline construction](https://github.com/Dao-AILab/flash-attention/blob/cb213fce11c3baf9168f7fa607bc7f22e3323554/flash_attn/cute/flash_bwd_sm100.py#L1078-L1281) | official repo |
| Backward MMA/compute/reduce | [MMA](https://github.com/Dao-AILab/flash-attention/blob/cb213fce11c3baf9168f7fa607bc7f22e3323554/flash_attn/cute/flash_bwd_sm100.py#L2194-L2695), [compute](https://github.com/Dao-AILab/flash-attention/blob/cb213fce11c3baf9168f7fa607bc7f22e3323554/flash_attn/cute/flash_bwd_sm100.py#L2810-L3290), [dQ reduce](https://github.com/Dao-AILab/flash-attention/blob/cb213fce11c3baf9168f7fa607bc7f22e3323554/flash_attn/cute/flash_bwd_sm100.py#L3416-L3654) | official repo |
| Pipeline wrappers and named barriers | [pipeline helpers](https://github.com/Dao-AILab/flash-attention/blob/cb213fce11c3baf9168f7fa607bc7f22e3323554/flash_attn/cute/pipeline.py#L38-L402) | official repo |
| Inline PTX for Blackwell MMA and async copy/reduce | [Blackwell helpers](https://github.com/Dao-AILab/flash-attention/blob/cb213fce11c3baf9168f7fa607bc7f22e3323554/flash_attn/cute/blackwell_helpers.py#L395-L611), [copy utilities](https://github.com/Dao-AILab/flash-attention/blob/cb213fce11c3baf9168f7fa607bc7f22e3323554/flash_attn/cute/copy_utils.py#L147-L288) | official repo |

## Part I: Forward Pass

### 1. Forward Math Algorithm

For a query tile `Q_i` and a key/value tile `(K_j, V_j)`:

```text
Q_i: [BM, D]
K_j: [BN, D]
V_j: [BN, DV]
S_ij = Q_i K_j^T
```

The forward pass uses online softmax. For each query row, it keeps:

```text
m = running max over all score tiles seen so far
l = running sum of exp-normalized scores
O = running unnormalized output accumulator
```

FA4 uses base-2 exponentials:

```text
scale_log2 = softmax_scale * log2(e)
```

For each KV tile:

```text
m_new = max(m_old, rowmax(S_ij))
alpha = exp2((m_old - m_new) * scale_log2)
P_ij  = exp2(S_ij * scale_log2 - m_new * scale_log2)

l_new = alpha * l_old + rowsum(P_ij)
O_new = alpha * O_old + P_ij V_j
```

Final output:

```text
O_out = O_final / l_final
```

The implementation does not literally compute this in one thread group. It splits the equation:

- QK produces `S_ij`.
- Softmax produces `m_new`, `alpha`, `P_ij`, and `l_new`.
- Correction rescales old `O` by `alpha`.
- PV computes `P_ij V_j` and accumulates into `O`.
- Epilogue divides by `l_final` and stores the output.

Intermediate PV does not notify `pipeline_o_acc`. The implementation relies on
the fixed MMA order `PV_i` before `QK_{i+1}` and on correction waiting for the
softmax stats handoff of tile `i+1`. That handoff is downstream of `QK_{i+1}`, so it
also implies the previous `O_i` accumulator has already been produced. The final
PV has no next softmax stats handoff to serve as this proxy, so the MMA warp does
commit `pipeline_o_acc` for `final_O_ready`.

The source states both sides explicitly: intermediate `O_full` notification is
commented out in the main loop, while tail PV does call
`pipeline_o_acc.producer_commit_w_index(...)` ([intermediate PV comment](https://github.com/Dao-AILab/flash-attention/blob/cb213fce11c3baf9168f7fa607bc7f22e3323554/flash_attn/cute/flash_fwd_sm100.py#L1714-L1719), [tail PV commit](https://github.com/Dao-AILab/flash-attention/blob/cb213fce11c3baf9168f7fa607bc7f22e3323554/flash_attn/cute/flash_fwd_sm100.py#L1781-L1785)).

Forward evidence: `SoftmaxSm100.update_row_max` implements `m_new` and `alpha` ([softmax code](https://github.com/Dao-AILab/flash-attention/blob/cb213fce11c3baf9168f7fa607bc7f22e3323554/flash_attn/cute/softmax.py#L227-L245)); `softmax_step` writes `P` to TMEM and signals MMA ([forward softmax step](https://github.com/Dao-AILab/flash-attention/blob/cb213fce11c3baf9168f7fa607bc7f22e3323554/flash_attn/cute/flash_fwd_sm100.py#L2242-L2319)).

### 2. Forward Scheduler And Pipeline Pseudocode

The forward kernel is organized around a CTA work tile:

```text
work_tile = (m_block, head, batch, split)
Q slots: q_slot in {0, 1}, QS = 2
Q_s[q_slot]: [BM, D] [SMEM]
K_s:       [BN, D] [SMEM]
V_s:       [BN, DV] [SMEM]
S_t, P_t:  [BM, BN] [TMEM]
O_t:       [BM, DV] [TMEM]
```

Forward warp specialization:

| Warps | Role |
|---|---|
| 0-3 | Softmax warpgroup for Q slot 0 |
| 4-7 | Softmax warpgroup for Q slot 1 |
| 8-11 | Correction warpgroup |
| 12 | MMA warp |
| 13 | Epilogue warp |
| 14 | Load warp |
| 15 | Empty, scheduler, or utility warp |

Role assignment is explicit in the forward constructor ([forward roles](https://github.com/Dao-AILab/flash-attention/blob/cb213fce11c3baf9168f7fa607bc7f22e3323554/flash_attn/cute/flash_fwd_sm100.py#L237-L256)).

Forward communication channels:

| Channel | CUDA feature | Meaning |
|---|---|---|
| `pipeline_q` | `[PipelineTmaUmma]` or `[PipelineAsyncUmma]` | Load warp says Q is ready in `[SMEM]`; MMA waits. |
| `pipeline_kv` | `[PipelineTmaUmma]` or `[PipelineAsyncUmma]` | Load warp says K/V is ready in `[SMEM]`; MMA waits. |
| `pipeline_s_p_o` | `[PipelineUmmaAsync]` | MMA says `S_t` is ready; later softmax/correction says `P_t` and rescaled `O_t` are ready. |
| `pipeline_p_lastsplit` | `[PipelineAsyncUmma]` plus `[mbarrier]` | Softmax says the second split of `P_t` is ready. |
| `pipeline_o_acc` | `[PipelineUmmaAsync]` | MMA says final `O_t` accumulator is ready. |
| `sm_stats_barrier` | `[NamedBarrier]` | Softmax says row `alpha`, row sum, and row max are ready for correction. |
| `pipeline_o_epi` | `[PipelineAsync]` | Correction says final output is staged in `[SMEM]`. |

The unusual bidirectional use of `pipeline_s_p_o` is documented in the source comments ([forward pipeline construction](https://github.com/Dao-AILab/flash-attention/blob/cb213fce11c3baf9168f7fa607bc7f22e3323554/flash_attn/cute/flash_fwd_sm100.py#L930-L940)).

Pipeline handoff notation used below:

```python
producer_acquire(pipeline_X, slot=slot_id, phase=producer_phase,
                 payload=tensor_or_buffer,
                 from=previous_consumer)
                                             # wait for empty-side release, then reserve slot
issue_async_copy(..., tma_bar_ptr=barrier)  # [TMA] completion will signal the full side
producer_ready(pipeline_X, slot=slot_id,
               payload=tensor_or_buffer,
               to=next_consumer)
                                             # copy/barrier makes payload visible to consumer
producer_commit(pipeline_X, slot=slot_id,
                payload=tensor_or_buffer,
                to=next_consumer)
                                             # producer makes payload visible on full side
consumer_wait(pipeline_X, slot=slot_id, phase=consumer_phase,
              payload=tensor_or_buffer,
              from=producer)
                                             # consumer waits on full side before reading
consumer_release(pipeline_X, slot=slot_id,
                 payload=tensor_or_buffer,
                 to=producer_or_next_reader)
                                             # empty-side release, or FA4 reverse signal
```

The important fields are `from=` and `to=`. They make the partner warp explicit:
`producer_commit(..., to=softmax_warpgroup)` should be matched by
`consumer_wait(..., from=mma_warp)`, and
`consumer_release(..., to=load_warp)` should be matched by a later
`producer_acquire(..., from=mma_warp)` on the producer side. The `payload=`
field names the storage protected by that pipeline edge.

The real CuTe `phase` is still present underneath this notation, but it is not
the payload name. `PipelineState(index, phase)` uses `index` as the physical
slot and `phase` as a generation/parity value so that reusing the same slot
cannot be confused with an older use of that slot. The phase comes from the
producer or consumer pipeline state:

```python
state.index = state.phase_index % num_stages    # physical slot
state.phase = state.phase_index // num_stages   # generation/parity
state.advance()                                 # move to next slot/generation
```

In the longer pseudocode, `phase=` is omitted when it would simply mean "the
current producer or consumer pipeline-state phase for this slot"; the
`payload/from/to` fields are the visual dependency trace.

Most ready handoffs use the full side of a pipeline:

```python
producer_commit(..., payload=S_t[q_slot], to=softmax_warpgroup)
consumer_wait(..., payload=S_t[q_slot], from=mma_warp)
```

Some FA4 handoffs deliberately use the empty side as a reverse signal. The
important forward example is `pipeline_s_p_o`: MMA commits `S_t` to
softmax on the full side; later softmax/correction release `P_t` plus
rescaled `O_t` back to MMA on the empty side; MMA acquires that empty-side
handoff before issuing `P @ V`.

Slot naming convention used below:

| Name | Meaning |
|---|---|
| `q_block`, `n_block` | Logical Q or KV block coordinate in the sequence. |
| `q_slot` | Physical Q/S/P/O circular-buffer slot, usually `0` or `1` in forward. |
| `k_slot`, `v_slot` | Physical `pipeline_kv` slots carrying K and V tiles. |
| `do_slot`, `stats_slot` | Backward slots carrying `dO` and row statistics. |
| `dV_slot`, `dK_slot` | Backward epilogue slots carrying final `dV_t` and `dK_t`. |
| `payload` | Tensor or buffer protected by the handoff, such as `S_t[q_slot]` or `Q_s`. |
| `phase` | Hardware/CuTe generation or parity for reusing a slot. |

Pipeline lifecycle audit used by the pseudocode:

| Pipeline | Full-side producer-to-consumer edge | Empty-side consumer-to-producer edge | Audit note |
|---|---|---|---|
| `pipeline_q` | Load warp sends `Q_s[q_slot]` to MMA. | MMA releases `Q_s[q_slot]` back to load warp. | Normal load pipeline. |
| `pipeline_kv` | Load warp sends `K_s[k_slot]` or `V_s[v_slot]` to MMA. | MMA releases K/V slots back to load warp. | K and V occupy separate logical slots. |
| `pipeline_s_p_o` | MMA sends `S_t[q_slot]` to softmax. | Softmax/correction send `P_t[q_slot]` plus rescaled `O_t[q_slot]` back to MMA. | Bidirectional pipeline. First S write can omit acquire because the slot is initially empty; later reuse is protected by the empty-side acquire. |
| `pipeline_p_lastsplit` | Softmax sends the last split of `P_t[q_slot]` to MMA. | PV MMA consumes it through the mbarrier passed to `tcgen05.mma`; the pseudocode shows this as `consumer_wait(..., from=softmax_warpgroup)`. | Optional split-P refinement, not the primary storage lifetime. |
| `pipeline_o_acc` | MMA sends final `O_t[q_slot]` to correction. | No loop acquire/release; final O readiness is a one-way full-side signal. | Intermediate O uses the next softmax-stats handoff as ordering proxy; final O needs this explicit signal. |
| `pipeline_o_epi` | Correction sends `O_s[q_slot]` to epilogue. | Epilogue releases `O_s[q_slot]` back to correction. | Normal output staging pipeline. |
| `pipeline_Q`, `pipeline_dO` | Backward load sends `Q_s` or `dO_s` to MMA. | MMA releases Q/dO slots back to load. | K and V may piggyback on the first Q/dO transaction. |
| `pipeline_LSE`, `pipeline_dPsum` | Backward load sends row statistics to compute. | Compute releases statistic slots back to load. | Row-stat load pipelines. |
| `pipeline_S_P`, `pipeline_dP`, `pipeline_dS`, `pipeline_dQ`, `pipeline_dKV` | Producer commits the ready accumulator or intermediate; consumer waits before reading. | Consumer releases once the TMEM/SMEM payload is no longer needed. | Some SM100 paths use raw `sync_object_empty/full` calls, but the same full/empty lifecycle is represented here. |

Barrier handoff notation used below:

```python
named_barrier_arrive(B, index)              # [NamedBarrier] cute.arch.barrier_arrive
named_barrier_sync(B, index)                # [NamedBarrier] cute.arch.barrier arrive+wait
mbarrier_arrive(B, phase_or_bytes)          # [mbarrier] arrive, often with expected bytes
mbarrier_arrive_expect_tx(B, bytes)         # [mbarrier] arrive and set async byte count
mbarrier_wait(B, phase)                     # [mbarrier] wait until phase is complete
fence_tmem_load/store()                     # [fence_view_async_tmem_load/store]
fence_shared()                              # [fence_view_async_shared]
```

Tensor-core notation used below:

```python
# [Tensor Core MMA: tcgen05.mma]
```

Only operations with this tag issue Blackwell tensor-core MMA. `tmem_load`,
`tmem_store`, `online_softmax`, `compute_P_from_S_and_LSE`, and `compute_dS`
are not tensor-core MMA operations.

Scheduler and pipeline pseudocode:

```python
def forward_cta_program():
    # All roles below run concurrently inside one CTA.
    # Work tiles are produced by TileScheduler:
    #   StaticPersistentTileScheduler, SingleTileLPTScheduler,
    #   varlen scheduler, or CLC-aware scheduler depending feature path.

    for work_tile in TileScheduler:
        # work_tile: [registers] (m_block, head, batch, split)
        # num_q_slots = QS, usually 2.

        # Load-warp schedule for this work tile.
        # n-blocks are visited from n_block_max - 1 down to n_block_min.
        # This is the order used in the load loop.
        load_warp_pipeline:
            k_slot0 = slot_for_K(n_block_max - 1)
            producer_acquire(pipeline_kv, slot=k_slot0,
                             payload=sK[k_slot0], from=mma_warp)
            K0_s = load_K(n_block_max - 1)      # [GMEM] -> [SMEM], [TMA/cp.async]
            producer_ready(pipeline_kv, slot=k_slot0,
                           payload=sK[k_slot0], to=mma_warp)
                                                 # [TMA mbarrier] or [cp.async mbarrier]

            for q_slot in [0, 1]:
                producer_acquire(pipeline_q, slot=q_slot,
                                 payload=Q_s[q_slot], from=mma_warp)
                Q_s[q_slot] = load_Q(q_slot)    # [GMEM] -> [SMEM], [TMA/cp.async]
                producer_ready(pipeline_q, slot=q_slot,
                               payload=Q_s[q_slot], to=mma_warp)
                                                 # Q_s[q_slot] ready for MMA

            v_slot0 = slot_for_V(n_block_max - 1)
            producer_acquire(pipeline_kv, slot=v_slot0,
                             payload=sV[v_slot0], from=mma_warp)
            V0_s = load_V(n_block_max - 1)      # [GMEM] -> [SMEM], [TMA/cp.async]
            producer_ready(pipeline_kv, slot=v_slot0,
                           payload=sV[v_slot0], to=mma_warp)
                                                 # V0_s ready for PV

            for n_block in reversed(remaining_n_blocks):
                k_slot = slot_for_K(n_block)
                producer_acquire(pipeline_kv, slot=k_slot,
                                 payload=sK[k_slot], from=mma_warp)
                K_s = load_K(n_block)           # [GMEM] -> [SMEM], [TMA/cp.async]
                producer_ready(pipeline_kv, slot=k_slot,
                               payload=sK[k_slot], to=mma_warp)

                v_slot = slot_for_V(n_block)
                producer_acquire(pipeline_kv, slot=v_slot,
                                 payload=sV[v_slot], from=mma_warp)
                V_s = load_V(n_block)           # [GMEM] -> [SMEM], [TMA/cp.async]
                producer_ready(pipeline_kv, slot=v_slot,
                               payload=sV[v_slot], to=mma_warp)

        # MMA/softmax/correction form a software pipeline:
        #
        #   QK(K_i) -> S_i
        #   softmax(S_i) -> P_i + alpha_i
        #   correction(alpha_i) rescales O_{i-1}
        #   PV(P_i, V_i) -> O_i
        #
        # FA4 overlaps PV for tile i with QK for tile i+1 whenever dependencies allow.

        mma_pipeline:
            for q_slot in [0, 1]:
                consumer_wait(pipeline_q, slot=q_slot,
                              payload=Q_s[q_slot], from=load_warp)
                                                 # Q_s[q_slot] ready in [SMEM]
            consumer_wait(pipeline_kv, slot=k_slot0,
                          payload=sK[k_slot0], from=load_warp)
                                                 # K0_s ready in [SMEM]
            K0_s = sK[k_slot0]
            for q_slot in [0, 1]:
                S_t[q_slot] = QK_MMA(Q_s[q_slot], K0_s)
                                                   # [Tensor Core MMA: tcgen05.mma]
                producer_commit(pipeline_s_p_o, slot=q_slot,
                                payload=S_t[q_slot], to=softmax_warpgroup)
                                                   # S_t[q_slot] ready in [TMEM]
            consumer_release(pipeline_kv, slot=k_slot0,
                             payload=sK[k_slot0], to=load_warp)
                                                 # K0_s no longer needed by QK

            for prev_n_block, next_n_block in consecutive_kv_block_pairs:
                v_prev_slot = slot_for_V(prev_n_block)
                consumer_wait(pipeline_kv, slot=v_prev_slot,
                              payload=sV[v_prev_slot], from=load_warp)
                                                   # V_prev_s ready in [SMEM]
                V_prev_s = sV[v_prev_slot]
                k_next_slot = slot_for_K(next_n_block)
                consumer_wait(pipeline_kv, slot=k_next_slot,
                              payload=sK[k_next_slot], from=load_warp)
                                                   # K_next_s ready in [SMEM]
                K_next_s = sK[k_next_slot]
                for q_slot in [0, 1]:
                    producer_acquire(pipeline_s_p_o, slot=q_slot,
                                     payload=P_t[q_slot] + O_t[q_slot],
                                     from=softmax_and_correction_warps)
                                                   # P_t ready and old O_t rescaled
                    if split_P_arrive:
                        consumer_wait(pipeline_p_lastsplit, slot=q_slot,
                                      payload=P_t[q_slot].last_split,
                                      from=softmax_warpgroup)
                                                   # [mbarrier] last P split gates PV MMA
                    O_t[q_slot] = PV_MMA(P_t[q_slot], V_prev_s)
                                                   # [Tensor Core MMA: tcgen05.mma]
                    # No pipeline_o_acc commit for intermediate O_t.
                    # Correction waits on softmax stats for the next tile; the
                    # MMA schedule issues PV_i before QK_{i+1}, so that stats
                    # handoff implies O_i is already available for correction.
                    # While softmax/correction work on this tile, issue next QK:
                    S_t[q_slot] = QK_MMA(Q_s[q_slot], K_next_s)
                                                   # [Tensor Core MMA: tcgen05.mma]
                    producer_commit(pipeline_s_p_o, slot=q_slot,
                                    payload=S_t[q_slot], to=softmax_warpgroup)
                                                   # next S_t ready for softmax
                consumer_release(pipeline_kv, slot=v_prev_slot,
                                 payload=sV[v_prev_slot], to=load_warp)
                                                   # V_prev_s consumed by PV
                consumer_release(pipeline_kv, slot=k_next_slot,
                                 payload=sK[k_next_slot], to=load_warp)
                                                   # K_next_s consumed by QK

            for q_slot in [0, 1]:
                consumer_release(pipeline_q, slot=q_slot,
                                 payload=Q_s[q_slot], to=load_warp)
                                                 # Q_s[q_slot] no longer needed by QK

            # Tail PV for final V tile.
            v_last_slot = slot_for_V(last_n_block)
            consumer_wait(pipeline_kv, slot=v_last_slot,
                          payload=sV[v_last_slot], from=load_warp)
            V_last_s = sV[v_last_slot]
            for q_slot in [0, 1]:
                producer_acquire(pipeline_s_p_o, slot=q_slot,
                                 payload=P_t[q_slot] + O_t[q_slot],
                                 from=softmax_and_correction_warps)
                if split_P_arrive:
                    consumer_wait(pipeline_p_lastsplit, slot=q_slot,
                                  payload=P_t[q_slot].last_split,
                                  from=softmax_warpgroup)
                                                   # [mbarrier] last P split gates PV MMA
                O_t[q_slot] = PV_MMA(P_t[q_slot], V_last_s)
                                                   # [Tensor Core MMA: tcgen05.mma]
                producer_commit(pipeline_o_acc, slot=q_slot,
                                payload=O_t[q_slot], to=correction_warpgroup)
                                                   # final O_t ready for correction
            consumer_release(pipeline_kv, slot=v_last_slot,
                             payload=sV[v_last_slot], to=load_warp)
                                                 # final V_s consumed by tail PV

        softmax_pipeline_for_each_q_slot:
            consumer_wait(pipeline_s_p_o, slot=q_slot,
                          payload=S_t[q_slot], from=mma_warp)
                                                   # S_t ready from MMA
            S_r = load_S_from_TMEM(q_slot)      # [TMEM] -> [RMEM], [TMEM copy]
            P_r, alpha, row_sum, row_max = online_softmax(S_r)
            store_alpha_to_sScale(alpha)        # [RMEM] -> [SMEM]
            named_barrier_arrive(
                sm_stats_barrier,
                index=q_slot * 4 + warp_idx,
            )                                   # [NamedBarrier] stats ready
            store_P_to_TMEM(P_r, q_slot)        # [RMEM] -> [TMEM], [TMEM copy]
            if split_P_arrive:
                consumer_release(pipeline_s_p_o, slot=q_slot,
                                 payload=P_t[q_slot].first_split, to=mma_warp)
                                                   # first P_t split ready for PV
                producer_commit(pipeline_p_lastsplit, slot=q_slot,
                                payload=P_t[q_slot].last_split, to=mma_warp)
                                                   # second P_t split ready
            else:
                consumer_release(pipeline_s_p_o, slot=q_slot,
                                 payload=P_t[q_slot], to=mma_warp)
                                                   # whole P_t ready for PV

        correction_pipeline:
            # First iteration has no old O to rescale, so it initially releases O-ready.
            for q_slot in [0, 1]:
                consumer_release(pipeline_s_p_o, slot=q_slot,
                                 payload=O_t[q_slot], to=mma_warp)
            for each softmax_stats_handoff:
                q_slot = softmax_stats_handoff.q_slot
                named_barrier_sync(
                    sm_stats_barrier,
                    index=q_slot * 4 + warp_idx,
                )                               # [NamedBarrier] alpha/row stats ready
                alpha = read_sScale(q_slot)     # [SMEM] -> [RMEM]
                if any_lane(alpha < 1.0):       # [vote_ballot_sync]
                    O_t[q_slot] *= alpha        # [TMEM] -> [RMEM] -> [TMEM]
                consumer_release(pipeline_s_p_o, slot=q_slot,
                                 payload=O_t[q_slot], to=mma_warp)
                                                   # old O_t safe for next PV MMA
            for q_slot in [0, 1]:
                consumer_wait(pipeline_o_acc, slot=q_slot,
                              payload=O_t[q_slot], from=mma_warp)
                                                       # final O_t ready in [TMEM]
                producer_acquire(pipeline_o_epi, slot=q_slot,
                                 payload=O_s[q_slot], from=epilogue_warp)
                                                       # reserve O_s slot before writing [SMEM]
                O_s[q_slot] = normalize_and_stage(O_t[q_slot], l[q_slot])
                                                       # [TMEM] -> [RMEM] -> [SMEM]
                consumer_release(pipeline_s_p_o, slot=q_slot,
                                 payload=O_t[q_slot], to=mma_warp)
                                                       # final O_t was read; slot reusable
                producer_commit(pipeline_o_epi, slot=q_slot,
                                payload=O_s[q_slot], to=epilogue_warp)
                                                       # O_s ready in [SMEM]

        epilogue_pipeline:
            for q_slot in [0, 1]:
                consumer_wait(pipeline_o_epi, slot=q_slot,
                              payload=O_s[q_slot], from=correction_warpgroup)
                                                     # O_s ready in [SMEM]
                store_O(O_s[q_slot], O_g)            # [SMEM] -> [GMEM], [TMA] or vector store
                consumer_release(pipeline_o_epi, slot=q_slot,
                                 payload=O_s[q_slot], to=correction_warpgroup)
```

Implementation anchors:

- Tile scheduler selection appears in [`flash_fwd_sm100.py`](https://github.com/Dao-AILab/flash-attention/blob/cb213fce11c3baf9168f7fa607bc7f22e3323554/flash_attn/cute/flash_fwd_sm100.py#L224-L233).
- Load order `K0, Q0, Q1, V0, K_i, V_i` appears in [`flash_fwd_sm100.py`](https://github.com/Dao-AILab/flash-attention/blob/cb213fce11c3baf9168f7fa607bc7f22e3323554/flash_attn/cute/flash_fwd_sm100.py#L1443-L1483).
- MMA loop overlaps PV and next QK in [`flash_fwd_sm100.py`](https://github.com/Dao-AILab/flash-attention/blob/cb213fce11c3baf9168f7fa607bc7f22e3323554/flash_attn/cute/flash_fwd_sm100.py#L1651-L1789).

### 3. Forward Specialized Warp Algorithms

#### 3.1 Load Warp

```python
def forward_load_warp():
    # warp 14, sometimes more warps on fallback paths.

    if elected_thread:
        prefetch_tma_descriptors(Q, K, V, O)       # [cpasync.prefetch_descriptor]

    for work_tile in TileScheduler:
        # Build tiled views. These are pointer/layout calculations in [RMEM].
        gQ = tile(Q_g, [QS * BM, D], work_tile)     # [GMEM view]
        gK = tile(K_g, [BN, D], kv_block)           # [GMEM view]
        gV = tile(V_g, [BN, DV], kv_block)          # [GMEM view]

        # Q load.
        for q_slot in [0, 1]:
            gQ_slot = slice_Q_slot(gQ, q_slot)       # [GMEM view], [BM, D]
            producer_acquire(pipeline_q, slot=q_slot,
                             payload=Q_s[q_slot], from=mma_warp) # [PipelineTmaUmma]
            copy_gmem_to_smem(gQ_slot, Q_s[q_slot])   # [TMA cute.copy(... tma_bar_ptr)]
            producer_ready(pipeline_q, slot=q_slot,
                           payload=Q_s[q_slot], to=mma_warp)
                                                     # Q_s[q_slot] ready

        # K/V load.
        k_slot = slot_for_K(kv_block)                # [RMEM] physical pipeline index
        producer_acquire(pipeline_kv, slot=k_slot,
                         payload=sK[k_slot], from=mma_warp) # [PipelineTmaUmma]
        copy_gmem_to_smem(gK, sK[k_slot])            # [TMA] or [cp.async]
        producer_ready(pipeline_kv, slot=k_slot,
                       payload=sK[k_slot], to=mma_warp)
                                                     # K_s ready for MMA consumer
        v_slot = slot_for_V(kv_block)                # [RMEM] physical pipeline index
        producer_acquire(pipeline_kv, slot=v_slot,
                         payload=sV[v_slot], from=mma_warp)
        copy_gmem_to_smem(gV, sV[v_slot])            # [TMA] or [cp.async]
        producer_ready(pipeline_kv, slot=v_slot,
                       payload=sV[v_slot], to=mma_warp)
                                                     # V_s ready for MMA consumer

    producer_tail(pipeline_kv)
```

Exact primitive examples:

- `cpasync.prefetch_descriptor` is called at kernel start ([forward descriptor prefetch](https://github.com/Dao-AILab/flash-attention/blob/cb213fce11c3baf9168f7fa607bc7f22e3323554/flash_attn/cute/flash_fwd_sm100.py#L825-L831)).
- TMA copy atoms are built with `cpasync.CopyBulkTensorTileG2SOp` and `cute.nvgpu.make_tiled_tma_atom_A/B` ([forward TMA setup](https://github.com/Dao-AILab/flash-attention/blob/cb213fce11c3baf9168f7fa607bc7f22e3323554/flash_attn/cute/flash_fwd_sm100.py#L560-L603)).

#### 3.2 MMA Warp

```python
def forward_mma_warp():
    # warp 12.

    tmem_ptr = allocate_tmem()                       # [TmemAllocator], [NamedBarrier]

    for work_tile in TileScheduler:
        first_kv_block = first_kv_block_for(work_tile) # [RMEM] logical KV block
        k_slot = slot_for_K(first_kv_block)          # [RMEM] physical pipeline index
        for q_slot in [0, 1]:
            consumer_wait(pipeline_q, slot=q_slot,
                          payload=Q_s[q_slot], from=load_warp)
                                                     # Q_s[q_slot] ready in [SMEM]
            consumer_wait(pipeline_kv, slot=k_slot,
                          payload=sK[k_slot], from=load_warp)
                                                     # K_s ready in [SMEM]
            K_s = sK[k_slot]
            S_t[q_slot] = tcgen05_mma_smem_smem(
                Q_s[q_slot], K_s
            )                                        # [tcgen05.mma], output [TMEM]
                                                     # [Tensor Core MMA: tcgen05.mma]
            producer_commit(pipeline_s_p_o, slot=q_slot,
                            payload=S_t[q_slot], to=softmax_warpgroup)
                                                     # S_t ready for softmax
        consumer_release(pipeline_kv, slot=k_slot,
                         payload=sK[k_slot], to=load_warp)
                                                     # K_s no longer needed by QK

        for kv_block in remaining_kv_blocks:
            v_slot = slot_for_V(kv_block)            # [RMEM] physical pipeline index
            consumer_wait(pipeline_kv, slot=v_slot,
                          payload=sV[v_slot], from=load_warp)
                                                     # V_s ready in [SMEM]
            V_s = sV[v_slot]
            next_kv_block = next_kv_block_after(kv_block)
            next_k_slot = slot_for_K(next_kv_block)
            consumer_wait(pipeline_kv, slot=next_k_slot,
                          payload=sK[next_k_slot], from=load_warp)
            K_next_s = sK[next_k_slot]
            for q_slot in [0, 1]:
                producer_acquire(pipeline_s_p_o, slot=q_slot,
                                 payload=P_t[q_slot] + O_t[q_slot],
                                 from=softmax_and_correction_warps)
                                                     # P_t and rescaled O_t ready
                if split_P_arrive:
                    consumer_wait(pipeline_p_lastsplit, slot=q_slot,
                                  payload=P_t[q_slot].last_split,
                                  from=softmax_warpgroup)
                                                     # [mbarrier] last P split gates PV MMA
                O_t[q_slot] = tcgen05_mma_tmem_smem(
                    P_t[q_slot], V_s
                )                                    # [tcgen05.mma], P [TMEM], V [SMEM]
                                                     # [Tensor Core MMA: tcgen05.mma]
                # No pipeline_o_acc commit for intermediate O_t.
                # The correction warpgroup is synchronized by softmax stats;
                # that stats handoff is downstream of the next QK, which is
                # scheduled after this PV.

                S_t[q_slot] = tcgen05_mma_smem_smem(Q_s[q_slot], K_next_s)
                                                     # [Tensor Core MMA: tcgen05.mma]
                producer_commit(pipeline_s_p_o, slot=q_slot,
                                payload=S_t[q_slot], to=softmax_warpgroup)
                                                     # next S_t ready for softmax
            consumer_release(pipeline_kv, slot=v_slot,
                             payload=sV[v_slot], to=load_warp)
                                                     # V_s consumed by PV
            consumer_release(pipeline_kv, slot=next_k_slot,
                             payload=sK[next_k_slot], to=load_warp)
                                                     # K_next_s consumed by QK

        for q_slot in [0, 1]:
            consumer_release(pipeline_q, slot=q_slot,
                             payload=Q_s[q_slot], to=load_warp)
                                                     # Q_s no longer needed by QK

        v_last_slot = slot_for_V(last_kv_block)
        consumer_wait(pipeline_kv, slot=v_last_slot,
                      payload=sV[v_last_slot], from=load_warp)
        V_last_s = sV[v_last_slot]
        for q_slot in [0, 1]:
            producer_acquire(pipeline_s_p_o, slot=q_slot,
                             payload=P_t[q_slot] + O_t[q_slot],
                             from=softmax_and_correction_warps)
                                                     # final P_t and rescaled O_t ready
            if split_P_arrive:
                consumer_wait(pipeline_p_lastsplit, slot=q_slot,
                              payload=P_t[q_slot].last_split,
                              from=softmax_warpgroup)
                                                     # [mbarrier] last P split gates PV MMA
            O_t[q_slot] = tcgen05_mma_tmem_smem(
                P_t[q_slot], V_last_s
            )                                        # [tcgen05.mma], P [TMEM], V [SMEM]
                                                     # [Tensor Core MMA: tcgen05.mma]
            producer_commit(pipeline_o_acc, slot=q_slot,
                            payload=O_t[q_slot], to=correction_warpgroup)
                                                     # final O_t ready for correction
        consumer_release(pipeline_kv, slot=v_last_slot,
                         payload=sV[v_last_slot], to=load_warp)
                                                     # final V_s consumed by tail PV
```

Exact primitive:

```ptx
tcgen05.mma.cta_group::{1 or 2}.kind::{kind}
```

The source emits the SMEM x SMEM and TMEM x SMEM variants in [`blackwell_helpers.py`](https://github.com/Dao-AILab/flash-attention/blob/cb213fce11c3baf9168f7fa607bc7f22e3323554/flash_attn/cute/blackwell_helpers.py#L470-L611). Forward binds QK and PV helpers in [`flash_fwd_sm100.py`](https://github.com/Dao-AILab/flash-attention/blob/cb213fce11c3baf9168f7fa607bc7f22e3323554/flash_attn/cute/flash_fwd_sm100.py#L1535-L1609).

#### 3.3 Softmax Warpgroups

```python
def forward_softmax_warpgroup(q_slot):
    # warps 0-3 for q_slot 0, warps 4-7 for q_slot 1.
    # Each active thread owns row-local fragments.

    for work_tile in TileScheduler:
        softmax_state.m = -inf                       # [RMEM]
        softmax_state.l = 0.0                        # [RMEM]

        for kv_block in scheduled_kv_blocks:
            consumer_wait(pipeline_s_p_o, slot=q_slot,
                          payload=S_t[q_slot], from=mma_warp)
                                                     # S_t ready in [TMEM]

            S_r = tmem_load(S_t[q_slot])             # [TMEM] -> [RMEM], [Ld32x32bOp]
            S_r = apply_mask(S_r)                    # [RMEM]

            m_new = max(softmax_state.m, rowmax(S_r))
            alpha = exp2((softmax_state.m - m_new) * scale_log2)
            sScale[row, q_slot] = alpha              # [RMEM] -> [SMEM]
            named_barrier_arrive(
                sm_stats_barrier,
                index=q_slot * 4 + warp_idx,
            )                                        # [NamedBarrier] alpha ready

            P_r = exp2(S_r * scale_log2 - m_new * scale_log2)
            P_r = convert_to_q_dtype(P_r)            # [RMEM]

            # P_t overlaps S_t; S must have been read first.
            tmem_store(P_t[q_slot], P_r)             # [RMEM] -> [TMEM], [St32x32bOp]
            fence_tmem_store()                       # [fence_view_async_tmem_store]

            if reached_split_P_boundary:
                consumer_release(pipeline_s_p_o, slot=q_slot,
                                 payload=P_t[q_slot].first_split, to=mma_warp)
                                                     # first P_t split ready for PV

            producer_commit(pipeline_p_lastsplit, slot=q_slot,
                            payload=P_t[q_slot].last_split, to=mma_warp)
                                                     # [elect_one] last P_t split ready
            # If there is no split-P path, consumer_release(..., payload=P_t[q_slot],
            # to=mma_warp) is the single P handoff instead of pipeline_p_lastsplit.

            softmax_state.l = softmax_state.l * alpha + rowsum(P_r)
            softmax_state.m = m_new
```

Softmax TMEM load/store atoms are built with `tcgen05.copy.Ld32x32bOp` and `tcgen05.copy.St32x32bOp` ([forward TMEM copy atoms](https://github.com/Dao-AILab/flash-attention/blob/cb213fce11c3baf9168f7fa607bc7f22e3323554/flash_attn/cute/flash_fwd_sm100.py#L1887-L1907)). The split-P release is in [`softmax_step`](https://github.com/Dao-AILab/flash-attention/blob/cb213fce11c3baf9168f7fa607bc7f22e3323554/flash_attn/cute/flash_fwd_sm100.py#L2302-L2317).

#### 3.4 Correction Warpgroup

```python
def forward_correction_warpgroup():
    # warps 8-11.

    # First tile has no old output to rescale.
    for q_slot in [0, 1]:
        consumer_release(pipeline_s_p_o, slot=q_slot,
                         payload=O_t[q_slot], to=mma_warp)
                                                     # MMA may issue first PV

    for work_tile in TileScheduler:
        for kv_block_after_first in remaining_kv_blocks:
            for q_slot in [0, 1]:
                named_barrier_sync(
                    sm_stats_barrier,
                    index=q_slot * 4 + warp_idx,
                )                                    # [NamedBarrier] alpha ready
                alpha = sScale[row, q_slot]          # [SMEM] -> [RMEM]

                if vote_ballot_sync(alpha < 1.0):
                    O_r = tmem_load(O_t[q_slot])     # [TMEM] -> [RMEM]
                    O_r = O_r * alpha                # [mul_packed_f32x2]
                    tmem_store(O_t[q_slot], O_r)     # [RMEM] -> [TMEM]
                    fence_tmem_store()               # [fence_view_async_tmem_store]

                consumer_release(pipeline_s_p_o, slot=q_slot,
                                 payload=O_t[q_slot], to=mma_warp)
                                                     # old O_t safe for PV consumer

        for q_slot in [0, 1]:
            consumer_wait(pipeline_o_acc, slot=q_slot,
                          payload=O_t[q_slot], from=mma_warp)
                                                     # final O_t ready in [TMEM]
            O_r = tmem_load(O_t[q_slot])             # [TMEM] -> [RMEM]
            O_r = O_r * rcp_approx(l_final)          # [rcp_approx], [mul_packed_f32x2]
            producer_acquire(pipeline_o_epi, slot=q_slot,
                             payload=O_s[q_slot], from=epilogue_warp)
                                                     # reserve O_s slot before writing [SMEM]
            O_s[q_slot] = convert_store_smem(O_r)    # [RMEM] -> [SMEM]
            fence_shared()                           # [fence_view_async_shared]
            consumer_release(pipeline_s_p_o, slot=q_slot,
                             payload=O_t[q_slot], to=mma_warp)
                                                     # final O_t was read; slot reusable
            producer_commit(pipeline_o_epi, slot=q_slot,
                            payload=O_s[q_slot], to=epilogue_warp)
                                                     # O_s ready for epilogue
```

The rescale routine uses TMEM load/store, packed multiply, and TMEM store fences ([forward correction rescale](https://github.com/Dao-AILab/flash-attention/blob/cb213fce11c3baf9168f7fa607bc7f22e3323554/flash_attn/cute/flash_fwd_sm100.py#L2630-L2678)). The final normalization and SMEM staging are in [`correction_epilogue`](https://github.com/Dao-AILab/flash-attention/blob/cb213fce11c3baf9168f7fa607bc7f22e3323554/flash_attn/cute/flash_fwd_sm100.py#L2680-L2767).

#### 3.5 Epilogue Warp

```python
def forward_epilogue_warp():
    # warp 13 unless correction warps own epilogue on some varlen paths.

    for work_tile in TileScheduler:
        for q_slot in [0, 1]:
            consumer_wait(pipeline_o_epi, slot=q_slot,
                          payload=O_s[q_slot], from=correction_warpgroup)
                                                     # O_s ready in [SMEM]
            store_O_to_global(O_s[q_slot], O_g)      # [SMEM] -> [GMEM], [TMA] or vector copy
            consumer_release(pipeline_o_epi, slot=q_slot,
                             payload=O_s[q_slot], to=correction_warpgroup)
                                                     # SMEM q_slot reusable
```

TMA store uses `cp_async_bulk_commit_group` and `cp_async_bulk_wait_group` when TMA output is enabled ([forward epilogue store](https://github.com/Dao-AILab/flash-attention/blob/cb213fce11c3baf9168f7fa607bc7f22e3323554/flash_attn/cute/flash_fwd_sm100.py#L2818-L2883)).

### 4. Forward Data And Synchronization Summary

Forward placement:

| Data | Shape | Main location | Notes |
|---|---:|---|---|
| `Q_s` | `[QS, BM, D]` | `[SMEM]` | Loaded from `[GMEM]` by TMA or `cp.async`. |
| `K_s`, `V_s` | `[BN, D]`, `[BN, DV]` | `[SMEM]` | Multi-slot buffering through `k_slot` and `v_slot`. |
| `S_t` | `[BM, BN]` per `q_slot` | `[TMEM]` | QK output. |
| `P_t` | `[BM, BN]` per `q_slot` | `[TMEM]` | Overlaps `S_t`; softmax output. |
| `m`, `l`, `alpha` | scalar per row | `[RMEM]` | `alpha` also copied to `[SMEM] sScale`. |
| `O_t` | `[BM, DV]` per `q_slot` | `[TMEM]` | PV accumulator and correction target. |
| `O_s` | `[BM, DV]` per `q_slot` | `[SMEM]` | Final staging before `[GMEM]` store. |

Forward synchronization:

| Dependency | CUDA feature |
|---|---|
| `[GMEM] -> [SMEM]` load completion | `[TMA]` transaction mbarrier through `PipelineTmaUmma`, or `[cp.async]` mbarrier |
| QK `S_t` ready for softmax | `pipeline_s_p_o.producer_commit_w_index` |
| `S_t` read before `P_t` overwrites it | `pipeline_s_p_o.consumer_wait`, then `[TMEM copy]` ordering |
| `P_t` ready for PV | `pipeline_s_p_o.consumer_release_w_index` and optionally `pipeline_p_lastsplit` |
| intermediate `O_t` ready for correction | No direct `pipeline_o_acc` commit; correction uses the later `sm_stats_barrier` handoff as a proxy because MMA orders `PV_i` before `QK_{i+1}`. |
| row scale ready for correction | `[NamedBarrier] sm_stats_barrier` |
| old `O_t` rescaled before PV | `pipeline_s_p_o.consumer_release_w_index` |
| final `O_t` ready | `pipeline_o_acc` |
| final `O_s` ready for store | `pipeline_o_epi` |

Forward barrier events:

| Barrier handoff | Producer side | Consumer side |
|---|---|---|
| TMEM pointer available | MMA warp allocates TMEM, then `tmem_alloc_barrier.arrive_and_wait()` | Softmax/correction/epilogue warps call `tmem_alloc_barrier.arrive()` before retrieving/using the pointer. |
| Q/K/V TMA load complete | Load warp issues `cute.copy(..., tma_bar_ptr=pipeline.producer_get_barrier(...))` | MMA warp waits through `pipeline_q.consumer_wait...` or `pipeline_kv.consumer_wait...`. |
| Softmax row stats ready | Softmax calls `sm_stats_barrier.arrive_w_index(index=q_slot * 4 + warp_idx)` | Correction calls `sm_stats_barrier.arrive_and_wait_w_index(index=q_slot * 4 + warp_idx)`. |
| Optional q-slot ordering | One softmax q-slot uses `pipeline_s0_s1_sequence.sync_object_full.arrive(...)` | The other q-slot uses `pipeline_s0_s1_sequence.sync_object_full.wait(...)`. |
| `O_s` visible before store | Correction calls `fence_view_async_shared()` and commits `pipeline_o_epi` | Epilogue waits `pipeline_o_epi`; overlap paths also use `cute.arch.barrier(NamedBarrierFwdSm100.Epilogue)`. |

## Part II: Backward Pass

### 1. Backward Math Algorithm

Backward starts with a preprocess step:

```text
dPsum_i[row] = sum_d O_i[row, d] * dO_i[row, d]
LSE_log2_i[row] = LSE_i[row] * log2(e)
```

The main backward kernel works naturally by KV tile. Let:

```text
K_j:  [BN, D]
V_j:  [BN, DV]
Q_i:  [BM, D]
dO_i: [BM, DV]
```

FA4 often stores score-like tiles transposed in the backward kernel:

```text
S_t = K_j Q_i^T          # [BN, BM]
P_t = exp2(S_t * scale_log2 - LSE_log2_i)   # [BN, BM]
dP_t = V_j dO_i^T        # [BN, BM]
dS_t = P_t * (dP_t - dPsum_i)               # [BN, BM]
```

Then the gradients are:

```text
dV_j += P_t dO_i         # [BN, DV]
dK_j += dS_t Q_i         # [BN, D]
dQ_i += dS_t^T K_j       # [BM, D]
```

The code's tiler comments list the same five MMA products: `S = K @ Q.T`, `dP = V @ dO.T`, `dV = P.T @ dO`, `dK = dS.T @ Q`, and `dQ = dS @ K` with orientation-specific layouts ([backward tilers](https://github.com/Dao-AILab/flash-attention/blob/cb213fce11c3baf9168f7fa607bc7f22e3323554/flash_attn/cute/flash_bwd_sm100.py#L95-L107)). The preprocess call computes `dPsum` and `LSE_log2` before the main kernel ([backward preprocess call](https://github.com/Dao-AILab/flash-attention/blob/cb213fce11c3baf9168f7fa607bc7f22e3323554/flash_attn/cute/interface.py#L1532-L1535)).

### 2. Backward Scheduler And Pipeline Pseudocode

Backward warp specialization:

| Warps | Role |
|---|---|
| 0-3 | dQ reduce warps |
| 4-11 | compute warps |
| 12 | MMA warp |
| 13 | load warp |
| 14 | relay warp for 2-CTA DSMEM synchronization |
| 15 | empty or utility warp |

Roles are defined in the backward constructor ([backward roles](https://github.com/Dao-AILab/flash-attention/blob/cb213fce11c3baf9168f7fa607bc7f22e3323554/flash_attn/cute/flash_bwd_sm100.py#L139-L156)).

Backward communication channels:

| Channel | CUDA feature | Meaning |
|---|---|---|
| `pipeline_Q`, `pipeline_Qt`, `pipeline_Kt` | `[PipelineTmaUmma]` | Load warp says Q/K transpose views are ready in `[SMEM]`. |
| `pipeline_dO` | `[PipelineTmaUmma]` | Load warp says dO is ready in `[SMEM]`. |
| `pipeline_LSE` | `[PipelineTmaAsync]` | Load warp says `LSE_log2` is ready in `[SMEM]`. |
| `pipeline_dPsum` | `[PipelineTmaAsync]` | Load warp says `dPsum` is ready in `[SMEM]`. |
| `pipeline_S_P` | `[PipelineUmmaAsync]` | MMA says `S_t` is ready; compute later releases after `P_t` is stored. |
| `pipeline_dP` | `[PipelineUmmaAsync]` | MMA says `dP_t` is ready; compute later releases after `dS_t` is stored. |
| `pipeline_dS` | `[PipelineAsyncUmma]` | Compute says `dS_t/dS_s` is ready for MMA. |
| `pipeline_dQ` | `[PipelineUmmaAsync]` | MMA says `dQ_t` is ready for reduce warps. |
| `pipeline_dKV` | `[PipelineUmmaAsync]` | MMA says `dK_t` or `dV_t` accumulator is ready for epilogue. |
| `compute_sync_barrier` | `[NamedBarrier]` | Compute warps make sure TMEM aliasing is safe before overwriting `S_t` with `P_t` or `dP_t` with `dS_t`. |
| `reduce_sync_barrier` | `[NamedBarrier]` | Reduce warps make shared-memory `dQ` staging visible before/after async global accumulation. |
| `dS_cluster_*_mbar_ptr` | `[mbarrier]` plus `[DSMEM]` | CTA pair synchronizes `dS` exchange in 2-CTA mode. |

These pipelines are constructed in [`flash_bwd_sm100.py`](https://github.com/Dao-AILab/flash-attention/blob/cb213fce11c3baf9168f7fa607bc7f22e3323554/flash_attn/cute/flash_bwd_sm100.py#L1151-L1281).

The backward pseudocode uses the same handoff vocabulary as forward: a load or MMA warp calls `producer_acquire`, issues the copy/MMA, then makes the item ready with a TMA barrier completion or `producer_commit`; the consuming warp calls `consumer_wait` before reading and `consumer_release` once the storage can be reused.

Backward scheduler and pipeline pseudocode:

```python
def backward_cta_program():
    # Work tile is keyed by a KV block:
    #   work_tile = (n_block, head, batch, split)
    # For each KV tile, stream relevant Q blocks.

    for kv_work_tile in TileScheduler:
        dV_slot = slot_for_dKV("dV", kv_work_tile)  # [RMEM] physical pipeline index
        dK_slot = slot_for_dKV("dK", kv_work_tile)  # [RMEM] physical pipeline index
        # K/V are often bundled with the first Q or dO load in the same
        # pipeline transaction through extra TMA byte counts.
        producer_acquire(pipeline_Q, slot=first_q_slot,
                         payload=K_s + Q_s, from=mma_warp)
        K_s = load_K(kv_work_tile)                # [GMEM] -> [SMEM], [TMA]
        Q_s = load_Q(first_q_block)               # [GMEM] -> [SMEM], [TMA]
        producer_commit(pipeline_Q, slot=first_q_slot,
                        payload=K_s + Q_s, to=mma_warp)
                                                   # Q_s ready; K_s piggybacks in same transaction

        producer_acquire(pipeline_dO, slot=first_do_slot,
                         payload=V_s + dO_s, from=mma_warp)
        V_s = load_V(kv_work_tile)                # [GMEM] -> [SMEM], [TMA]
        dO_s = load_dO(first_q_block)             # [GMEM] -> [SMEM], [TMA]
        producer_commit(pipeline_dO, slot=first_do_slot,
                        payload=V_s + dO_s, to=mma_warp)
                                                   # dO_s ready; V_s piggybacks in same transaction

        # dK_t and dV_t are long-lived accumulators for this KV tile.
        dK_t = zero_tmem([BN, D])                 # [TMEM]
        dV_t = zero_tmem([BN, DV])                # [TMEM]

        for q_block in scheduled_q_blocks(kv_work_tile):
            # Load warp streams operands and row statistics.
            q_slot = slot_for_Q(q_block)          # [RMEM] physical pipeline index
            do_slot = slot_for_dO(q_block)        # [RMEM] physical pipeline index
            stats_slot = slot_for_stats(q_block)  # [RMEM] physical pipeline index

            producer_acquire(pipeline_Q, slot=q_slot,
                             payload=Q_s, from=mma_warp)
            Q_s = load_Q(q_block)                 # [GMEM] -> [SMEM], [TMA]
            producer_commit(pipeline_Q, slot=q_slot,
                            payload=Q_s, to=mma_warp)
                                                   # Q_s ready for S MMA

            producer_acquire(pipeline_dO, slot=do_slot,
                             payload=dO_s, from=mma_warp)
            dO_s = load_dO(q_block)               # [GMEM] -> [SMEM], [TMA]
            producer_commit(pipeline_dO, slot=do_slot,
                            payload=dO_s, to=mma_warp)
                                                   # dO_s ready for dP/dV MMA

            producer_acquire(pipeline_LSE, slot=stats_slot,
                             payload=LSE_log2_s, from=compute_warps)
            LSE_log2_s = load_LSE_log2(q_block)   # [GMEM] -> [SMEM], [TMA async]
            producer_ready(pipeline_LSE, slot=stats_slot,
                           payload=LSE_log2_s, to=compute_warps)

            producer_acquire(pipeline_dPsum, slot=stats_slot,
                             payload=dPsum_s, from=compute_warps)
            dPsum_s = load_dPsum(q_block)         # [GMEM] -> [SMEM], [TMA async]
            producer_ready(pipeline_dPsum, slot=stats_slot,
                           payload=dPsum_s, to=compute_warps)

            # MMA warp produces S and dP.
            consumer_wait(pipeline_Q, slot=q_slot,
                          payload=Q_s, from=load_warp)
                                                   # Q_s and K_s ready in [SMEM]
            S_t = MMA(K_s, Q_s.T)                 # [SMEM] x [SMEM] -> [TMEM]
                                                   # [Tensor Core MMA: tcgen05.mma]
            producer_commit(pipeline_S_P, slot=q_slot,
                            payload=S_t, to=compute_warps)
                                                   # S_t ready for compute

            consumer_wait(pipeline_dO, slot=do_slot,
                          payload=dO_s, from=load_warp)
                                                   # dO_s and V_s ready in [SMEM]
            dP_t = MMA(V_s, dO_s.T)               # [SMEM] x [SMEM] -> [TMEM]
                                                   # [Tensor Core MMA: tcgen05.mma]
            producer_commit(pipeline_dP, slot=q_slot,
                            payload=dP_t, to=compute_warps)
                                                   # dP_t ready for compute

            # Compute warps derive P and dS.
            consumer_wait(pipeline_S_P, slot=q_slot,
                          payload=S_t, from=mma_warp)
            consumer_wait(pipeline_LSE, slot=stats_slot,
                          payload=LSE_log2_s, from=load_warp)
            P_t = compute_P_from_S_and_LSE()      # [TMEM] -> [RMEM] -> [TMEM]
            fence_tmem_load()                     # all S_t loads visible before P_t stores
            named_barrier_sync(compute_sync_barrier)
                                                   # [NamedBarrier Compute] no warp still reads S_t
            consumer_release(pipeline_LSE, slot=stats_slot,
                             payload=LSE_log2_s, to=load_warp)
                                                   # LSE_log2_s no longer needed
            consumer_release(pipeline_S_P, slot=q_slot,
                             payload=P_t, to=mma_warp)
                                                   # P_t ready for dV MMA

            consumer_wait(pipeline_dP, slot=q_slot,
                          payload=dP_t, from=mma_warp)
            consumer_wait(pipeline_dPsum, slot=stats_slot,
                          payload=dPsum_s, from=load_warp)
            dS_t, dS_s = compute_dS()             # [TMEM] -> [RMEM] -> [TMEM]/[SMEM]
            fence_tmem_load()                     # all dP_t loads visible before dS_t stores
            fence_shared()                        # dS_s store visible before DSMEM/MMA handoff
            named_barrier_sync(compute_sync_barrier)
                                                   # [NamedBarrier Compute] dS storage consistent
            consumer_release(pipeline_dP, slot=q_slot,
                             payload=dP_t, to=mma_warp)
                                                   # dP_t no longer needed
            consumer_release(pipeline_dPsum, slot=stats_slot,
                             payload=dPsum_s, to=load_warp)
                                                   # dPsum_s no longer needed
            if use_2cta:
                mbarrier_arrive_expect_tx(
                    dS_cluster_full_mbar_ptr,
                    bytes=dS_exchange_bytes,
                )                                # [mbarrier] peer copy will complete this phase
                exchange_dS_with_peer_cta()       # [DSMEM], [cp.async.bulk.shared::cluster]
            producer_commit(pipeline_dS, slot=q_slot,
                            payload=dS_t + dS_s, to=mma_warp)
                                                   # dS_t/dS_s ready for MMA

            # MMA warp consumes P and dS.
            consumer_wait(pipeline_S_P, slot=q_slot,
                          payload=P_t, from=compute_warps)
            dV_t += MMA(P_t, dO_s)                # [TMEM] x [SMEM] -> [TMEM]
                                                   # [Tensor Core MMA: tcgen05.mma]
            consumer_release(pipeline_dO, slot=do_slot,
                             payload=dO_s, to=load_warp)
                                                   # dO_s no longer needed

            consumer_wait(pipeline_dS, slot=q_slot,
                          payload=dS_t + dS_s, from=compute_warps)
            dK_t += MMA(dS_t, Q_s)                # [TMEM] x [SMEM] -> [TMEM]
                                                   # [Tensor Core MMA: tcgen05.mma]
            consumer_release(pipeline_Q, slot=q_slot,
                             payload=Q_s, to=load_warp)
                                                   # Q_s no longer needed

            if use_2cta:
                mbarrier_wait(
                    dS_cluster_leader_mbar_ptr,
                    phase=dS_cluster_phase,
                )                                # peer dS visible for dQ MMA
            dQ_t = MMA(dS_s, K_s)                 # [SMEM] x [SMEM] -> [TMEM]
                                                   # [Tensor Core MMA: tcgen05.mma]
            producer_commit(pipeline_dQ, slot=q_slot,
                            payload=dQ_t, to=reduce_warps)
                                                   # dQ_t ready for reduce warps
            consumer_release(pipeline_dS, slot=q_slot,
                             payload=dS_t + dS_s, to=compute_warps)
                                                   # dS_t/dS_s no longer needed

            # Reduce warps accumulate dQ globally.
            consumer_wait(pipeline_dQ, slot=q_slot,
                          payload=dQ_t, from=mma_warp)
            reduce_dQ_to_global(dQ_t)             # [TMEM] -> [RMEM] -> [SMEM] -> [GMEM]
            consumer_release(pipeline_dQ, slot=q_slot,
                             payload=dQ_t, to=mma_warp)

        # Compute/epilogue warps store dV and dK.
        for acc_slot, acc_t in [(dV_slot, dV_t), (dK_slot, dK_t)]:
            producer_commit(pipeline_dKV, slot=acc_slot,
                            payload=acc_t, to=compute_epilogue_warps)
                                                   # acc_t ready for epilogue
            consumer_wait(pipeline_dKV, slot=acc_slot,
                          payload=acc_t, from=mma_warp)
            store_dKV(acc_t)                      # [TMEM] -> [RMEM] -> [SMEM] -> [GMEM]
            consumer_release(pipeline_dKV, slot=acc_slot,
                             payload=acc_t, to=mma_warp)
```

The 2-CTA head-dim-192 code comments show the exact five-step order in the main loop ([2-CTA hdim192 schedule](https://github.com/Dao-AILab/flash-attention/blob/cb213fce11c3baf9168f7fa607bc7f22e3323554/flash_attn/cute/flash_bwd_sm100.py#L2376-L2452)). The generic 2-CTA path overlaps current `dK/dQ` with next `S/dP/dV` using prologue/main/tail structure ([generic 2-CTA schedule](https://github.com/Dao-AILab/flash-attention/blob/cb213fce11c3baf9168f7fa607bc7f22e3323554/flash_attn/cute/flash_bwd_sm100.py#L2453-L2576)).

### 3. Backward Specialized Warp Algorithms

#### 3.1 Load Warp

```python
def backward_load_warp():
    # warp 13.

    if elected_thread:
        prefetch_tma_descriptors(Q, K, V, dO, dK, dV)
                                                    # [cpasync.prefetch_descriptor]

    for kv_work_tile in TileScheduler:
        # K is normally made ready through pipeline_Q, V through pipeline_dO.
        producer_acquire(pipeline_Q, slot=first_q_slot,
                         payload=K_s + Q_s, from=mma_warp)
        load_K_for_kv_block()                      # [GMEM] -> [SMEM], [TMA]
        load_Q(first_q_block)                      # [GMEM] -> [SMEM], [TMA]
        producer_commit(pipeline_Q, slot=first_q_slot,
                        payload=K_s + Q_s, to=mma_warp)
                                                    # first Q_s ready; K_s piggybacks

        producer_acquire(pipeline_dO, slot=first_do_slot,
                         payload=V_s + dO_s, from=mma_warp)
        load_V_for_kv_block()                      # [GMEM] -> [SMEM], [TMA]
        load_dO(first_q_block)                     # [GMEM] -> [SMEM], [TMA]
        producer_commit(pipeline_dO, slot=first_do_slot,
                        payload=V_s + dO_s, to=mma_warp)
                                                    # first dO_s ready; V_s piggybacks

        for q_block in scheduled_q_blocks:
            q_slot = slot_for_Q(q_block)            # [RMEM] physical pipeline index
            do_slot = slot_for_dO(q_block)          # [RMEM] physical pipeline index
            stats_slot = slot_for_stats(q_block)    # [RMEM] physical pipeline index

            producer_acquire(pipeline_Q, slot=q_slot,
                             payload=Q_s, from=mma_warp)
            load_Q(q_block)                         # [GMEM] -> [SMEM], [TMA]
            producer_commit(pipeline_Q, slot=q_slot,
                            payload=Q_s, to=mma_warp)
                                                    # Q_s ready for MMA

            producer_acquire(pipeline_dO, slot=do_slot,
                             payload=dO_s, from=mma_warp)
            load_dO(q_block)                        # [GMEM] -> [SMEM], [TMA]
            producer_commit(pipeline_dO, slot=do_slot,
                            payload=dO_s, to=mma_warp)
                                                    # dO_s ready for MMA

            producer_acquire(pipeline_LSE, slot=stats_slot,
                             payload=LSE_log2_s, from=compute_warps)
            load_LSE_log2(q_block)                  # [GMEM] -> [SMEM], [PipelineTmaAsync]
            producer_ready(pipeline_LSE, slot=stats_slot,
                           payload=LSE_log2_s, to=compute_warps)
                                                    # LSE_log2_s ready for compute

            producer_acquire(pipeline_dPsum, slot=stats_slot,
                             payload=dPsum_s, from=compute_warps)
            load_dPsum(q_block)                     # [GMEM] -> [SMEM], [PipelineTmaAsync]
            producer_ready(pipeline_dPsum, slot=stats_slot,
                           payload=dPsum_s, to=compute_warps)
                                                    # dPsum_s ready for compute
```

Backward descriptor prefetch and pipeline setup are in [`flash_bwd_sm100.py`](https://github.com/Dao-AILab/flash-attention/blob/cb213fce11c3baf9168f7fa607bc7f22e3323554/flash_attn/cute/flash_bwd_sm100.py#L1078-L1281). Load sequencing for LSE, dPsum, Q, dO, Qt, Kt, and dOt is in [`flash_bwd_sm100.py`](https://github.com/Dao-AILab/flash-attention/blob/cb213fce11c3baf9168f7fa607bc7f22e3323554/flash_attn/cute/flash_bwd_sm100.py#L1951-L2174).

#### 3.2 MMA Warp

```python
def backward_mma_warp():
    # warp 12.

    allocate_tmem()                                  # [TmemAllocator]

    for kv_work_tile in TileScheduler:
        dV_slot = slot_for_dKV("dV", kv_work_tile)   # [RMEM] physical pipeline index
        dK_slot = slot_for_dKV("dK", kv_work_tile)   # [RMEM] physical pipeline index
        dK_t = zero_tmem([BN, D])                    # [TMEM]
        dV_t = zero_tmem([BN, DV])                   # [TMEM]

        for q_block in scheduled_q_blocks:
            q_slot = slot_for_Q(q_block)             # [RMEM] physical pipeline index
            do_slot = slot_for_dO(q_block)           # [RMEM] physical pipeline index
            consumer_wait(pipeline_Q, slot=q_slot,
                          payload=Q_s, from=load_warp)
                                                     # Q_s and K_s ready [SMEM]
            S_t = tcgen05_mma_smem_smem(K_s, Q_s.T)  # [TMEM]
                                                     # [Tensor Core MMA: tcgen05.mma]
            producer_commit(pipeline_S_P, slot=q_slot,
                            payload=S_t, to=compute_warps)
                                                     # S_t ready for compute

            consumer_wait(pipeline_dO, slot=do_slot,
                          payload=dO_s, from=load_warp)
                                                     # dO_s and V_s ready [SMEM]
            dP_t = tcgen05_mma_smem_smem(V_s, dO_s.T)# [TMEM]
                                                     # [Tensor Core MMA: tcgen05.mma]
            producer_commit(pipeline_dP, slot=q_slot,
                            payload=dP_t, to=compute_warps)
                                                     # dP_t ready for compute

            consumer_wait(pipeline_dS, slot=q_slot,
                          payload=dS_t + dS_s, from=compute_warps)
                                                     # dS_t/dS_s ready
            dK_t += tcgen05_mma_tmem_smem(dS_t, Q_s) # [TMEM] x [SMEM] -> [TMEM]
                                                     # [Tensor Core MMA: tcgen05.mma]
            consumer_release(pipeline_Q, slot=q_slot,
                             payload=Q_s, to=load_warp)
                                                     # Q_s no longer needed

            consumer_wait(pipeline_S_P, slot=q_slot,
                          payload=P_t, from=compute_warps)
                                                     # P_t ready after compute release
            dV_t += tcgen05_mma_tmem_smem(P_t, dO_s) # [TMEM] x [SMEM] -> [TMEM]
                                                     # [Tensor Core MMA: tcgen05.mma]
            consumer_release(pipeline_dO, slot=do_slot,
                             payload=dO_s, to=load_warp)
                                                     # dO_s no longer needed

            if use_2cta:
                mbarrier_wait(dS_cluster_leader_mbar)
                                                     # [mbarrier], peer dS arrived
            dQ_t = tcgen05_mma_smem_smem(dS_s, K_s)  # [SMEM] x [SMEM] -> [TMEM]
                                                     # [Tensor Core MMA: tcgen05.mma]
            producer_commit(pipeline_dQ, slot=q_slot,
                            payload=dQ_t, to=reduce_warps)
                                                     # dQ_t ready for reduce warps
            consumer_release(pipeline_dS, slot=q_slot,
                             payload=dS_t + dS_s, to=compute_warps)
                                                     # dS_t/dS_s no longer needed

        producer_commit(pipeline_dKV, slot=dV_slot,
                        payload=dV_t, to=compute_epilogue_warps)
                                                     # dV_t ready for epilogue
        producer_commit(pipeline_dKV, slot=dK_slot,
                        payload=dK_t, to=compute_epilogue_warps)
                                                     # dK_t ready for epilogue
```

The backward MMA function binds the five products to SMEM or TMEM operands in [`flash_bwd_sm100.py`](https://github.com/Dao-AILab/flash-attention/blob/cb213fce11c3baf9168f7fa607bc7f22e3323554/flash_attn/cute/flash_bwd_sm100.py#L2239-L2323).

#### 3.3 Compute Warps

```python
def backward_compute_warps():
    # warps 4-11.

    for kv_work_tile in TileScheduler:
        dV_slot = slot_for_dKV("dV", kv_work_tile)   # [RMEM] physical pipeline index
        dK_slot = slot_for_dKV("dK", kv_work_tile)   # [RMEM] physical pipeline index
        for q_block in scheduled_q_blocks:
            q_slot = slot_for_Q(q_block)             # [RMEM] physical pipeline index
            stats_slot = slot_for_stats(q_block)     # [RMEM] physical pipeline index
            consumer_wait(pipeline_LSE, slot=stats_slot,
                          payload=LSE_log2_s, from=load_warp)
                                                     # LSE_log2_s ready [SMEM]
            consumer_wait(pipeline_S_P, slot=q_slot,
                          payload=S_t, from=mma_warp)
                                                     # S_t ready [TMEM]

            S_r = tmem_load(S_t)                     # [TMEM] -> [RMEM], [Ld32x32bOp]
            S_r = apply_mask_and_score_mod(S_r)      # [RMEM]
            LSE_r = load_from_smem(LSE_log2_s)        # [SMEM] -> [RMEM]
            P_r = exp2(S_r * scale_log2 - LSE_r)     # [RMEM], [fma_packed_f32x2], [exp2]

            # S_t and P_t overlap. All S loads must finish before P stores.
            fence_tmem_load()                        # [fence_view_async_tmem_load]
            named_barrier_sync(compute_sync_barrier) # [NamedBarrier Compute]
            P_t = tmem_store(convert(P_r))           # [RMEM] -> [TMEM], [St32x32bOp]
            fence_tmem_store()                       # [fence_view_async_tmem_store]
            fence_shared()                           # [fence_view_async_shared]
            named_barrier_sync(compute_sync_barrier) # P_t visible; S_t no longer read
            consumer_release(pipeline_LSE, slot=stats_slot,
                             payload=LSE_log2_s, to=load_warp)
                                                     # LSE_log2_s no longer needed
            consumer_release(pipeline_S_P, slot=q_slot,
                             payload=P_t, to=mma_warp)
                                                     # P_t ready for dV MMA

            consumer_wait(pipeline_dPsum, slot=stats_slot,
                          payload=dPsum_s, from=load_warp)
                                                     # dPsum_s ready [SMEM]
            consumer_wait(pipeline_dP, slot=q_slot,
                          payload=dP_t, from=mma_warp)
                                                     # dP_t ready [TMEM]
            dP_r = tmem_load(dP_t)                   # [TMEM] -> [RMEM]
            dPsum_r = load_from_smem(dPsum_s)        # [SMEM] -> [RMEM]
            dS_r = P_r * (dP_r - dPsum_r)            # [RMEM], [sub_packed], [mul_packed]

            # dP_t and dS_t overlap.
            fence_tmem_load()                        # [fence_view_async_tmem_load]
            named_barrier_sync(compute_sync_barrier) # no warp still reads dP_t
            dS_t = tmem_store(convert(dS_r))         # [RMEM] -> [TMEM]

            dS_s_local = store_dS_to_smem(dS_r)      # [RMEM] -> [SMEM]
            if use_2cta:
                mbarrier_arrive_expect_tx(
                    dS_cluster_full_mbar_ptr,
                    bytes=dS_exchange_bytes,
                )                                    # [mbarrier] DSMEM tx count
                copy_dS_to_peer_cta(dS_s_local)      # [DSMEM], [cp.async.bulk.shared::cluster]
            fence_shared()                           # [fence_view_async_shared]
            named_barrier_sync(compute_sync_barrier) # dS_s/dS_t visible to consumers
            consumer_release(pipeline_dP, slot=q_slot,
                             payload=dP_t, to=mma_warp)
                                                     # dP_t no longer needed
            consumer_release(pipeline_dPsum, slot=stats_slot,
                             payload=dPsum_s, to=load_warp)
                                                     # dPsum_s no longer needed
            producer_commit(pipeline_dS, slot=q_slot,
                            payload=dS_t + dS_s, to=mma_warp)
                                                     # dS ready for MMA

        # dK/dV epilogue also runs on compute warps.
        for acc_slot in [dV_slot, dK_slot]:
            consumer_wait(pipeline_dKV, slot=acc_slot,
                          payload=dK_t_or_dV_t, from=mma_warp)
                                                     # dK_t or dV_t ready [TMEM]
            dK_r_or_dV_r = tmem_load(dK_t_or_dV_t)    # [TMEM] -> [RMEM]
            scaled = scale_and_convert(dK_r_or_dV_r)  # [RMEM]
            smem_stage = store_to_smem(scaled)        # [RMEM] -> [SMEM]
            TMA_or_reduce_store(smem_stage)           # [SMEM] -> [GMEM]
            consumer_release(pipeline_dKV, slot=acc_slot,
                             payload=dK_t_or_dV_t, to=mma_warp)
                                                     # TMEM accumulator reusable
```

The S/P aliasing barrier and `P = exp2(S * scale_log2 - LSE_log2)` computation are in [`compute_loop`](https://github.com/Dao-AILab/flash-attention/blob/cb213fce11c3baf9168f7fa607bc7f22e3323554/flash_attn/cute/flash_bwd_sm100.py#L3036-L3133). The `dS = P * (dP - dPsum)` computation and dS stores are in [`compute_loop`](https://github.com/Dao-AILab/flash-attention/blob/cb213fce11c3baf9168f7fa607bc7f22e3323554/flash_attn/cute/flash_bwd_sm100.py#L3148-L3261).

#### 3.4 Relay Warp For 2-CTA

```python
def backward_relay_warp():
    # warp 14, only meaningful in 2-CTA paths.

    for q_block in scheduled_q_blocks:
        mbarrier_wait(dS_cluster_full_mbar)          # [mbarrier], peer DSMEM copy complete
        mbarrier_arrive(dS_cluster_leader_mbar)      # [mbarrier], let leader MMA issue dQ
```

The 2-CTA DSMEM exchange uses:

```ptx
mapa.shared::cluster.u32
cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes
```

These helpers are in [`copy_utils.py`](https://github.com/Dao-AILab/flash-attention/blob/cb213fce11c3baf9168f7fa607bc7f22e3323554/flash_attn/cute/copy_utils.py#L147-L239). The compute-side call site is in [`flash_bwd_sm100.py`](https://github.com/Dao-AILab/flash-attention/blob/cb213fce11c3baf9168f7fa607bc7f22e3323554/flash_attn/cute/flash_bwd_sm100.py#L3263-L3283).

#### 3.5 Reduce Warps

```python
def backward_reduce_warps():
    # warps 0-3.

    for kv_work_tile in TileScheduler:
        for q_block in scheduled_q_blocks:
            q_slot = slot_for_Q(q_block)             # [RMEM] physical pipeline index
            consumer_wait(pipeline_dQ, slot=q_slot,
                          payload=dQ_t, from=mma_warp)
                                                     # dQ_t ready [TMEM]
            dQ_r = tmem_load(dQ_t)                    # [TMEM] -> [RMEM], [Ld32x32bOp]
            fence_tmem_load()                         # [fence_view_async_tmem_load]
            consumer_release(pipeline_dQ, slot=q_slot,
                             payload=dQ_t, to=mma_warp)
                                                     # MMA may reuse dQ TMEM

            dQ_s = store_to_smem(dQ_r)                # [RMEM] -> [SMEM]
            fence_shared()                            # [fence_view_async_shared]

            if deterministic:
                wait_global_order_semaphore()         # [barrier.wait_eq]

            named_barrier_sync(reduce_sync_barrier)   # [NamedBarrier dQaccReduce]
            if is_tma_warp:
                cp_reduce_async_bulk_add_f32(
                    src=dQ_s, dst=dQaccum_g[q_block]
                )                                     # [cp.reduce.async.bulk...add.f32]
                cp_async_bulk_commit_group()
                cp_async_bulk_wait_group()
            named_barrier_sync(reduce_sync_barrier)   # async reduce done before reuse

            if deterministic:
                release_global_order_semaphore()      # [barrier.arrive_inc]
```

The async bulk global add is emitted as:

```ptx
cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32
```

The helper is in [`copy_utils.py`](https://github.com/Dao-AILab/flash-attention/blob/cb213fce11c3baf9168f7fa607bc7f22e3323554/flash_attn/cute/copy_utils.py#L266-L288), and the reduce loop is in [`dQacc_reduce`](https://github.com/Dao-AILab/flash-attention/blob/cb213fce11c3baf9168f7fa607bc7f22e3323554/flash_attn/cute/flash_bwd_sm100.py#L3436-L3585).

### 4. Backward Data And Synchronization Summary

Backward placement:

| Data | Shape | Main location | Notes |
|---|---:|---|---|
| `K_s`, `V_s` | `[BN, D]`, `[BN, DV]` | `[SMEM]` | Loaded once for KV tile, reused across Q blocks. |
| `Q_s`, `dO_s` | `[BM, D]`, `[BM, DV]` | `[SMEM]` | Streamed for each Q block. |
| `LSE_log2_s`, `dPsum_s` | `[BM]` | `[SMEM]` | Preprocessed row stats. |
| `S_t`, `P_t` | `[BN, BM]` | `[TMEM]` | Overlap in TMEM; protected by compute barrier. |
| `dP_t`, `dS_t` | `[BN, BM]` | `[TMEM]` | Overlap in TMEM; protected by compute barrier. |
| `dS_s` | layout-dependent | `[SMEM]` and sometimes `[DSMEM]` | Used by dQ MMA in 2-CTA mode. |
| `dQ_t` | `[BM, D]` | `[TMEM]` | Reduced into global `dQaccum`. |
| `dK_t`, `dV_t` | `[BN, D]`, `[BN, DV]` | `[TMEM]` | Long-lived accumulators for the KV tile. |

Backward synchronization:

| Dependency | CUDA feature |
|---|---|
| Q/K/V/dO/stats loads complete | `[PipelineTmaUmma]`, `[PipelineTmaAsync]`, TMA mbarriers |
| `S_t` ready for compute | `pipeline_S_P` |
| `P_t` stored before dV MMA consumes it | `pipeline_S_P.consumer_release` |
| `dP_t` ready for compute | `pipeline_dP` |
| `dS_t/dS_s` stored before dK/dQ MMA consumes it | `pipeline_dS` |
| `S_t` not overwritten by `P_t` too early | `fence_view_async_tmem_load` plus `Compute` named barrier |
| `dP_t` not overwritten by `dS_t` too early | `fence_view_async_tmem_load` plus `Compute` named barrier |
| peer CTA dS visible before 2-CTA dQ MMA | DSMEM `cp.async.bulk.shared::cluster` plus `[mbarrier]` |
| `dQ_t` copied before MMA reuses TMEM | `pipeline_dQ.consumer_wait/release` |
| global dQ accumulation safe | `cp.reduce.async.bulk...add.f32`, reduce named barrier, optional deterministic semaphores |

Backward barrier events:

| Barrier handoff | Producer side | Consumer side |
|---|---|---|
| TMEM pointer available | MMA warp allocates TMEM, then `tmem_alloc_barrier.arrive_and_wait()` | Compute/reduce warps call `tmem_alloc_barrier.arrive()` before retrieving/using the pointer. |
| `S_t` fully read before `P_t` store | Compute warps call `fence_view_async_tmem_load()` after loading `S_t` | All compute warps synchronize with `compute_sync_barrier.arrive_and_wait()`. |
| `P_t` visible before dV MMA | Compute warps store `P_t`, call `fence_view_async_tmem_store()` and `fence_view_async_shared()` | Compute warps synchronize again with `compute_sync_barrier.arrive_and_wait()`, then release `pipeline_S_P`. |
| `dP_t` fully read before `dS_t` store | Compute warps call `fence_view_async_tmem_load()` after loading `dP_t` | All compute warps synchronize with `compute_sync_barrier.arrive_and_wait()`. |
| Peer CTA `dS` exchange | Compute warp calls `mbarrier_arrive_and_expect_tx(dS_cluster_full_mbar_ptr, bytes)` then `cp.async.bulk.shared::cluster` | Relay warp waits `dS_cluster_full_mbar_ptr`, then arrives `dS_cluster_leader_mbar_ptr`; MMA waits `dS_cluster_leader_mbar_ptr`. |
| dQ async reduce | Reduce warps call `fence_view_async_shared()` after staging `dQ_s` | `reduce_sync_barrier.arrive_and_wait()` guards both before and after `cp.reduce.async.bulk...add.f32`. |
| deterministic dQ order | Prior CTA calls `barrier.arrive_inc` on the global semaphore | Current CTA calls `barrier.wait_eq` before issuing the reduce. |

## Cross-Cutting Notes

### Exact Primitive Glossary

| Purpose | Exact primitive names in FA4 |
|---|---|
| TMA global-to-shared | `cpasync.CopyBulkTensorTileG2SOp`, `cute.nvgpu.make_tiled_tma_atom_A`, `cute.nvgpu.make_tiled_tma_atom_B`, `cpasync.tma_partition`, `cute.copy(..., tma_bar_ptr=...)` |
| TMA shared-to-global | `cpasync.CopyBulkTensorTileS2GOp`, `cpasync.make_tiled_tma_atom`, `cute.arch.cp_async_bulk_commit_group`, `cute.arch.cp_async_bulk_wait_group` |
| cp.async fallback | `cute.nvgpu.cpasync.CopyG2SOp`, `cute.arch.cp_async_commit_group`, `arrive_cp_async_mbarrier` |
| SM100 tensor-core MMA | Inline PTX `tcgen05.mma.cta_group::{1 or 2}.kind::{kind}`; CuTe uses `tcgen05.CtaGroup.ONE/TWO`, `tcgen05.OperandSource.SMEM/TMEM`, `tcgen05.OperandMajorMode` |
| TMEM allocation | `cutlass.utils.TmemAllocator`, `tmem.allocate`, `tmem.wait_for_alloc`, `tmem.retrieve_ptr`, `tmem.free` |
| TMEM load/store | `tcgen05.copy.Ld32x32bOp`, `tcgen05.copy.St32x32bOp`, `tcgen05.make_tmem_copy`, `cute.copy` |
| Memory visibility | `cute.arch.fence_view_async_tmem_store`, `cute.arch.fence_view_async_tmem_load`, `cute.arch.fence_view_async_shared` |
| Named CTA barriers | `cutlass.pipeline.NamedBarrier`, `pipeline.NamedBarrier.arrive_w_index`, `pipeline.NamedBarrier.arrive_and_wait_w_index`, `cute.arch.barrier_arrive`, `cute.arch.barrier` |
| Pipeline synchronization | `PipelineTmaUmma`, `PipelineAsyncUmma`, `PipelineUmmaAsync`, `PipelineTmaAsync`, `PipelineAsync`, pipeline full/empty `mbarrier` storage |
| mbarriers | `cute.arch.mbarrier_init`, `cute.arch.mbarrier_arrive`, `cute.arch.mbarrier_arrive_and_expect_tx`, `cute.arch.mbarrier_wait`, inline PTX `mbarrier.try_wait.parity.shared::cta.b64` |
| Warp election | `cute.arch.sync_warp`, `cute.arch.elect_one`, inline PTX `elect.sync` |
| DSMEM | Inline PTX `mapa.shared::cluster.u32`, `cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes` |
| Async global reduction | Inline PTX `cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32` |
| Packed math | `cute.arch.fma_packed_f32x2`, `cute.arch.mul_packed_f32x2`, `cute.arch.add_packed_f32x2`, `cute.arch.vote_ballot_sync`, `cute.arch.rcp_approx` |

### Why This Mapping Works

The forward pass is dominated by the need to overlap QK, softmax, correction, PV, and memory movement. FA4 puts Q/K/V in `[SMEM]`, score/probability/output accumulator tiles in `[TMEM]`, row fragments in `[RMEM]`, and uses pipeline barriers so each specialized warp group can run independently.

The backward pass is dominated by a five-MMA graph and by `dQ` reduction. FA4 recomputes `S` and `P` in `[TMEM]`, aliases `S/P` and `dP/dS` carefully, keeps `dK/dV` as long-lived `[TMEM]` accumulators, exchanges `dS` through `[DSMEM]` in 2-CTA mode, and reduces `dQ` with async bulk global add.

### Evidence Boundaries

This note focuses on the public FA4 CuTe DSL SM100/SM110 implementation at commit `cb213fce11c3baf9168f7fa607bc7f22e3323554`. It summarizes the generic SM100 forward/backward kernels and the major 2-CTA paths. Dedicated head-dim-256 kernels and every feature-specific branch, such as block sparsity and split-KV details, are referenced only where they affect the main pipeline story.

## References

1. Ted Zadouri, Markus Hoehnerbach, Jay Shah, Timmy Liu, Vijay Thakkar, and Tri Dao. [FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling](https://arxiv.org/abs/2603.05451). arXiv:2603.05451, submitted 2026-03-05.
2. Dao-AILab. [FlashAttention CuTe/FA4 implementation at commit `cb213fce11c3baf9168f7fa607bc7f22e3323554`](https://github.com/Dao-AILab/flash-attention/tree/cb213fce11c3baf9168f7fa607bc7f22e3323554/flash_attn/cute).
3. NVIDIA. [CuTe DSL `tcgen05` submodule documentation](https://docs.nvidia.com/cutlass/4.4.1/media/docs/pythonDSL/cute_dsl_api/cute_nvgpu_tcgen05.html).
4. NVIDIA. [CUTLASS Blackwell-specific documentation](https://docs.nvidia.com/cutlass/4.3.2/media/docs/cpp/blackwell.html).
5. NVIDIA. [CUDA Programming Guide: Distributed Shared Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#distributed-shared-memory).
