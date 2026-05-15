# CUDA Synchronization and Ordering Patterns

Source cutoff: May 14, 2026

Audience: CUDA kernel authors, performance engineers, and systems researchers who already know the CUDA execution hierarchy but want a practical map of synchronization and memory-ordering choices.

Scope: CUDA C++, PTX-level memory semantics, NCCL, NVSHMEM, and CUDA-aware MPI patterns where they affect ordering across a single GPU, a node, or multiple nodes.

## 1. Abstract and Scope

CUDA synchronization is not one mechanism. It is a family of control barriers, memory-ordering operations, scoped atomics, stream dependencies, cooperative-group barriers, and communication-library protocols. The most common bugs happen when code accidentally combines a primitive from one scope with data or participants from another scope: a warp shuffle used after divergence, a block barrier expected to order other blocks, a device fence treated as a grid barrier, or a NCCL collective launched in a different order on different ranks.

This survey classifies synchronization by the scope of participants:

| Scope | Representative primitives | Canonical algorithm in this survey | Main hazard |
|---|---|---|---|
| Thread | program order, `cuda::atomic` acquire/release, `__threadfence*` | producer-consumer message passing | flag is visible before payload or uses wrong scope |
| Warp | `__syncwarp`, `__shfl_sync`, `__ballot_sync`, `__match_any_sync` | warp reduction and ballot compaction | incorrect mask after divergence |
| Block | `__syncthreads`, block cooperative groups, shared memory | block reduction and tiled transpose | divergent barrier or missing shared-memory barrier |
| Block async pipeline | `cuda::barrier`, `cuda::pipeline`, `memcpy_async`, PTX `mbarrier` | double-buffered tiled stencil or GEMM prefetch | using a tile before the async copy completed |
| Cluster | thread-block clusters, distributed shared memory, `cluster.sync()` | intra-cluster halo exchange | assuming cluster sync is grid sync |
| Grid | cooperative launch, `cooperative_groups::this_grid().sync()` | single-kernel global reduction or iterative solver phase | launch cannot satisfy full-grid residency |
| Device and streams | stream order, events, graph dependencies, kernel-launch boundaries | copy-compute-overlap pipeline | accidental default-stream synchronization |
| Device inter-block memory | device-scope atomics, fences, counters | last-block reduction | deadlock/progress hazard in spin barriers |
| Node | peer access, CUDA IPC, NCCL, stream/event ordering | multi-GPU all-reduce | mismatched collective order or stream lifetime bugs |
| Cross-node | NVSHMEM, NCCL net, CUDA-aware MPI | halo exchange, custom symmetric-memory barriers, and distributed reductions | confusing remote completion, local completion, and global visibility |

Evidence labels used below:

| Label | Meaning |
|---|---|
| official docs | NVIDIA or standards-adjacent documentation for an API or ISA |
| official blog | NVIDIA developer blog, useful for design intent and migration guidance |
| paper | peer-reviewed or preprint literature |
| inferred | a practical implication derived from documented semantics rather than directly stated as a pattern |

## 2. Shared Model: Synchronization Is Control Plus Memory

Every CUDA ordering pattern has two separable parts.

Control synchronization asks which execution agents have reached a point. Examples include `__syncthreads()` for a thread block, `grid.sync()` for a cooperatively launched grid, `cudaEventRecord` plus `cudaStreamWaitEvent` for stream tasks, and `nvshmem_barrier_all` for processing elements.

Memory ordering asks which memory operations become ordered or visible to which observers. Examples include `cuda::atomic` release/acquire operations, `__threadfence_block`, `__threadfence`, `__threadfence_system`, PTX `fence` scopes, NVSHMEM `fence`/`quiet`, and NCCL's stream-ordered enqueue semantics.

These dimensions often line up, but they are not the same thing. A barrier usually gives both convergence and ordering for its participants in a particular memory domain. A fence gives ordering but not convergence. An atomic read-modify-write can serialize updates to one address without making unrelated non-atomic payload visible unless the operation has the right memory order and scope. A kernel launch boundary gives device-wide ordering between kernels in the same stream but does not make two independent streams ordered without an explicit dependency.

The CUDA Programming Guide documents device synchronization functions, fences, cooperative groups, thread-block clusters, asynchronous execution, and memory synchronization domains in CUDA 13.2 [official docs](https://docs.nvidia.com/cuda/cuda-programming-guide/). The PTX ISA 9.2 specifies lower-level memory consistency, operation kinds, scopes, and proxy fences [official docs](https://docs.nvidia.com/cuda/parallel-thread-execution/). The libcu++ extended API documents `cuda::atomic`, `cuda::barrier`, `cuda::pipeline`, and thread scopes [official docs](https://nvidia.github.io/cccl/libcudacxx/).

### 2.1 Scope and Visibility Vocabulary

CUDA code commonly uses the following scopes:

| Scope | Typical participants | Examples |
|---|---|---|
| Thread | one CUDA thread | program order, register/local memory dependencies |
| Warp | active lanes in one warp | warp vote, shuffle, `__syncwarp(mask)` |
| Block | threads in one thread block | shared memory, `__syncthreads()` |
| Cluster | blocks in one thread-block cluster | distributed shared memory, `cluster_group::sync()` |
| Grid | all blocks in one cooperatively launched grid | `this_grid().sync()` |
| Device | all agents on one GPU | device-scope atomics, `__threadfence()` |
| System | host plus GPUs and peer agents that participate in system scope | `__threadfence_system()`, system-scope atomics |
| Node | multiple GPUs/processes in one host | peer access, CUDA IPC, NCCL local ranks |
| Cross-node | GPUs/processes across hosts | NCCL network transport, NVSHMEM, CUDA-aware MPI |

CUDA memory spaces also matter. Shared memory is naturally block-scoped unless distributed shared memory is used in clusters. Global memory can be observed by any block on the device, peer GPUs, the CPU, or network peers only when the allocation and API path permit it. Texture, surface, constant, and read-only cache paths have their own coherency constraints; do not assume a store through one path is immediately coherent through another path without a documented synchronization point or a new kernel phase.

### 2.2 Ordering Cost Heuristic

The ordering cost grows with scope:

$$
C_{\mathrm{order}}(\mathrm{thread}) < C_{\mathrm{order}}(\mathrm{warp}) < C_{\mathrm{order}}(\mathrm{block}) < C_{\mathrm{order}}(\mathrm{cluster}) < C_{\mathrm{order}}(\mathrm{grid}) < C_{\mathrm{order}}(\mathrm{device}) < C_{\mathrm{order}}(\mathrm{system})
$$

Symbols: $C_{\mathrm{order}}(s)$ denotes the rough latency and opportunity cost of enforcing ordering at scope $s$. This is a qualitative inequality, not a benchmark. Real cost depends on architecture, occupancy, memory system traffic, cache state, and communication topology.

The practical rule is to synchronize at the smallest scope that contains all producers and consumers of the data. If a warp reduction never leaves the warp, use warp collectives. If values are exchanged through shared memory by all threads in a block, use a block barrier. If phases cross all blocks, prefer a kernel boundary or cooperative-grid synchronization rather than hand-rolled global barriers.

## 3. Thread and Memory-Ordering Scope

### Concept

Thread scope starts with program order: a single CUDA thread observes its own operations in sequence, subject to compiler and hardware transformations that preserve single-thread semantics. Communication between threads requires an ordering primitive. The modern CUDA C++ form is a scoped atomic with an explicit memory order, such as a producer storing a payload and then storing a flag with `memory_order_release`, while a consumer spins on the flag with `memory_order_acquire`.

The same message-passing pattern appears in C++ concurrency literature and in GPU memory-model papers because it is the minimal test for whether payload writes are ordered before a notification. PTX exposes this through acquire/release and scope-qualified operations; libcu++ exposes it through `cuda::atomic` and `cuda::thread_scope_*` [official docs](https://nvidia.github.io/cccl/libcudacxx/extended_api/memory_model.html).

### Canonical Algorithm: Publish a Work Item to Another Block

One block writes a descriptor into global memory and publishes a flag. Another block polls the flag and then consumes the descriptor. The canonical uses are persistent work queues, producer-consumer pipelines, and device-side scheduling.

```cuda
#include <cuda/atomic>

struct WorkItem {
  int offset;
  int count;
};

__device__ WorkItem item;
__device__ cuda::atomic<int, cuda::thread_scope_device> ready;

__global__ void producer() {
  if (threadIdx.x == 0) {
    item = WorkItem{1024, 256};                       // payload
    ready.store(1, cuda::memory_order_release);       // publish
  }
}

__global__ void consumer(int *out) {
  if (threadIdx.x == 0) {
    while (ready.load(cuda::memory_order_acquire) == 0) {
      // Optional backoff or bounded polling in real kernels.
    }
    WorkItem local = item;                            // sees payload
    out[0] = local.offset + local.count;
  }
}
```

This example uses device scope because the producer and consumer are on the same GPU. If a CPU thread or peer GPU must observe the flag, use system-capable allocation plus system-scope atomics or API-level synchronization. If all communication is inside one block, block scope is cheaper and sufficient.

### Memory-Model Notes

Acquire/release is a directed ordering contract. A release store orders prior writes before the flag publication, and an acquire load that reads that store orders subsequent reads after the flag observation. Sequential consistency is stronger but not automatically required for all producer-consumer cases. Relaxed atomics serialize the atomic object but do not carry payload ordering.

Fence functions are older but still useful. `__threadfence_block()` orders memory operations as observed by block peers, `__threadfence()` by device peers, and `__threadfence_system()` by system peers where supported [official docs](https://docs.nvidia.com/cuda/cuda-programming-guide/). A fence is not a notification: another thread still needs an atomic, flag, barrier, or stream dependency to know when to read.

PTX distinguishes operation type, scope, semantic strength, and proxy. This matters for advanced features such as tensor-map copies, asynchronous copies, and multimem operations, where a normal fence may not order accesses through a different proxy unless the ISA calls for a proxy fence [official docs](https://docs.nvidia.com/cuda/parallel-thread-execution/).

### Pitfalls

- Using `volatile` as a synchronization primitive. It can affect compiler caching behavior in limited contexts, but it is not a replacement for scoped atomics or fences.
- Using `memory_order_relaxed` on the flag and expecting the payload to be ordered.
- Publishing with device scope and consuming from the host, another GPU, or network peer.
- Spinning in a way that requires blocks not currently resident to make progress. Device-scope ordering does not guarantee scheduling progress for all blocks.
- Treating `__threadfence()` as a barrier. It orders the caller's memory operations; it does not wait for other threads.

### Explore Further

- CUDA Programming Guide, memory fence functions and synchronization behavior [official docs](https://docs.nvidia.com/cuda/cuda-programming-guide/).
- libcu++ memory model and `cuda::atomic` scopes [official docs](https://nvidia.github.io/cccl/libcudacxx/extended_api/memory_model.html).
- PTX ISA memory consistency model, scopes, acquire/release, and proxy fences [official docs](https://docs.nvidia.com/cuda/parallel-thread-execution/).
- GPU memory model work such as scoped synchronization and weak-memory analyses [paper](https://dl.acm.org/doi/10.1145/3297858.3304043).

## 4. Warp Scope

### Concept

A warp is the smallest SIMD-like scheduling group in CUDA. Warp-level intrinsics exchange values between lanes without shared memory. Since Volta introduced independent thread scheduling, warp-synchronous programming must use explicit masks and synchronization intrinsics when communication crosses lanes. NVIDIA's warp primitive guidance emphasizes `_sync` variants such as `__shfl_down_sync`, `__ballot_sync`, and `__syncwarp` [official blog](https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/).

Warp synchronization has two practical roles. First, warp collectives move values through registers. Second, `__syncwarp(mask)` reconverges a set of lanes and orders memory among participating lanes at warp scope. It is not a block barrier.

### Canonical Algorithm A: Warp Reduction

Each lane has a partial value. A tree reduction uses shuffle-down operations to sum values within one warp.

```cuda
__device__ float warp_sum(float x, unsigned mask) {
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    x += __shfl_down_sync(mask, x, offset);
  }
  return x;  // valid in the lowest participating lane
}

__global__ void reduce_warps(const float *in, float *warp_out, int n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned mask = __ballot_sync(0xffffffffu, tid < n);
  float x = (tid < n) ? in[tid] : 0.0f;
  float sum = warp_sum(x, mask);
  if ((threadIdx.x & 31) == 0) {
    warp_out[(blockIdx.x * blockDim.x + threadIdx.x) / warpSize] = sum;
  }
}
```

The mask names the lanes that participate. It should be computed before the collective and reused consistently by all participating lanes. The algorithm is canonical because it replaces shared-memory reduction steps with register exchange.

### Canonical Algorithm B: Ballot-Based Warp Compaction

Each lane decides whether it contributes an item. `__ballot_sync` builds a bitmask, and `__popc` computes the local compacted rank.

```cuda
__device__ int warp_compact_rank(bool keep, unsigned full_mask) {
  unsigned active = __ballot_sync(full_mask, keep);
  unsigned lane_mask_lt = (1u << (threadIdx.x & 31)) - 1u;
  int rank = __popc(active & lane_mask_lt);
  return keep ? rank : -1;
}
```

This is the warp-local core of stream compaction, frontier expansion in graph traversal, and warp-aggregated atomics. A leader can reserve a contiguous output segment with one atomic add, then lanes write at `base + rank`.

### Memory-Model Notes

Warp intrinsics are not a general memory model. Shuffle and vote move register values; they do not publish arbitrary shared or global memory. If lanes write shared memory and then other lanes read it, use `__syncwarp(mask)` when the participating lanes are a strict warp subset, or `__syncthreads()` when block-wide participation is required. Do not use implicit lockstep as a memory fence.

### Pitfalls

- Passing `0xffffffffu` when only a subset of lanes actually reaches the intrinsic. Use an active mask derived from the same control condition.
- Computing a mask inside divergent code where not all intended lanes execute the computation.
- Assuming lane 0 is always the leader when lane 0 may not be active. Use `__ffs(mask) - 1` when the active set is sparse.
- Mixing warp-local and block-local communication. Once data crosses warp boundaries through shared memory, a block barrier is usually needed.
- Relying on pre-Volta implicit warp convergence. Use `_sync` intrinsics and masks.

### Explore Further

- NVIDIA warp-level primitives blog [official blog](https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/).
- CUDA Programming Guide, warp vote, match, reduce, and shuffle functions [official docs](https://docs.nvidia.com/cuda/cuda-programming-guide/).
- NVIDIA filtering and warp-aggregated atomics discussion [official blog](https://developer.nvidia.com/blog/cuda-pro-tip-optimized-filtering-warp-aggregated-atomics/).

## 5. Block Scope

### Concept

A thread block is the first CUDA scope with a full barrier available to ordinary kernels. `__syncthreads()` waits until all non-exited threads in the block reach the call, and it orders shared-memory and relevant memory operations among participating threads at block scope. Cooperative groups provide a typed `thread_block` abstraction with `block.sync()` [official docs](https://docs.nvidia.com/cuda/cuda-programming-guide/).

Block barriers are fundamental for tiled algorithms because shared memory is block-scoped and fast. The canonical pattern is: load a tile into shared memory, synchronize, compute from the tile, synchronize again if the shared buffer will be reused.

### Canonical Algorithm A: Block Reduction

Threads load one element each, reduce in shared memory, and have one thread write the block result.

```cuda
template <int BLOCK>
__global__ void block_reduce(const float *in, float *block_out, int n) {
  __shared__ float s[BLOCK];
  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x + tid;

  s[tid] = (i < n) ? in[i] : 0.0f;
  __syncthreads();

  for (int stride = BLOCK / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      s[tid] += s[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    block_out[blockIdx.x] = s[0];
  }
}
```

The barrier after each reduction step prevents a thread from reading `s[tid + stride]` before the writer has completed the previous step.

### Canonical Algorithm B: Tiled Matrix Transpose

Threads cooperatively load a tile, synchronize, and then write it transposed. Padding avoids shared-memory bank conflicts in the common square-tile case.

```cuda
template <int TILE>
__global__ void transpose_tile(const float *in, float *out, int width, int height) {
  __shared__ float tile[TILE][TILE + 1];

  int x = blockIdx.x * TILE + threadIdx.x;
  int y = blockIdx.y * TILE + threadIdx.y;
  if (x < width && y < height) {
    tile[threadIdx.y][threadIdx.x] = in[y * width + x];
  }

  __syncthreads();

  int ox = blockIdx.y * TILE + threadIdx.x;
  int oy = blockIdx.x * TILE + threadIdx.y;
  if (ox < height && oy < width) {
    out[oy * height + ox] = tile[threadIdx.x][threadIdx.y];
  }
}
```

This algorithm is canonical because the tile load and tile store are separated by a true block-wide data dependency.

### Memory-Model Notes

`__syncthreads()` is a barrier and a memory-ordering point for threads in the same block. It does not make other blocks wait. For partial-block groups, cooperative groups tiles or warp primitives can be safer than conditional `__syncthreads()`, but the group object must match the threads that actually participate.

### Pitfalls

- Calling `__syncthreads()` in a branch that is not uniform across the block.
- Reusing shared memory for the next tile without a second barrier after consumers finish.
- Assuming block barrier orders peer blocks or host code.
- Using block-level algorithms with dynamic block sizes that do not match compile-time shared-memory arrays.
- Forgetting that early returns before a later barrier can deadlock the remaining block.

### Explore Further

- CUDA Programming Guide synchronization functions [official docs](https://docs.nvidia.com/cuda/cuda-programming-guide/).
- Cooperative Groups programming model [official docs](https://docs.nvidia.com/cuda/cuda-programming-guide/#cooperative-groups).
- NVIDIA Cooperative Groups blog for modular block-level synchronization [official blog](https://developer.nvidia.com/blog/cooperative-groups/).

## 6. Block-Local Async Pipelines

### Concept

Modern CUDA exposes asynchronous copy and barrier primitives that let a block overlap global-to-shared-memory movement with computation. At the CUDA C++ level, common interfaces include `cuda::pipeline`, `cuda::barrier`, and `cuda::memcpy_async`; at the PTX level, related concepts include `cp.async`, `mbarrier`, and proxy ordering. The primitive is not merely a faster load. It introduces a producer-consumer protocol between asynchronous memory operations and threads that later read the shared-memory tile [official docs](https://nvidia.github.io/cccl/libcudacxx/extended_api/synchronization_primitives.html).

### Canonical Algorithm: Double-Buffered Tiled Stencil

A block computes a 1D stencil. While it computes tile `t`, it asynchronously prefetches tile `t + 1` into the other shared-memory buffer.

```cuda
#include <cuda/pipeline>

template <int BLOCK, int TILE>
__global__ void stencil_async(const float *in, float *out, int n) {
  __shared__ float smem[2][TILE + 2];  // halo on each side
  auto pipe = cuda::make_pipeline();
  int tid = threadIdx.x;
  int tile0 = blockIdx.x * TILE;

  for (int t = tile0, stage = 0; t < n; t += gridDim.x * TILE, stage ^= 1) {
    pipe.producer_acquire();
    int g = t + tid - 1;
    if (tid < TILE + 2 && 0 <= g && g < n) {
      cuda::memcpy_async(&smem[stage][tid], &in[g], sizeof(float), pipe);
    }
    pipe.producer_commit();

    pipe.consumer_wait();
    __syncthreads();  // all copied elements are visible to block threads

    int center = t + tid;
    if (tid < TILE && center > 0 && center + 1 < n) {
      out[center] = 0.25f * smem[stage][tid]
                  + 0.50f * smem[stage][tid + 1]
                  + 0.25f * smem[stage][tid + 2];
    }

    __syncthreads();  // do not overwrite the buffer before consumers finish
    pipe.consumer_release();
  }
}
```

This is illustrative rather than a fully tuned stencil. The algorithmic shape is the important part: async copy publishes a tile, the consumer waits, a block barrier aligns all readers, and a second barrier protects buffer reuse.

### Memory-Model Notes

Asynchronous copies may use a different memory path than ordinary loads and stores. PTX models some of these with proxy concepts, so advanced inline PTX code must use the fences documented for the relevant operation kind. At the CUDA C++ level, prefer `cuda::pipeline` or `cuda::barrier` APIs because they encode the intended producer-consumer protocol.

### Pitfalls

- Treating `memcpy_async` issue as completion. The consumer must wait before reading the shared-memory destination.
- Omitting the block barrier when multiple threads consume data loaded by other threads.
- Overwriting a staging buffer before all consumers finish.
- Initializing `cuda::barrier` incorrectly or by only a subset of threads.
- Mixing inline PTX `cp.async` with C++ barriers without understanding proxy ordering.

### Explore Further

- libcu++ synchronization primitives, barriers, pipelines, and asynchronous copies [official docs](https://nvidia.github.io/cccl/libcudacxx/extended_api/synchronization_primitives.html).
- CUDA Programming Guide asynchronous SIMT programming model [official docs](https://docs.nvidia.com/cuda/cuda-programming-guide/).
- PTX ISA `mbarrier`, async copy, and proxy fence sections [official docs](https://docs.nvidia.com/cuda/parallel-thread-execution/).

## 7. Cluster Scope

### Concept

Thread-block clusters group blocks so they can synchronize and access distributed shared memory. This creates a scope between a block and a full grid. Cluster synchronization is useful when neighboring blocks exchange small halo regions or partial results without going through global memory and a second kernel. CUDA documents thread-block clusters, distributed shared memory, and cluster cooperative groups for architectures that support them [official docs](https://docs.nvidia.com/cuda/cuda-programming-guide/).

Cluster scope is not automatic. Kernels must be launched with cluster dimensions or compiled with attributes, and the cluster's blocks must be scheduled in a way that supports the cluster contract.

### Canonical Algorithm: Intra-Cluster Halo Exchange

Each block computes a tile of a 1D stencil. Neighboring blocks in the same cluster publish boundary values into distributed shared memory, synchronize the cluster, and read neighbor halos.

```cuda
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

extern "C" __global__
__cluster_dims__(4, 1, 1)
void cluster_halo(const float *in, float *out, int n) {
  cg::cluster_group cluster = cg::this_cluster();
  cg::thread_block block = cg::this_thread_block();

  extern __shared__ float local[];
  int rank = cluster.block_rank();
  int tid = threadIdx.x;
  int block_base = blockIdx.x * blockDim.x;
  int g = block_base + tid;

  local[tid] = (g < n) ? in[g] : 0.0f;
  block.sync();

  // Publish left and right boundary values in each block's shared segment.
  if (tid == 0) {
    local[blockDim.x] = local[0];
  }
  if (tid == blockDim.x - 1) {
    local[blockDim.x + 1] = local[tid];
  }

  cluster.sync();

  float left = (tid == 0 && rank > 0)
      ? *cluster.map_shared_rank(&local[blockDim.x + 1], rank - 1)
      : ((tid > 0) ? local[tid - 1] : local[tid]);

  float right = (tid == blockDim.x - 1 && rank + 1 < cluster.num_blocks())
      ? *cluster.map_shared_rank(&local[blockDim.x], rank + 1)
      : ((tid + 1 < blockDim.x) ? local[tid + 1] : local[tid]);

  if (g > 0 && g + 1 < n) {
    out[g] = 0.25f * left + 0.5f * local[tid] + 0.25f * right;
  }
}
```

The cluster barrier is the phase boundary: after it, each block can safely read neighbor boundaries published in distributed shared memory.

### Memory-Model Notes

Distributed shared memory extends shared-memory addressing across blocks in one cluster, but its synchronization scope is still the cluster. Reads through `map_shared_rank` depend on the producing block having written the value and the cluster having synchronized. For data needed outside the cluster, use global memory plus device/grid synchronization.

### Pitfalls

- Assuming `cluster.sync()` waits for blocks outside the cluster.
- Launching more blocks than can participate in the intended cluster topology and then indexing neighbors as if the whole grid were one cluster.
- Accessing a remote block's shared memory before a cluster synchronization point.
- Forgetting that distributed shared memory lifetime is tied to the blocks in the cluster.
- Writing cluster-specific kernels without a fallback for devices that do not support clusters.

### Explore Further

- CUDA Programming Guide, thread-block clusters and distributed shared memory [official docs](https://docs.nvidia.com/cuda/cuda-programming-guide/).
- Cooperative Groups cluster APIs [official docs](https://docs.nvidia.com/cuda/cuda-programming-guide/#cooperative-groups).
- PTX ISA cluster-scope barriers and memory operations [official docs](https://docs.nvidia.com/cuda/parallel-thread-execution/).

## 8. Grid Scope

### Concept

Ordinary CUDA kernels do not provide a global barrier between all blocks. Kernel launch boundaries often serve as grid-wide phase boundaries: kernel A completes, then kernel B starts in the same stream. Cooperative groups add a true in-kernel grid barrier through cooperative launch and `cooperative_groups::this_grid().sync()` [official docs](https://docs.nvidia.com/cuda/cuda-programming-guide/#cooperative-groups).

Grid synchronization is useful when a multi-phase algorithm benefits from staying in one kernel: iterative solvers, global reductions with a final broadcast, graph algorithms with repeated frontiers, or producer-consumer loops that would otherwise pay repeated launch overheads.

### Canonical Algorithm: Single-Kernel Global Reduction

Each block reduces its tile, writes one partial sum, synchronizes the grid, and one block reduces the partials.

```cuda
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

extern "C" __global__
void cooperative_reduce(const float *in, float *partials, float *out, int n) {
  cg::grid_group grid = cg::this_grid();
  extern __shared__ float s[];
  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x + tid;

  s[tid] = (i < n) ? in[i] : 0.0f;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) s[tid] += s[tid + stride];
    __syncthreads();
  }

  if (tid == 0) partials[blockIdx.x] = s[0];

  grid.sync();  // every partial is now written

  if (blockIdx.x == 0) {
    float total = 0.0f;
    for (int b = tid; b < gridDim.x; b += blockDim.x) {
      total += partials[b];
    }
    s[tid] = total;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (tid < stride) s[tid] += s[tid + stride];
      __syncthreads();
    }
    if (tid == 0) *out = s[0];
  }
}
```

The host side must use cooperative launch and choose a grid size that can be resident under the cooperative-launch constraints.

```cuda
void *args[] = {&in, &partials, &out, &n};
cudaLaunchCooperativeKernel((void*)cooperative_reduce,
                            grid, block, args, block.x * sizeof(float), stream);
```

### Memory-Model Notes

`grid.sync()` is a control and memory synchronization point for the cooperatively launched grid. It is not available for arbitrary kernels. When the algorithm can tolerate separate kernels, a kernel boundary in one stream is usually simpler and more portable.

### Pitfalls

- Using `this_grid().sync()` without cooperative launch.
- Launching a grid too large to satisfy cooperative residency. The required number of blocks must be able to run concurrently according to device properties and kernel resource usage.
- Letting any block return before a later `grid.sync()`.
- Expecting a cooperative grid barrier to synchronize with host work or other streams.
- Replacing grid sync with a spin barrier over global memory. That can deadlock if some blocks required to release the barrier are not resident.

### Explore Further

- CUDA Programming Guide cooperative groups and cooperative launch [official docs](https://docs.nvidia.com/cuda/cuda-programming-guide/#cooperative-groups).
- NVIDIA Cooperative Groups blog [official blog](https://developer.nvidia.com/blog/cooperative-groups/).
- Research on GPU synchronization methods comparing blocking, lock-free, and occupancy-aware barriers [paper](https://ieeexplore.ieee.org/document/8091043).

## 9. Device-Scope Inter-Block Ordering

### Concept

Device-scope ordering covers communication among blocks on one GPU without requiring a full grid barrier. Typical primitives include device-scope atomics, release/acquire flags, and device fences. These are appropriate when the algorithm has a one-way handoff or a monotonic counter, not when all blocks must wait for all other blocks in a reusable global barrier.

### Canonical Algorithm: Last-Block Reduction

Each block writes a partial result. A device-scope atomic counter identifies the last block. The last block then consumes all partials and writes the final result.

```cuda
#include <cuda/atomic>

__device__ cuda::atomic<int, cuda::thread_scope_device> done_count;

__global__ void last_block_reduce(const float *in, float *partials, float *out, int n) {
  extern __shared__ float s[];
  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x + tid;

  s[tid] = (i < n) ? in[i] : 0.0f;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) s[tid] += s[tid + stride];
    __syncthreads();
  }

  if (tid == 0) {
    partials[blockIdx.x] = s[0];
    int old = done_count.fetch_add(1, cuda::memory_order_acq_rel);

    if (old == gridDim.x - 1) {
      float total = 0.0f;
      for (int b = 0; b < gridDim.x; ++b) {
        total += partials[b];
      }
      *out = total;
      done_count.store(0, cuda::memory_order_release);
    }
  }
}
```

In production, reset `done_count` before launch or inside a carefully ordered initialization phase. A two-kernel reduction is often easier: kernel 1 writes partials; kernel 2 reduces partials. The one-kernel form is useful when launch overhead is material and the one-way "last block does final work" structure is safe.

### Memory-Model Notes

The counter's atomicity does not automatically order unrelated partial writes unless the operation has release/acquire semantics or is paired with a fence. The last block must observe all previous partial writes before consuming them. Older CUDA examples used `volatile` plus `__threadfence()` before an atomic increment. Modern code should prefer scoped atomics where possible because the ordering contract is explicit.

This pattern is not a reusable global barrier. It works because only the last block proceeds to a final sequential phase; earlier blocks do not wait and then continue.

### Pitfalls

- Using a relaxed atomic counter and reading stale partials.
- Forgetting to reset the counter between launches.
- Turning the pattern into a spin barrier where all blocks wait for `done_count == gridDim.x`; that can deadlock when not all blocks are resident.
- Assuming device-scope atomics synchronize with CPU reads before the kernel completes.
- Overusing `__threadfence_system()` when only device peers need the ordering.

### Explore Further

- CUDA Programming Guide memory fence examples and atomic operations [official docs](https://docs.nvidia.com/cuda/cuda-programming-guide/).
- libcu++ atomics and thread scopes [official docs](https://nvidia.github.io/cccl/libcudacxx/extended_api/synchronization_primitives/atomic.html).
- Stuart and Owens, GPU synchronization primitives and applications [paper](https://escholarship.org/uc/item/2r01h7gf).
- GPU progress-model work on why blocking GPU algorithms need scheduler-awareness [paper](https://dl.acm.org/doi/10.1145/3297858.3304043).

## 10. Device and Host Ordering Through Streams

### Concept

CUDA streams order work submitted to the same stream. Different streams can execute concurrently unless ordered by events, explicit waits, graph dependencies, or API semantics. Host calls such as `cudaStreamSynchronize`, `cudaDeviceSynchronize`, synchronous copies, and some allocation or memory-management operations can introduce broader synchronization.

This is the main device-scope ordering mechanism for most applications: split a global phase into kernels, connect phases with stream order or events, and let the runtime handle scheduling.

### Canonical Algorithm: Copy-Compute-Overlap Pipeline

A host pipeline uses two streams and two buffers. While one chunk is copied to the GPU, the previous chunk is computed and copied back. Events express cross-stream dependencies.

```cuda
cudaStream_t h2d_stream, compute_stream;
cudaEvent_t copied[2], computed[2];
cudaStreamCreate(&h2d_stream);
cudaStreamCreate(&compute_stream);

for (int chunk = 0; chunk < chunks; ++chunk) {
  int b = chunk & 1;

  cudaMemcpyAsync(d_in[b], h_in + offset(chunk), bytes,
                  cudaMemcpyHostToDevice, h2d_stream);
  cudaEventRecord(copied[b], h2d_stream);

  cudaStreamWaitEvent(compute_stream, copied[b], 0);
  transform_kernel<<<grid, block, 0, compute_stream>>>(d_in[b], d_out[b], count);
  cudaEventRecord(computed[b], compute_stream);

  cudaStreamWaitEvent(h2d_stream, computed[b], 0);
  cudaMemcpyAsync(h_out + offset(chunk), d_out[b], bytes,
                  cudaMemcpyDeviceToHost, h2d_stream);
}
```

The canonical insight is that stream order handles intra-stream sequencing, while events connect only the dependencies that cross streams. This avoids device-wide synchronization inside the loop.

### Memory-Model Notes

A kernel completion in a stream orders its global-memory writes before later work in the same stream. An event recorded after a kernel and waited on by another stream transfers that ordering to the waiting stream. Host observation requires a host synchronization point or a completed asynchronous copy into host-visible memory. Pinned host memory is typically required for true asynchronous host-device copies.

CUDA Graphs express the same dependency structure as a graph rather than repeated host enqueue calls. They preserve dependency ordering between nodes, but graph capture can also preserve accidental synchronization if the original stream sequence contains it.

### Pitfalls

- Assuming operations in different streams are ordered because they are submitted by one host thread.
- Accidentally using the legacy default stream, which can synchronize more broadly than intended depending on the default-stream mode.
- Reusing an event or buffer before all streams that depend on it have passed the dependency.
- Calling `cudaDeviceSynchronize()` in a hot path when an event wait would suffice.
- Capturing a graph from code with hidden allocations or synchronous operations and then expecting full overlap.

### Explore Further

- CUDA Programming Guide asynchronous execution and streams [official docs](https://docs.nvidia.com/cuda/cuda-programming-guide/#asynchronous-concurrent-execution).
- CUDA Runtime API stream and event management [official docs](https://docs.nvidia.com/cuda/cuda-runtime-api/).
- CUDA Graphs programming guide material [official docs](https://docs.nvidia.com/cuda/cuda-programming-guide/).

## 11. Node Scope: Multiple GPUs in One Host

### Concept

Node-level synchronization spans multiple GPUs and often multiple host threads or processes. Ordering can be expressed through CUDA peer access and events, CUDA interprocess communication, or communication libraries such as NCCL. NCCL operations are enqueued into CUDA streams, so their execution is ordered by CUDA stream semantics and by NCCL's collective ordering requirements [official docs](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/).

The main design choice is whether GPUs communicate explicitly through peer memory operations or collectively through a library. For reductions, broadcasts, all-gathers, and reduce-scatters, NCCL is usually the canonical path.

### Canonical Algorithm A: Data-Parallel All-Reduce with NCCL

Each GPU computes local gradients, enqueues an all-reduce on the same stream, then launches an optimizer kernel after the collective.

```cuda
for (int dev = 0; dev < local_gpus; ++dev) {
  cudaSetDevice(dev);
  compute_gradients<<<grid, block, 0, stream[dev]>>>(grad[dev], model[dev]);
}

ncclGroupStart();
for (int dev = 0; dev < local_gpus; ++dev) {
  cudaSetDevice(dev);
  ncclAllReduce(grad[dev], grad[dev], count, ncclFloat,
                ncclSum, comm[dev], stream[dev]);
}
ncclGroupEnd();

for (int dev = 0; dev < local_gpus; ++dev) {
  cudaSetDevice(dev);
  optimizer_step<<<grid, block, 0, stream[dev]>>>(model[dev], grad[dev]);
}
```

The all-reduce is ordered after `compute_gradients` and before `optimizer_step` because all three operations are in the same per-GPU stream. The NCCL group brackets let the host enqueue related operations together, especially when one host thread manages multiple devices.

### Canonical Algorithm B: Peer-to-Peer Staging with Events

GPU 0 produces a halo, GPU 1 waits for an event and copies the halo through peer access before launching its consumer kernel.

```cuda
cudaSetDevice(0);
pack_halo<<<grid, block, 0, s0>>>(halo0, field0);
cudaEventRecord(halo_ready, s0);

cudaSetDevice(1);
cudaStreamWaitEvent(s1, halo_ready, 0);
cudaMemcpyPeerAsync(halo1, 1, halo0, 0, bytes, s1);
consume_halo<<<grid, block, 0, s1>>>(field1, halo1);
```

This pattern uses CUDA stream/event ordering rather than a device-wide host synchronization.

### Memory-Model Notes

NCCL operations are CUDA work items. Their ordering relative to kernels is stream ordering, not a hidden global device barrier. Correctness across ranks also requires all ranks to call matching collectives in compatible order. When one host thread controls multiple GPUs, `ncclGroupStart` and `ncclGroupEnd` are not a memory fence by themselves; they manage collective launch grouping and avoid host-side blocking pathologies.

Peer-to-peer memory visibility depends on peer access support and the transfer API path. A producer event recorded after a kernel is the usual way to order a peer copy after the producer kernel.

### Pitfalls

- Launching collectives in different orders across devices or processes.
- Assuming a NCCL call on stream A orders kernels on stream B without an event or other dependency.
- Destroying or reusing buffers before the stream containing the collective has completed.
- Forgetting `cudaSetDevice` around per-device stream and buffer operations.
- Mixing host threads, CUDA contexts, and NCCL communicators without a consistent launch policy.

### Explore Further

- NCCL user guide, streams and group calls [official docs](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/streams.html).
- NCCL group-call usage and operation ordering [official docs](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/groups.html).
- CUDA Programming Guide multi-device and peer access material [official docs](https://docs.nvidia.com/cuda/cuda-programming-guide/).

## 12. Cross-Node Scope

### Concept

Cross-node GPU synchronization involves network-visible memory, process-level rank coordination, and communication libraries. CUDA alone does not provide a grid barrier across nodes. The usual mechanisms are NCCL collectives, NVSHMEM one-sided communication, and CUDA-aware MPI. These differ in progress, completion, and ordering semantics.

NVSHMEM is especially relevant to kernel-level communication because GPU threads can issue puts, gets, signals, waits, and barriers against symmetric memory. Its ordering vocabulary includes `fence`, `quiet`, barriers, signals, and wait/test operations [official docs](https://docs.nvidia.com/nvshmem/api/). NCCL is the canonical collective path for dense deep-learning communication. CUDA-aware MPI is common in HPC codes where CPU-side MPI orchestration and GPU buffers coexist.

### Canonical Algorithm A: NVSHMEM Halo Exchange

Each processing element owns a subdomain. A kernel writes boundary values into a neighbor's symmetric buffer, ensures completion, and waits for the neighbor's signal before consuming the halo.

```cuda
#include <nvshmem.h>
#include <nvshmemx.h>

__global__ void halo_exchange(float *field, float *remote_halo,
                              uint64_t *signal, int n, int left_pe, int right_pe) {
  int tid = threadIdx.x;

  if (tid == 0) {
    float left_value = field[1];
    float right_value = field[n - 2];

    nvshmem_float_p(remote_halo + 0, left_value, left_pe);
    nvshmem_float_p(remote_halo + 1, right_value, right_pe);

    nvshmem_fence();   // order puts before signals to each PE
    nvshmemx_signal_op(signal, 1, NVSHMEM_SIGNAL_SET, left_pe);
    nvshmemx_signal_op(signal, 1, NVSHMEM_SIGNAL_SET, right_pe);
    nvshmem_quiet();   // wait for issued remote operations to complete

    nvshmem_signal_wait_until(signal, NVSHMEM_CMP_EQ, 1);
  }

  __syncthreads();
  // Threads may now use remote_halo values placed by neighbors.
}
```

The exact signal API variant depends on the NVSHMEM version and whether the operation is host-side or device-side. The algorithmic contract is stable: put payload, order payload before signal, wait for remote completion or notification, then consume.

### Canonical Algorithm B: Distributed Reduction with NCCL Across Nodes

Each rank launches local computation, enqueues a NCCL all-reduce on the rank's CUDA stream, and then launches the dependent kernel in that same stream.

```cuda
compute_local<<<grid, block, 0, stream>>>(x, partial);
ncclAllReduce(partial, global, count, ncclFloat, ncclSum, comm, stream);
normalize<<<grid, block, 0, stream>>>(global, x);
```

This is identical in shape to the node-local all-reduce, but communicator setup and transport span processes and hosts. Correctness still depends on every rank entering matching collectives in the same sequence.

### Canonical Algorithm C: Dissemination or Butterfly Barrier in Symmetric Memory

A dissemination barrier is the canonical all-to-all notification barrier for small metadata. In round `r`, processing element `pe` notifies `(pe + 2^r) mod P` and waits for `(pe - 2^r) mod P`. After `ceil(log2(P))` rounds, every PE has indirectly learned that every other PE entered the barrier. This is the same pattern commonly called a butterfly barrier when drawn as rank-pair exchanges.

The example assumes `round_flags` is symmetric memory with at least `rounds` `uint64_t` slots per PE, and `epoch` is monotonically increasing so flags do not need to be reset between barriers.

```cuda
__device__ void symmetric_dissemination_barrier(uint64_t *round_flags,
                                                uint64_t epoch) {
  int me = nvshmem_my_pe();
  int npes = nvshmem_n_pes();
  int rounds = 0;
  while ((1 << rounds) < npes) ++rounds;

  // If this barrier is meant to order earlier NVSHMEM writes, complete them
  // before telling peers that this PE has arrived.
  nvshmem_quiet();

  for (int r = 0; r < rounds; ++r) {
    int distance = 1 << r;
    int dst = (me + distance) % npes;

    nvshmemx_signal_op(&round_flags[r], epoch, NVSHMEM_SIGNAL_SET, dst);
    nvshmem_signal_wait_until(&round_flags[r], NVSHMEM_CMP_EQ, epoch);
  }
}
```

The algorithm sends `P * ceil(log2(P))` notifications and has `ceil(log2(P))` dependent rounds. It is attractive when every PE can afford the same amount of work and the barrier metadata is tiny. It is less attractive on hierarchical systems where intra-node and inter-node links have very different costs.

### Canonical Algorithm D: Combining Tree Barrier in Symmetric Memory

A tree barrier separates arrival from release. Children notify their parent after their subtrees have arrived; the root then releases the tree downward. This reduces hot-spot pressure compared with a centralized counter and maps well to hierarchical topologies.

The example uses a `fanout`-ary tree. Each PE owns symmetric `arrive[fanout]` slots for child arrival notifications and one symmetric `release` slot for parent release.

```cuda
__device__ void symmetric_tree_barrier(uint64_t *arrive,
                                       uint64_t *release,
                                       int fanout,
                                       uint64_t epoch) {
  int me = nvshmem_my_pe();
  int npes = nvshmem_n_pes();

  nvshmem_quiet();  // complete prior NVSHMEM operations before arrival

  for (int slot = 0; slot < fanout; ++slot) {
    int child = fanout * me + slot + 1;
    if (child < npes) {
      nvshmem_signal_wait_until(&arrive[slot], NVSHMEM_CMP_EQ, epoch);
    }
  }

  if (me != 0) {
    int parent = (me - 1) / fanout;
    int parent_slot = (me - 1) % fanout;
    nvshmemx_signal_op(&arrive[parent_slot], epoch, NVSHMEM_SIGNAL_SET, parent);
    nvshmem_signal_wait_until(release, NVSHMEM_CMP_EQ, epoch);
  }

  for (int slot = 0; slot < fanout; ++slot) {
    int child = fanout * me + slot + 1;
    if (child < npes) {
      nvshmemx_signal_op(release, epoch, NVSHMEM_SIGNAL_SET, child);
    }
  }
}
```

The tree shape can be topology-aware: a two-level tree can first combine within a node, then across node leaders, then release locally. That is usually better than pretending all ranks are equidistant.

### Canonical Algorithm E: Tree All-Reduce in Symmetric Memory

A tree all-reduce uses the same tree barrier skeleton but carries payload. The up-sweep reduces child values into the parent; the down-sweep broadcasts the root's final value. This is canonical for scalar reductions, convergence flags, and small control-plane reductions.

The example reduces one `float`. `value`, `child_values`, `reduce_ready`, and `bcast_ready` are symmetric objects on every PE.

```cuda
__device__ void symmetric_tree_allreduce_sum(float *value,
                                             float *child_values,
                                             uint64_t *reduce_ready,
                                             uint64_t *bcast_ready,
                                             int fanout,
                                             uint64_t epoch) {
  int me = nvshmem_my_pe();
  int npes = nvshmem_n_pes();
  float sum = *value;

  for (int slot = 0; slot < fanout; ++slot) {
    int child = fanout * me + slot + 1;
    if (child < npes) {
      nvshmem_signal_wait_until(&reduce_ready[slot], NVSHMEM_CMP_EQ, epoch);
      sum += child_values[slot];
    }
  }

  if (me != 0) {
    int parent = (me - 1) / fanout;
    int parent_slot = (me - 1) % fanout;
    nvshmem_float_p(&child_values[parent_slot], sum, parent);
    nvshmem_fence();  // payload before notification to parent
    nvshmemx_signal_op(&reduce_ready[parent_slot], epoch, NVSHMEM_SIGNAL_SET, parent);

    nvshmem_signal_wait_until(bcast_ready, NVSHMEM_CMP_EQ, epoch);
    sum = *value;  // parent wrote the final result into this PE's value
  } else {
    *value = sum;
  }

  for (int slot = 0; slot < fanout; ++slot) {
    int child = fanout * me + slot + 1;
    if (child < npes) {
      nvshmem_float_p(value, sum, child);
      nvshmem_fence();  // result before release notification
      nvshmemx_signal_op(bcast_ready, epoch, NVSHMEM_SIGNAL_SET, child);
    }
  }
}
```

For vectors, each edge usually transfers a contiguous segment, and production implementations pipeline chunks so the network and reduction work overlap. For large tensors, use NCCL, NVSHMEM collectives, or MPI collectives unless a custom reduction has a specific semantic need.

### Canonical Algorithm F: Recursive-Doubling All-Reduce in Symmetric Memory

Recursive doubling is the reduction analogue of the butterfly barrier. On each round, every PE exchanges its current partial with a partner `pe xor 2^r` and combines the received value. After `log2(P)` rounds on a power-of-two PE count, every PE has the complete reduction.

The example reduces one scalar and uses one inbox and flag per round. Non-power-of-two process counts require a prologue/epilogue or a different algorithm such as reduce-scatter plus all-gather.

```cuda
__device__ void symmetric_recursive_doubling_sum(float *value,
                                                 float *round_inbox,
                                                 uint64_t *round_ready,
                                                 uint64_t epoch) {
  int me = nvshmem_my_pe();
  int npes = nvshmem_n_pes();  // assume power of two for this compact form
  float sum = *value;

  for (int r = 0; (1 << r) < npes; ++r) {
    int partner = me ^ (1 << r);

    nvshmem_float_p(&round_inbox[r], sum, partner);
    nvshmem_fence();  // inbox payload before ready flag
    nvshmemx_signal_op(&round_ready[r], epoch, NVSHMEM_SIGNAL_SET, partner);

    nvshmem_signal_wait_until(&round_ready[r], NVSHMEM_CMP_EQ, epoch);
    sum += round_inbox[r];
  }

  *value = sum;
}
```

Recursive doubling has low latency for small messages because all PEs communicate every round. For large arrays, it sends increasingly large partials unless paired with segmentation or replaced by reduce-scatter plus all-gather, the pattern used by many optimized all-reduce implementations.

### Memory-Model Notes

NVSHMEM `fence` orders previously issued operations before later operations to the same destination PE; `quiet` waits for completion of previously issued operations according to the API contract. Barriers provide collective synchronization among PEs. Signals decouple data movement from notification, but the signal must be ordered after the payload. CUDA-aware MPI often needs explicit stream synchronization or MPI extensions that understand CUDA streams; otherwise the host may call MPI before the GPU has produced the buffer.

Remote completion and local completion are different. A GPU thread may have issued a network operation; that does not mean the remote PE can read the data, nor that all PEs have reached a global phase.

For hand-built symmetric-memory primitives, epoching is part of correctness. Reusing a flag value without a monotonically increasing epoch creates an ABA race where a wait observes a stale signal from a prior barrier or collective. If ordinary CUDA stores produce the payload and NVSHMEM operations only publish the notification, add the documented CUDA/NVSHMEM ordering point between the local store and the remote notification; using NVSHMEM puts for the payload and `fence` before the signal is usually easier to reason about.

### Pitfalls

- Using `fence` when `quiet` or a barrier is required for completion.
- Signaling a neighbor before the payload is ordered.
- Forgetting symmetric allocation requirements in NVSHMEM.
- Reusing symmetric-memory flags without epochs or without double-buffered rounds.
- Implementing a flat butterfly across a topology where node-local and cross-node links have very different latency.
- Using recursive doubling for large tensors when reduce-scatter plus all-gather or NCCL's tuned algorithms are more appropriate.
- Assuming GPU-side communication makes progress while all SMs are occupied by a spin loop that prevents the communication engine or helper work from advancing.
- Calling MPI on a GPU buffer from the host without synchronizing the stream that produced it, unless the MPI implementation has a documented stream-aware path.
- Mismatching collective calls across ranks; this is a correctness bug, not just a performance issue.

### Explore Further

- NVSHMEM API documentation for memory ordering, fence, quiet, barriers, and signals [official docs](https://docs.nvidia.com/nvshmem/api/).
- NVSHMEM programming model and CUDA kernel communication [official docs](https://docs.nvidia.com/nvshmem/).
- NCCL user guide for multi-node collectives and streams [official docs](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/).
- MPI and CUDA-aware communication surveys in HPC literature [paper](https://dl.acm.org/doi/10.1145/2807591.2807677).
- Mellor-Crummey and Scott on scalable tree and dissemination barriers [paper](https://doi.org/10.1145/103727.103729).
- Rabenseifner on all-reduce decomposition through reduce-scatter and all-gather [paper](https://doi.org/10.1007/978-3-540-24685-5_1).

## 13. Subtle CUDA Memory-Model Aspects

This section collects issues that cut across scopes.

### 13.1 Scope Mismatch

The producer and consumer must be inside the scope of the ordering primitive. A block-scope atomic or fence is insufficient when another block consumes the data. A device-scope atomic is insufficient for a host or remote PE unless the allocation and operation participate in system scope. System-scope operations can be expensive; use them only when the consumer is outside the GPU device scope.

Canonical algorithm: choose the smallest scope that contains all consumers in a work queue.

```cuda
// Same block: block scope is enough.
cuda::atomic<int, cuda::thread_scope_block> block_flag;

// Same GPU, different blocks: device scope.
cuda::atomic<int, cuda::thread_scope_device> device_flag;

// Host, peer GPU, or other system observer: system scope when supported.
cuda::atomic<int, cuda::thread_scope_system> system_flag;
```

Pitfall: increasing memory order from acquire/release to sequential consistency does not fix a scope mismatch.

### 13.2 Fence Versus Visibility Versus Notification

A fence orders the calling thread's memory operations. It does not tell another thread that work is ready. A notification primitive usually needs an atomic flag, event, signal, barrier, or stream dependency.

Canonical algorithm: fence-plus-flag in legacy code.

```cuda
payload[idx] = value;
__threadfence();             // order payload before flag at device scope
atomicExch(&ready[idx], 1);  // notify
```

Prefer release/acquire atomics when they express the pattern directly:

```cuda
payload[idx] = value;
ready[idx].store(1, cuda::memory_order_release);
```

Pitfall: adding `__threadfence()` after the flag store orders the wrong thing for message passing.

### 13.3 Atomics Are Not Barriers

Atomics serialize operations to one address and can carry memory ordering. They do not imply that all threads have reached a phase boundary. Algorithms such as global queues, ticket locks, and counters can be correct with atomics; global phase barriers usually need kernel boundaries, cooperative-grid sync, or library collectives.

Canonical algorithm: a global work queue.

```cuda
int ticket = work_head.fetch_add(1, cuda::memory_order_relaxed);
if (ticket < total_work) {
  process(work_items[ticket]);
}
```

This does not need a barrier because each ticket is independent. A later phase that reads all processed outputs does need a separate phase boundary.

Pitfall: using atomics to build a reusable global barrier can deadlock under normal CUDA scheduling.

### 13.4 Acquire/Release Is Directional

Acquire/release works when the acquire reads from, or is otherwise ordered after, the release sequence that published the data. It is not magic global freshness.

Canonical algorithm: ring-buffer slot publication.

```cuda
slot.data = value;
slot.sequence.store(seq + 1, cuda::memory_order_release);

while (slot.sequence.load(cuda::memory_order_acquire) != seq + 1) {}
consume(slot.data);
```

Pitfall: loading a different flag from the one the producer released may not order the payload.

### 13.5 Async and Proxy Ordering

Asynchronous copies, tensor-memory operations, texture/surface paths, and other specialized memory routes can have ordering rules beyond ordinary global loads and stores. PTX models this through operation kinds and proxy fences. CUDA C++ abstractions such as `cuda::pipeline` are safer than hand-rolled inline PTX unless a kernel author is deliberately targeting an ISA feature.

Canonical algorithm: async tile load before compute.

```cuda
pipe.producer_acquire();
cuda::memcpy_async(dst_smem, src_gmem, bytes, pipe);
pipe.producer_commit();
pipe.consumer_wait();
__syncthreads();
compute_from(dst_smem);
pipe.consumer_release();
```

Pitfall: using a normal block barrier where the async copy requires a pipeline or async barrier wait.

### 13.6 Memory Synchronization Domains

CUDA memory synchronization domains reduce unnecessary interference between independent traffic classes on architectures that support the feature. The practical idea is that broad fences can wait on more memory traffic than the algorithm logically needs; domains help isolate fence effects [official docs](https://docs.nvidia.com/cuda/cuda-programming-guide/).

Canonical algorithm: separate communication and compute streams with domain-aware attributes where supported.

```cuda
// Pseudocode: actual attribute setup depends on CUDA runtime version and device support.
cudaLaunchAttribute attrs[1] = {};
attrs[0].id = cudaLaunchAttributeMemSyncDomain;
attrs[0].val.memSyncDomain = cudaLaunchMemSyncDomainRemote;
```

Pitfall: treating domains as a correctness mechanism. They are a performance/isolation tool layered under already-correct synchronization.

### 13.7 Texture, Surface, and Read-Only Paths

Texture and surface memory paths are not a substitute for synchronization. A common safe pattern is phase separation: write global memory in one kernel, then read through a texture or read-only path in a later kernel or after a documented synchronization boundary.

Canonical algorithm: two-phase image processing.

```cuda
write_image<<<grid, block, 0, stream>>>(linear_storage);
read_texture_phase<<<grid, block, 0, stream>>>(texture_object, output);
```

The stream order between kernels is the synchronization. Within one kernel, do not assume a store through one path is immediately visible through another cache path unless the CUDA documentation for that path explicitly permits it.

## 14. Literature and Pattern Map

The table below connects the practical patterns to source families. It is not exhaustive; it is a map of high-signal sources for implementation work.

| Pattern family | Primary source type | Useful sources |
|---|---|---|
| CUDA block, warp, grid, cluster synchronization | official docs/blog | CUDA Programming Guide [official docs](https://docs.nvidia.com/cuda/cuda-programming-guide/), Cooperative Groups blog [official blog](https://developer.nvidia.com/blog/cooperative-groups/), warp-level primitives blog [official blog](https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/) |
| CUDA C++ memory model and scoped atomics | official docs | libcu++ memory model [official docs](https://nvidia.github.io/cccl/libcudacxx/extended_api/memory_model.html), libcu++ atomics [official docs](https://nvidia.github.io/cccl/libcudacxx/extended_api/synchronization_primitives/atomic.html) |
| PTX memory consistency | official docs and papers | PTX ISA [official docs](https://docs.nvidia.com/cuda/parallel-thread-execution/), formal GPU memory-model literature [paper](https://dl.acm.org/doi/10.1145/3297858.3304043) |
| In-kernel global synchronization | papers and official docs | Zhang, Wahib, Zhang, and Matsuoka synchronization-method evaluation [paper](https://ieeexplore.ieee.org/document/8091043), CUDA cooperative groups [official docs](https://docs.nvidia.com/cuda/cuda-programming-guide/#cooperative-groups) |
| GPU locks, barriers, queues | papers | Stuart and Owens GPU synchronization primitives [paper](https://escholarship.org/uc/item/2r01h7gf), GPU progress model work [paper](https://dl.acm.org/doi/10.1145/3297858.3304043) |
| Communication collectives | official docs | NCCL user guide [official docs](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/), NCCL streams [official docs](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/streams.html) |
| One-sided GPU communication | official docs | NVSHMEM API [official docs](https://docs.nvidia.com/nvshmem/api/), NVSHMEM programming guide [official docs](https://docs.nvidia.com/nvshmem/) |
| Symmetric-memory barriers and reductions | papers and official docs | Mellor-Crummey and Scott scalable barriers [paper](https://doi.org/10.1145/103727.103729), Rabenseifner all-reduce decomposition [paper](https://doi.org/10.1007/978-3-540-24685-5_1), NVSHMEM API [official docs](https://docs.nvidia.com/nvshmem/api/) |
| CUDA bug patterns | papers | Empirical CUDA bug studies and concurrency-bug taxonomies [paper](https://dl.acm.org/doi/10.1145/3133956.3134053) |

## 15. Operational Decision Guide

Use this checklist when choosing a primitive:

1. Identify producers and consumers. Are they lanes, threads in a block, blocks in a cluster, all blocks in a grid, host work, peer GPUs, or remote ranks?
2. Identify the communication medium. Is the payload in registers, shared memory, distributed shared memory, global memory, host memory, symmetric memory, or a communication library buffer?
3. Pick the smallest scope containing all consumers.
4. Decide whether you need a barrier, a one-way notification, or only memory ordering.
5. Prefer structured primitives over hand-rolled protocols: cooperative groups over ad hoc global barriers, `cuda::atomic` over volatile flags, streams/events over device-wide host synchronization, NCCL/NVSHMEM collectives over custom cross-node polling.
6. Check progress. Any algorithm where resident blocks spin while waiting for non-resident blocks is suspect.
7. Check masks and uniformity. Warp masks must match participating lanes; block, cluster, and grid barriers must be reached by all participants.
8. Check stream order. If work is in different streams, add an event or graph dependency.
9. Check completion semantics. For communication, know whether the primitive means issued, locally complete, remotely visible, or globally synchronized.
10. Test under low occupancy, high occupancy, multiple stream modes, and multiple GPU counts. Synchronization bugs often appear only when scheduling changes.

## 16. Common Anti-Patterns

| Anti-pattern | Why it fails | Safer replacement |
|---|---|---|
| Spin-based global barrier inside an ordinary kernel | Non-resident blocks may be required to release resident blocks | kernel boundary or cooperative launch `grid.sync()` |
| `volatile` flag without atomic order | Does not provide a complete inter-thread memory-ordering contract | `cuda::atomic` release/acquire |
| Full warp mask after divergence | Names lanes that may not execute the intrinsic | compute and pass the actual participation mask |
| Conditional `__syncthreads()` | Deadlocks if branch is not uniform | restructure so all block threads reach the barrier or use a smaller cooperative group |
| Device-wide synchronization after every copy or kernel | Serializes unrelated work | stream order and events |
| NCCL collective in a side stream without dependency | Dependent kernels may race with communication | enqueue dependent work in the same stream or use events |
| NVSHMEM signal before payload ordering | Remote PE may observe notification before data is usable | payload, `fence`, signal, `quiet` or wait as required |
| Custom symmetric-memory barrier without epochs | Later waits can consume stale notifications from an earlier round | monotonic epochs or double-buffered flag storage |
| System-scope fence for device-only communication | Correct but often unnecessarily expensive | block or device scope |

## 17. Verification Notes

This document is a research and engineering survey, not a conformance test suite. Code snippets are intentionally short and omit production details such as error checking, architecture guards, communicator setup, bounds-specialized tuning, and launch-configuration validation unless those details are central to the synchronization pattern.

Claims about CUDA primitives are grounded in the CUDA Programming Guide, PTX ISA, libcu++ documentation, NCCL user guide, and NVSHMEM documentation current to the source cutoff. Academic references are used to explain broader design tradeoffs, known hazards, and progress or memory-model subtleties.

## 18. Bibliography

- NVIDIA, CUDA C++ Programming Guide, v13.2 [official docs](https://docs.nvidia.com/cuda/cuda-programming-guide/).
- NVIDIA, Parallel Thread Execution ISA, v9.2 [official docs](https://docs.nvidia.com/cuda/parallel-thread-execution/).
- NVIDIA, libcu++ extended API: memory model, atomics, barriers, and pipelines [official docs](https://nvidia.github.io/cccl/libcudacxx/).
- Mark Harris and Kyrylo Perelygin, Cooperative Groups: Flexible CUDA Thread Programming [official blog](https://developer.nvidia.com/blog/cooperative-groups/).
- Yuan Lin and Vinod Grover, Using CUDA Warp-Level Primitives [official blog](https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/).
- NVIDIA, NCCL User Guide [official docs](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/).
- NVIDIA, NVSHMEM Documentation [official docs](https://docs.nvidia.com/nvshmem/).
- Guoyang Zhang, Mohamed Wahib, Haohuan Fu, and Satoshi Matsuoka, A Study of Single and Multiple Device Synchronization Methods in NVIDIA GPUs [paper](https://ieeexplore.ieee.org/document/8091043).
- Jeff A. Stuart and John D. Owens, Efficient Synchronization Primitives for GPUs [paper](https://escholarship.org/uc/item/2r01h7gf).
- John M. Mellor-Crummey and Michael L. Scott, Algorithms for Scalable Synchronization on Shared-Memory Multiprocessors [paper](https://doi.org/10.1145/103727.103729).
- Rolf Rabenseifner, Optimization of Collective Reduction Operations [paper](https://doi.org/10.1007/978-3-540-24685-5_1).
- Tyler Sorensen, Mark Batty, Ganesh Gopalakrishnan, and others, GPU memory consistency and progress-model work [paper](https://dl.acm.org/doi/10.1145/3297858.3304043).
- Yuxi Liu and Shan Lu, empirical studies of CUDA bugs and concurrency hazards [paper](https://dl.acm.org/doi/10.1145/3133956.3134053).
