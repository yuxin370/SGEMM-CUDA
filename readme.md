# CUDA Matrix Multiplication Optimization (SGEMM)


## Objective
Implement a CUDA-based SGEMM (Single-precision General Matrix Multiply) to explore CUDA programming models and optimize memory access patterns using shared memory and tiling techniques.


## Environment
- **Hardware:** School server with NVIDIA RTX 2080 Ti GPU
- **CUDA Version:** 11.0+

## Experiment Design

### Baseline Implementation Analysis
- **Thread Mapping:** Each thread computes one element of the output matrix `C[i][j]`.
- **Memory Access:** Directly reads data from global memory for every computation step.
- **Limitation:** Redundant global memory accesses lead to low bandwidth utilization.

### Optimization Strategy
   - Divide matrices into `TILE_WIDTH x TILE_WIDTH` blocks.
   - Load mini-sub-matrices fit to shared memory and registers to reduce global memory access.
   - Synchronize threads (`__syncthreads()`) after loading data to shared memory.


## Performance Results

### Performance Comparison
#### Baseline (GFLOPS/s)
| Matrix Size | 512       | 1024       | 2048        |
|-------------|-----------|------------|-------------|
| GFLOPS/s    | 667.28    | 1,134.14   | 1,091.68    |
| GB/s        | 2.61      | 2.22       | 1.07        |

#### Optimized (GFLOPS/s)
| Tile Size \ Matrix | 512       | 1024       | 2048        |
|--------------------|-----------|------------|-------------|
| **16x16**          | 834.19    | 1,342.43   | 1,421.77    |
| **32x32**          | 1,401.59  | 1,452.09   | 1,515.54    |

### Key Observations
1. **Speedup:** Optimized version achieves **~1.4x higher GFLOPS/s** compared to baseline.
2. **Tile Size Impact:** Larger tiles (32x32) show better performance due to increased parallelism.
3. **Bandwidth Trend:** Bandwidth utilization decreases with larger matrix sizes in both versions.


## Conclusion
- **Optimization Effectiveness:** Shared memory tiling significantly reduces global memory access latency.
- **Trade-offs:** Larger tiles improve compute intensity but may increase shared memory contention.
- **opportunities:** Implement double-buffering to overlap computation and memory transfers: By partitioning the shared memory into two sections: one section for loading data for the next iteration and another section for current computations, thereby implementing data prefetching to hide memory access latency.
