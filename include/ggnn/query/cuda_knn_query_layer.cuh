/* Copyright 2019 ComputerGraphics Tuebingen. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
// Authors: Fabian Groh, Lukas Ruppert, Patrick Wieschollek, Hendrik P.A. Lensch

#ifndef INCLUDE_GGNN_QUERY_CUDA_KNN_QUERY_LAYER_CUH_
#define INCLUDE_GGNN_QUERY_CUDA_KNN_QUERY_LAYER_CUH_

#include <algorithm>
#include <limits>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

// #include "ggnn/cache/cuda_knn_sorted_buffer_cache.cuh"
#include "ggnn/cache/cuda_simple_knn_cache.cuh"
#include "ggnn/utils/cuda_knn_constants.cuh"
#include "ggnn/utils/cuda_knn_utils.cuh"

template <typename T>
__global__ void query(const T kernel) {
  kernel();
}

template <DistanceMeasure measure, typename ValueT, typename KeyT, int D, int DBA, int DA, int K,
          int KF, int KQuery, int S, int BLOCK_DIM_X, typename BaseT = ValueT,
          typename BAddrT = KeyT, typename GAddrT = KeyT,
          bool DIST_STATS = false, bool OVERFLOW_STATS = false,
          int MAX_ITERATIONS = 400, int CACHE_SIZE = 512, int SORTED_SIZE = 256,
          bool WRITE_DISTS = false>
struct QueryKernel {
  static constexpr int KL = K - KF;
  static constexpr int KS = (K > S) ? K : S;

  static constexpr int BEST_SIZE = KQuery; // KQuery
  static constexpr int VISITED_SIZE = CACHE_SIZE - SORTED_SIZE;
  static constexpr int PRIOQ_SIZE = SORTED_SIZE - BEST_SIZE;

  static constexpr int ITERATIONS_FOR_K = (K + BLOCK_DIM_X - 1) / BLOCK_DIM_X;
  static constexpr int ITERATIONS_FOR_S = (S + BLOCK_DIM_X - 1) / BLOCK_DIM_X;
  static constexpr int ITERATIONS_FOR_KS = (KS + BLOCK_DIM_X - 1) / BLOCK_DIM_X;

  static constexpr int ATTRS_PER_THREAD = (DA - 1) / BLOCK_DIM_X + 1;

  typedef SimpleKNNCache<measure, ValueT, KeyT, KQuery, D, DA, BLOCK_DIM_X,
                         VISITED_SIZE, PRIOQ_SIZE, BEST_SIZE, BaseT, BAddrT,
                         DIST_STATS, OVERFLOW_STATS>
      Cache;

  void launch(const cudaStream_t stream = 0) {
    VLOG(1) << "QueryKernel -- BLOCK_DIM_X: " << BLOCK_DIM_X
            << " || KQuery: " << KQuery << " MAX_ITERATIONS: " << MAX_ITERATIONS
            << " CACHE_SIZE: " << CACHE_SIZE << " SORTED_SIZE: " << SORTED_SIZE
            << " || BEST_SIZE: " << BEST_SIZE << " PRIOQ_SIZE: " << PRIOQ_SIZE
            << " VISITED_SIZE: " << VISITED_SIZE;
    //111111111111111111111111111111111111111111111111111111111111
    query<<<N, BLOCK_DIM_X, 0, stream>>>((*this));
  }

  __device__ __forceinline__ void operator()() const {
    int num_dist = 0;
    for (int blockid = blockIdx.x; blockid < N; blockid += gridDim.x) {
      clock_t clc_kernel_start, clc_kernel_end;
      clock_t clc_fetch_start, clc_fetch_end;
      clock_t clc_nlist_start, clc_nlist_end;
      int clc_kernel = 0;
      int clc_fetch = 0;
      int clc_nlist = 0;
      clc_kernel_start = clock();

      const int ATTRS_SIZE = ceil(log2(DBA));
      const int ATTRS_PER_INT = (sizeof(unsigned) * 8 / ATTRS_SIZE);
      const float xi =
          (measure == Euclidean)
              ? (d_nn1_stats[1] * d_nn1_stats[1]) * c_tau_query * c_tau_query
              : d_nn1_stats[1] * c_tau_query;

      const KeyT n = N_offset + static_cast<int>(blockid);

      Cache cache(d_base, d_query, n, xi);
      __syncthreads();

      int r_query_attr[ATTRS_PER_THREAD];
      for (int item = 0; item < ATTRS_PER_THREAD; ++item) {
        const int read_dim = item * BLOCK_DIM_X + threadIdx.x;
        if (read_dim < DA) {
          r_query_attr[item] = *(d_query_attr + n*DA + read_dim);
        }
      }
      __syncthreads();

      const int start_attr = *(d_query_attr + n*DA);
      __shared__ KeyT s_knn[KS];
      for (int i = 0; i < ITERATIONS_FOR_S; ++i) {
        const int s = i * BLOCK_DIM_X + threadIdx.x;
        if (s < S) s_knn[s] = d_graph[N_base * K + start_attr * S + s];
      }
      __syncthreads();

      __shared__ int s_att[KS];
      for (int i = 0; i < ITERATIONS_FOR_S; ++i) {
        const int s = i * BLOCK_DIM_X + threadIdx.x;
        if (s < S) s_att[s] = *(d_base_attr + static_cast<BAddrT>(s_knn[s]));
      }
      __syncthreads();

      clc_fetch_start = clock();
      cache.fetch(s_knn, s_att, r_query_attr, nullptr, S);
      __syncthreads();
      clc_fetch_end = clock();
      clc_fetch += (int)(clc_fetch_end - clc_fetch_start);
      

      for (int ite = 0; ite < MAX_ITERATIONS; ++ite) {
        __syncthreads();

        if (measure == Euclidean) {
          cache.xi = min(xi, cache.s_dists[0] * c_tau_query * c_tau_query);
        } else if (measure == Cosine) {
          cache.xi = min(xi, cache.s_dists[0] * c_tau_query);
        }
        // const KeyT anchor1 = cache.pop1();
        const KeyT anchor = cache.pop();
        if (anchor == Cache::EMPTY_KEY) {
          break;
        }
        __syncthreads();

        clc_nlist_start = clock();
        unsigned shift = (ATTRS_PER_INT - 1 - anchor % ATTRS_PER_INT) * ATTRS_SIZE;
        unsigned mask = pow(2, ATTRS_SIZE)-1;
        for (int i = 0; i < ITERATIONS_FOR_K; ++i) {
          const int k = i * BLOCK_DIM_X + threadIdx.x;
          if (k < K) s_knn[k] = d_graph[static_cast<GAddrT>(anchor) * K + k];
          if (k < K) {
            unsigned attr_tmp = d_per_attr[static_cast<GAddrT>(anchor) / ATTRS_PER_INT * K + k];
            s_att[k] = (attr_tmp & (mask<<shift))>>shift;
          }
          // if (k < K) s_att[k] = *(d_base_attr + static_cast<BAddrT>(s_knn[k]));
        }
        __syncthreads();
        clc_nlist_end = clock();
        clc_nlist += (int)(clc_nlist_end - clc_nlist_start);
        
        clc_fetch_start = clock();
        cache.fetch(s_knn, s_att, r_query_attr, nullptr, K);
        clc_fetch_end = clock();
        clc_fetch += (int)(clc_fetch_end - clc_fetch_start);
      }  // end iterations

      __syncthreads();
      cache.write_best(d_query_results, n * num_parts + part, KQuery,
                      part * N_base);

      if (WRITE_DISTS) {
        if (threadIdx.x < KQuery) {
          d_query_results_dists[(n * num_parts + part) * KQuery + threadIdx.x] =
              cache.s_dists[threadIdx.x];
        }
      }

      if (DIST_STATS) {
        if (!threadIdx.x) {
          d_dist_stats[n] = cache.get_dist_stats();
        }
      }

      clc_kernel_end = clock();
      clc_kernel = (int)(clc_kernel_end - clc_kernel_start);
      // if (!blockIdx.x && !threadIdx.x) printf("c_tau_query > %f    %d    %d    %d    %d    %d    %d    %d\n", 
      //                                         c_tau_query, clc_kernel, clc_nlist, clc_fetch, cache.clc_re, cache.clc_filter, cache.clc_dist, cache.clc_push);
      num_dist += cache.num_dist;
    }
    // if (!blockIdx.x && !threadIdx.x) printf("number of distance: %d\n", num_dist / 10000); 
  }

  const BaseT* d_base;        // [Nall,D]
  const BaseT* d_query;       // [Nq,D]
  const int* d_base_attr;        //
  const int* d_query_attr;       //
  const KeyT* d_translation;  // [Nall]

  const KeyT* d_graph;            // [Nall,K]
  const unsigned* d_per_attr;
  KeyT* d_query_results;          // [Nq,KQuery]
  ValueT* d_query_results_dists;  // [Nq,KQuery]

  const float* d_nn1_stats;  // [sum,max]

  int* d_dist_stats;          // [Nq]

  int N;         // number of points to query for -> Nq
  int N_offset;  // gpu offset in N
  int N_base;    // number of points in the dataset

  int num_parts {1};
  int part      {0};
};

#endif  // INCLUDE_GGNN_QUERY_CUDA_KNN_QUERY_LAYER_CUH_
