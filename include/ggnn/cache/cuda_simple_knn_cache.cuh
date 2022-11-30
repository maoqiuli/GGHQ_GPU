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

#ifndef INCLUDE_GGNN_CACHE_CUDA_SIMPLE_KNN_CACHE_CUH_
#define INCLUDE_GGNN_CACHE_CUDA_SIMPLE_KNN_CACHE_CUH_

#include <cuda.h>
#include <cuda_runtime.h>

#include <cub/cub.cuh>
#include <limits>

#include "ggnn/utils/cuda_knn_distance.cuh"
#include "ggnn/utils/cuda_knn_utils.cuh"

template <DistanceMeasure measure,
          typename ValueT, typename KeyT, int KQuery, int D, int DA, int BLOCK_DIM_X,
          int VISITED_SIZE = 256, int PRIOQ_SIZE = 128, int BEST_SIZE = 32,
          typename BaseT = ValueT, typename BAddrT = KeyT,
          bool DIST_STATS = false, bool OVERFLOW_STATS = false>
struct SimpleKNNCache {
  static constexpr KeyT EMPTY_KEY = (KeyT)-1;
  static constexpr ValueT EMPTY_DIST = std::numeric_limits<ValueT>::infinity();

 private:
  static constexpr int CACHE_SIZE = BEST_SIZE + PRIOQ_SIZE + VISITED_SIZE;
  static constexpr int SORTED_SIZE = BEST_SIZE + PRIOQ_SIZE;

  static constexpr int DIST_ITEMS_PER_THREAD = (D - 1) / BLOCK_DIM_X + 1;
  static constexpr int BEST_ITEMS_PER_THREAD =
      (BEST_SIZE - 1) / BLOCK_DIM_X + 1;
  static constexpr int PRIOQ_ITEMS_PER_THREAD =
      (PRIOQ_SIZE - 1) / BLOCK_DIM_X + 1;

  static constexpr int CACHE_ITEMS_PER_THREAD =
      (CACHE_SIZE - 1) / BLOCK_DIM_X + 1;
  static constexpr int SORTED_ITEMS_PER_THREAD =
      (SORTED_SIZE - 1) / BLOCK_DIM_X + 1;

  static constexpr int BEST_END = BEST_SIZE - 1;

  typedef Distance<measure, ValueT, KeyT, D, DA, BLOCK_DIM_X, BaseT, BAddrT> Distance;

  union SyncTempStorage {
    KeyT cache;
    bool flag;
  };

 public:
  KeyT* s_cache;
  ValueT* s_dists;
  // KeyT* s_cache1;
  // ValueT* s_dists1;

  clock_t clc_dist_start, clc_dist_end;
  clock_t clc_push_start, clc_push_end;
  clock_t clc_re_start, clc_re_end;
  int clc_dist = 0;
  int clc_push = 0;
  int clc_re = 0;
  int num_dist = 0;

  int& s_prioQ_head;
  int& s_visited_head;
  int& s_overflow_counter;

  SyncTempStorage& s_sync;

  ValueT xi;

  Distance rs_dist_calc;

  const int* d_base_attr;
  const int* d_query_attr;

  const bool is_search{false};

  //# threadIdx.x == 0 stats registers only
  int dist_calc_counter;

  __device__ __forceinline__ void initSharedStorage() {
    __shared__ KeyT s_cache_tmp[CACHE_SIZE];
    __shared__ ValueT s_dists_tmp[SORTED_SIZE];

    s_cache = reinterpret_cast<KeyT*>(s_cache_tmp);
    s_dists = reinterpret_cast<ValueT*>(s_dists_tmp);

    // __shared__ KeyT s_cache_tmp1[CACHE_SIZE];
    // __shared__ ValueT s_dists_tmp1[SORTED_SIZE];

    // s_cache1 = reinterpret_cast<KeyT*>(s_cache_tmp1);
    // s_dists1 = reinterpret_cast<ValueT*>(s_dists_tmp1);
  }

  __device__ __forceinline__ SyncTempStorage& SyncPrivateTmpStorage() {
    __shared__ SyncTempStorage s_sync_tmp;
    return s_sync_tmp;
  }

  __device__ __forceinline__ int& PrioQRingPrivateTmpStorage() {
    __shared__ int s_prioQ_head_tmp;
    return s_prioQ_head_tmp;
  }

  __device__ __forceinline__ int& CacheRingPrivateTmpStorage() {
    __shared__ int s_visited_head_tmp;
    return s_visited_head_tmp;
  }

  __device__ __forceinline__ int& OverflowPrivateTmpStorage() {
    __shared__ int s_overflow_tmp;
    return s_overflow_tmp;
  }

  __device__ __forceinline__ void init() {
    for (int i = threadIdx.x; i < CACHE_SIZE; i += BLOCK_DIM_X) {
      s_cache[i] = EMPTY_KEY;
      // s_cache1[i] = EMPTY_KEY;
    }
    for (int i = threadIdx.x; i < SORTED_SIZE; i += BLOCK_DIM_X) {
      s_dists[i] = EMPTY_DIST;
      // s_dists1[i] = EMPTY_DIST;
    }
    if (DIST_STATS && !threadIdx.x) dist_calc_counter = 0;
    if (OVERFLOW_STATS && !threadIdx.x) s_overflow_counter = 0;
    if (!threadIdx.x) {
      s_prioQ_head = 0;
      s_visited_head = 0;
    }
    __syncthreads();
  }

  __device__ __forceinline__ SimpleKNNCache(const BaseT* d_base, const KeyT n,
                                            const ValueT xi_criteria)
      : xi(xi_criteria),
        s_prioQ_head(PrioQRingPrivateTmpStorage()),
        s_visited_head(CacheRingPrivateTmpStorage()),
        s_overflow_counter(OverflowPrivateTmpStorage()),
        s_sync(SyncPrivateTmpStorage()),
        rs_dist_calc(d_base, n) {
    initSharedStorage();
    init();
  }

  __device__ __forceinline__ SimpleKNNCache(const BaseT* d_base, const int* d_base_attr, const KeyT n,
                                            const ValueT xi_criteria)
      : xi(xi_criteria),
        s_prioQ_head(PrioQRingPrivateTmpStorage()),
        s_visited_head(CacheRingPrivateTmpStorage()),
        s_overflow_counter(OverflowPrivateTmpStorage()),
        s_sync(SyncPrivateTmpStorage()),
        d_base_attr(d_base_attr),
        rs_dist_calc(d_base, d_base_attr, n) {
    initSharedStorage();
    init();
  }

  __device__ __forceinline__ SimpleKNNCache(const BaseT* d_base,
                                            const BaseT* d_query, const KeyT n,
                                            const ValueT xi_criteria)
      : xi(xi_criteria),
        s_prioQ_head(PrioQRingPrivateTmpStorage()),
        s_visited_head(CacheRingPrivateTmpStorage()),
        s_overflow_counter(OverflowPrivateTmpStorage()),
        s_sync(SyncPrivateTmpStorage()),
        rs_dist_calc(d_base, d_query, n){
    initSharedStorage();
    init();
  }

  __device__ __forceinline__ SimpleKNNCache(const BaseT* d_base,
                                            const BaseT* d_query, 
                                            const int* d_base_attr,
                                            const int* d_query_attr,
                                            const KeyT n,
                                            const ValueT xi_criteria)
      : xi(xi_criteria),
        s_prioQ_head(PrioQRingPrivateTmpStorage()),
        s_visited_head(CacheRingPrivateTmpStorage()),
        s_overflow_counter(OverflowPrivateTmpStorage()),
        s_sync(SyncPrivateTmpStorage()),
        rs_dist_calc(d_base, d_query, d_base_attr, d_query_attr, n),
        is_search(true),
        d_base_attr(d_base_attr),
        d_query_attr(d_query_attr){
    initSharedStorage();
    init();
  }

  __device__ __forceinline__ bool criteria(ValueT dist) {
    if (dist < s_dists[KQuery - 1] + xi) return true;
    return false;
  }

  __device__ __forceinline__ bool is_end(int tid) {
    const int prev_prioQ_ring =
        (s_prioQ_head - 1 < 0) ? PRIOQ_SIZE - 1 : s_prioQ_head - 1;
    return tid == BEST_END || tid == BEST_SIZE + prev_prioQ_ring;
  }

  __device__ __forceinline__ void push(const KeyT key, const ValueT dist) {
    __syncthreads();
    clock_t clc_push_1, clc_push_2, clc_push_3, clc_push_4, clc_push_5, clc_push_6, clc_push_7, clc_push_8;
    int clc_push_add_1 = 0, clc_push_add_2 = 0, clc_push_add_3 = 0;
    clc_push_1 = clock();

    // Register for insertion in best and prioq
    KeyT r_cache[SORTED_ITEMS_PER_THREAD];
    ValueT r_dists[SORTED_ITEMS_PER_THREAD];

    int r_write_item_best = -1;
    int r_write_item_prioQ = -1;
    if (!threadIdx.x) s_sync.flag = true;
    __syncthreads();

    // Load items for insertion.
    for (int item = 0; item < SORTED_ITEMS_PER_THREAD && s_sync.flag; ++item) {
      const int idx = item * BLOCK_DIM_X + threadIdx.x;
      if (idx < SORTED_SIZE) {
        r_cache[item] = s_cache[idx];
        r_dists[item] = s_dists[idx];
        if (r_cache[item] == key) s_sync.flag = false;
      }
    }
    __syncthreads();
    clc_push_2 = clock();
    // TODO(fabi) return on s_sync.flag = true?
    for (int item = 0; item < SORTED_ITEMS_PER_THREAD && s_sync.flag; ++item) {
      const int idx = item * BLOCK_DIM_X + threadIdx.x;
      if (idx < SORTED_SIZE) {
        if (r_dists[item] >= dist) {
          clc_push_5 = clock();
          // Don't move if no entry or end of best or prioq.
          if ((r_cache[item] != EMPTY_KEY) && !is_end(idx)) {
            const int idx_next = (idx + 1 == SORTED_SIZE) ? BEST_SIZE : idx + 1;
            s_cache[idx_next] = r_cache[item];
            s_dists[idx_next] = r_dists[item];
          }
          clc_push_6 = clock();
          // Find insert points.
          const int idx_prev = idx - 1;
          const ValueT dist_prev =
              ((idx_prev == -1) || (idx_prev == BEST_SIZE + s_prioQ_head - 1))
                  ? -1.f
                  : (idx_prev == BEST_END) ? s_dists[SORTED_SIZE - 1]
                                           : s_dists[idx_prev];
          clc_push_7 = clock();
          if (dist_prev < dist) {
            if (idx < BEST_SIZE)
              r_write_item_best = item;
            else
              r_write_item_prioQ = item;
          }
          clc_push_8 = clock();
          clc_push_add_1 += (int)(clc_push_6 - clc_push_5);
          clc_push_add_2 += (int)(clc_push_7 - clc_push_6);
          clc_push_add_3 += (int)(clc_push_8 - clc_push_7);
        }
      }
    }
    __syncthreads();
    clc_push_3 = clock();

    // Insert into best and prioq.
    if (r_write_item_best >= 0) {
      const int idx = r_write_item_best * BLOCK_DIM_X + threadIdx.x;
      s_cache[idx] = key;
      s_dists[idx] = dist;
    }
    if (r_write_item_prioQ >= 0) {
      const int idx = r_write_item_prioQ * BLOCK_DIM_X + threadIdx.x;
      s_cache[idx] = key;
      s_dists[idx] = dist;
    }
    __syncthreads();
    clc_push_4 = clock();
    // if (!blockIdx.x && !threadIdx.x) printf("%d    %d    %d\n", (int)(clc_push_2 - clc_push_1), 
    //                                                             (int)(clc_push_3 - clc_push_2), 
    //                                                             (int)(clc_push_4 - clc_push_3));
    // if (!blockIdx.x) printf("%d    %d    %d    %d\n", threadIdx.x, clc_push_add_1, clc_push_add_2, clc_push_add_3);
  }

  __device__ __forceinline__ KeyT pop() {
    __syncthreads();

    if (!threadIdx.x) {
      const int head_idx_prioQ = BEST_SIZE + s_prioQ_head;
      const ValueT dist = s_dists[head_idx_prioQ];
      if (dist == EMPTY_DIST) {
        // Pop on empty prioQ.
        s_sync.cache = EMPTY_KEY;
      } else {
        if (!criteria(dist)) {
          s_sync.cache = EMPTY_KEY;
        } else {
          const KeyT key = s_cache[head_idx_prioQ];
          s_sync.cache = key;
          const int head_idx_visited = SORTED_SIZE + s_visited_head;
          s_cache[head_idx_visited] = key;
          s_visited_head = (s_visited_head + 1) % VISITED_SIZE;
        }
        s_cache[head_idx_prioQ] = EMPTY_KEY;
        s_dists[head_idx_prioQ] = EMPTY_DIST;
        // Move ring-buffer head forward.
        s_prioQ_head = (s_prioQ_head + 1) % PRIOQ_SIZE;
      }
    }
    __syncthreads();
    return s_sync.cache;
  }


  __device__ __forceinline__ void fetch(KeyT* s_keys, const KeyT* d_translation,
                                        int len) {
    __syncthreads();
    clc_re_start = clock();
    for (int item = 0; item < CACHE_ITEMS_PER_THREAD; ++item) {
      const int i = item * BLOCK_DIM_X + threadIdx.x;
      if (i < CACHE_SIZE) {
        const KeyT n = s_cache[i];
        for (int k = 0; n != EMPTY_KEY && k < len; k++) {
          if (n == s_keys[k]) {
            s_keys[k] = EMPTY_KEY;
          }
        }
      }
    }
    clc_re_end = clock();
    clc_re += (int)(clc_re_end - clc_re_start);

    for (int k = 0; k < len; k++) {
      __syncthreads();
      const KeyT other_n = s_keys[k];
      if (other_n == EMPTY_KEY) continue;
      const KeyT other_m =
          (d_translation == nullptr) ? other_n : d_translation[other_n];
      clc_dist_start = clock();
      const ValueT dist = is_search ? rs_dist_calc.distance_synced(other_m, DA) : rs_dist_calc.distance_synced(other_m);
      num_dist += 1;
      clc_dist_end = clock();
      clc_dist += (int)(clc_dist_end - clc_dist_start);

      if (criteria(dist)) {
        clc_push_start = clock();
        push(other_n, dist);
        __syncthreads();
        clc_push_end = clock();
        clc_push += (int)(clc_push_end - clc_push_start);
      }

    }
    __syncthreads();
  }

  __device__ __forceinline__ void transform(const KeyT* transform) {
    for (int item = 0; item < CACHE_ITEMS_PER_THREAD; ++item) {
      const int i = item * BLOCK_DIM_X + threadIdx.x;

      if (i < BEST_SIZE) {
        // transform best
        KeyT key = s_cache[i];
        if (key != EMPTY_KEY)
          key = transform[key];
        s_cache[i] = key;

        // copy best into prio queue
        if (i+BEST_SIZE < SORTED_SIZE) {
          s_cache[i+BEST_SIZE] = key;
          s_dists[i+BEST_SIZE] = s_dists[i];
        }
      }
      else if (i < 2*BEST_SIZE && i < SORTED_SIZE) {
        // do nothing (handled by previous threads)
      }
      else if (i < CACHE_SIZE) {
        // reset remainder of the prio queue and visited cache
        s_cache[i] = EMPTY_KEY;
        if (i < SORTED_SIZE)
          s_dists[i] = EMPTY_DIST;
      }
    }

    // reset heads.
    if (!threadIdx.x) {
      s_prioQ_head = 0;
      s_visited_head = 0;
    }
  }


  __device__ __forceinline__ void write_best_graph(KeyT* d_buffer, const KeyT n,
                                                   int K, int offset = 1) {
    for (int i = threadIdx.x; i < K; i += BLOCK_DIM_X) {
      const KeyT idx = s_cache[i + offset];
      d_buffer[n * K + i] = (idx != EMPTY_KEY) ? idx : n;
    }
  }

  __device__ __forceinline__ void write_best(KeyT* d_buffer, const KeyT n,
                                             int stride) {
    for (int i = threadIdx.x; i < KQuery; i += BLOCK_DIM_X) {
      const KeyT idx = s_cache[i];
      d_buffer[n * stride + i] = idx;
    }
  }

  __device__ __forceinline__ void write_best(KeyT* d_buffer, const KeyT n,
                                             int stride, int idx_offset) {
    for (int i = threadIdx.x; i < KQuery; i += BLOCK_DIM_X) {
      const KeyT idx = s_cache[i];
      d_buffer[n * stride + i] = idx + idx_offset;
    }
  }

  __device__ __forceinline__ void filter_and_write_best(KeyT* d_buffer, const KeyT n,
                                             int stride, int idx_offset) {
    __shared__ int filter_number;
    if (!threadIdx.x) filter_number = 0;
    __syncthreads();
    for (int i = 0; i < BEST_SIZE && filter_number < KQuery; i++) {
      const KeyT idx = s_cache[i];
      const int base_attr = d_base_attr[idx];
      for (int j = threadIdx.x; j < DA; j += BLOCK_DIM_X) {
        if (d_query_attr[n * DA + j] == base_attr) {
          d_buffer[n * stride + filter_number] = idx + idx_offset;
          filter_number++;
        }
        __syncthreads();
      }
    }
  }

  template <DistanceMeasure m = measure, typename std::enable_if<m == Euclidean, int>::type = 0> // euclidean distance version
  __device__ __forceinline__ float get_nn1_dist() {
    return sqrtf(s_dists[1]);
  }

  template <DistanceMeasure m = measure, typename std::enable_if<m == Cosine, int>::type = 0> // cosine similarity version
  __device__ __forceinline__ float get_nn1_dist() {
    return s_dists[1];
  }

  __device__ __forceinline__ int get_dist_stats() { return dist_calc_counter; }
  __device__ __forceinline__ int get_overflow_stats() {
    return s_overflow_counter;
  }

  /**
   * Prints first 'len' elements in the Cache. [parallel call]:
   * cash.print(8);
   *
   */
  __device__ __forceinline__ void print(int len = CACHE_SIZE) {
    __syncthreads();
    if (!threadIdx.x) printf("print \n");
    if (!threadIdx.x) {
      printf("Cache: ring: %d KQuery: %f (+xi -> %f) \n", s_prioQ_head,
             s_dists[KQuery - 1], s_dists[KQuery - 1] + xi);
      for (int i = 0; i < len; ++i) {
        if (i < BEST_SIZE) {
          printf("%d -> %d %f \n", i, s_cache[i], s_dists[i]);
        } else {
          if (i < SORTED_SIZE) {
            printf("%d -> %d %f | ", i, s_cache[i], s_dists[i]);
            if (i - BEST_SIZE == s_prioQ_head) printf("X");
            printf("\n");
          } else {
            printf("%d -> %d | ", i, s_cache[i]);
            if (i - SORTED_SIZE == s_visited_head) printf("X");
            printf("\n");
          }
        }
      }
    }
    __syncthreads();
  }
};

#endif  // INCLUDE_GGNN_CACHE_CUDA_SIMPLE_KNN_CACHE_CUH_
