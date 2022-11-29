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

#ifndef INCLUDE_GGNN_UTILS_CUDA_KNN_DISTANCE_CUH_
#define INCLUDE_GGNN_UTILS_CUDA_KNN_DISTANCE_CUH_

#include <cuda.h>
#include <cuda_runtime.h>

#include <cub/cub.cuh>
#include <limits>

// helper structs to avoid having the register when not necessary
struct Nothing {
};
template <typename ValueT>
struct QueryNorm {
  // only valid in thread 0, only needed if measure == Cosine
  ValueT query_norm;
};

/**
 * Distance calculates the distance/difference between the base vector and
 * other_id vector.
 */
template <DistanceMeasure measure,
          typename ValueT, typename KeyT, int D, int BLOCK_DIM_X,
          typename BaseT = ValueT, typename AddrT = KeyT>
struct Distance : std::conditional<measure == Cosine, QueryNorm<ValueT>, Nothing>::type {
  enum { ITEMS_PER_THREAD = (D - 1) / BLOCK_DIM_X + 1 };

  struct DistanceAndNorm {
    ValueT r_dist;
    ValueT r_norm;

    __device__ __forceinline__ DistanceAndNorm(const ValueT dist, const ValueT norm)
        : r_dist(dist), r_norm(norm) {}

    __device__ __forceinline__ DistanceAndNorm() {}

    struct Sum {
      __host__ __device__ __forceinline__ DistanceAndNorm operator()(const DistanceAndNorm& a,
                                                                     const DistanceAndNorm& b) const {
        return DistanceAndNorm(a.r_dist + b.r_dist, a.r_norm + b.r_norm);
      }
    };
  };

  typedef cub::BlockReduce<ValueT, BLOCK_DIM_X> BlockReduceDist;
  typedef typename std::conditional<measure == Cosine, cub::BlockReduce<DistanceAndNorm, BLOCK_DIM_X>, BlockReduceDist>::type BlockReduceDistAndNorm;

  union TempStorage {
    typename BlockReduceDist::TempStorage dist_temp_storage;
    typename BlockReduceDistAndNorm::TempStorage dist_and_norm_temp_storage;
    ValueT dist;
  };

  const BaseT* d_base;
  BaseT r_query[ITEMS_PER_THREAD];

  TempStorage& s_temp_storage;
  __device__ __forceinline__ TempStorage& PrivateTmpStorage() {
    __shared__ TempStorage s_tmp;
    return s_tmp;
  }

  /**
   * Distance dist_calc(d_base, d_query, blockIdx.x);
   */
  __device__ __forceinline__ Distance(const BaseT* d_base, const BaseT* d_query, const KeyT n)
      : d_base(d_base), s_temp_storage(PrivateTmpStorage()) {
    loadQueryPos(d_query+static_cast<AddrT>(n) * D);
    // loadQueryPos(d_query+static_cast<AddrT>(n) * D);
  }

  /**
   * Distance dist_calc(d_base, blockIdx.x);
   */
  __device__ __forceinline__ Distance(const BaseT* d_base, const KeyT n)
      : d_base(d_base), s_temp_storage(PrivateTmpStorage()) {
    loadQueryPos(d_base+static_cast<AddrT>(n) * D);
    // loadQueryPos(d_base+static_cast<AddrT>(n) * D);
  }

  template <DistanceMeasure m = measure, typename std::enable_if<m == Euclidean, int>::type = 0> // euclidean distance version
  __device__ __forceinline__ void loadQueryPos(const BaseT* d_query)
  {
    for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
      const int read_dim = item * BLOCK_DIM_X + threadIdx.x;
      if (read_dim < D) {
        r_query[item] = *(d_query+read_dim);
      }
    }
  }
  template <DistanceMeasure m = measure, typename std::enable_if<m == Cosine, int>::type = 0> // cosine similarity version
  __device__ __forceinline__ void loadQueryPos(const BaseT* d_query)
  {
    ValueT r_query_norm = 0.0f;
    for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
      const int read_dim = item * BLOCK_DIM_X + threadIdx.x;
      if (read_dim < D) {
        r_query[item] = *(d_query+read_dim);
        r_query_norm += r_query[item]*r_query[item];
      }
    }
    // only needed by thread 0
    this->query_norm = BlockReduceDist(s_temp_storage.dist_temp_storage).Sum(r_query_norm);
  }

  /**
   * Calculates distance of base vector to other_id vector.
   *
   * [parallel call]:
   * ValueT dist = distCalc.distance(other_id)
   *
   * Return:
   *   ValueT distance
   *
   * Note: distance only valid in first thread.
   */


  __device__ __forceinline__ void warpRecude(volatile ValueT* s_data, int tid){
    if(BLOCK_DIM_X >= 64) s_data[tid] += s_data[tid + 32];
    if(BLOCK_DIM_X >= 32) s_data[tid] += s_data[tid + 16];
    if(BLOCK_DIM_X >= 16) s_data[tid] += s_data[tid + 8];
    if(BLOCK_DIM_X >= 8) s_data[tid] += s_data[tid + 4];
    if(BLOCK_DIM_X >= 4) s_data[tid] += s_data[tid + 2];
    if(BLOCK_DIM_X >= 2) s_data[tid] += s_data[tid + 1];
  }

  template <DistanceMeasure m = measure, typename std::enable_if<m == Euclidean, int>::type = 0> // euclidean distance version
  __device__ __forceinline__ ValueT distance(const KeyT other_id) {
    ValueT r_dist = 0.0f;
    for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
      const int read_dim = item * BLOCK_DIM_X + threadIdx.x;
      if (read_dim < D) {
        ValueT pos_other =
            r_query[item] - d_base[static_cast<AddrT>(other_id) * D + read_dim];
        r_dist += pos_other * pos_other;
      }
    }

    return BlockReduceDist(s_temp_storage.dist_temp_storage).Sum(r_dist);
  }

  
  // __device__ __forceinline__ ValueT distance(const KeyT other_id) {
  //   ValueT dist = 0.0f;
  //   for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
  //     const int read_dim = item * BLOCK_DIM_X + threadIdx.x;
  //     if (read_dim < D) {
  //       ValueT pos_other =
  //           r_query[item] - d_base[static_cast<AddrT>(other_id) * D + read_dim];
  //       dist += pos_other * pos_other;
  //     }
  //   }
  //   // __syncthreads();

  //   volatile ValueT* vshm;
  //   ValueT val;

  //   if (BLOCK_DIM_X > 32) {
  //     __shared__ ValueT r_dist[BLOCK_DIM_X];
  //     r_dist[threadIdx.x] = dist;
  //     __syncthreads();

  //     for (int stride = BLOCK_DIM_X / 2; stride > 16; stride /= 2)
  //     {
  //       if (threadIdx.x < stride)
  //       {
  //         r_dist[threadIdx.x] += r_dist[threadIdx.x + stride];
  //       }
  //       __syncthreads();
  //     }
  //     vshm = r_dist;
  //     val = vshm[threadIdx.x];
  //   }
  //   else {
  //     val = dist;
  //   }

  //   val += __shfl_xor_sync(0xffffffff, val, 16);
  //   val += __shfl_xor_sync(0xffffffff, val, 8);
  //   val += __shfl_xor_sync(0xffffffff, val, 4);
  //   val += __shfl_xor_sync(0xffffffff, val, 2);
  //   val += __shfl_xor_sync(0xffffffff, val, 1);
  //   return val;
  // }


  // __device__ __forceinline__ ValueT distance(const KeyT other_id) {
  //   __shared__ ValueT r_dist[BLOCK_DIM_X];
  //   r_dist[threadIdx.x] = 0.0f;
  //   __syncthreads();
  //   for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
  //     const int read_dim = item * BLOCK_DIM_X + threadIdx.x;
  //     if (read_dim < D) {
  //       ValueT pos_other =
  //           r_query[item] - d_base[static_cast<AddrT>(other_id) * D + read_dim];
  //       r_dist[threadIdx.x] += pos_other * pos_other;
  //     }
  //   }
  //   __syncthreads();

  //   for (int stride = BLOCK_DIM_X / 2; stride > 0; stride /= 2)
  //   {
  //       if (threadIdx.x < stride)
  //       {
  //           r_dist[threadIdx.x] += r_dist[threadIdx.x + stride];
  //       }
  //       __syncthreads();
  //   }

  //   return r_dist[threadIdx.x];
  // }
  
  template <DistanceMeasure m = measure, typename std::enable_if<m == Cosine, int>::type = 0> // cosine similarity version
  __device__ __forceinline__ ValueT distance(const KeyT other_id) {
    DistanceAndNorm r_dist_and_norm(0.0f, 0.0f);
    for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
      const int read_dim = item * BLOCK_DIM_X + threadIdx.x;
      if (read_dim < D) {
        r_dist_and_norm.r_dist += r_query[item] * d_base[static_cast<AddrT>(other_id) * D + read_dim];
        r_dist_and_norm.r_norm += d_base[static_cast<AddrT>(other_id) * D + read_dim]*d_base[static_cast<AddrT>(other_id) * D + read_dim];
      }
    }

    DistanceAndNorm dist_and_norm = BlockReduceDistAndNorm(s_temp_storage.dist_and_norm_temp_storage).Reduce(r_dist_and_norm, DistanceAndNorm::Sum());
    // need to normalize by the vectors' lengths (in high dimensions, no vector has length 1.0f)
    ValueT norm_sqr = this->query_norm*dist_and_norm.r_norm;
    // use negative dot product, as larger values are closer to each other
    if (!threadIdx.x) {
      if (norm_sqr > 0.0f)
        dist_and_norm.r_dist = fabs(1.0f-dist_and_norm.r_dist/sqrt(norm_sqr));
      else
        dist_and_norm.r_dist = 1.0f;
    }

    return dist_and_norm.r_dist;
  }

  /**
   * Calculates synced distance of base vector to other_id vector.
   *
   * [parallel call]:
   * ValueT dist = distCalc.distance(other_id)
   *
   * Return:
   *   ValueT distance
   *
   * Note: distance valid in all threads.
   */
  __device__ __forceinline__ ValueT distance_synced(const KeyT other_id) {
    ValueT dist = distance(other_id);
    if (!threadIdx.x)
      s_temp_storage.dist = dist;
    __syncthreads();

    // dist = distance(other_id);
    // if (!threadIdx.x)
    //   s_temp_storage.dist = dist;
    // __syncthreads();

    return s_temp_storage.dist;
  }
};

#endif  // INCLUDE_GGNN_UTILS_CUDA_KNN_DISTANCE_CUH_
