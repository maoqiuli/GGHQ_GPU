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
#ifndef INCLUDE_GGNN_UTILS_CUDA_KNN_UTILS_CUH_
#define INCLUDE_GGNN_UTILS_CUDA_KNN_UTILS_CUH_

#include <iostream>
#include <vector>
#include <string>


#include <cuda.h>
#include <cuda_runtime.h>

#include <cub/cub.cuh>
#include <glog/logging.h>

enum DistanceMeasure : int {
  Euclidean = 0,
  Cosine = 1
};

template <typename T>
__global__ void launcher(const T kernel) {
  kernel();
}

#define CHECK_CUDA(ans) \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    if (abort)
      LOG(FATAL) << "GPUassert: " << cudaGetErrorString(code) << " " << file << " " << line << "\n";
    else
      LOG(ERROR) << "GPUassert: " << cudaGetErrorString(code) << " " << file << " " << line << "\n";
  }
}

template <typename T>
float time_launcher(const int log_level, T* kernel, int N, cudaStream_t stream = 0) {
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start, stream);
  kernel->launch(stream);
  cudaEventRecord(stop, stream);
  cudaEventSynchronize(stop);

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  VLOG(log_level) << milliseconds << " ms for " << N << " queries -> " << milliseconds*1000.0f/N << " us/query \n";
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return milliseconds;
}

template <typename T>
void launcher(const int log_level, T* kernel, int N, cudaStream_t stream = 0) {
  kernel->launch(stream);
}

template <typename KeyT>
void clusterLabel(const std::string& path, const size_t& N_cluster, std::vector<std::vector<KeyT>>& cluster_base, std::vector<std::vector<KeyT>>& cluster_query) {
  std::ifstream f(path, std::ios_base::in | std::ios_base::binary);
  if(!f.is_open()) throw std::runtime_error("Dataset file " + path + " does not exists");

  int32_t n_cluster, cluster_size;
  f.read((char *) &n_cluster, sizeof(int32_t));
  CHECK_EQ(N_cluster, n_cluster);

  for (size_t i=0; i < n_cluster; i++) {
    f.read((char *) &cluster_size, sizeof(int32_t));
    std::vector<KeyT> cluster(cluster_size);
    for (size_t j=0; j < cluster_size; j++) {
      KeyT data;
      f.read((char *) &data, sizeof(KeyT));
      cluster[j] = data;
    }
    sort(cluster.begin(), cluster.end());
    cluster_base[i] = cluster;
  }

  for (size_t i=0; i < n_cluster; i++) {
    f.read((char *) &cluster_size, sizeof(int32_t));
    std::vector<KeyT> cluster(cluster_size);
    for (size_t j=0; j < cluster_size; j++) {
      KeyT data;
      f.read((char *) &data, sizeof(KeyT));
      cluster[j] = data;
    }
    sort(cluster.begin(), cluster.end());
    cluster_query[i] = cluster;
  }

  int n_query = 0;
  for (size_t i=0; i < n_cluster; i++) n_query += cluster_query[i].size();
  for (size_t i=0; i < n_cluster; i++) {
    for (size_t j=0; cluster_query[i].size() < n_query; j++) {
      cluster_query[i].push_back(cluster_query[i][j]);
    }
  }

  f.close();
}


#endif  // INCLUDE_GGNN_UTILS_CUDA_KNN_UTILS_CUH_


