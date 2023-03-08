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
//

#ifndef CUDA_API_PER_THREAD_DEFAULT_STREAM
#define CUDA_API_PER_THREAD_DEFAULT_STREAM
#endif

#include <cuda.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <stdio.h>

#include <cub/cub.cuh>

// only needed for file_exists check
#include <sys/stat.h>

inline bool file_exists(const std::string& name) {
  struct stat buffer;
  return (stat(name.c_str(), &buffer) == 0);
}

#include <iostream>
#include <vector>

#include "ggnn/cuda_knn_ggnn_multi_gpu.cuh"
#include "ggnn/utils/cuda_knn_constants.cuh"

DEFINE_string(
    mode, "bq",
    "Mode: bq -> build_and_query, bs -> build_and_store, lq -> load_and_query");
DEFINE_string(base_filename, "/home/maoqiuli21/nfs/data/turing/", "path to file with base vectors");
DEFINE_string(query_filename, "/home/maoqiuli21/nfs/data/turing/", "path to file with perform_query vectors");
DEFINE_string(base_attr_filename, "/home/maoqiuli21/nfs/data/label/", "path to file with base attributes");
DEFINE_string(query_attr_filename, "/home/maoqiuli21/nfs/data/label/", "path to file with query attributes");
DEFINE_string(groundtruth_filename, "/home/maoqiuli21/nfs/data/label/", "path to file with groundtruth");
DEFINE_string(graph_dir, "/home/maoqiuli21/nfs/index_ggnnlbsearch/adage/", "directory to store and load ggnn graph files.");
DEFINE_double(tau, 0.5, "Parameter tau");
DEFINE_int32(factor, 1000000, "Factor");
DEFINE_int32(base, 1, "N_base: base x factor");
DEFINE_int32(shard, 1, "N_shard: shard x factor");
DEFINE_int32(refinement_iterations, 2, "Number of refinement iterations");
DEFINE_string(gpu_ids, "0", "GPU id");
DEFINE_bool(grid_search, false,
            "Perform queries for a wide range of parameters.");

int main(int argc, char* argv[]) {
  FLAGS_log_dir = "/home/maoqiuli21/project/ggnn_lbsearch/build_local/log/";
  google::InitGoogleLogging(argv[0]);
  google::LogToStderr();

  gflags::SetUsageMessage(
      "GGNN: Graph-based GPU Nearest Neighbor Search\n"
      "by Fabian Groh, Lukas Ruppert, Patrick Wieschollek, Hendrik P.A. "
      "Lensch\n"
      "(c) 2020 Computer Graphics University of Tuebingen");
  gflags::SetVersionString("1.0.0");
  google::ParseCommandLineFlags(&argc, &argv, true);

  LOG(INFO) << "Reading files";
  // CHECK(file_exists(FLAGS_base_filename))
  //     << "File for base vectors has to exist";
  // CHECK(file_exists(FLAGS_query_filename))
  //     << "File for perform_query vectors has to exist";

  CHECK_GE(FLAGS_tau, 0) << "Tau has to be bigger or equal 0.";
  CHECK_GE(FLAGS_refinement_iterations, 0)
      << "The number of refinement iterations has to be non-negative.";

  // ####################################################################
  // compile-time configuration
  //
  // data types
  //
  /// data type for addressing points (needs to be able to represent N)
  using KeyT = int32_t;
  /// data type of the dataset (e.g., char, int, float)
  using BaseT = float;
  /// data type of computed distances
  using ValueT = float;
  /// data type for addressing base-vectors (needs to be able to represent N*D)
  using BAddrT = uint64_t;
  /// data type for addressing the graph (needs to be able to represent
  /// N*KBuild)
  using GAddrT = uint64_t;
  //
  // dataset configuration (here: TURING+16)
  //
  /// dimension of the vector
  const int D = 100;
  /// dimension of the attribute
  const int DBA = 16;
  /// dimension of the qurey attribute
  const int DA = 1;
  /// distance measure (Euclidean or Cosine)
  const DistanceMeasure measure = Euclidean;
  const int DistPar = 4;
  //
  // search-graph configuration
  //
  /// number of neighbors per point in the global graph
  const int KBuild = 24;
  /// maximum number of inverse/symmetric links (KBuild / 2 usually works best)
  const int KF = KBuild / 2;
  /// number of neighbors per point in the local graph
  const int KBuild_ = 24;
  /// maximum number of inverse/symmetric links (KBuild / 2 usually works best)
  const int KF_ = KBuild_ / 2;
  /// segment/batch size (needs to be > KBuild-KF)
  const int S = 32;
  /// graph height / number of layers (4 usually performs best)
  const int L = 4;

  //
  // query configuration
  //
  /// number of neighbors to search for
  const int KQuery = 10;

  static_assert(KBuild - KF < S,
                "there are not enough points to fill the local neighbor list!");

  LOG(INFO) << "Using the following parameters " << KBuild << " (KBuild) " << KF
            << " (KF) " << S << " (S) " << L << " (L) " << D << " (D) ";

  std::istringstream iss(FLAGS_gpu_ids);
  std::vector<std::string> results(std::istream_iterator<std::string>{iss},
                                   std::istream_iterator<std::string>());

  int numGpus;
  cudaGetDeviceCount(&numGpus);

  std::vector<int> gpus;
  for (auto&& r : results) {
    int gpu_id = atoi(r.c_str());
    printf("GPU %d: ", gpu_id);
    {
      CHECK_GE(gpu_id, 0) << "This GPU does not exist";
      CHECK_LT(gpu_id, numGpus) << "This GPU does not exist";

      cudaDeviceProp prop;
      cudaGetDeviceProperties(&prop, gpu_id);
      printf("Found device name: %s\n", prop.name);

      gpus.push_back(gpu_id);
    }
  }

  const size_t N_base = FLAGS_base * FLAGS_factor;
  const size_t N_base_fac = FLAGS_base * FLAGS_factor / 1000000;
  const int N_shard = FLAGS_shard * FLAGS_factor;

  std::string base = (N_base_fac == 1)?"":std::to_string(N_base_fac);
  std::string base_filename = FLAGS_base_filename + 
              "turing" + std::to_string(N_base_fac) + "m/base." + std::to_string(N_base_fac) + "m.fbin";
  std::string query_filename = FLAGS_query_filename + 
              "query100K.fbin";
  std::string base_attr_filename = FLAGS_base_attr_filename + 
              "label_turing" + base + "_base_value_" + std::to_string(DBA) + ".txt";
  std::string query_attr_filename = FLAGS_query_attr_filename + 
              "label_turing" + base + "_query_value_" + std::to_string(DBA) + "_labeldim_" + std::to_string(DA) + ".txt";
  std::string groundtruth_filename = FLAGS_groundtruth_filename + 
              "turing" + base + "_groundtruth_label_value_" + std::to_string(DBA) + "_labeldim_" + std::to_string(DA) + ".bin";
  std::string graph_dir = FLAGS_graph_dir + 
              "turing" + std::to_string(N_base_fac) + "m_" + std::to_string(DBA) + "/";

  typedef GGNNMultiGPU<measure, KeyT, ValueT, GAddrT, BaseT, BAddrT, DistPar, D, DBA, DA, KBuild,
                       KF, KBuild_, KF_, KQuery, S>
      GGNN;
  GGNN ggnn{
      base_filename,
      query_filename,
      base_attr_filename,
      query_attr_filename,
      file_exists(groundtruth_filename) ? groundtruth_filename : "",
      L,
      static_cast<float>(FLAGS_tau),
      N_base};

  ggnn.ggnnMain(gpus, FLAGS_mode, N_shard, graph_dir,
                FLAGS_refinement_iterations, FLAGS_grid_search);

  printf("done! \n");
  gflags::ShutDownCommandLineFlags();
  return 0;
}
