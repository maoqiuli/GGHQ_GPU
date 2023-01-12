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

#ifndef INCLUDE_IO_LOADER_ANN_HPP_
#define INCLUDE_IO_LOADER_ANN_HPP_

#include <fstream>
#include <string>
#include <vector>
#include "loader.hpp"


void SplitString(const std::string &s, std::vector<int> &v, const std::string &c)
{
    std::string::size_type pos1, pos2;
    pos2 = s.find(c);
    pos1 = 0;

    while (std::string::npos != pos2)
    {
        v.push_back(atof(s.substr(pos1, pos2 - pos1).c_str()));

        pos1 = pos2 + c.size();
        pos2 = s.find(c, pos1);
    }
    if (pos1 != s.length())
        v.push_back(atof(s.substr(pos1).c_str()));
}


template <typename ValueT, typename KeyT>
class XVecsLoader : public Loader<ValueT, KeyT> {
 public:
  explicit XVecsLoader(const std::string& path, const bool is_txt) : Loader<ValueT, KeyT>(path, is_txt) {

    if (is_txt)
    {
      std::string temp;
      getline(*(this->hnd), temp);
      std::vector<int> tmp2;
      SplitString(temp, tmp2, " ");
      this->dimension = tmp2[1];
      this->num_elements = tmp2[0];

      DLOG(INFO) << "Open " << path << " with " << this->num_elements << " "
                << this->dimension << "-dim attributes.";
    }
    else
    {  
      int32_t num, dim;
      this->hnd->read((char *) &num, sizeof(int32_t));
      this->hnd->read((char *) &dim, sizeof(int32_t));
      this->dimension = dim;
      this->num_elements = num;

      DLOG(INFO) << "Open " << path << " with " << this->num_elements << " "
                << this->dimension << "-dim vectors.";
    }
  }

  void load(ValueT* dst, size_t skip, size_t num) override {
    {
      DLOG(INFO) << "Loading " << num << " vectors starting at " << skip
                << " ...";

      this->hnd->read((char *) dst, num * this->dimension * sizeof(ValueT));
    }
    
    DLOG(INFO) << "Done";
  }

  void load(ValueT* dst, size_t skip, size_t num, const std::vector<KeyT>& cluster) override {
    {
      DLOG(INFO) << "Loading " << num << " vectors starting at " << skip
                << " ...";

      size_t num_data = cluster.size();
      this->num_elements = num_data;
      for (size_t i=0; i < num_data; i++) {
        int offset = (i==0 ? cluster[i] : cluster[i] - cluster[i-1] - 1) * this->dimension * sizeof(ValueT);
        this->hnd->seekg(offset, std::ios::cur);
        this->hnd->read((char *) dst, this->dimension * sizeof(ValueT));
        dst += this->dimension;
      }
    }
    
    DLOG(INFO) << "Done";
  }

  void load_attr(ValueT* dst, size_t skip, size_t num) override {
    DLOG(INFO) << "Loading " << num << " attributes starting at " << skip
              << " ...";

    std::string temp;
    int count = 0;
    while (getline(*(this->hnd), temp))
    {
      std::vector<int> tmp2;
      SplitString(temp, tmp2, " ");
      for (int i = 0;i < this->dimension;i++)
        dst[count + i] = tmp2[i];
      count += this->dimension;
    }

    DLOG(INFO) << "Done";
  }

  void load_attr(ValueT* dst, size_t skip, size_t num, const std::vector<KeyT>& cluster) override {
    DLOG(INFO) << "Loading " << num << " attributes starting at " << skip
              << " ...";

    std::vector<std::string> temp(this->num_elements);
    for (int i = 0;i < this->num_elements;i++) {
      getline(*(this->hnd), temp[i]);
    }
    for (int i = 0;i < cluster.size();i++) {
      std::vector<int> tmp2;
      SplitString(temp[cluster[i]], tmp2, " ");
      for (int j = 0;j < this->dimension;j++) {
        dst[i * this->dimension + j] = tmp2[j];
      }
    }
    this->num_elements = cluster.size();

    DLOG(INFO) << "Done";
  }
};





using FVecsLoader = XVecsLoader<float, int32_t>;
using IVecsLoader = XVecsLoader<int, int32_t>;
using BVecsLoader = XVecsLoader<uint8_t, int32_t>;

#endif  // INCLUDE_IO_LOADER_ANN_HPP_
