
# GGHQ: GPU-Friendly Graph-Based Hybrid Query with Vector and Attribute


## Environmental configuration

GGHQ(GPU) is implemented based on GGNN. Therefore, please refer to GGNN (https://github.com/cgtuebingen/ggnn.git) for environmental configuration.

## vector + attribute hybrid search  dataset format
base.vector is a binary file in the following format:
```bash
Number of vectors N, Dimension of vectors D, {vector_1，vector_2，...，vector_N}
```
query.vector is a binary file in the following format:
```bash
Number of vectors N, Dimension of vectors D, {vector_1，vector_2，...，vector_N}
```
base.attribute is a TXT file in the following format:
```bash
Number of attribute data N, Number of dataset attributes D
attribute_1
attribute_2
...
attribute_N
% The number of dataset attributes D of base data is 1. 
% Attribute value is a natural number. 
% If the attribute is in string format, it needs to be mapped to a natural number first
% If a data has multiple attributes, we need to map each possible base attribute to a natural number.
```
query.attribute is a TXT file in the following format:
```bash
Number of attribute data N, Number of query attributes D
attribute_1.1 attribute_1.2 ... attribute_1.D
attribute_2.1 attribute_2.2 ... attribute_2.D
...
attribute_N.1 attribute_N.2 ... attribute_N.D
% The number of query attributes D of query data may not be 1. 
% This means that a query has different attribute conditions, and the relationship between them is “or”.
```

Please use brute force calculation to generate the groundtruth file and store it in binary format.


## vector + attribute hybrid search  compile and run test
To compile
```bash
mkdir build_local
cd build_local
cmake ..
make 
```

To run
```bash
# Example for SIFT1M on GPU 0:
./sift1m_multi  --gpu_ids="0" --mode="bs"
./sift1m_multi  --gpu_ids="0" --mode="lq" --grid_search
# The attrsize and attrdim are set in the .cu file
```

