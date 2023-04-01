#!/bin/bash

# Snappy.
cd snappy
rm -rf build
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release .. || { echo 'Failed to build Snappy.'; exit 1; }
make -j
cd ..
