#!/bin/bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install matplotlib numpy scipy

echo "Configuring and building Thirdparty/DBoW2 ..."

cd Thirdparty/DBoW2
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j10

cd ../../g2o

echo "Configuring and building Thirdparty/g2o ..."

mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j10

cd ../../Sophus

echo "Configuring and building Thirdparty/Sophus ..."

mkdir -p build && cd build
# cmake .. -DCMAKE_BUILD_TYPE=Release
# make -j10

cd ../../../

echo "Uncompress vocabulary ..."

cd Vocabulary
tar -xf ORBvoc.txt.tar.gz
cd ..

echo "Configuring and building ORB_SLAM3 ..."

mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j5
