#!/bin/bash

# Build python wheels for all environments:
# ./bin/build_wheels.sh
# Because our shared libraries are pre-built, we have to customize each environment and just
# bundle what is required for that runtime, instead of including all binaries
#
# Refernece:
# https://github.com/numpy/numpy/releases/tag/v1.23.5

cd ~/dev/bmds
source venv/bin/activate

# build mac
sed -i '' -e "s/recursive-include bmds\/bin \*.so/# recursive-include bmds\/bin \*.so/g" MANIFEST.in
sed -i '' -e "s/recursive-include bmds\/bin \*.exe \*.dll/# recursive-include bmds\/bin \*.exe \*.dll/g" MANIFEST.in
make clean
python setup.py bdist_wheel --plat-name=macosx_11_0_arm64
mv dist/*.whl .
sed -i '' -e "s/# recursive-include bmds\/bin \*.so/recursive-include bmds\/bin \*.so/g" MANIFEST.in
sed -i '' -e "s/# recursive-include bmds\/bin \*.exe \*.dll/recursive-include bmds\/bin \*.exe \*.dll/g" MANIFEST.in

# build linux
sed -i '' -e "s/recursive-include bmds\/bin \*.dylib/# recursive-include bmds\/bin \*.dylib/g" MANIFEST.in
sed -i '' -e "s/recursive-include bmds\/bin \*.exe \*.dll/# recursive-include bmds\/bin \*.exe \*.dll/g" MANIFEST.in
make clean
python setup.py bdist_wheel --plat-name=manylinux_2_17_x86_64.manylinux2014_x86_64
mv dist/*.whl .
sed -i '' -e "s/# recursive-include bmds\/bin \*.dylib/recursive-include bmds\/bin \*.dylib/g" MANIFEST.in
sed -i '' -e "s/# recursive-include bmds\/bin \*.exe \*.dll/recursive-include bmds\/bin \*.exe \*.dll/g" MANIFEST.in

# build windows
sed -i '' -e "s/recursive-include bmds\/bin \*.so/# recursive-include bmds\/bin \*.so/g" MANIFEST.in
sed -i '' -e "s/recursive-include bmds\/bin \*.dylib/# recursive-include bmds\/bin \*.dylib/g" MANIFEST.in
make clean
python setup.py bdist_wheel --plat-name=win32
mv dist/*.whl .
sed -i '' -e "s/# recursive-include bmds\/bin \*.so/recursive-include bmds\/bin \*.so/g" MANIFEST.in
sed -i '' -e "s/# recursive-include bmds\/bin \*.dylib/recursive-include bmds\/bin \*.dylib/g" MANIFEST.in

# cleanup and move dists back to dist folder
make clean
mkdir ./dist
mv *.whl ./dist

pip install -e .
