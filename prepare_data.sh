#!/usr/bin/env bash

apt-get install libgl1 -y
apt install unzip -y
pip install cupy-cuda11x -f https://pip.cupy.dev/aarch64

#unzip FGT_data.zip
unzip weights.zip
mkdir FGT/checkpoint
mkdir FGT/flowCheckPoint
mkdir LAFC/checkpoint
mv weights/fgt/* FGT/checkpoint
mv weights/lafc/* LAFC/checkpoint
mv weights/lafc_single/* FGT/flowCheckPoint
rm -r weights
#rm FGT_data.zip
rm weights.zip
