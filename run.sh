#!/bin/bash
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/targets/x86_64-linux/lib:/home/surfai/dev/clipHelper/downloads/onnxruntime-linux-x64-gpu-1.18.0/lib:$LD_LIBRARY_PATH
export LD_PRELOAD=/usr/local/cuda-11.8/targets/x86_64-linux/lib/libcublasLt.so.11.11.3.6:/usr/local/cuda-11.8/targets/x86_64-linux/lib/libcublas.so.11.11.3.6
exec build/face_replacer $@
