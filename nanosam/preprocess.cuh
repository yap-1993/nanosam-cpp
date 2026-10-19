#pragma once
#include <cuda_runtime.h>

void launchPreprocess(const uint8_t* d_src, int srcW, int srcH, int srcStep,
                      float* d_dst, int dstW, int dstH, cudaStream_t stream);