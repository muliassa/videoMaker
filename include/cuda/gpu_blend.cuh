#pragma once

#include <opencv2/core/cuda.hpp>
#include <cuda_runtime.h>

namespace facereplacer {
namespace cuda {

// Feathered alpha blending on GPU
void featheredBlend(const cv::cuda::GpuMat& src,
                    const cv::cuda::GpuMat& dst,
                    const cv::cuda::GpuMat& mask,
                    cv::cuda::GpuMat& result,
                    int featherRadius);

// Simple alpha blending
void alphaBlend(const cv::cuda::GpuMat& src,
                const cv::cuda::GpuMat& dst,
                const cv::cuda::GpuMat& mask,
                cv::cuda::GpuMat& result);

// Color transfer on GPU
void colorTransfer(const cv::cuda::GpuMat& src,
                   const cv::cuda::GpuMat& target,
                   cv::cuda::GpuMat& result);

// Gaussian blur on mask
void blurMask(cv::cuda::GpuMat& mask, int radius);

// Check CUDA availability
bool isCudaAvailable();
int getCudaDeviceCount();
void printCudaInfo();

} // namespace cuda
} // namespace facereplacer
