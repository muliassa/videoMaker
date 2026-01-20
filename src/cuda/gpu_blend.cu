#include "cuda/gpu_blend.cuh"
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafilters.hpp>
#include <iostream>

namespace facereplacer {
namespace cuda {

// CUDA kernel for feathered alpha blending
__global__ void featheredBlendKernel(const uchar3* src, const uchar3* dst,
                                      const uchar* mask, uchar3* result,
                                      int width, int height, int srcStep,
                                      int dstStep, int maskStep, int resStep) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int srcIdx = y * srcStep / sizeof(uchar3) + x;
    int dstIdx = y * dstStep / sizeof(uchar3) + x;
    int maskIdx = y * maskStep + x;
    int resIdx = y * resStep / sizeof(uchar3) + x;
    
    float alpha = mask[maskIdx] / 255.0f;
    
    uchar3 srcPixel = src[srcIdx];
    uchar3 dstPixel = dst[dstIdx];
    
    result[resIdx].x = static_cast<uchar>(srcPixel.x * alpha + dstPixel.x * (1.0f - alpha));
    result[resIdx].y = static_cast<uchar>(srcPixel.y * alpha + dstPixel.y * (1.0f - alpha));
    result[resIdx].z = static_cast<uchar>(srcPixel.z * alpha + dstPixel.z * (1.0f - alpha));
}

// CUDA kernel for simple alpha blending
__global__ void alphaBlendKernel(const uchar3* src, const uchar3* dst,
                                  const uchar* mask, uchar3* result,
                                  int width, int height, int srcStep,
                                  int dstStep, int maskStep, int resStep) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int srcIdx = y * srcStep / sizeof(uchar3) + x;
    int dstIdx = y * dstStep / sizeof(uchar3) + x;
    int maskIdx = y * maskStep + x;
    int resIdx = y * resStep / sizeof(uchar3) + x;
    
    float alpha = mask[maskIdx] / 255.0f;
    
    uchar3 srcPixel = src[srcIdx];
    uchar3 dstPixel = dst[dstIdx];
    
    result[resIdx].x = static_cast<uchar>(srcPixel.x * alpha + dstPixel.x * (1.0f - alpha));
    result[resIdx].y = static_cast<uchar>(srcPixel.y * alpha + dstPixel.y * (1.0f - alpha));
    result[resIdx].z = static_cast<uchar>(srcPixel.z * alpha + dstPixel.z * (1.0f - alpha));
}

// CUDA kernel for color statistics (mean calculation)
__global__ void calcMeanKernel(const uchar3* img, const uchar* mask,
                                float3* partialSums, int* counts,
                                int width, int height, int imgStep, int maskStep) {
    __shared__ float3 sharedSum[256];
    __shared__ int sharedCount[256];
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    
    sharedSum[tid] = make_float3(0, 0, 0);
    sharedCount[tid] = 0;
    
    if (x < width && y < height) {
        int imgIdx = y * imgStep / sizeof(uchar3) + x;
        int maskIdx = y * maskStep + x;
        
        if (mask == nullptr || mask[maskIdx] > 127) {
            uchar3 pixel = img[imgIdx];
            sharedSum[tid] = make_float3(pixel.x, pixel.y, pixel.z);
            sharedCount[tid] = 1;
        }
    }
    
    __syncthreads();
    
    // Reduction
    for (int s = blockDim.x * blockDim.y / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sharedSum[tid].x += sharedSum[tid + s].x;
            sharedSum[tid].y += sharedSum[tid + s].y;
            sharedSum[tid].z += sharedSum[tid + s].z;
            sharedCount[tid] += sharedCount[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        int blockId = blockIdx.y * gridDim.x + blockIdx.x;
        partialSums[blockId] = sharedSum[0];
        counts[blockId] = sharedCount[0];
    }
}

void featheredBlend(const cv::cuda::GpuMat& src,
                    const cv::cuda::GpuMat& dst,
                    const cv::cuda::GpuMat& mask,
                    cv::cuda::GpuMat& result,
                    int featherRadius) {
    CV_Assert(src.size() == dst.size());
    CV_Assert(src.type() == CV_8UC3 && dst.type() == CV_8UC3);
    CV_Assert(mask.type() == CV_8UC1);
    
    // Resize mask if needed
    cv::cuda::GpuMat resizedMask;
    if (mask.size() != src.size()) {
        cv::cuda::resize(mask, resizedMask, src.size());
    } else {
        resizedMask = mask;
    }
    
    // Apply Gaussian blur to feather the mask
    cv::cuda::GpuMat blurredMask;
    auto filter = cv::cuda::createGaussianFilter(CV_8UC1, CV_8UC1, 
                                                  cv::Size(featherRadius * 2 + 1, featherRadius * 2 + 1),
                                                  featherRadius / 2.0);
    filter->apply(resizedMask, blurredMask);
    
    // Allocate result
    result.create(src.size(), src.type());
    
    // Launch kernel
    dim3 block(16, 16);
    dim3 grid((src.cols + block.x - 1) / block.x,
              (src.rows + block.y - 1) / block.y);
    
    featheredBlendKernel<<<grid, block>>>(
        src.ptr<uchar3>(),
        dst.ptr<uchar3>(),
        blurredMask.ptr<uchar>(),
        result.ptr<uchar3>(),
        src.cols, src.rows,
        static_cast<int>(src.step),
        static_cast<int>(dst.step),
        static_cast<int>(blurredMask.step),
        static_cast<int>(result.step)
    );
    
    cudaDeviceSynchronize();
}

void alphaBlend(const cv::cuda::GpuMat& src,
                const cv::cuda::GpuMat& dst,
                const cv::cuda::GpuMat& mask,
                cv::cuda::GpuMat& result) {
    CV_Assert(src.size() == dst.size());
    CV_Assert(src.type() == CV_8UC3 && dst.type() == CV_8UC3);
    
    cv::cuda::GpuMat resizedMask;
    if (mask.size() != src.size()) {
        cv::cuda::resize(mask, resizedMask, src.size());
    } else {
        resizedMask = mask;
    }
    
    result.create(src.size(), src.type());
    
    dim3 block(16, 16);
    dim3 grid((src.cols + block.x - 1) / block.x,
              (src.rows + block.y - 1) / block.y);
    
    alphaBlendKernel<<<grid, block>>>(
        src.ptr<uchar3>(),
        dst.ptr<uchar3>(),
        resizedMask.ptr<uchar>(),
        result.ptr<uchar3>(),
        src.cols, src.rows,
        static_cast<int>(src.step),
        static_cast<int>(dst.step),
        static_cast<int>(resizedMask.step),
        static_cast<int>(result.step)
    );
    
    cudaDeviceSynchronize();
}

void colorTransfer(const cv::cuda::GpuMat& src,
                   const cv::cuda::GpuMat& target,
                   cv::cuda::GpuMat& result) {
    // Convert to LAB color space
    cv::cuda::GpuMat srcLab, targetLab;
    cv::cuda::cvtColor(src, srcLab, cv::COLOR_BGR2Lab);
    cv::cuda::cvtColor(target, targetLab, cv::COLOR_BGR2Lab);
    
    // Split channels
    std::vector<cv::cuda::GpuMat> srcChannels, targetChannels;
    cv::cuda::split(srcLab, srcChannels);
    cv::cuda::split(targetLab, targetChannels);
    
    // Calculate mean and std for each channel
    for (int i = 0; i < 3; i++) {
        cv::Scalar srcMean, srcStd, tgtMean, tgtStd;
        
        // Download to CPU for statistics (can be optimized with custom kernels)
        cv::Mat srcCh, tgtCh;
        srcChannels[i].download(srcCh);
        targetChannels[i].download(tgtCh);
        
        cv::meanStdDev(srcCh, srcMean, srcStd);
        cv::meanStdDev(tgtCh, tgtMean, tgtStd);
        
        if (srcStd[0] > 1e-6) {
            // Transfer statistics
            srcCh.convertTo(srcCh, CV_32F);
            srcCh = ((srcCh - srcMean[0]) * (tgtStd[0] / srcStd[0])) + tgtMean[0];
            srcCh.convertTo(srcCh, CV_8U);
            srcChannels[i].upload(srcCh);
        }
    }
    
    // Merge and convert back
    cv::cuda::GpuMat resultLab;
    cv::cuda::merge(srcChannels, resultLab);
    cv::cuda::cvtColor(resultLab, result, cv::COLOR_Lab2BGR);
}

void blurMask(cv::cuda::GpuMat& mask, int radius) {
    auto filter = cv::cuda::createGaussianFilter(mask.type(), mask.type(),
                                                  cv::Size(radius * 2 + 1, radius * 2 + 1),
                                                  radius / 2.0);
    cv::cuda::GpuMat blurred;
    filter->apply(mask, blurred);
    mask = blurred;
}

bool isCudaAvailable() {
    return cv::cuda::getCudaEnabledDeviceCount() > 0;
}

int getCudaDeviceCount() {
    return cv::cuda::getCudaEnabledDeviceCount();
}

void printCudaInfo() {
    int deviceCount = cv::cuda::getCudaEnabledDeviceCount();
    std::cout << "CUDA Device Count: " << deviceCount << std::endl;
    
    for (int i = 0; i < deviceCount; i++) {
        cv::cuda::DeviceInfo info(i);
        std::cout << "Device " << i << ": " << info.name() << std::endl;
        std::cout << "  Compute Capability: " << info.majorVersion() << "." << info.minorVersion() << std::endl;
        std::cout << "  Total Memory: " << info.totalMemory() / (1024 * 1024) << " MB" << std::endl;
        std::cout << "  Free Memory: " << info.freeMemory() / (1024 * 1024) << " MB" << std::endl;
    }
}

} // namespace cuda
} // namespace facereplacer
