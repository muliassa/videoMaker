#include "face_replacer.hpp"
#include "face_detector.hpp"
#include "segmentation.hpp"

#ifdef USE_CUDA
#include "cuda/gpu_blend.cuh"
#endif

#include <iostream>

namespace facereplacer {

FaceReplacer::FaceReplacer(const Config& config) : m_config(config) {
    m_detector = std::make_unique<FaceDetector>(config);
    m_segmentation = std::make_unique<Segmentation>(config);
    
#ifdef USE_CUDA
    if (m_config.useGPU && !cuda::isCudaAvailable()) {
        std::cerr << "Warning: CUDA not available, falling back to CPU" << std::endl;
        m_config.useGPU = false;
    }
    if (m_config.useGPU) {
        cuda::printCudaInfo();
    }
#else
    if (m_config.useGPU) {
        std::cerr << "Note: Built without CUDA, using CPU" << std::endl;
        m_config.useGPU = false;
    }
#endif
}

FaceReplacer::~FaceReplacer() = default;

std::vector<FaceInfo> FaceReplacer::detectFaces(const cv::Mat& image) {
    return m_detector->detect(image);
}

cv::Mat FaceReplacer::markFace(const cv::Mat& frame, int faceIndex) {
    cv::Mat result = frame.clone();
    auto faces = m_detector->detect(frame);
    
    if (faces.empty()) {
        std::cerr << "No faces detected!" << std::endl;
        return result;
    }
    
    if (faceIndex >= static_cast<int>(faces.size())) {
        faceIndex = 0;
    }
    
    for (size_t i = 0; i < faces.size(); i++) {
        cv::Scalar color = (i == static_cast<size_t>(faceIndex)) ? 
                           cv::Scalar(0, 255, 0) : cv::Scalar(255, 0, 0);
        int thickness = (i == static_cast<size_t>(faceIndex)) ? 3 : 1;
        
        cv::rectangle(result, faces[i].boundingBox, color, thickness);
        cv::putText(result, "Face " + std::to_string(i), 
                    cv::Point(faces[i].boundingBox.x, faces[i].boundingBox.y - 10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 2);
    }
    
    m_targetFace = faces[faceIndex];
    return result;
}

void FaceReplacer::setSourceImage(const cv::Mat& selfie) {
    m_sourceImage = selfie.clone();
    
    auto faces = m_detector->detect(selfie);
    if (!faces.empty()) {
        m_sourceFace = faces[0];
        std::cout << "Source face: " << m_sourceFace.boundingBox << std::endl;
        
        // Create mask for source face
        if (m_config.mode != ReplacementMode::RECT_TO_RECT) {
            m_sourceMask = m_segmentation->segmentHead(selfie, m_sourceFace.boundingBox);
            int nonZero = cv::countNonZero(m_sourceMask);
            std::cout << "Source mask pixels: " << nonZero << std::endl;
        }
    } else {
        std::cerr << "Warning: No face detected in selfie!" << std::endl;
    }
}

void FaceReplacer::setTargetFace(const FaceInfo& targetFace) {
    m_targetFace = targetFace;
}

cv::Mat FaceReplacer::processFrame(const cv::Mat& frame) {
    if (m_sourceImage.empty() || m_sourceFace.boundingBox.width <= 0) {
        std::cerr << "Error: Source not initialized" << std::endl;
        return frame.clone();
    }
    
    if (m_targetFace.boundingBox.width <= 0) {
        std::cerr << "Error: Target face not set" << std::endl;
        return frame.clone();
    }
    
    cv::Mat result;
    
    switch (m_config.mode) {
        case ReplacementMode::RECT_TO_RECT:
            result = replaceRectToRect(frame, m_sourceImage, m_targetFace.boundingBox);
            break;
        case ReplacementMode::HEAD_SEGMENTED:
            result = replaceSegmented(frame, m_sourceImage, m_sourceMask, m_targetFace.boundingBox);
            break;
        case ReplacementMode::LIVE_ANIMATED:
            result = replaceLive(frame, m_targetFace);
            break;
        default:
            result = frame.clone();
    }
    
    return result;
}

cv::Mat FaceReplacer::replaceRectToRect(const cv::Mat& frame, const cv::Mat& source,
                                         const cv::Rect& targetRect) {
    cv::Mat result = frame.clone();
    
    // Source face region
    cv::Rect srcRect = m_sourceFace.boundingBox;
    srcRect = clampRect(srcRect, source.size());
    if (srcRect.width <= 0 || srcRect.height <= 0) return result;
    
    // Target region (where to place)
    cv::Rect tgtRect = targetRect;
    tgtRect = clampRect(tgtRect, frame.size());
    if (tgtRect.width <= 0 || tgtRect.height <= 0) return result;
    
    // Extract and resize source face to target size
    cv::Mat srcFace = source(srcRect).clone();
    cv::Mat resizedFace;
    cv::resize(srcFace, resizedFace, tgtRect.size());
    
    // Color correction
    if (m_config.colorCorrection) {
        resizedFace = matchColors(resizedFace, frame(tgtRect));
    }
    
    // Create elliptical mask for soft blending
    cv::Mat mask = cv::Mat::zeros(resizedFace.size(), CV_8UC1);
    cv::ellipse(mask, 
                cv::Point(mask.cols / 2, mask.rows / 2),
                cv::Size(mask.cols / 2 - 2, mask.rows / 2 - 2),
                0, 0, 360, cv::Scalar(255), -1);
    cv::GaussianBlur(mask, mask, cv::Size(21, 21), 10);
    
    // Alpha blend
    cv::Mat targetROI = result(tgtRect);
    for (int y = 0; y < targetROI.rows; y++) {
        for (int x = 0; x < targetROI.cols; x++) {
            float alpha = mask.at<uchar>(y, x) / 255.0f;
            cv::Vec3b& dst = targetROI.at<cv::Vec3b>(y, x);
            cv::Vec3b src = resizedFace.at<cv::Vec3b>(y, x);
            dst[0] = static_cast<uchar>(src[0] * alpha + dst[0] * (1 - alpha));
            dst[1] = static_cast<uchar>(src[1] * alpha + dst[1] * (1 - alpha));
            dst[2] = static_cast<uchar>(src[2] * alpha + dst[2] * (1 - alpha));
        }
    }
    
    return result;
}

cv::Mat FaceReplacer::replaceSegmented(const cv::Mat& frame, const cv::Mat& source,
                                        const cv::Mat& sourceMask, const cv::Rect& targetRect) {
    cv::Mat result = frame.clone();
    
    // Get expanded regions (include some context around face)
    cv::Rect srcRect = scaleRect(m_sourceFace.boundingBox, 1.5f);
    srcRect = clampRect(srcRect, source.size());
    
    cv::Rect tgtRect = scaleRect(targetRect, 1.5f);
    tgtRect = clampRect(tgtRect, frame.size());
    
    if (srcRect.width <= 0 || srcRect.height <= 0 ||
        tgtRect.width <= 0 || tgtRect.height <= 0) {
        return result;
    }
    
    // Extract source face region and its mask
    cv::Mat srcRegion = source(srcRect).clone();
    cv::Mat maskRegion;
    
    if (!sourceMask.empty()) {
        maskRegion = sourceMask(srcRect).clone();
    } else {
        // Create elliptical mask as fallback
        maskRegion = cv::Mat::zeros(srcRect.size(), CV_8UC1);
        cv::ellipse(maskRegion,
                    cv::Point(maskRegion.cols / 2, maskRegion.rows / 2),
                    cv::Size(maskRegion.cols * 2 / 5, maskRegion.rows * 2 / 5),
                    0, 0, 360, cv::Scalar(255), -1);
    }
    
    // Resize to target size
    cv::Mat resizedSrc, resizedMask;
    cv::resize(srcRegion, resizedSrc, tgtRect.size());
    cv::resize(maskRegion, resizedMask, tgtRect.size());
    
    // Color correction
    if (m_config.colorCorrection) {
        resizedSrc = matchColors(resizedSrc, frame(tgtRect), resizedMask);
    }
    
    // Feather the mask edges
    resizedMask = m_segmentation->featherMask(resizedMask, m_config.featherRadius);
    
    // Blend into target
    cv::Mat targetROI = result(tgtRect);
    
    for (int y = 0; y < targetROI.rows; y++) {
        for (int x = 0; x < targetROI.cols; x++) {
            float alpha = resizedMask.at<uchar>(y, x) / 255.0f;
            if (alpha > 0.01f) {
                cv::Vec3b& dst = targetROI.at<cv::Vec3b>(y, x);
                cv::Vec3b src = resizedSrc.at<cv::Vec3b>(y, x);
                dst[0] = static_cast<uchar>(src[0] * alpha + dst[0] * (1 - alpha));
                dst[1] = static_cast<uchar>(src[1] * alpha + dst[1] * (1 - alpha));
                dst[2] = static_cast<uchar>(src[2] * alpha + dst[2] * (1 - alpha));
            }
        }
    }
    
    return result;
}

cv::Mat FaceReplacer::replaceLive(const cv::Mat& frame, const FaceInfo& targetFace) {
    // For live mode, use segmented replacement with face tracking
    return replaceSegmented(frame, m_sourceImage, m_sourceMask, targetFace.boundingBox);
}

cv::Mat FaceReplacer::matchColors(const cv::Mat& source, const cv::Mat& target,
                                   const cv::Mat& mask) {
    cv::Mat result = source.clone();
    cv::Mat srcLab, tgtLab;
    
    cv::cvtColor(source, srcLab, cv::COLOR_BGR2Lab);
    cv::cvtColor(target, tgtLab, cv::COLOR_BGR2Lab);
    
    cv::Scalar srcMean, srcStd, tgtMean, tgtStd;
    
    if (mask.empty()) {
        cv::meanStdDev(srcLab, srcMean, srcStd);
        cv::meanStdDev(tgtLab, tgtMean, tgtStd);
    } else {
        cv::Mat resizedMask;
        if (mask.size() != source.size()) {
            cv::resize(mask, resizedMask, source.size());
        } else {
            resizedMask = mask;
        }
        cv::meanStdDev(srcLab, srcMean, srcStd, resizedMask);
        
        cv::Mat tgtMaskResized;
        if (mask.size() != target.size()) {
            cv::resize(mask, tgtMaskResized, target.size());
        } else {
            tgtMaskResized = mask;
        }
        cv::meanStdDev(tgtLab, tgtMean, tgtStd, tgtMaskResized);
    }
    
    std::vector<cv::Mat> channels;
    cv::split(srcLab, channels);
    
    for (int i = 0; i < 3; i++) {
        if (srcStd[i] > 1e-6) {
            channels[i].convertTo(channels[i], CV_32F);
            channels[i] = ((channels[i] - srcMean[i]) * (tgtStd[i] / srcStd[i])) + tgtMean[i];
            channels[i].convertTo(channels[i], CV_8U);
        }
    }
    
    cv::Mat resultLab;
    cv::merge(channels, resultLab);
    cv::cvtColor(resultLab, result, cv::COLOR_Lab2BGR);
    
    return result;
}

cv::Mat FaceReplacer::adjustLighting(const cv::Mat& source, const cv::Mat& target,
                                      const cv::Rect& region) {
    cv::Mat result = source.clone();
    
    cv::Rect clamped = clampRect(region, target.size());
    if (clamped.width <= 0 || clamped.height <= 0) return result;
    
    cv::Mat tgtGray, srcGray;
    cv::cvtColor(target(clamped), tgtGray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(source, srcGray, cv::COLOR_BGR2GRAY);
    
    double tgtMean = cv::mean(tgtGray)[0];
    double srcMean = cv::mean(srcGray)[0];
    
    if (srcMean > 1e-6) {
        double ratio = std::clamp(tgtMean / srcMean, 0.5, 2.0);
        result.convertTo(result, -1, ratio, 0);
    }
    
    return result;
}

cv::Mat FaceReplacer::poissonBlend(const cv::Mat& source, const cv::Mat& target,
                                    const cv::Mat& mask, const cv::Point& center) {
    cv::Mat result;
    try {
        cv::seamlessClone(source, target, mask, center, result, cv::NORMAL_CLONE);
    } catch (...) {
        result = target.clone();
    }
    return result;
}

#ifdef USE_CUDA
cv::Mat FaceReplacer::blendGPU(const cv::Mat& source, const cv::Mat& target,
                                const cv::Mat& mask) {
    cv::cuda::GpuMat gpuSrc, gpuDst, gpuMask, gpuResult;
    gpuSrc.upload(source);
    gpuDst.upload(target);
    gpuMask.upload(mask);
    
    cuda::featheredBlend(gpuSrc, gpuDst, gpuMask, gpuResult, m_config.featherRadius);
    
    cv::Mat result;
    gpuResult.download(result);
    return result;
}
#endif

cv::Mat FaceReplacer::warpFaceToTarget(const cv::Mat& source, const FaceInfo& sourceFace,
                                        const FaceInfo& targetFace) {
    cv::Mat result = cv::Mat::zeros(source.size(), source.type());
    
    cv::Point2f srcPts[3], dstPts[3];
    
    srcPts[0] = cv::Point2f(sourceFace.boundingBox.x, sourceFace.boundingBox.y);
    srcPts[1] = cv::Point2f(sourceFace.boundingBox.x + sourceFace.boundingBox.width, sourceFace.boundingBox.y);
    srcPts[2] = cv::Point2f(sourceFace.boundingBox.x, sourceFace.boundingBox.y + sourceFace.boundingBox.height);
    
    dstPts[0] = cv::Point2f(targetFace.boundingBox.x, targetFace.boundingBox.y);
    dstPts[1] = cv::Point2f(targetFace.boundingBox.x + targetFace.boundingBox.width, targetFace.boundingBox.y);
    dstPts[2] = cv::Point2f(targetFace.boundingBox.x, targetFace.boundingBox.y + targetFace.boundingBox.height);
    
    cv::Mat warpMat = cv::getAffineTransform(srcPts, dstPts);
    cv::warpAffine(source, result, warpMat, result.size());
    
    return result;
}

cv::Mat FaceReplacer::applyTemporalSmoothing(const cv::Mat& currentResult) {
    if (m_frameBuffer.empty()) return currentResult;
    
    float decay = 0.7f;
    std::vector<float> weights(m_frameBuffer.size() + 1);
    float total = 0;
    
    for (size_t i = 0; i < weights.size(); i++) {
        weights[i] = std::pow(decay, static_cast<float>(weights.size() - 1 - i));
        total += weights[i];
    }
    for (auto& w : weights) w /= total;
    
    cv::Mat result = cv::Mat::zeros(currentResult.size(), CV_32FC3);
    
    for (size_t i = 0; i < m_frameBuffer.size(); i++) {
        cv::Mat f;
        m_frameBuffer[i].convertTo(f, CV_32FC3);
        result += f * weights[i];
    }
    
    cv::Mat curr;
    currentResult.convertTo(curr, CV_32FC3);
    result += curr * weights.back();
    result.convertTo(result, CV_8UC3);
    
    return result;
}

void FaceReplacer::updateBuffers(const cv::Mat& frame, const FaceInfo& face) {
    m_frameBuffer.push_back(frame.clone());
    m_faceBuffer.push_back(face);
    
    if (static_cast<int>(m_frameBuffer.size()) > m_config.temporalSmoothing) {
        m_frameBuffer.erase(m_frameBuffer.begin());
        m_faceBuffer.erase(m_faceBuffer.begin());
    }
}

// LiveFaceReplacer
LiveFaceReplacer::LiveFaceReplacer(const Config& config) : FaceReplacer(config) {}

cv::Mat LiveFaceReplacer::processWithExpression(const cv::Mat& frame, const FaceInfo& targetFace) {
    return replaceLive(frame, targetFace);
}

cv::Mat LiveFaceReplacer::warpToPose(const cv::Mat& sourceFace,
                                      const std::vector<cv::Point2f>& sourceLandmarks,
                                      const std::vector<cv::Point2f>& targetLandmarks) {
    if (sourceLandmarks.size() != targetLandmarks.size() || sourceLandmarks.size() < 3) {
        return sourceFace.clone();
    }
    
    cv::Mat result = cv::Mat::zeros(sourceFace.size(), sourceFace.type());
    cv::Rect bounds(0, 0, sourceFace.cols, sourceFace.rows);
    calculateDelaunay(targetLandmarks, bounds);
    
    for (const auto& tri : m_triangles) {
        std::vector<cv::Point2f> srcTri(3), dstTri(3);
        bool valid = true;
        
        for (int i = 0; i < 3 && valid; i++) {
            cv::Point2f pt(tri[i * 2], tri[i * 2 + 1]);
            int idx = -1;
            float minDist = 1e9;
            
            for (size_t j = 0; j < targetLandmarks.size(); j++) {
                float d = cv::norm(pt - targetLandmarks[j]);
                if (d < minDist) { minDist = d; idx = j; }
            }
            
            if (idx < 0 || minDist > 5.0f) valid = false;
            else { srcTri[i] = sourceLandmarks[idx]; dstTri[i] = targetLandmarks[idx]; }
        }
        
        if (valid) result = warpTriangle(sourceFace, result, srcTri, dstTri);
    }
    
    return result;
}

void LiveFaceReplacer::calculateDelaunay(const std::vector<cv::Point2f>& points, const cv::Rect& bounds) {
    m_triangles.clear();
    cv::Subdiv2D subdiv(bounds);
    
    for (const auto& pt : points) {
        if (bounds.contains(pt)) subdiv.insert(pt);
    }
    
    subdiv.getTriangleList(m_triangles);
    
    m_triangles.erase(
        std::remove_if(m_triangles.begin(), m_triangles.end(),
            [&bounds](const cv::Vec6f& tri) {
                for (int i = 0; i < 3; i++) {
                    if (!bounds.contains(cv::Point2f(tri[i*2], tri[i*2+1]))) return true;
                }
                return false;
            }),
        m_triangles.end());
}

cv::Mat LiveFaceReplacer::warpTriangle(const cv::Mat& src, const cv::Mat& dst,
                                        const std::vector<cv::Point2f>& srcTri,
                                        const std::vector<cv::Point2f>& dstTri) {
    cv::Mat result = dst.clone();
    
    cv::Rect srcR = clampRect(cv::boundingRect(srcTri), src.size());
    cv::Rect dstR = clampRect(cv::boundingRect(dstTri), result.size());
    
    if (srcR.width <= 0 || srcR.height <= 0 || dstR.width <= 0 || dstR.height <= 0) return result;
    
    std::vector<cv::Point2f> srcLocal(3), dstLocal(3);
    std::vector<cv::Point> dstInt(3);
    
    for (int i = 0; i < 3; i++) {
        srcLocal[i] = cv::Point2f(srcTri[i].x - srcR.x, srcTri[i].y - srcR.y);
        dstLocal[i] = cv::Point2f(dstTri[i].x - dstR.x, dstTri[i].y - dstR.y);
        dstInt[i] = cv::Point(dstLocal[i].x, dstLocal[i].y);
    }
    
    cv::Mat warpMat = cv::getAffineTransform(srcLocal, dstLocal);
    cv::Mat warped;
    cv::warpAffine(src(srcR), warped, warpMat, dstR.size());
    
    cv::Mat mask = cv::Mat::zeros(dstR.height, dstR.width, CV_8UC1);
    cv::fillConvexPoly(mask, dstInt, cv::Scalar(255));
    
    warped.copyTo(result(dstR), mask);
    return result;
}

} // namespace facereplacer
