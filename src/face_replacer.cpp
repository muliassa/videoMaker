#include "face_replacer.hpp"
#include "face_detector.hpp"
#include "segmentation.hpp"
#include <iostream>

namespace facereplacer {

FaceReplacer::FaceReplacer(const Config& config) : m_config(config) {
    m_detector = std::make_unique<FaceDetector>(config);
    m_segmentation = std::make_unique<Segmentation>(config);
}

FaceReplacer::~FaceReplacer() = default;

std::vector<FaceInfo> FaceReplacer::detectFaces(const cv::Mat& image) {
    return m_detector->detect(image);
}

cv::Mat FaceReplacer::markFace(const cv::Mat& frame, int faceIndex) {
    cv::Mat result = frame.clone();
    auto faces = m_detector->detect(frame);
    
    for (size_t i = 0; i < faces.size(); i++) {
        cv::Scalar color = (i == static_cast<size_t>(faceIndex)) ? 
                           cv::Scalar(0, 255, 0) : cv::Scalar(255, 0, 0);
        cv::rectangle(result, faces[i].boundingBox, color, 2);
    }
    
    if (faceIndex < static_cast<int>(faces.size())) {
        m_targetFace = faces[faceIndex];
    }
    return result;
}

void FaceReplacer::setSourceImage(const cv::Mat& selfie) {
    m_sourceImage = selfie.clone();
    
    // Detect face in selfie
    auto faces = m_detector->detect(selfie);
    if (faces.empty()) {
        std::cerr << "ERROR: No face detected in selfie!" << std::endl;
        return;
    }
    
    m_sourceFace = faces[0];
    std::cout << "Selfie face: " << m_sourceFace.boundingBox << std::endl;
    
    // Create expanded region around face (include forehead, chin, sides)
    cv::Rect faceRect = m_sourceFace.boundingBox;
    
    // Expand: 40% up (forehead/hair), 25% sides (ears), 20% down (chin/neck)
    int expandTop = static_cast<int>(faceRect.height * 0.4);
    int expandSide = static_cast<int>(faceRect.width * 0.25);
    int expandBottom = static_cast<int>(faceRect.height * 0.2);
    
    cv::Rect headRect;
    headRect.x = std::max(0, faceRect.x - expandSide);
    headRect.y = std::max(0, faceRect.y - expandTop);
    headRect.width = std::min(faceRect.width + expandSide * 2, selfie.cols - headRect.x);
    headRect.height = std::min(faceRect.height + expandTop + expandBottom, selfie.rows - headRect.y);
    
    std::cout << "Head region: " << headRect << std::endl;
    
    // Extract head region from selfie
    m_selfieHead = selfie(headRect).clone();
    
    // Create elliptical mask for the head (relative to extracted region)
    m_selfieMask = cv::Mat::zeros(m_selfieHead.size(), CV_8UC1);
    
    cv::Point center(m_selfieHead.cols / 2, m_selfieHead.rows / 2);
    cv::Size axes(m_selfieHead.cols / 2 - 5, m_selfieHead.rows / 2 - 5);
    cv::ellipse(m_selfieMask, center, axes, 0, 0, 360, cv::Scalar(255), -1);
    
    // Feather the mask edges
    cv::GaussianBlur(m_selfieMask, m_selfieMask, cv::Size(31, 31), 15);
    
    int maskPixels = cv::countNonZero(m_selfieMask);
    std::cout << "Selfie mask: " << m_selfieMask.cols << "x" << m_selfieMask.rows 
              << ", pixels=" << maskPixels << std::endl;
    
    // Debug: save mask
    cv::imwrite("debug_selfie_mask.jpg", m_selfieMask);
    cv::imwrite("debug_selfie_head.jpg", m_selfieHead);
    std::cout << "Debug images saved: debug_selfie_mask.jpg, debug_selfie_head.jpg" << std::endl;
}

void FaceReplacer::setTargetFace(const FaceInfo& targetFace) {
    m_targetFace = targetFace;
}

cv::Mat FaceReplacer::processFrame(const cv::Mat& frame) {
    if (m_selfieHead.empty() || m_selfieMask.empty()) {
        std::cerr << "Source not initialized!" << std::endl;
        return frame.clone();
    }
    
    if (m_targetFace.boundingBox.width <= 0) {
        return frame.clone();
    }
    
    return replaceSegmented(frame, m_targetFace.boundingBox);
}

cv::Mat FaceReplacer::replaceSegmented(const cv::Mat& frame, const cv::Rect& targetRect) {
    cv::Mat result = frame.clone();
    
    // Expand target rect similar to source (to match proportions)
    int expandTop = static_cast<int>(targetRect.height * 0.4);
    int expandSide = static_cast<int>(targetRect.width * 0.25);
    int expandBottom = static_cast<int>(targetRect.height * 0.2);
    
    cv::Rect tgtHead;
    tgtHead.x = std::max(0, targetRect.x - expandSide);
    tgtHead.y = std::max(0, targetRect.y - expandTop);
    tgtHead.width = std::min(targetRect.width + expandSide * 2, frame.cols - tgtHead.x);
    tgtHead.height = std::min(targetRect.height + expandTop + expandBottom, frame.rows - tgtHead.y);
    
    if (tgtHead.width <= 0 || tgtHead.height <= 0) {
        return result;
    }
    
    // Resize selfie head and mask to match target size
    cv::Mat resizedHead, resizedMask;
    cv::resize(m_selfieHead, resizedHead, tgtHead.size());
    cv::resize(m_selfieMask, resizedMask, tgtHead.size());
    
    // Color correction: match selfie colors to target region
    if (m_config.colorCorrection) {
        resizedHead = matchColors(resizedHead, frame(tgtHead), resizedMask);
    }
    
    // Alpha blend into result
    cv::Mat targetROI = result(tgtHead);
    
    for (int y = 0; y < targetROI.rows; y++) {
        for (int x = 0; x < targetROI.cols; x++) {
            float alpha = resizedMask.at<uchar>(y, x) / 255.0f;
            if (alpha > 0.01f) {
                cv::Vec3b& dst = targetROI.at<cv::Vec3b>(y, x);
                const cv::Vec3b& src = resizedHead.at<cv::Vec3b>(y, x);
                
                dst[0] = static_cast<uchar>(src[0] * alpha + dst[0] * (1 - alpha));
                dst[1] = static_cast<uchar>(src[1] * alpha + dst[1] * (1 - alpha));
                dst[2] = static_cast<uchar>(src[2] * alpha + dst[2] * (1 - alpha));
            }
        }
    }
    
    return result;
}

cv::Mat FaceReplacer::matchColors(const cv::Mat& source, const cv::Mat& target, 
                                   const cv::Mat& mask) {
    cv::Mat result = source.clone();
    
    // Convert to LAB
    cv::Mat srcLab, tgtLab;
    cv::cvtColor(source, srcLab, cv::COLOR_BGR2Lab);
    cv::cvtColor(target, tgtLab, cv::COLOR_BGR2Lab);
    
    // Calculate mean/std with mask
    cv::Scalar srcMean, srcStd, tgtMean, tgtStd;
    
    cv::Mat useMask;
    if (!mask.empty() && mask.size() == source.size()) {
        useMask = mask;
    }
    
    if (useMask.empty()) {
        cv::meanStdDev(srcLab, srcMean, srcStd);
        cv::meanStdDev(tgtLab, tgtMean, tgtStd);
    } else {
        cv::meanStdDev(srcLab, srcMean, srcStd, useMask);
        cv::Mat tgtMask;
        cv::resize(useMask, tgtMask, target.size());
        cv::meanStdDev(tgtLab, tgtMean, tgtStd, tgtMask);
    }
    
    // Transfer colors
    std::vector<cv::Mat> channels;
    cv::split(srcLab, channels);
    
    for (int i = 0; i < 3; i++) {
        if (srcStd[i] > 1.0) {
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

// Unused methods - keeping for interface compatibility
cv::Mat FaceReplacer::replaceRectToRect(const cv::Mat& frame, const cv::Mat& source,
                                         const cv::Rect& targetRect) {
    return replaceSegmented(frame, targetRect);
}

cv::Mat FaceReplacer::replaceLive(const cv::Mat& frame, const FaceInfo& targetFace) {
    return replaceSegmented(frame, targetFace.boundingBox);
}

cv::Mat FaceReplacer::adjustLighting(const cv::Mat& source, const cv::Mat& target,
                                      const cv::Rect& region) {
    return source.clone();
}

cv::Mat FaceReplacer::poissonBlend(const cv::Mat& source, const cv::Mat& target,
                                    const cv::Mat& mask, const cv::Point& center) {
    return target.clone();
}

cv::Mat FaceReplacer::warpFaceToTarget(const cv::Mat& source, const FaceInfo& sourceFace,
                                        const FaceInfo& targetFace) {
    return source.clone();
}

cv::Mat FaceReplacer::applyTemporalSmoothing(const cv::Mat& currentResult) {
    return currentResult.clone();
}

void FaceReplacer::updateBuffers(const cv::Mat& frame, const FaceInfo& face) {
}

#ifdef USE_CUDA
cv::Mat FaceReplacer::blendGPU(const cv::Mat& source, const cv::Mat& target,
                                const cv::Mat& mask) {
    return target.clone();
}
#endif

// LiveFaceReplacer stubs
LiveFaceReplacer::LiveFaceReplacer(const Config& config) : FaceReplacer(config) {}

cv::Mat LiveFaceReplacer::processWithExpression(const cv::Mat& frame, const FaceInfo& targetFace) {
    return replaceSegmented(frame, targetFace.boundingBox);
}

cv::Mat LiveFaceReplacer::warpToPose(const cv::Mat& sourceFace,
                                      const std::vector<cv::Point2f>& sourceLandmarks,
                                      const std::vector<cv::Point2f>& targetLandmarks) {
    return sourceFace.clone();
}

void LiveFaceReplacer::calculateDelaunay(const std::vector<cv::Point2f>& points,
                                          const cv::Rect& bounds) {
}

cv::Mat LiveFaceReplacer::warpTriangle(const cv::Mat& src, const cv::Mat& dst,
                                        const std::vector<cv::Point2f>& srcTri,
                                        const std::vector<cv::Point2f>& dstTri) {
    return dst.clone();
}

} // namespace facereplacer
