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
    cv::Rect faceRect = m_sourceFace.boundingBox;
    std::cout << "Selfie face detected: " << faceRect << std::endl;
    
    // TIGHTER expansion - just enough to cover face + small margin
    // 15% up (forehead), 10% sides, 10% down (chin)
    int expandTop = static_cast<int>(faceRect.height * 0.15);
    int expandSide = static_cast<int>(faceRect.width * 0.10);
    int expandBottom = static_cast<int>(faceRect.height * 0.10);
    
    cv::Rect headRect;
    headRect.x = std::max(0, faceRect.x - expandSide);
    headRect.y = std::max(0, faceRect.y - expandTop);
    headRect.width = std::min(faceRect.width + expandSide * 2, selfie.cols - headRect.x);
    headRect.height = std::min(faceRect.height + expandTop + expandBottom, selfie.rows - headRect.y);
    
    std::cout << "Head region: " << headRect << std::endl;
    
    // Extract head region
    m_selfieHead = selfie(headRect).clone();
    
    // Create TIGHT elliptical mask - covers face closely
    m_selfieMask = cv::Mat::zeros(m_selfieHead.size(), CV_8UC1);
    
    cv::Point center(m_selfieHead.cols / 2, m_selfieHead.rows / 2);
    
    // Tighter ellipse: 85% of region size (was 95%)
    cv::Size axes(
        static_cast<int>(m_selfieHead.cols * 0.42),  // width/2 * 0.85
        static_cast<int>(m_selfieHead.rows * 0.45)   // height/2 * 0.9
    );
    
    cv::ellipse(m_selfieMask, center, axes, 0, 0, 360, cv::Scalar(255), -1);
    
    // Feather edges (smaller blur for sharper edge)
    cv::GaussianBlur(m_selfieMask, m_selfieMask, cv::Size(21, 21), 10);
    
    int maskPixels = cv::countNonZero(m_selfieMask);
    std::cout << "Mask size: " << m_selfieMask.cols << "x" << m_selfieMask.rows 
              << ", non-zero: " << maskPixels << std::endl;
    
    // Debug output
    cv::imwrite("debug_selfie_head.jpg", m_selfieHead);
    cv::imwrite("debug_selfie_mask.jpg", m_selfieMask);
    
    // Also save overlay for visualization
    cv::Mat overlay;
    cv::cvtColor(m_selfieMask, overlay, cv::COLOR_GRAY2BGR);
    cv::addWeighted(m_selfieHead, 0.7, overlay, 0.3, 0, overlay);
    cv::imwrite("debug_selfie_overlay.jpg", overlay);
    
    std::cout << "Debug images saved: debug_selfie_*.jpg" << std::endl;
}

void FaceReplacer::setTargetFace(const FaceInfo& targetFace) {
    m_targetFace = targetFace;
}

cv::Mat FaceReplacer::processFrame(const cv::Mat& frame) {
    if (m_selfieHead.empty() || m_selfieMask.empty()) {
        return frame.clone();
    }
    
    if (m_targetFace.boundingBox.width <= 0) {
        return frame.clone();
    }
    
    return replaceSegmented(frame, m_targetFace.boundingBox);
}

cv::Mat FaceReplacer::replaceSegmented(const cv::Mat& frame, const cv::Rect& targetRect) {
    cv::Mat result = frame.clone();
    
    // Match expansion ratio used for selfie
    int expandTop = static_cast<int>(targetRect.height * 0.15);
    int expandSide = static_cast<int>(targetRect.width * 0.10);
    int expandBottom = static_cast<int>(targetRect.height * 0.10);
    
    cv::Rect tgtHead;
    tgtHead.x = std::max(0, targetRect.x - expandSide);
    tgtHead.y = std::max(0, targetRect.y - expandTop);
    tgtHead.width = std::min(targetRect.width + expandSide * 2, frame.cols - tgtHead.x);
    tgtHead.height = std::min(targetRect.height + expandTop + expandBottom, frame.rows - tgtHead.y);
    
    if (tgtHead.width <= 0 || tgtHead.height <= 0) {
        return result;
    }
    
    // Resize selfie to match target
    cv::Mat resizedHead, resizedMask;
    cv::resize(m_selfieHead, resizedHead, tgtHead.size(), 0, 0, cv::INTER_LINEAR);
    cv::resize(m_selfieMask, resizedMask, tgtHead.size(), 0, 0, cv::INTER_LINEAR);
    
    // Color matching
    if (m_config.colorCorrection) {
        resizedHead = matchColors(resizedHead, frame(tgtHead), resizedMask);
    }
    
    // Blend
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
    
    cv::Mat srcLab, tgtLab;
    cv::cvtColor(source, srcLab, cv::COLOR_BGR2Lab);
    cv::cvtColor(target, tgtLab, cv::COLOR_BGR2Lab);
    
    cv::Scalar srcMean, srcStd, tgtMean, tgtStd;
    
    cv::Mat useMask = mask;
    if (!mask.empty() && mask.size() != source.size()) {
        cv::resize(mask, useMask, source.size());
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

// Stub implementations for interface compatibility
cv::Mat FaceReplacer::replaceRectToRect(const cv::Mat& frame, const cv::Mat& source,
                                         const cv::Rect& targetRect) {
    return replaceSegmented(frame, targetRect);
}

cv::Mat FaceReplacer::replaceLive(const cv::Mat& frame, const FaceInfo& targetFace) {
    return replaceSegmented(frame, targetFace.boundingBox);
}

cv::Mat FaceReplacer::adjustLighting(const cv::Mat& source, const cv::Mat& target,
                                      const cv::Rect& region) { return source.clone(); }

cv::Mat FaceReplacer::poissonBlend(const cv::Mat& source, const cv::Mat& target,
                                    const cv::Mat& mask, const cv::Point& center) { return target.clone(); }

cv::Mat FaceReplacer::warpFaceToTarget(const cv::Mat& source, const FaceInfo& sourceFace,
                                        const FaceInfo& targetFace) { return source.clone(); }

cv::Mat FaceReplacer::applyTemporalSmoothing(const cv::Mat& currentResult) { return currentResult.clone(); }

void FaceReplacer::updateBuffers(const cv::Mat& frame, const FaceInfo& face) {}

#ifdef USE_CUDA
cv::Mat FaceReplacer::blendGPU(const cv::Mat& source, const cv::Mat& target,
                                const cv::Mat& mask) { return target.clone(); }
#endif

LiveFaceReplacer::LiveFaceReplacer(const Config& config) : FaceReplacer(config) {}
cv::Mat LiveFaceReplacer::processWithExpression(const cv::Mat& frame, const FaceInfo& targetFace) {
    return replaceSegmented(frame, targetFace.boundingBox);
}
cv::Mat LiveFaceReplacer::warpToPose(const cv::Mat& sourceFace,
    const std::vector<cv::Point2f>& sourceLandmarks,
    const std::vector<cv::Point2f>& targetLandmarks) { return sourceFace.clone(); }
void LiveFaceReplacer::calculateDelaunay(const std::vector<cv::Point2f>& points,
    const cv::Rect& bounds) {}
cv::Mat LiveFaceReplacer::warpTriangle(const cv::Mat& src, const cv::Mat& dst,
    const std::vector<cv::Point2f>& srcTri,
    const std::vector<cv::Point2f>& dstTri) { return dst.clone(); }

} // namespace facereplacer
