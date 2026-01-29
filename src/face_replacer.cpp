#include "face_replacer.hpp"
#include "face_detector.hpp"
#include "segmentation.hpp"
#include <iostream>
#include <deque>

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
    
    auto faces = m_detector->detect(selfie);
    if (faces.empty()) {
        std::cerr << "ERROR: No face detected in selfie!" << std::endl;
        return;
    }
    
    m_sourceFace = faces[0];
    cv::Rect faceRect = m_sourceFace.boundingBox;
    std::cout << "Selfie face bbox: " << faceRect << std::endl;
    
    // Expand for head region (GrabCut needs context)
    int expandTop = static_cast<int>(faceRect.height * 0.6);
    int expandSide = static_cast<int>(faceRect.width * 0.4);
    int expandBottom = static_cast<int>(faceRect.height * 0.25);
    
    cv::Rect headRect;
    headRect.x = std::max(0, faceRect.x - expandSide);
    headRect.y = std::max(0, faceRect.y - expandTop);
    headRect.width = std::min(faceRect.width + expandSide * 2, selfie.cols - headRect.x);
    headRect.height = std::min(faceRect.height + expandTop + expandBottom, selfie.rows - headRect.y);
    
    m_selfieHead = selfie(headRect).clone();
    
    // Face rect relative to extracted region
    cv::Rect faceInRegion(
        faceRect.x - headRect.x,
        faceRect.y - headRect.y,
        faceRect.width,
        faceRect.height
    );
    
    m_selfieFaceCenterInRegion = cv::Point(
        faceInRegion.x + faceInRegion.width/2,
        faceInRegion.y + faceInRegion.height/2
    );
    
    m_faceToRegionRatio = static_cast<float>(faceRect.width) / headRect.width;
    
    std::cout << "Extracted region: " << headRect << std::endl;
    std::cout << "Face in region: " << faceInRegion << std::endl;
    
    // === GRABCUT SEGMENTATION ===
    std::cout << "Running GrabCut segmentation..." << std::endl;
    
    cv::Mat mask = cv::Mat::zeros(m_selfieHead.size(), CV_8UC1);
    cv::Mat bgModel, fgModel;
    
    // Initialize mask: face area is probably foreground
    // Shrink faceInRegion slightly for definite foreground
    cv::Rect defForeground = faceInRegion;
    defForeground.x += defForeground.width / 6;
    defForeground.y += defForeground.height / 6;
    defForeground.width = defForeground.width * 2 / 3;
    defForeground.height = defForeground.height * 2 / 3;
    
    // GrabCut rect - the area that MIGHT contain foreground
    cv::Rect grabRect(5, 5, m_selfieHead.cols - 10, m_selfieHead.rows - 10);
    
    try {
        // First pass with rect
        cv::grabCut(m_selfieHead, mask, grabRect, bgModel, fgModel, 3, cv::GC_INIT_WITH_RECT);
        
        // Mark face center as definite foreground
        mask(defForeground).setTo(cv::GC_FGD);
        
        // Second pass with mask
        cv::grabCut(m_selfieHead, mask, grabRect, bgModel, fgModel, 2, cv::GC_INIT_WITH_MASK);
        
    } catch (const cv::Exception& e) {
        std::cerr << "GrabCut failed: " << e.what() << std::endl;
    }
    
    // Extract foreground mask
    m_selfieMask = cv::Mat::zeros(m_selfieHead.size(), CV_8UC1);
    m_selfieMask.setTo(255, (mask == cv::GC_FGD) | (mask == cv::GC_PR_FGD));
    
    // Clean up mask
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::morphologyEx(m_selfieMask, m_selfieMask, cv::MORPH_CLOSE, kernel, cv::Point(-1,-1), 2);
    cv::morphologyEx(m_selfieMask, m_selfieMask, cv::MORPH_OPEN, kernel);
    
    // Keep only largest contour
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(m_selfieMask.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    
    if (!contours.empty()) {
        int maxIdx = 0;
        double maxArea = 0;
        for (size_t i = 0; i < contours.size(); i++) {
            double area = cv::contourArea(contours[i]);
            if (area > maxArea) {
                maxArea = area;
                maxIdx = static_cast<int>(i);
            }
        }
        m_selfieMask = cv::Mat::zeros(m_selfieMask.size(), CV_8UC1);
        cv::drawContours(m_selfieMask, contours, maxIdx, cv::Scalar(255), -1);
    }
    
    // Feather edges
    cv::GaussianBlur(m_selfieMask, m_selfieMask, cv::Size(21, 21), 10);
    
    int maskPixels = cv::countNonZero(m_selfieMask);
    std::cout << "Segmentation complete. Mask pixels: " << maskPixels << std::endl;
    
    // Debug output
    cv::imwrite("debug_selfie_head.jpg", m_selfieHead);
    cv::imwrite("debug_selfie_mask.jpg", m_selfieMask);
    
    // Masked result
    cv::Mat masked;
    m_selfieHead.copyTo(masked);
    for (int y = 0; y < masked.rows; y++) {
        for (int x = 0; x < masked.cols; x++) {
            float alpha = m_selfieMask.at<uchar>(y, x) / 255.0f;
            cv::Vec3b& p = masked.at<cv::Vec3b>(y, x);
            p[0] = static_cast<uchar>(p[0] * alpha + 128 * (1 - alpha));
            p[1] = static_cast<uchar>(p[1] * alpha + 128 * (1 - alpha));
            p[2] = static_cast<uchar>(p[2] * alpha + 128 * (1 - alpha));
        }
    }
    cv::imwrite("debug_selfie_segmented.jpg", masked);
    
    std::cout << "Debug images saved" << std::endl;
}

void FaceReplacer::setTargetFace(const FaceInfo& targetFace) {
    // 5-frame smoothing buffer
    m_positionBuffer.push_back(targetFace.boundingBox);
    while (m_positionBuffer.size() > 5) {
        m_positionBuffer.pop_front();
    }
    
    float weights[] = {0.1f, 0.15f, 0.2f, 0.25f, 0.3f};
    int startIdx = 5 - static_cast<int>(m_positionBuffer.size());
    
    float sumX = 0, sumY = 0, sumW = 0, sumH = 0;
    float totalWeight = 0;
    
    int idx = 0;
    for (const auto& r : m_positionBuffer) {
        float w = weights[startIdx + idx];
        sumX += r.x * w;
        sumY += r.y * w;
        sumW += r.width * w;
        sumH += r.height * w;
        totalWeight += w;
        idx++;
    }
    
    m_targetFace.boundingBox = cv::Rect(
        static_cast<int>(sumX / totalWeight),
        static_cast<int>(sumY / totalWeight),
        static_cast<int>(sumW / totalWeight),
        static_cast<int>(sumH / totalWeight)
    );
}

cv::Mat FaceReplacer::processFrame(const cv::Mat& frame) {
    if (m_selfieHead.empty() || m_selfieMask.empty()) {
        return frame.clone();
    }
    
    if (m_targetFace.boundingBox.width <= 0) {
        return frame.clone();
    }
    
    return replaceWithBlur(frame, m_targetFace.boundingBox);
}

cv::Mat FaceReplacer::replaceWithBlur(const cv::Mat& frame, const cv::Rect& targetRect) {
    cv::Mat result = frame.clone();
    
    cv::Point targetCenter(
        targetRect.x + targetRect.width / 2,
        targetRect.y + targetRect.height / 2
    );
    
    // === STEP 1: BLUR ORIGINAL FACE ===
    int blurMarginX = static_cast<int>(targetRect.width * 0.5);
    int blurMarginTop = static_cast<int>(targetRect.height * 0.7);
    int blurMarginBottom = static_cast<int>(targetRect.height * 0.3);
    
    cv::Rect blurRect;
    blurRect.x = std::max(0, targetRect.x - blurMarginX);
    blurRect.y = std::max(0, targetRect.y - blurMarginTop);
    blurRect.width = std::min(targetRect.width + blurMarginX * 2, frame.cols - blurRect.x);
    blurRect.height = std::min(targetRect.height + blurMarginTop + blurMarginBottom, frame.rows - blurRect.y);
    
    if (blurRect.width <= 0 || blurRect.height <= 0) {
        return result;
    }
    
    // Triple blur
    cv::Mat roiBlurred = result(blurRect).clone();
    cv::GaussianBlur(roiBlurred, roiBlurred, cv::Size(71, 71), 35);
    cv::GaussianBlur(roiBlurred, roiBlurred, cv::Size(71, 71), 35);
    cv::GaussianBlur(roiBlurred, roiBlurred, cv::Size(71, 71), 35);
    
    // Gradient mask for blur
    cv::Mat blurMask = cv::Mat::zeros(blurRect.size(), CV_8UC1);
    cv::Point blurCenter(blurRect.width / 2, static_cast<int>(blurRect.height * 0.45));
    cv::Size blurAxes(blurRect.width / 2 - 5, blurRect.height / 2 - 5);
    cv::ellipse(blurMask, blurCenter, blurAxes, 0, 0, 360, cv::Scalar(255), -1);
    cv::GaussianBlur(blurMask, blurMask, cv::Size(51, 51), 25);
    
    cv::Mat roiOriginal = result(blurRect);
    for (int y = 0; y < roiOriginal.rows; y++) {
        for (int x = 0; x < roiOriginal.cols; x++) {
            float alpha = blurMask.at<uchar>(y, x) / 255.0f;
            if (alpha > 0.01f) {
                cv::Vec3b& dst = roiOriginal.at<cv::Vec3b>(y, x);
                const cv::Vec3b& blur = roiBlurred.at<cv::Vec3b>(y, x);
                dst[0] = static_cast<uchar>(blur[0] * alpha + dst[0] * (1 - alpha));
                dst[1] = static_cast<uchar>(blur[1] * alpha + dst[1] * (1 - alpha));
                dst[2] = static_cast<uchar>(blur[2] * alpha + dst[2] * (1 - alpha));
            }
        }
    }
    
    // === STEP 2: PLACE SEGMENTED SELFIE ===
    float targetRegionWidth = targetRect.width / m_faceToRegionRatio;
    float scale = targetRegionWidth / m_selfieHead.cols;
    
    cv::Size newSize(
        static_cast<int>(m_selfieHead.cols * scale),
        static_cast<int>(m_selfieHead.rows * scale)
    );
    
    if (newSize.width <= 0 || newSize.height <= 0) {
        return result;
    }
    
    cv::Mat resizedHead, resizedMask;
    cv::resize(m_selfieHead, resizedHead, newSize, 0, 0, cv::INTER_LINEAR);
    cv::resize(m_selfieMask, resizedMask, newSize, 0, 0, cv::INTER_LINEAR);
    
    cv::Point scaledFaceCenter(
        static_cast<int>(m_selfieFaceCenterInRegion.x * scale),
        static_cast<int>(m_selfieFaceCenterInRegion.y * scale)
    );
    
    int placeX = targetCenter.x - scaledFaceCenter.x;
    int placeY = targetCenter.y - scaledFaceCenter.y;
    
    int srcX = 0, srcY = 0;
    int dstX = placeX, dstY = placeY;
    int copyW = resizedHead.cols, copyH = resizedHead.rows;
    
    if (dstX < 0) { srcX = -dstX; copyW += dstX; dstX = 0; }
    if (dstY < 0) { srcY = -dstY; copyH += dstY; dstY = 0; }
    if (dstX + copyW > result.cols) { copyW = result.cols - dstX; }
    if (dstY + copyH > result.rows) { copyH = result.rows - dstY; }
    
    if (copyW <= 0 || copyH <= 0) {
        return result;
    }
    
    cv::Rect srcROI(srcX, srcY, copyW, copyH);
    cv::Rect dstROI(dstX, dstY, copyW, copyH);
    
    cv::Mat srcRegion = resizedHead(srcROI).clone();
    cv::Mat maskRegion = resizedMask(srcROI);
    cv::Mat dstRegion = result(dstROI);
    
    if (m_config.colorCorrection) {
        srcRegion = matchColors(srcRegion, frame(blurRect), maskRegion);
    }
    
    for (int y = 0; y < dstRegion.rows; y++) {
        for (int x = 0; x < dstRegion.cols; x++) {
            float alpha = maskRegion.at<uchar>(y, x) / 255.0f;
            if (alpha > 0.01f) {
                cv::Vec3b& dst = dstRegion.at<cv::Vec3b>(y, x);
                const cv::Vec3b& src = srcRegion.at<cv::Vec3b>(y, x);
                dst[0] = static_cast<uchar>(src[0] * alpha + dst[0] * (1 - alpha));
                dst[1] = static_cast<uchar>(src[1] * alpha + dst[1] * (1 - alpha));
                dst[2] = static_cast<uchar>(src[2] * alpha + dst[2] * (1 - alpha));
            }
        }
    }
    
    return result;
}

cv::Mat FaceReplacer::replaceSegmented(const cv::Mat& frame, const cv::Rect& targetRect) {
    return replaceWithBlur(frame, targetRect);
}

cv::Mat FaceReplacer::matchColors(const cv::Mat& source, const cv::Mat& target, const cv::Mat& mask) {
    cv::Mat result = source.clone();
    
    cv::Mat srcLab, tgtLab;
    cv::cvtColor(source, srcLab, cv::COLOR_BGR2Lab);
    cv::cvtColor(target, tgtLab, cv::COLOR_BGR2Lab);
    
    cv::Scalar srcMean, srcStd, tgtMean, tgtStd;
    
    if (mask.empty()) {
        cv::meanStdDev(srcLab, srcMean, srcStd);
    } else {
        cv::meanStdDev(srcLab, srcMean, srcStd, mask);
    }
    cv::meanStdDev(tgtLab, tgtMean, tgtStd);
    
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

// Stubs
cv::Mat FaceReplacer::replaceRectToRect(const cv::Mat& frame, const cv::Mat& source, const cv::Rect& targetRect) { return replaceWithBlur(frame, targetRect); }
cv::Mat FaceReplacer::replaceLive(const cv::Mat& frame, const FaceInfo& targetFace) { return replaceWithBlur(frame, targetFace.boundingBox); }
cv::Mat FaceReplacer::adjustLighting(const cv::Mat& source, const cv::Mat& target, const cv::Rect& region) { return source.clone(); }
cv::Mat FaceReplacer::poissonBlend(const cv::Mat& source, const cv::Mat& target, const cv::Mat& mask, const cv::Point& center) { return target.clone(); }
cv::Mat FaceReplacer::warpFaceToTarget(const cv::Mat& source, const FaceInfo& sourceFace, const FaceInfo& targetFace) { return source.clone(); }
cv::Mat FaceReplacer::applyTemporalSmoothing(const cv::Mat& currentResult) { return currentResult.clone(); }
void FaceReplacer::updateBuffers(const cv::Mat& frame, const FaceInfo& face) {}

#ifdef USE_CUDA
cv::Mat FaceReplacer::blendGPU(const cv::Mat& source, const cv::Mat& target, const cv::Mat& mask) { return target.clone(); }
#endif

LiveFaceReplacer::LiveFaceReplacer(const Config& config) : FaceReplacer(config) {}
cv::Mat LiveFaceReplacer::processWithExpression(const cv::Mat& frame, const FaceInfo& targetFace) { return replaceWithBlur(frame, targetFace.boundingBox); }
cv::Mat LiveFaceReplacer::warpToPose(const cv::Mat& sourceFace, const std::vector<cv::Point2f>& sourceLandmarks, const std::vector<cv::Point2f>& targetLandmarks) { return sourceFace.clone(); }
void LiveFaceReplacer::calculateDelaunay(const std::vector<cv::Point2f>& points, const cv::Rect& bounds) {}
cv::Mat LiveFaceReplacer::warpTriangle(const cv::Mat& src, const cv::Mat& dst, const std::vector<cv::Point2f>& srcTri, const std::vector<cv::Point2f>& dstTri) { return dst.clone(); }

} // namespace facereplacer
