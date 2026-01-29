#include "segmentation.hpp"
#include <iostream>

namespace facereplacer {

Segmentation::Segmentation(const Config& config) : m_config(config) {
}

bool Segmentation::loadModel(const std::string& modelPath) {
    try {
        m_segNet = cv::dnn::readNetFromONNX(modelPath);
        if (m_config.useGPU) {
            m_segNet.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
            m_segNet.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
        }
        m_hasModel = true;
        return true;
    } catch (const cv::Exception& e) {
        std::cerr << "Failed to load segmentation model: " << e.what() << std::endl;
        return false;
    }
}

cv::Mat Segmentation::segmentHead(const cv::Mat& image, const cv::Rect& faceRect) {
    // Expand face rect to include full head (hair, ears, neck)
    cv::Rect headRect = faceRect;
    
    // Expand upward for hair (50% of face height)
    int expandTop = static_cast<int>(faceRect.height * 0.5);
    // Expand sides for ears (25% each side)
    int expandSide = static_cast<int>(faceRect.width * 0.25);
    // Expand down for neck (20% of face height)
    int expandBottom = static_cast<int>(faceRect.height * 0.2);
    
    headRect.x = std::max(0, headRect.x - expandSide);
    headRect.y = std::max(0, headRect.y - expandTop);
    headRect.width = std::min(faceRect.width + expandSide * 2, image.cols - headRect.x);
    headRect.height = std::min(faceRect.height + expandTop + expandBottom, image.rows - headRect.y);
    
    // Try GrabCut for good segmentation
    cv::Mat grabCutMask = grabCutSegment(image, headRect);
    int grabCutPixels = cv::countNonZero(grabCutMask);
    
    // Create elliptical mask as fallback/base
    cv::Mat ellipseMask = createEllipticalMask(image.size(), headRect);
    int ellipsePixels = cv::countNonZero(ellipseMask);
    
    cv::Mat result;
    
    // Use GrabCut if it found reasonable amount of pixels
    if (grabCutPixels > ellipsePixels * 0.3 && grabCutPixels < ellipsePixels * 3.0) {
        // Combine GrabCut with ellipse (union)
        cv::bitwise_or(grabCutMask, ellipseMask, result);
    } else {
        // GrabCut failed, use ellipse only
        result = ellipseMask;
    }
    
    // Refine the mask
    result = refineMask(result, image);
    
    std::cout << "Segmentation: GrabCut=" << grabCutPixels << " Ellipse=" << ellipsePixels 
              << " Final=" << cv::countNonZero(result) << std::endl;
    
    return result;
}

cv::Mat Segmentation::segmentFace(const cv::Mat& image, const cv::Rect& faceRect) {
    return createEllipticalMask(image.size(), faceRect);
}

cv::Mat Segmentation::grabCutSegment(const cv::Mat& image, const cv::Rect& rect) {
    cv::Mat result = cv::Mat::zeros(image.size(), CV_8UC1);
    
    if (rect.width <= 10 || rect.height <= 10) {
        return result;
    }
    
    // Ensure rect is within image bounds
    cv::Rect safeRect = rect;
    safeRect.x = std::max(1, safeRect.x);
    safeRect.y = std::max(1, safeRect.y);
    safeRect.width = std::min(safeRect.width, image.cols - safeRect.x - 1);
    safeRect.height = std::min(safeRect.height, image.rows - safeRect.y - 1);
    
    if (safeRect.width <= 10 || safeRect.height <= 10) {
        return result;
    }
    
    try {
        cv::Mat mask = cv::Mat::zeros(image.size(), CV_8UC1);
        cv::Mat bgdModel, fgdModel;
        
        // Run GrabCut with rect initialization
        cv::grabCut(image, mask, safeRect, bgdModel, fgdModel, 5, cv::GC_INIT_WITH_RECT);
        
        // Extract foreground (both definite and probable)
        result = (mask == cv::GC_FGD) | (mask == cv::GC_PR_FGD);
        result.convertTo(result, CV_8UC1, 255);
        
    } catch (const cv::Exception& e) {
        std::cerr << "GrabCut failed: " << e.what() << std::endl;
    }
    
    return result;
}

cv::Mat Segmentation::createEllipticalMask(const cv::Size& size, const cv::Rect& rect) {
    cv::Mat mask = cv::Mat::zeros(size, CV_8UC1);
    
    // Center of the rectangle
    cv::Point center(rect.x + rect.width / 2, rect.y + rect.height / 2);
    
    // Ellipse axes - head shaped (slightly taller than wide)
    cv::Size axes(rect.width / 2, static_cast<int>(rect.height * 0.55));
    
    // Draw filled ellipse
    cv::ellipse(mask, center, axes, 0, 0, 360, cv::Scalar(255), -1);
    
    return mask;
}

cv::Mat Segmentation::skinColorMask(const cv::Mat& image) {
    cv::Mat ycrcb;
    cv::cvtColor(image, ycrcb, cv::COLOR_BGR2YCrCb);
    
    // YCrCb skin detection (works for various skin tones)
    cv::Mat mask;
    cv::inRange(ycrcb, cv::Scalar(0, 133, 77), cv::Scalar(255, 173, 127), mask);
    
    // Clean up
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
    
    return mask;
}

cv::Mat Segmentation::refineMask(const cv::Mat& mask, const cv::Mat& image) {
    cv::Mat result = mask.clone();
    
    // Morphological cleanup
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7));
    
    // Close holes
    cv::morphologyEx(result, result, cv::MORPH_CLOSE, kernel, cv::Point(-1, -1), 2);
    
    // Remove small noise
    cv::morphologyEx(result, result, cv::MORPH_OPEN, kernel);
    
    // Keep only the largest connected component
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(result.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    
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
        
        result = cv::Mat::zeros(mask.size(), CV_8UC1);
        cv::drawContours(result, contours, maxIdx, cv::Scalar(255), -1);
        
        // Fill any holes inside the contour
        cv::Mat filled = result.clone();
        cv::floodFill(filled, cv::Point(0, 0), cv::Scalar(255));
        cv::bitwise_not(filled, filled);
        cv::bitwise_or(result, filled, result);
    }
    
    return result;
}

cv::Mat Segmentation::featherMask(const cv::Mat& mask, int radius) {
    if (radius <= 0) return mask.clone();
    
    cv::Mat result;
    int kernelSize = radius * 2 + 1;
    cv::GaussianBlur(mask, result, cv::Size(kernelSize, kernelSize), radius / 2.0);
    
    return result;
}

} // namespace facereplacer
