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
        std::cout << "Segmentation model loaded: " << modelPath << std::endl;
        return true;
    } catch (const cv::Exception& e) {
        std::cerr << "Failed to load segmentation model: " << e.what() << std::endl;
        return false;
    }
}

cv::Mat Segmentation::segmentHead(const cv::Mat& image, const cv::Rect& faceRect) {
    // Try GrabCut first for better results, fall back to elliptical mask
    cv::Mat mask = grabCutSegment(image, faceRect);
    
    if (cv::countNonZero(mask) < faceRect.area() * 0.1) {
        // GrabCut failed, use elliptical mask
        mask = createEllipticalMask(image.size(), faceRect);
    }
    
    // Combine with skin color detection for refinement
    cv::Mat skinMask = skinColorMask(image);
    
    // Expand face rect for head region (include hair)
    cv::Rect headRect = faceRect;
    headRect.y = std::max(0, headRect.y - headRect.height / 2);
    headRect.height = std::min(static_cast<int>(headRect.height * 1.8), 
                                image.rows - headRect.y);
    headRect.x = std::max(0, headRect.x - headRect.width / 4);
    headRect.width = std::min(static_cast<int>(headRect.width * 1.5),
                               image.cols - headRect.x);
    
    // Create combined mask
    cv::Mat headMask = cv::Mat::zeros(image.size(), CV_8UC1);
    cv::Mat ellipse = createEllipticalMask(image.size(), headRect);
    
    // Blend masks
    cv::Mat finalMask;
    cv::bitwise_or(mask, ellipse, finalMask);
    
    // Apply skin mask in face region only
    cv::Mat faceRegionMask = cv::Mat::zeros(image.size(), CV_8UC1);
    faceRegionMask(faceRect) = 255;
    
    cv::Mat skinInFace;
    cv::bitwise_and(skinMask, faceRegionMask, skinInFace);
    cv::bitwise_or(finalMask, skinInFace, finalMask);
    
    return finalMask;
}

cv::Mat Segmentation::segmentFace(const cv::Mat& image, const cv::Rect& faceRect) {
    // Just the face, not the full head
    cv::Mat mask = createEllipticalMask(image.size(), faceRect);
    cv::Mat skinMask = skinColorMask(image);
    
    // Combine
    cv::Mat result;
    cv::bitwise_and(mask, skinMask, result);
    
    // Clean up
    cv::morphologyEx(result, result, cv::MORPH_CLOSE, 
                     cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));
    
    return result;
}

cv::Mat Segmentation::grabCutSegment(const cv::Mat& image, const cv::Rect& faceRect) {
    cv::Mat result = cv::Mat::zeros(image.size(), CV_8UC1);
    
    if (faceRect.width <= 0 || faceRect.height <= 0) {
        return result;
    }
    
    // Expand rect for GrabCut
    cv::Rect grabRect = faceRect;
    int expandX = faceRect.width / 3;
    int expandY = faceRect.height / 2;
    
    grabRect.x = std::max(0, grabRect.x - expandX);
    grabRect.y = std::max(0, grabRect.y - expandY);
    grabRect.width = std::min(grabRect.width + expandX * 2, image.cols - grabRect.x);
    grabRect.height = std::min(grabRect.height + expandY * 2, image.rows - grabRect.y);
    
    if (grabRect.width <= 0 || grabRect.height <= 0) {
        return result;
    }
    
    try {
        cv::Mat mask = cv::Mat::zeros(image.size(), CV_8UC1);
        cv::Mat bgdModel, fgdModel;
        
        // Run GrabCut
        cv::grabCut(image, mask, grabRect, bgdModel, fgdModel, 5, cv::GC_INIT_WITH_RECT);
        
        // Extract foreground
        cv::compare(mask, cv::GC_PR_FGD, result, cv::CMP_EQ);
        cv::Mat fg;
        cv::compare(mask, cv::GC_FGD, fg, cv::CMP_EQ);
        cv::bitwise_or(result, fg, result);
        
    } catch (const cv::Exception& e) {
        std::cerr << "GrabCut failed: " << e.what() << std::endl;
    }
    
    return result;
}

cv::Mat Segmentation::createEllipticalMask(const cv::Size& size, const cv::Rect& faceRect) {
    cv::Mat mask = cv::Mat::zeros(size, CV_8UC1);
    
    // Create ellipse covering head region
    cv::Point center(faceRect.x + faceRect.width / 2,
                     faceRect.y + faceRect.height / 2);
    
    // Ellipse axes - wider horizontally for head shape
    cv::Size axes(static_cast<int>(faceRect.width * 0.55),
                  static_cast<int>(faceRect.height * 0.65));
    
    cv::ellipse(mask, center, axes, 0, 0, 360, cv::Scalar(255), -1);
    
    return mask;
}

cv::Mat Segmentation::skinColorMask(const cv::Mat& image) {
    cv::Mat hsv, ycrcb;
    cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);
    cv::cvtColor(image, ycrcb, cv::COLOR_BGR2YCrCb);
    
    // HSV skin detection
    cv::Mat hsvMask;
    cv::inRange(hsv, cv::Scalar(0, 20, 70), cv::Scalar(20, 255, 255), hsvMask);
    
    // YCrCb skin detection (more robust)
    cv::Mat ycrcbMask;
    cv::inRange(ycrcb, cv::Scalar(0, 133, 77), cv::Scalar(255, 173, 127), ycrcbMask);
    
    // Combine both
    cv::Mat result;
    cv::bitwise_and(hsvMask, ycrcbMask, result);
    
    // Clean up
    cv::morphologyEx(result, result, cv::MORPH_OPEN,
                     cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3)));
    cv::morphologyEx(result, result, cv::MORPH_CLOSE,
                     cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));
    
    return result;
}

cv::Mat Segmentation::refineMask(const cv::Mat& mask, const cv::Mat& image) {
    cv::Mat result = mask.clone();
    
    // Morphological operations
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    
    // Close holes
    cv::morphologyEx(result, result, cv::MORPH_CLOSE, kernel, cv::Point(-1, -1), 2);
    
    // Remove small noise
    cv::morphologyEx(result, result, cv::MORPH_OPEN, kernel);
    
    // Find largest contour and keep only that
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(result.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    
    if (!contours.empty()) {
        // Find largest contour
        int maxIdx = 0;
        double maxArea = 0;
        for (size_t i = 0; i < contours.size(); i++) {
            double area = cv::contourArea(contours[i]);
            if (area > maxArea) {
                maxArea = area;
                maxIdx = i;
            }
        }
        
        // Create mask with only largest contour
        result = cv::Mat::zeros(mask.size(), CV_8UC1);
        cv::drawContours(result, contours, maxIdx, cv::Scalar(255), -1);
    }
    
    return result;
}

cv::Mat Segmentation::featherMask(const cv::Mat& mask, int radius) {
    cv::Mat result;
    
    // Gaussian blur for soft edges
    int kernelSize = radius * 2 + 1;
    cv::GaussianBlur(mask, result, cv::Size(kernelSize, kernelSize), radius / 2.0);
    
    return result;
}

} // namespace facereplacer
