#pragma once

#include "face_replacer.hpp"
#include <opencv2/dnn.hpp>

namespace facereplacer {

class Segmentation {
public:
    explicit Segmentation(const Config& config);
    ~Segmentation() = default;
    
    // Main segmentation methods
    cv::Mat segmentHead(const cv::Mat& image, const cv::Rect& faceRect);
    cv::Mat segmentFace(const cv::Mat& image, const cv::Rect& faceRect);
    
    // Mask refinement
    cv::Mat refineMask(const cv::Mat& mask, const cv::Mat& image);
    cv::Mat featherMask(const cv::Mat& mask, int radius);
    
    // Load segmentation model (optional - for deep learning based segmentation)
    bool loadModel(const std::string& modelPath);
    
    // GrabCut-based segmentation (no model required)
    cv::Mat grabCutSegment(const cv::Mat& image, const cv::Rect& faceRect);

private:
    Config m_config;
    cv::dnn::Net m_segNet;
    bool m_hasModel = false;
    
    // Create elliptical mask based on face rect
    cv::Mat createEllipticalMask(const cv::Size& size, const cv::Rect& faceRect);
    
    // Skin color based segmentation
    cv::Mat skinColorMask(const cv::Mat& image);
};

} // namespace facereplacer
