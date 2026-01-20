#pragma once

#include "face_replacer.hpp"
#include <opencv2/dnn.hpp>
#include <string>

namespace facereplacer {

class FaceDetector {
public:
    explicit FaceDetector(const Config& config);
    ~FaceDetector() = default;
    
    // Detect all faces in an image
    std::vector<FaceInfo> detect(const cv::Mat& image);
    
    // Detect and return landmarks
    std::vector<FaceInfo> detectWithLandmarks(const cv::Mat& image);
    
    // Load models
    bool loadFaceDetector(const std::string& modelPath, const std::string& configPath = "");
    bool loadLandmarkDetector(const std::string& modelPath);
    
    // Use built-in Haar cascade (fallback)
    void useHaarCascade();

private:
    Config m_config;
    
    // DNN-based detector (ONNX, Caffe, etc.)
    cv::dnn::Net m_faceNet;
    cv::dnn::Net m_landmarkNet;
    
    // Haar cascade fallback
    cv::CascadeClassifier m_haarCascade;
    
    // Detection parameters
    float m_confThreshold;
    cv::Size m_inputSize{300, 300};
    
    // Flags
    bool m_useDNN = false;
    bool m_hasLandmarkModel = false;
    
    // Internal methods
    std::vector<FaceInfo> detectDNN(const cv::Mat& image);
    std::vector<FaceInfo> detectHaar(const cv::Mat& image);
    std::vector<cv::Point2f> detectLandmarks(const cv::Mat& image, const cv::Rect& faceRect);
};

} // namespace facereplacer
