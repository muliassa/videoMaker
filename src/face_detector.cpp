#include "face_detector.hpp"
#include <iostream>

namespace facereplacer {

FaceDetector::FaceDetector(const Config& config) 
    : m_config(config), m_confThreshold(config.detectionConfidence) {
    // Try to load default Haar cascade
    useHaarCascade();
}

bool FaceDetector::loadFaceDetector(const std::string& modelPath, const std::string& configPath) {
    try {
        if (configPath.empty()) {
            // ONNX model
            m_faceNet = cv::dnn::readNetFromONNX(modelPath);
        } else {
            // Caffe or TensorFlow model
            m_faceNet = cv::dnn::readNet(modelPath, configPath);
        }
        
        if (m_config.useGPU) {
            m_faceNet.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
            m_faceNet.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
        }
        
        m_useDNN = true;
        std::cout << "Face detector model loaded: " << modelPath << std::endl;
        return true;
    } catch (const cv::Exception& e) {
        std::cerr << "Failed to load face detector: " << e.what() << std::endl;
        return false;
    }
}

bool FaceDetector::loadLandmarkDetector(const std::string& modelPath) {
    try {
        m_landmarkNet = cv::dnn::readNetFromONNX(modelPath);
        
        if (m_config.useGPU) {
            m_landmarkNet.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
            m_landmarkNet.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
        }
        
        m_hasLandmarkModel = true;
        std::cout << "Landmark detector model loaded: " << modelPath << std::endl;
        return true;
    } catch (const cv::Exception& e) {
        std::cerr << "Failed to load landmark detector: " << e.what() << std::endl;
        return false;
    }
}

void FaceDetector::useHaarCascade() {
    // Try common paths for Haar cascade
    std::vector<std::string> cascadePaths = {
        "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
        "/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml",
        "/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
        "haarcascade_frontalface_default.xml"
    };
    
    for (const auto& path : cascadePaths) {
        if (m_haarCascade.load(path)) {
            std::cout << "Haar cascade loaded: " << path << std::endl;
            m_useDNN = false;
            return;
        }
    }
    
    std::cerr << "Warning: Could not load Haar cascade" << std::endl;
}

std::vector<FaceInfo> FaceDetector::detect(const cv::Mat& image) {
    if (m_useDNN) {
        return detectDNN(image);
    } else {
        return detectHaar(image);
    }
}

std::vector<FaceInfo> FaceDetector::detectWithLandmarks(const cv::Mat& image) {
    auto faces = detect(image);
    
    for (auto& face : faces) {
        face.landmarks = detectLandmarks(image, face.boundingBox);
    }
    
    return faces;
}

std::vector<FaceInfo> FaceDetector::detectDNN(const cv::Mat& image) {
    std::vector<FaceInfo> faces;
    
    // Prepare input blob
    cv::Mat blob = cv::dnn::blobFromImage(image, 1.0, m_inputSize,
                                           cv::Scalar(104, 177, 123),
                                           false, false);
    
    m_faceNet.setInput(blob);
    cv::Mat detection = m_faceNet.forward();
    
    // Parse detections
    // Format depends on the model used
    cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());
    
    for (int i = 0; i < detectionMat.rows; i++) {
        float confidence = detectionMat.at<float>(i, 2);
        
        if (confidence > m_confThreshold) {
            int x1 = static_cast<int>(detectionMat.at<float>(i, 3) * image.cols);
            int y1 = static_cast<int>(detectionMat.at<float>(i, 4) * image.rows);
            int x2 = static_cast<int>(detectionMat.at<float>(i, 5) * image.cols);
            int y2 = static_cast<int>(detectionMat.at<float>(i, 6) * image.rows);
            
            FaceInfo face;
            face.boundingBox = cv::Rect(x1, y1, x2 - x1, y2 - y1);
            face.confidence = confidence;
            faces.push_back(face);
        }
    }
    
    return faces;
}

std::vector<FaceInfo> FaceDetector::detectHaar(const cv::Mat& image) {
    std::vector<FaceInfo> faces;
    
    if (m_haarCascade.empty()) {
        return faces;
    }
    
    cv::Mat gray;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = image;
    }
    
    cv::equalizeHist(gray, gray);
    
    std::vector<cv::Rect> detections;
    m_haarCascade.detectMultiScale(gray, detections, 1.1, 3, 0,
                                    cv::Size(30, 30));
    
    for (const auto& rect : detections) {
        FaceInfo face;
        face.boundingBox = rect;
        face.confidence = 1.0f;  // Haar doesn't provide confidence
        faces.push_back(face);
    }
    
    return faces;
}

std::vector<cv::Point2f> FaceDetector::detectLandmarks(const cv::Mat& image, const cv::Rect& faceRect) {
    std::vector<cv::Point2f> landmarks;
    
    if (!m_hasLandmarkModel) {
        // Return basic 5-point landmarks estimated from face rect
        // Eye centers, nose tip, mouth corners
        float w = faceRect.width;
        float h = faceRect.height;
        float x = faceRect.x;
        float y = faceRect.y;
        
        landmarks.push_back(cv::Point2f(x + w * 0.3f, y + h * 0.35f));   // Left eye
        landmarks.push_back(cv::Point2f(x + w * 0.7f, y + h * 0.35f));   // Right eye
        landmarks.push_back(cv::Point2f(x + w * 0.5f, y + h * 0.55f));   // Nose tip
        landmarks.push_back(cv::Point2f(x + w * 0.3f, y + h * 0.75f));   // Left mouth
        landmarks.push_back(cv::Point2f(x + w * 0.7f, y + h * 0.75f));   // Right mouth
        
        return landmarks;
    }
    
    // Extract face region
    cv::Rect expandedRect = faceRect;
    expandedRect.x = std::max(0, faceRect.x - faceRect.width / 4);
    expandedRect.y = std::max(0, faceRect.y - faceRect.height / 4);
    expandedRect.width = std::min(faceRect.width * 3 / 2, image.cols - expandedRect.x);
    expandedRect.height = std::min(faceRect.height * 3 / 2, image.rows - expandedRect.y);
    
    cv::Mat faceROI = image(expandedRect);
    
    // Prepare input
    cv::Mat blob = cv::dnn::blobFromImage(faceROI, 1.0 / 255.0, cv::Size(112, 112),
                                           cv::Scalar(0, 0, 0), true, false);
    
    m_landmarkNet.setInput(blob);
    cv::Mat output = m_landmarkNet.forward();
    
    // Parse output (assuming 68 or 98 point format)
    int numPoints = output.total() / 2;
    for (int i = 0; i < numPoints; i++) {
        float px = output.at<float>(0, i * 2) * expandedRect.width + expandedRect.x;
        float py = output.at<float>(0, i * 2 + 1) * expandedRect.height + expandedRect.y;
        landmarks.push_back(cv::Point2f(px, py));
    }
    
    return landmarks;
}

} // namespace facereplacer
