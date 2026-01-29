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
    auto faces = m_detector->detect(selfie);
    if (!faces.empty()) {
        m_sourceFace = faces[0];
    }
}

void FaceReplacer::setTargetFace(const FaceInfo& targetFace) {
    m_targetFace = targetFace;
}

cv::Mat FaceReplacer::processFrame(const cv::Mat& frame) {
    return frame.clone();
}

// Stub implementations - main.cpp handles actual replacement
cv::Mat FaceReplacer::replaceWithBlur(const cv::Mat& frame, const cv::Rect&) { return frame.clone(); }
cv::Mat FaceReplacer::replaceSegmented(const cv::Mat& frame, const cv::Rect&) { return frame.clone(); }
cv::Mat FaceReplacer::replaceRectToRect(const cv::Mat& frame, const cv::Mat&, const cv::Rect&) { return frame.clone(); }
cv::Mat FaceReplacer::replaceLive(const cv::Mat& frame, const FaceInfo&) { return frame.clone(); }
cv::Mat FaceReplacer::matchColors(const cv::Mat& source, const cv::Mat&, const cv::Mat&) { return source.clone(); }
cv::Mat FaceReplacer::adjustLighting(const cv::Mat& source, const cv::Mat&, const cv::Rect&) { return source.clone(); }
cv::Mat FaceReplacer::poissonBlend(const cv::Mat&, const cv::Mat& target, const cv::Mat&, const cv::Point&) { return target.clone(); }
cv::Mat FaceReplacer::warpFaceToTarget(const cv::Mat& source, const FaceInfo&, const FaceInfo&) { return source.clone(); }
cv::Mat FaceReplacer::applyTemporalSmoothing(const cv::Mat& result) { return result.clone(); }
void FaceReplacer::updateBuffers(const cv::Mat&, const FaceInfo&) {}

#ifdef USE_CUDA
cv::Mat FaceReplacer::blendGPU(const cv::Mat&, const cv::Mat& target, const cv::Mat&) { return target.clone(); }
#endif

LiveFaceReplacer::LiveFaceReplacer(const Config& config) : FaceReplacer(config) {}
cv::Mat LiveFaceReplacer::processWithExpression(const cv::Mat& frame, const FaceInfo&) { return frame.clone(); }
cv::Mat LiveFaceReplacer::warpToPose(const cv::Mat& face, const std::vector<cv::Point2f>&, const std::vector<cv::Point2f>&) { return face.clone(); }
void LiveFaceReplacer::calculateDelaunay(const std::vector<cv::Point2f>&, const cv::Rect&) {}
cv::Mat LiveFaceReplacer::warpTriangle(const cv::Mat&, const cv::Mat& dst, const std::vector<cv::Point2f>&, const std::vector<cv::Point2f>&) { return dst.clone(); }

} // namespace facereplacer
