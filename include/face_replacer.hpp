#pragma once

#include <opencv2/opencv.hpp>
#ifdef USE_CUDA
#include <opencv2/core/cuda.hpp>
#endif
#include <memory>
#include <vector>
#include <deque>

namespace facereplacer {

enum class ReplacementMode {
    RECT_TO_RECT,
    HEAD_SEGMENTED,
    LIVE_ANIMATED
};

struct Config {
    ReplacementMode mode = ReplacementMode::HEAD_SEGMENTED;
    bool useGPU = true;
    bool colorCorrection = true;
    bool preserveLighting = true;
    int featherRadius = 15;
    int temporalSmoothing = 5;
    float detectionConfidence = 0.5f;
};

struct FaceInfo {
    cv::Rect boundingBox;
    std::vector<cv::Point2f> landmarks;
    float confidence = 0.0f;
    float yaw = 0.0f;
    float pitch = 0.0f;
    float roll = 0.0f;
};

class FaceDetector;
class Segmentation;

inline cv::Rect clampRect(const cv::Rect& rect, const cv::Size& size) {
    int x = std::max(0, rect.x);
    int y = std::max(0, rect.y);
    int w = std::min(rect.width, size.width - x);
    int h = std::min(rect.height, size.height - y);
    return cv::Rect(x, y, std::max(0, w), std::max(0, h));
}

inline cv::Rect scaleRect(const cv::Rect& rect, float scale) {
    int newW = static_cast<int>(rect.width * scale);
    int newH = static_cast<int>(rect.height * scale);
    int newX = rect.x - (newW - rect.width) / 2;
    int newY = rect.y - (newH - rect.height) / 2;
    return cv::Rect(newX, newY, newW, newH);
}

class FaceReplacer {
public:
    explicit FaceReplacer(const Config& config);
    virtual ~FaceReplacer();

    void setSourceImage(const cv::Mat& selfie);
    void setTargetFace(const FaceInfo& targetFace);
    
    std::vector<FaceInfo> detectFaces(const cv::Mat& image);
    
    cv::Mat processFrame(const cv::Mat& frame);
    cv::Mat markFace(const cv::Mat& frame, int faceIndex = 0);
    
    const Config& getConfig() const { return m_config; }
    const FaceInfo& getSourceFace() const { return m_sourceFace; }
    const FaceInfo& getTargetFace() const { return m_targetFace; }

protected:
    cv::Mat replaceRectToRect(const cv::Mat& frame, const cv::Mat& source, const cv::Rect& targetRect);
    cv::Mat replaceSegmented(const cv::Mat& frame, const cv::Rect& targetRect);
    cv::Mat replaceLive(const cv::Mat& frame, const FaceInfo& targetFace);
    cv::Mat replaceWithBlur(const cv::Mat& frame, const cv::Rect& targetRect);
    
    cv::Mat matchColors(const cv::Mat& source, const cv::Mat& target, const cv::Mat& mask = cv::Mat());
    cv::Mat adjustLighting(const cv::Mat& source, const cv::Mat& target, const cv::Rect& region);
    cv::Mat poissonBlend(const cv::Mat& source, const cv::Mat& target, const cv::Mat& mask, const cv::Point& center);
#ifdef USE_CUDA
    cv::Mat blendGPU(const cv::Mat& source, const cv::Mat& target, const cv::Mat& mask);
#endif
    
    cv::Mat warpFaceToTarget(const cv::Mat& source, const FaceInfo& sourceFace, const FaceInfo& targetFace);
    cv::Mat applyTemporalSmoothing(const cv::Mat& currentResult);
    void updateBuffers(const cv::Mat& frame, const FaceInfo& face);

protected:
    Config m_config;
    
    cv::Mat m_sourceImage;
    cv::Mat m_sourceMask;
    cv::Mat m_selfieHead;
    cv::Mat m_selfieMask;
    FaceInfo m_sourceFace;
    
    FaceInfo m_targetFace;
    std::deque<cv::Rect> m_positionBuffer;

#ifdef USE_CUDA
    cv::cuda::GpuMat m_gpuSource;
#endif
    
    std::unique_ptr<FaceDetector> m_detector;
    std::unique_ptr<Segmentation> m_segmentation;
    
    std::vector<cv::Mat> m_frameBuffer;
    std::vector<FaceInfo> m_faceBuffer;
};

class LiveFaceReplacer : public FaceReplacer {
public:
    explicit LiveFaceReplacer(const Config& config);
    cv::Mat processWithExpression(const cv::Mat& frame, const FaceInfo& targetFace);
    
protected:
    cv::Mat warpToPose(const cv::Mat& sourceFace, const std::vector<cv::Point2f>& sourceLandmarks,
                       const std::vector<cv::Point2f>& targetLandmarks);
    void calculateDelaunay(const std::vector<cv::Point2f>& points, const cv::Rect& bounds);
    cv::Mat warpTriangle(const cv::Mat& src, const cv::Mat& dst,
                         const std::vector<cv::Point2f>& srcTri, const std::vector<cv::Point2f>& dstTri);

private:
    std::vector<cv::Vec6f> m_triangles;
};

} // namespace facereplacer
