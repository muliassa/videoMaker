#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <memory>
#include <vector>

namespace facereplacer {

// Replacement modes
enum class ReplacementMode {
    RECT_TO_RECT,       // Mode 1: Simple rectangle replacement
    HEAD_SEGMENTED,     // Mode 2: Head segmentation with background preservation  
    LIVE_ANIMATED       // Mode 3: Advanced live/animated faces
};

// Configuration structure
struct Config {
    ReplacementMode mode = ReplacementMode::HEAD_SEGMENTED;
    bool useGPU = true;
    bool colorCorrection = true;
    bool preserveLighting = true;
    int featherRadius = 15;
    int temporalSmoothing = 5;  // Number of frames for smoothing (Mode 3)
    float detectionConfidence = 0.5f;
};

// Face information structure
struct FaceInfo {
    cv::Rect boundingBox;
    std::vector<cv::Point2f> landmarks;  // 68 or 5 point landmarks
    float confidence = 0.0f;
    float yaw = 0.0f;    // Head pose
    float pitch = 0.0f;
    float roll = 0.0f;
};

// Forward declarations
class FaceDetector;
class Segmentation;

// Utility functions
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

// Main Face Replacer class
class FaceReplacer {
public:
    explicit FaceReplacer(const Config& config);
    virtual ~FaceReplacer() = default;

    // Setup
    void setSourceImage(const cv::Mat& selfie);
    void setTargetFace(const FaceInfo& targetFace);
    
    // Detection
    std::vector<FaceInfo> detectFaces(const cv::Mat& image);
    
    // Processing
    cv::Mat processFrame(const cv::Mat& frame);
    cv::Mat markFace(const cv::Mat& frame, int faceIndex = 0);
    
    // Getters
    const Config& getConfig() const { return m_config; }
    const FaceInfo& getSourceFace() const { return m_sourceFace; }
    const FaceInfo& getTargetFace() const { return m_targetFace; }

protected:
    // Replacement methods for each mode
    cv::Mat replaceRectToRect(const cv::Mat& frame, const cv::Mat& source,
                              const cv::Rect& targetRect);
    cv::Mat replaceSegmented(const cv::Mat& frame, const cv::Mat& source,
                             const cv::Mat& sourceMask, const cv::Rect& targetRect);
    cv::Mat replaceLive(const cv::Mat& frame, const FaceInfo& targetFace);
    
    // Image processing utilities
    cv::Mat matchColors(const cv::Mat& source, const cv::Mat& target,
                        const cv::Mat& mask = cv::Mat());
    cv::Mat adjustLighting(const cv::Mat& source, const cv::Mat& target,
                           const cv::Rect& region);
    cv::Mat poissonBlend(const cv::Mat& source, const cv::Mat& target,
                         const cv::Mat& mask, const cv::Point& center);
    cv::Mat blendGPU(const cv::Mat& source, const cv::Mat& target,
                     const cv::Mat& mask);
    
    // Face warping
    cv::Mat warpFaceToTarget(const cv::Mat& source, const FaceInfo& sourceFace,
                             const FaceInfo& targetFace);
    
    // Temporal processing
    cv::Mat applyTemporalSmoothing(const cv::Mat& currentResult);
    void updateBuffers(const cv::Mat& frame, const FaceInfo& face);

protected:
    Config m_config;
    
    // Source (selfie) data
    cv::Mat m_sourceImage;
    cv::Mat m_sourceMask;
    FaceInfo m_sourceFace;
    
    // Target data
    FaceInfo m_targetFace;
    
    // GPU resources
    cv::cuda::GpuMat m_gpuSource;
    
    // Components
    std::unique_ptr<FaceDetector> m_detector;
    std::unique_ptr<Segmentation> m_segmentation;
    
    // Temporal buffers (for Mode 3)
    std::vector<cv::Mat> m_frameBuffer;
    std::vector<FaceInfo> m_faceBuffer;
};

// Extended class for live face replacement with expression transfer
class LiveFaceReplacer : public FaceReplacer {
public:
    explicit LiveFaceReplacer(const Config& config);
    
    cv::Mat processWithExpression(const cv::Mat& frame, const FaceInfo& targetFace);
    
protected:
    cv::Mat warpToPose(const cv::Mat& sourceFace,
                       const std::vector<cv::Point2f>& sourceLandmarks,
                       const std::vector<cv::Point2f>& targetLandmarks);
    
    void calculateDelaunay(const std::vector<cv::Point2f>& points,
                           const cv::Rect& bounds);
    
    cv::Mat warpTriangle(const cv::Mat& src, const cv::Mat& dst,
                         const std::vector<cv::Point2f>& srcTri,
                         const std::vector<cv::Point2f>& dstTri);

private:
    std::vector<cv::Vec6f> m_triangles;
};

} // namespace facereplacer
