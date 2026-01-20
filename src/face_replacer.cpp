#include "face_replacer.hpp"
#include "face_detector.hpp"
#include "segmentation.hpp"
#include "cuda/gpu_blend.cuh"
#include <iostream>

namespace facereplacer {

FaceReplacer::FaceReplacer(const Config& config) : m_config(config) {
    m_detector = std::make_unique<FaceDetector>(config);
    m_segmentation = std::make_unique<Segmentation>(config);
    
    // Check GPU availability
    if (m_config.useGPU && !cuda::isCudaAvailable()) {
        std::cerr << "Warning: CUDA not available, falling back to CPU" << std::endl;
        m_config.useGPU = false;
    }
    
    if (m_config.useGPU) {
        cuda::printCudaInfo();
    }
}

std::vector<FaceInfo> FaceReplacer::detectFaces(const cv::Mat& image) {
    return m_detector->detect(image);
}

cv::Mat FaceReplacer::markFace(const cv::Mat& frame, int faceIndex) {
    cv::Mat result = frame.clone();
    
    auto faces = m_detector->detect(frame);
    
    if (faces.empty()) {
        std::cerr << "No faces detected!" << std::endl;
        return result;
    }
    
    if (faceIndex >= static_cast<int>(faces.size())) {
        faceIndex = 0;
    }
    
    // Draw all faces
    for (size_t i = 0; i < faces.size(); i++) {
        cv::Scalar color = (i == static_cast<size_t>(faceIndex)) ? 
                           cv::Scalar(0, 255, 0) : cv::Scalar(255, 0, 0);
        int thickness = (i == static_cast<size_t>(faceIndex)) ? 3 : 1;
        
        cv::rectangle(result, faces[i].boundingBox, color, thickness);
        cv::putText(result, "Face " + std::to_string(i), 
                    cv::Point(faces[i].boundingBox.x, faces[i].boundingBox.y - 10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 2);
    }
    
    m_targetFace = faces[faceIndex];
    
    return result;
}

void FaceReplacer::setSourceImage(const cv::Mat& selfie) {
    m_sourceImage = selfie.clone();
    
    // Detect face in source
    auto faces = m_detector->detect(selfie);
    if (!faces.empty()) {
        m_sourceFace = faces[0];
        
        // Generate segmentation mask for modes 2 and 3
        if (m_config.mode != ReplacementMode::RECT_TO_RECT) {
            m_sourceMask = m_segmentation->segmentHead(selfie, m_sourceFace.boundingBox);
        }
        
        std::cout << "Source face detected: " << m_sourceFace.boundingBox << std::endl;
    } else {
        std::cerr << "Warning: No face detected in source image" << std::endl;
    }
    
    // Upload to GPU
    if (m_config.useGPU) {
        m_gpuSource.upload(m_sourceImage);
    }
}

void FaceReplacer::setTargetFace(const FaceInfo& targetFace) {
    m_targetFace = targetFace;
}

cv::Mat FaceReplacer::processFrame(const cv::Mat& frame) {
    cv::Mat result;
    
    switch (m_config.mode) {
        case ReplacementMode::RECT_TO_RECT:
            result = replaceRectToRect(frame, m_sourceImage, m_targetFace.boundingBox);
            break;
            
        case ReplacementMode::HEAD_SEGMENTED:
            result = replaceSegmented(frame, m_sourceImage, m_sourceMask, 
                                      m_targetFace.boundingBox);
            break;
            
        case ReplacementMode::LIVE_ANIMATED:
            result = replaceLive(frame, m_targetFace);
            break;
    }
    
    return result;
}

cv::Mat FaceReplacer::replaceRectToRect(const cv::Mat& frame, const cv::Mat& source,
                                         const cv::Rect& targetRect) {
    cv::Mat result = frame.clone();
    
    // Extract source face region
    cv::Rect sourceRect = m_sourceFace.boundingBox;
    sourceRect = clampRect(sourceRect, source.size());
    
    if (sourceRect.width <= 0 || sourceRect.height <= 0) {
        return result;
    }
    
    cv::Mat sourceFace = source(sourceRect).clone();
    
    // Expand target region slightly
    cv::Rect expandedTarget = scaleRect(targetRect, 1.2f);
    expandedTarget = clampRect(expandedTarget, frame.size());
    
    if (expandedTarget.width <= 0 || expandedTarget.height <= 0) {
        return result;
    }
    
    // Resize source to match target
    cv::Mat resizedSource;
    cv::resize(sourceFace, resizedSource, expandedTarget.size());
    
    // Color correction
    if (m_config.colorCorrection) {
        cv::Mat targetRegion = frame(expandedTarget);
        resizedSource = matchColors(resizedSource, targetRegion);
    }
    
    // Lighting adjustment
    if (m_config.preserveLighting) {
        resizedSource = adjustLighting(resizedSource, frame, expandedTarget);
    }
    
    // Create gradient mask for blending
    cv::Mat mask = cv::Mat::zeros(resizedSource.size(), CV_8UC1);
    cv::ellipse(mask, 
                cv::Point(mask.cols / 2, mask.rows / 2),
                cv::Size(mask.cols / 2 - 5, mask.rows / 2 - 5),
                0, 0, 360, cv::Scalar(255), -1);
    
    // Feather the mask
    cv::GaussianBlur(mask, mask, cv::Size(21, 21), 10);
    
    // Blend
    cv::Point center(expandedTarget.x + expandedTarget.width / 2,
                     expandedTarget.y + expandedTarget.height / 2);
    
    if (m_config.useGPU) {
        result = blendGPU(resizedSource, frame, mask);
    } else {
        result = poissonBlend(resizedSource, frame, mask, center);
    }
    
    return result;
}

cv::Mat FaceReplacer::replaceSegmented(const cv::Mat& frame, const cv::Mat& source,
                                        const cv::Mat& sourceMask, const cv::Rect& targetRect) {
    cv::Mat result = frame.clone();
    
    // Get source face region with mask
    cv::Rect sourceRect = m_sourceFace.boundingBox;
    cv::Rect expandedSource = scaleRect(sourceRect, 2.0f);
    expandedSource = clampRect(expandedSource, source.size());
    
    if (expandedSource.width <= 0 || expandedSource.height <= 0) {
        return result;
    }
    
    cv::Mat sourceFaceRegion = source(expandedSource).clone();
    cv::Mat maskRegion = sourceMask(expandedSource).clone();
    
    // Target region
    cv::Rect expandedTarget = scaleRect(targetRect, 2.0f);
    expandedTarget = clampRect(expandedTarget, frame.size());
    
    if (expandedTarget.width <= 0 || expandedTarget.height <= 0) {
        return result;
    }
    
    // Resize source and mask to match target
    cv::Mat resizedSource, resizedMask;
    cv::resize(sourceFaceRegion, resizedSource, expandedTarget.size());
    cv::resize(maskRegion, resizedMask, expandedTarget.size());
    
    // Color correction
    if (m_config.colorCorrection) {
        cv::Mat targetRegion = frame(expandedTarget);
        resizedSource = matchColors(resizedSource, targetRegion, resizedMask);
    }
    
    // Lighting adjustment
    if (m_config.preserveLighting) {
        resizedSource = adjustLighting(resizedSource, frame, expandedTarget);
    }
    
    // Refine mask edges
    resizedMask = m_segmentation->refineMask(resizedMask, resizedSource);
    resizedMask = m_segmentation->featherMask(resizedMask, m_config.featherRadius);
    
    // Blend into target region
    cv::Mat targetRegion = result(expandedTarget);
    
    if (m_config.useGPU) {
        cv::cuda::GpuMat gpuSrc, gpuDst, gpuMask, gpuResult;
        gpuSrc.upload(resizedSource);
        gpuDst.upload(targetRegion);
        gpuMask.upload(resizedMask);
        
        cuda::featheredBlend(gpuSrc, gpuDst, gpuMask, gpuResult, m_config.featherRadius);
        gpuResult.download(targetRegion);
    } else {
        // CPU alpha blending
        for (int y = 0; y < targetRegion.rows; y++) {
            for (int x = 0; x < targetRegion.cols; x++) {
                float alpha = resizedMask.at<uchar>(y, x) / 255.0f;
                cv::Vec3b srcPixel = resizedSource.at<cv::Vec3b>(y, x);
                cv::Vec3b dstPixel = targetRegion.at<cv::Vec3b>(y, x);
                
                targetRegion.at<cv::Vec3b>(y, x) = cv::Vec3b(
                    static_cast<uchar>(srcPixel[0] * alpha + dstPixel[0] * (1 - alpha)),
                    static_cast<uchar>(srcPixel[1] * alpha + dstPixel[1] * (1 - alpha)),
                    static_cast<uchar>(srcPixel[2] * alpha + dstPixel[2] * (1 - alpha))
                );
            }
        }
    }
    
    return result;
}

cv::Mat FaceReplacer::replaceLive(const cv::Mat& frame, const FaceInfo& targetFace) {
    cv::Mat result = frame.clone();
    
    // Warp source face to match target pose
    cv::Mat warpedSource = warpFaceToTarget(m_sourceImage, m_sourceFace, targetFace);
    
    // Generate mask for warped face
    cv::Mat warpedMask;
    if (!m_sourceMask.empty()) {
        // Warp the mask using the same transformation
        warpedMask = warpFaceToTarget(m_sourceMask, m_sourceFace, targetFace);
    } else {
        warpedMask = m_segmentation->segmentHead(warpedSource, targetFace.boundingBox);
    }
    
    // Color and lighting correction
    if (m_config.colorCorrection) {
        cv::Rect targetRect = scaleRect(targetFace.boundingBox, 1.5f);
        targetRect = clampRect(targetRect, frame.size());
        if (targetRect.width > 0 && targetRect.height > 0) {
            warpedSource = matchColors(warpedSource, frame(targetRect), warpedMask);
        }
    }
    
    // Apply temporal smoothing
    updateBuffers(warpedSource, targetFace);
    if (m_config.temporalSmoothing > 1) {
        warpedSource = applyTemporalSmoothing(warpedSource);
    }
    
    // Final blend
    cv::Rect blendRect = scaleRect(targetFace.boundingBox, 2.0f);
    blendRect = clampRect(blendRect, frame.size());
    
    if (blendRect.width <= 0 || blendRect.height <= 0) {
        return result;
    }
    
    // Ensure warpedSource and warpedMask are large enough
    if (warpedSource.cols < blendRect.x + blendRect.width ||
        warpedSource.rows < blendRect.y + blendRect.height) {
        return result;
    }
    
    cv::Mat blendRegion = warpedSource(blendRect);
    cv::Mat maskRegion = warpedMask(blendRect);
    cv::Mat targetRegion = result(blendRect);
    
    // GPU blending
    if (m_config.useGPU) {
        cv::cuda::GpuMat gpuSrc, gpuDst, gpuMask, gpuResult;
        gpuSrc.upload(blendRegion);
        gpuDst.upload(targetRegion);
        gpuMask.upload(maskRegion);
        
        cuda::featheredBlend(gpuSrc, gpuDst, gpuMask, gpuResult, m_config.featherRadius);
        gpuResult.download(targetRegion);
    } else {
        // CPU blending with seamless clone
        cv::Point center(blendRect.x + blendRect.width / 2, 
                         blendRect.y + blendRect.height / 2);
        try {
            cv::seamlessClone(blendRegion, result, maskRegion, center, result, cv::NORMAL_CLONE);
        } catch (const cv::Exception& e) {
            std::cerr << "Seamless clone failed: " << e.what() << std::endl;
        }
    }
    
    return result;
}

cv::Mat FaceReplacer::matchColors(const cv::Mat& source, const cv::Mat& target,
                                   const cv::Mat& mask) {
    cv::Mat result = source.clone();
    cv::Mat sourceLab, targetLab;
    
    cv::cvtColor(source, sourceLab, cv::COLOR_BGR2Lab);
    cv::cvtColor(target, targetLab, cv::COLOR_BGR2Lab);
    
    // Calculate statistics
    cv::Scalar srcMean, srcStd, tgtMean, tgtStd;
    
    if (mask.empty()) {
        cv::meanStdDev(sourceLab, srcMean, srcStd);
        cv::meanStdDev(targetLab, tgtMean, tgtStd);
    } else {
        cv::Mat resizedMask;
        if (mask.size() != source.size()) {
            cv::resize(mask, resizedMask, source.size());
        } else {
            resizedMask = mask;
        }
        cv::meanStdDev(sourceLab, srcMean, srcStd, resizedMask);
        
        if (mask.size() != target.size()) {
            cv::resize(mask, resizedMask, target.size());
        }
        cv::meanStdDev(targetLab, tgtMean, tgtStd, resizedMask);
    }
    
    // Transfer color statistics
    std::vector<cv::Mat> labChannels;
    cv::split(sourceLab, labChannels);
    
    for (int i = 0; i < 3; i++) {
        if (srcStd[i] > 1e-6) {
            labChannels[i].convertTo(labChannels[i], CV_32F);
            labChannels[i] = ((labChannels[i] - srcMean[i]) * (tgtStd[i] / srcStd[i])) + tgtMean[i];
            labChannels[i].convertTo(labChannels[i], CV_8U);
        }
    }
    
    cv::Mat resultLab;
    cv::merge(labChannels, resultLab);
    cv::cvtColor(resultLab, result, cv::COLOR_Lab2BGR);
    
    return result;
}

cv::Mat FaceReplacer::adjustLighting(const cv::Mat& source, const cv::Mat& target,
                                      const cv::Rect& region) {
    cv::Mat result = source.clone();
    
    cv::Rect clampedRegion = clampRect(region, target.size());
    if (clampedRegion.width <= 0 || clampedRegion.height <= 0) {
        return result;
    }
    
    // Get target region luminance
    cv::Mat targetRegion = target(clampedRegion);
    cv::Mat targetGray, sourceGray;
    cv::cvtColor(targetRegion, targetGray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(source, sourceGray, cv::COLOR_BGR2GRAY);
    
    // Calculate luminance ratio
    double targetMean = cv::mean(targetGray)[0];
    double sourceMean = cv::mean(sourceGray)[0];
    
    if (sourceMean > 1e-6) {
        double ratio = targetMean / sourceMean;
        ratio = std::clamp(ratio, 0.5, 2.0);  // Limit adjustment
        result.convertTo(result, -1, ratio, 0);
    }
    
    return result;
}

cv::Mat FaceReplacer::poissonBlend(const cv::Mat& source, const cv::Mat& target,
                                    const cv::Mat& mask, const cv::Point& center) {
    cv::Mat result;
    
    try {
        cv::seamlessClone(source, target, mask, center, result, cv::NORMAL_CLONE);
    } catch (const cv::Exception& e) {
        std::cerr << "Poisson blending failed: " << e.what() << std::endl;
        result = target.clone();
        
        // Fallback to alpha blending
        cv::Rect roi(center.x - source.cols / 2, center.y - source.rows / 2,
                     source.cols, source.rows);
        roi = clampRect(roi, target.size());
        
        if (roi.width > 0 && roi.height > 0) {
            cv::Mat srcRoi = source(cv::Rect(0, 0, roi.width, roi.height));
            cv::Mat dstRoi = result(roi);
            cv::Mat maskRoi = mask(cv::Rect(0, 0, roi.width, roi.height));
            
            cv::Mat maskF;
            maskRoi.convertTo(maskF, CV_32F, 1.0 / 255.0);
            cv::cvtColor(maskF, maskF, cv::COLOR_GRAY2BGR);
            
            cv::Mat srcF, dstF;
            srcRoi.convertTo(srcF, CV_32F);
            dstRoi.convertTo(dstF, CV_32F);
            
            cv::Mat blended = srcF.mul(maskF) + dstF.mul(cv::Scalar(1, 1, 1) - maskF);
            blended.convertTo(dstRoi, CV_8U);
        }
    }
    
    return result;
}

cv::Mat FaceReplacer::blendGPU(const cv::Mat& source, const cv::Mat& target,
                                const cv::Mat& mask) {
    cv::cuda::GpuMat gpuSrc, gpuDst, gpuMask, gpuResult;
    gpuSrc.upload(source);
    gpuDst.upload(target);
    gpuMask.upload(mask);
    
    cuda::featheredBlend(gpuSrc, gpuDst, gpuMask, gpuResult, m_config.featherRadius);
    
    cv::Mat result;
    gpuResult.download(result);
    
    return result;
}

cv::Mat FaceReplacer::warpFaceToTarget(const cv::Mat& source, const FaceInfo& sourceFace,
                                        const FaceInfo& targetFace) {
    cv::Mat result = cv::Mat::zeros(source.size(), source.type());
    
    if (sourceFace.landmarks.size() < 5 || targetFace.landmarks.size() < 5) {
        // Simple affine warp based on bounding boxes
        cv::Point2f srcPoints[3], dstPoints[3];
        
        srcPoints[0] = cv::Point2f(static_cast<float>(sourceFace.boundingBox.x), 
                                    static_cast<float>(sourceFace.boundingBox.y));
        srcPoints[1] = cv::Point2f(static_cast<float>(sourceFace.boundingBox.x + sourceFace.boundingBox.width), 
                                    static_cast<float>(sourceFace.boundingBox.y));
        srcPoints[2] = cv::Point2f(static_cast<float>(sourceFace.boundingBox.x), 
                                    static_cast<float>(sourceFace.boundingBox.y + sourceFace.boundingBox.height));
        
        dstPoints[0] = cv::Point2f(static_cast<float>(targetFace.boundingBox.x), 
                                    static_cast<float>(targetFace.boundingBox.y));
        dstPoints[1] = cv::Point2f(static_cast<float>(targetFace.boundingBox.x + targetFace.boundingBox.width),
                                    static_cast<float>(targetFace.boundingBox.y));
        dstPoints[2] = cv::Point2f(static_cast<float>(targetFace.boundingBox.x),
                                    static_cast<float>(targetFace.boundingBox.y + targetFace.boundingBox.height));
        
        cv::Mat warpMat = cv::getAffineTransform(srcPoints, dstPoints);
        cv::warpAffine(source, result, warpMat, result.size());
    } else {
        // Use Delaunay triangulation for landmark-based warping
        // (simplified version - full implementation in LiveFaceReplacer)
        cv::Mat warpMat = cv::estimateAffine2D(sourceFace.landmarks, targetFace.landmarks);
        if (!warpMat.empty()) {
            cv::warpAffine(source, result, warpMat, result.size());
        } else {
            result = source.clone();
        }
    }
    
    return result;
}

cv::Mat FaceReplacer::applyTemporalSmoothing(const cv::Mat& currentResult) {
    if (m_frameBuffer.empty()) {
        return currentResult;
    }
    
    // Exponential moving average weights
    std::vector<float> weights(m_frameBuffer.size() + 1);
    float totalWeight = 0;
    float decay = 0.7f;
    
    for (size_t i = 0; i < weights.size(); i++) {
        weights[i] = std::pow(decay, static_cast<float>(weights.size() - 1 - i));
        totalWeight += weights[i];
    }
    
    // Normalize weights
    for (auto& w : weights) {
        w /= totalWeight;
    }
    
    cv::Mat result = cv::Mat::zeros(currentResult.size(), CV_32FC3);
    
    for (size_t i = 0; i < m_frameBuffer.size(); i++) {
        cv::Mat frameF;
        m_frameBuffer[i].convertTo(frameF, CV_32FC3);
        result += frameF * weights[i];
    }
    
    cv::Mat currentF;
    currentResult.convertTo(currentF, CV_32FC3);
    result += currentF * weights.back();
    
    result.convertTo(result, CV_8UC3);
    
    return result;
}

void FaceReplacer::updateBuffers(const cv::Mat& frame, const FaceInfo& face) {
    m_frameBuffer.push_back(frame.clone());
    m_faceBuffer.push_back(face);
    
    if (static_cast<int>(m_frameBuffer.size()) > m_config.temporalSmoothing) {
        m_frameBuffer.erase(m_frameBuffer.begin());
        m_faceBuffer.erase(m_faceBuffer.begin());
    }
}

// LiveFaceReplacer implementation
LiveFaceReplacer::LiveFaceReplacer(const Config& config) : FaceReplacer(config) {
}

cv::Mat LiveFaceReplacer::processWithExpression(const cv::Mat& frame, const FaceInfo& targetFace) {
    // For now, use the standard live replacement
    return replaceLive(frame, targetFace);
}

cv::Mat LiveFaceReplacer::warpToPose(const cv::Mat& sourceFace,
                                      const std::vector<cv::Point2f>& sourceLandmarks,
                                      const std::vector<cv::Point2f>& targetLandmarks) {
    if (sourceLandmarks.size() != targetLandmarks.size() || sourceLandmarks.size() < 3) {
        return sourceFace.clone();
    }
    
    cv::Mat result = cv::Mat::zeros(sourceFace.size(), sourceFace.type());
    
    // Create Delaunay triangulation on target landmarks
    cv::Rect bounds(0, 0, sourceFace.cols, sourceFace.rows);
    calculateDelaunay(targetLandmarks, bounds);
    
    // Warp each triangle
    for (const auto& tri : m_triangles) {
        std::vector<cv::Point2f> srcTri(3), dstTri(3);
        bool valid = true;
        
        for (int i = 0; i < 3; i++) {
            cv::Point2f pt(tri[i * 2], tri[i * 2 + 1]);
            
            // Find closest landmark
            int idx = -1;
            float minDist = std::numeric_limits<float>::max();
            
            for (size_t j = 0; j < targetLandmarks.size(); j++) {
                float dist = static_cast<float>(cv::norm(pt - targetLandmarks[j]));
                if (dist < minDist) {
                    minDist = dist;
                    idx = static_cast<int>(j);
                }
            }
            
            if (idx < 0 || minDist > 5.0f) {
                valid = false;
                break;
            }
            
            srcTri[i] = sourceLandmarks[idx];
            dstTri[i] = targetLandmarks[idx];
        }
        
        if (valid) {
            result = warpTriangle(sourceFace, result, srcTri, dstTri);
        }
    }
    
    return result;
}

void LiveFaceReplacer::calculateDelaunay(const std::vector<cv::Point2f>& points,
                                          const cv::Rect& bounds) {
    m_triangles.clear();
    
    cv::Subdiv2D subdiv(bounds);
    
    for (const auto& pt : points) {
        if (bounds.contains(pt)) {
            subdiv.insert(pt);
        }
    }
    
    subdiv.getTriangleList(m_triangles);
    
    // Filter triangles outside bounds
    m_triangles.erase(
        std::remove_if(m_triangles.begin(), m_triangles.end(),
                       [&bounds](const cv::Vec6f& tri) {
                           for (int i = 0; i < 3; i++) {
                               cv::Point2f pt(tri[i * 2], tri[i * 2 + 1]);
                               if (!bounds.contains(pt)) return true;
                           }
                           return false;
                       }),
        m_triangles.end()
    );
}

cv::Mat LiveFaceReplacer::warpTriangle(const cv::Mat& src, const cv::Mat& dst,
                                        const std::vector<cv::Point2f>& srcTri,
                                        const std::vector<cv::Point2f>& dstTri) {
    cv::Mat result = dst.clone();
    
    cv::Rect srcRect = cv::boundingRect(srcTri);
    cv::Rect dstRect = cv::boundingRect(dstTri);
    
    srcRect = clampRect(srcRect, src.size());
    dstRect = clampRect(dstRect, result.size());
    
    if (srcRect.width <= 0 || srcRect.height <= 0 ||
        dstRect.width <= 0 || dstRect.height <= 0) {
        return result;
    }
    
    std::vector<cv::Point2f> srcTriRect(3), dstTriRect(3);
    std::vector<cv::Point> dstTriInt(3);
    
    for (int i = 0; i < 3; i++) {
        srcTriRect[i] = cv::Point2f(srcTri[i].x - srcRect.x, srcTri[i].y - srcRect.y);
        dstTriRect[i] = cv::Point2f(dstTri[i].x - dstRect.x, dstTri[i].y - dstRect.y);
        dstTriInt[i] = cv::Point(static_cast<int>(dstTriRect[i].x), 
                                  static_cast<int>(dstTriRect[i].y));
    }
    
    cv::Mat warpMat = cv::getAffineTransform(srcTriRect, dstTriRect);
    
    cv::Mat srcCrop = src(srcRect);
    cv::Mat dstCrop;
    cv::warpAffine(srcCrop, dstCrop, warpMat, dstRect.size());
    
    cv::Mat mask = cv::Mat::zeros(dstRect.height, dstRect.width, CV_8UC1);
    cv::fillConvexPoly(mask, dstTriInt, cv::Scalar(255));
    
    dstCrop.copyTo(result(dstRect), mask);
    
    return result;
}

} // namespace facereplacer
