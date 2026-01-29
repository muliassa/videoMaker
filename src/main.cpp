/**
 * Face Replacer - Two-Phase Pipeline
 * 
 * Phase 1 (Preprocess): Detect and label all faces with #1, #2, #3...
 *   ./face_replacer --preprocess video.mp4 preview.mp4
 * 
 * Phase 2 (Production): Replace specified face(s) with selfie
 *   ./face_replacer video.mp4 selfie.jpg output.mp4 [face_id] [mode]
 */

#include "face_replacer.hpp"
#ifdef USE_CUDA
#include "cuda/gpu_blend.cuh"
#endif
#include <iostream>
#include <string>
#include <chrono>
#include <algorithm>
#include <map>
#include <cmath>

void printUsage(const char* programName) {
    std::cout << "Face Replacer - Two-Phase Pipeline\n\n";
    std::cout << "=== PHASE 1: PREPROCESS (detect and label faces) ===\n";
    std::cout << "  " << programName << " --preprocess <input_video> <output_preview>\n\n";
    std::cout << "  This creates a video with all faces marked #1, #2, #3...\n";
    std::cout << "  Review this to find which face number you want to replace.\n\n";
    std::cout << "=== PHASE 2: PRODUCTION (replace faces) ===\n";
    std::cout << "  " << programName << " <input_video> <selfie.jpg> <output_video> [face_id] [mode]\n\n";
    std::cout << "  face_id  - Which face to replace (from preprocess, default: 1)\n";
    std::cout << "  mode     - 1=RECT, 2=SEGMENTED (default), 3=LIVE\n\n";
    std::cout << "Examples:\n";
    std::cout << "  " << programName << " --preprocess video.mp4 preview.mp4\n";
    std::cout << "  " << programName << " video.mp4 selfie.jpg output.mp4 2 2\n";
}

//------------------------------------------------------------------------------
// Simple Face Tracker - assigns consistent IDs across frames
//------------------------------------------------------------------------------
class FaceTracker {
public:
    struct TrackedFace {
        int id;
        cv::Rect lastPosition;
        int framesLost;
        cv::Scalar color;
    };
    
    FaceTracker() : m_nextId(1) {
        // Predefined colors for faces
        m_colors = {
            cv::Scalar(0, 255, 0),    // Green
            cv::Scalar(255, 0, 0),    // Blue
            cv::Scalar(0, 0, 255),    // Red
            cv::Scalar(255, 255, 0),  // Cyan
            cv::Scalar(255, 0, 255),  // Magenta
            cv::Scalar(0, 255, 255),  // Yellow
            cv::Scalar(128, 255, 0),  // Lime
            cv::Scalar(255, 128, 0),  // Orange
        };
    }
    
    // Update tracker with new detections, returns map of detection index -> face ID
    std::map<int, int> update(const std::vector<facereplacer::FaceInfo>& detections) {
        std::map<int, int> assignments;  // detection idx -> face ID
        std::vector<bool> matched(detections.size(), false);
        
        // Try to match existing tracks
        for (auto& track : m_tracks) {
            float bestDist = 1e9;
            int bestIdx = -1;
            
            for (size_t i = 0; i < detections.size(); i++) {
                if (matched[i]) continue;
                
                float dist = rectDistance(track.lastPosition, detections[i].boundingBox);
                if (dist < bestDist && dist < 150) {  // Max distance threshold
                    bestDist = dist;
                    bestIdx = static_cast<int>(i);
                }
            }
            
            if (bestIdx >= 0) {
                track.lastPosition = detections[bestIdx].boundingBox;
                track.framesLost = 0;
                assignments[bestIdx] = track.id;
                matched[bestIdx] = true;
            } else {
                track.framesLost++;
            }
        }
        
        // Create new tracks for unmatched detections
        for (size_t i = 0; i < detections.size(); i++) {
            if (!matched[i]) {
                TrackedFace newTrack;
                newTrack.id = m_nextId++;
                newTrack.lastPosition = detections[i].boundingBox;
                newTrack.framesLost = 0;
                newTrack.color = m_colors[(newTrack.id - 1) % m_colors.size()];
                m_tracks.push_back(newTrack);
                assignments[static_cast<int>(i)] = newTrack.id;
            }
        }
        
        // Remove tracks lost for too long
        m_tracks.erase(
            std::remove_if(m_tracks.begin(), m_tracks.end(),
                [](const TrackedFace& t) { return t.framesLost > 30; }),
            m_tracks.end());
        
        return assignments;
    }
    
    cv::Scalar getColor(int faceId) {
        for (const auto& track : m_tracks) {
            if (track.id == faceId) return track.color;
        }
        return cv::Scalar(0, 255, 0);
    }
    
    // Get face position by ID
    bool getFacePosition(int faceId, cv::Rect& outRect) {
        for (const auto& track : m_tracks) {
            if (track.id == faceId) {
                outRect = track.lastPosition;
                return true;
            }
        }
        return false;
    }
    
private:
    float rectDistance(const cv::Rect& a, const cv::Rect& b) {
        float cx1 = a.x + a.width / 2.0f;
        float cy1 = a.y + a.height / 2.0f;
        float cx2 = b.x + b.width / 2.0f;
        float cy2 = b.y + b.height / 2.0f;
        return std::sqrt((cx1-cx2)*(cx1-cx2) + (cy1-cy2)*(cy1-cy2));
    }
    
    std::vector<TrackedFace> m_tracks;
    int m_nextId;
    std::vector<cv::Scalar> m_colors;
};

//------------------------------------------------------------------------------
// PHASE 1: Preprocess - detect and label all faces
//------------------------------------------------------------------------------
int preprocessVideo(const std::string& inputPath, const std::string& outputPath) {
    std::cout << "\n=== PHASE 1: PREPROCESSING ===" << std::endl;
    std::cout << "Detecting and labeling all faces...\n" << std::endl;
    
    cv::VideoCapture cap(inputPath);
    if (!cap.isOpened()) {
        std::cerr << "Error: Cannot open " << inputPath << std::endl;
        return 1;
    }
    
    int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);
    int totalFrames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    
    std::cout << "Video: " << width << "x" << height << " @ " << fps << " FPS, " 
              << totalFrames << " frames" << std::endl;
    
    int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
    cv::VideoWriter writer(outputPath, fourcc, fps, cv::Size(width, height));
    if (!writer.isOpened()) {
        std::cerr << "Error: Cannot create output video" << std::endl;
        return 1;
    }
    
    facereplacer::Config config;
    config.useGPU = false;
    facereplacer::FaceReplacer detector(config);
    FaceTracker tracker;
    
    cv::Mat frame;
    int frameNum = 0;
    std::map<int, int> faceAppearances;  // face_id -> frame count
    
    while (true) {
        cap >> frame;
        if (frame.empty()) break;
        
        auto faces = detector.detectFaces(frame);
        auto assignments = tracker.update(faces);
        
        // Draw rectangles and labels
        for (size_t i = 0; i < faces.size(); i++) {
            int faceId = assignments[static_cast<int>(i)];
            cv::Scalar color = tracker.getColor(faceId);
            cv::Rect& box = faces[i].boundingBox;
            
            // Draw thick rectangle
            cv::rectangle(frame, box, color, 3);
            
            // Draw label background
            std::string label = "#" + std::to_string(faceId);
            int baseline;
            cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 1.0, 2, &baseline);
            cv::rectangle(frame, 
                         cv::Point(box.x, box.y - textSize.height - 10),
                         cv::Point(box.x + textSize.width + 10, box.y),
                         color, -1);
            
            // Draw label text
            cv::putText(frame, label, cv::Point(box.x + 5, box.y - 5),
                       cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255), 2);
            
            faceAppearances[faceId]++;
        }
        
        // Draw frame number
        cv::putText(frame, "Frame: " + std::to_string(frameNum), cv::Point(10, 30),
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
        
        writer.write(frame);
        frameNum++;
        
        if (frameNum % 30 == 0) {
            std::cout << "\rProcessing: " << frameNum << "/" << totalFrames << std::flush;
        }
    }
    
    cap.release();
    writer.release();
    
    std::cout << "\n\n=== FACE SUMMARY ===" << std::endl;
    for (const auto& pair : faceAppearances) {
        std::cout << "  Face #" << pair.first << ": appeared in " << pair.second << " frames" << std::endl;
    }
    
    std::cout << "\nPreview saved: " << outputPath << std::endl;
    std::cout << "\nNext step: Watch the preview and note which face # to replace." << std::endl;
    std::cout << "Then run: face_replacer " << inputPath << " selfie.jpg output.mp4 <face#>" << std::endl;
    
    return 0;
}

//------------------------------------------------------------------------------
// PHASE 2: Production - replace specified face with selfie
//------------------------------------------------------------------------------
int processVideo(const std::string& inputPath, const std::string& selfiePath,
                 const std::string& outputPath, int targetFaceId, int mode) {
    
    std::cout << "\n=== PHASE 2: PRODUCTION ===" << std::endl;
    std::cout << "Replacing face #" << targetFaceId << "...\n" << std::endl;
    
    cv::VideoCapture cap(inputPath);
    if (!cap.isOpened()) {
        std::cerr << "Error: Cannot open " << inputPath << std::endl;
        return 1;
    }
    
    int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);
    int totalFrames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    
    std::cout << "Video: " << width << "x" << height << " @ " << fps << " FPS" << std::endl;
    
    int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
    cv::VideoWriter writer(outputPath, fourcc, fps, cv::Size(width, height));
    if (!writer.isOpened()) {
        std::cerr << "Error: Cannot create output video" << std::endl;
        return 1;
    }
    
    // Load selfie
    cv::Mat selfie = cv::imread(selfiePath);
    if (selfie.empty()) {
        std::cerr << "Error: Cannot load selfie " << selfiePath << std::endl;
        return 1;
    }
    std::cout << "Selfie: " << selfie.cols << "x" << selfie.rows << std::endl;
    
    // Configure replacer
    facereplacer::Config config;
    switch (mode) {
        case 1: config.mode = facereplacer::ReplacementMode::RECT_TO_RECT; break;
        case 3: config.mode = facereplacer::ReplacementMode::LIVE_ANIMATED; break;
        default: config.mode = facereplacer::ReplacementMode::HEAD_SEGMENTED; break;
    }
    config.useGPU = false;
    config.colorCorrection = true;
    config.featherRadius = 15;
    
    std::cout << "Mode: " << mode << std::endl;
    
    facereplacer::FaceReplacer replacer(config);
    replacer.setSourceImage(selfie);
    
    FaceTracker tracker;
    
    cv::Mat frame;
    int frameNum = 0;
    int replacedFrames = 0;
    
    auto startTime = std::chrono::high_resolution_clock::now();
    
    while (true) {
        cap >> frame;
        if (frame.empty()) break;
        
        auto faces = replacer.detectFaces(frame);
        auto assignments = tracker.update(faces);
        
        // Find the detection that matches our target face ID
        int targetDetectionIdx = -1;
        for (const auto& pair : assignments) {
            if (pair.second == targetFaceId) {
                targetDetectionIdx = pair.first;
                break;
            }
        }
        
        cv::Mat result = frame.clone();
        
        if (targetDetectionIdx >= 0 && targetDetectionIdx < static_cast<int>(faces.size())) {
            // Set target and replace
            replacer.setTargetFace(faces[targetDetectionIdx]);
            result = replacer.processFrame(frame);
            replacedFrames++;
        }
        
        writer.write(result);
        frameNum++;
        
        if (frameNum % 30 == 0) {
            float progress = 100.0f * frameNum / totalFrames;
            std::cout << "\rProgress: " << frameNum << "/" << totalFrames 
                      << " (" << static_cast<int>(progress) << "%)" << std::flush;
        }
    }
    
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(endTime - startTime);
    
    cap.release();
    writer.release();
    
    std::cout << "\n\nProcessed " << frameNum << " frames in " << duration.count() << "s" << std::endl;
    std::cout << "Face #" << targetFaceId << " replaced in " << replacedFrames << " frames" << std::endl;
    std::cout << "Output: " << outputPath << std::endl;
    
    return 0;
}

//------------------------------------------------------------------------------
// Main
//------------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    if (argc < 3) {
        printUsage(argv[0]);
        return 1;
    }
    
    std::string arg1 = argv[1];
    
    // Phase 1: Preprocess
    if (arg1 == "--preprocess" || arg1 == "-p") {
        if (argc < 4) {
            std::cerr << "Usage: " << argv[0] << " --preprocess <input.mp4> <preview.mp4>" << std::endl;
            return 1;
        }
        return preprocessVideo(argv[2], argv[3]);
    }
    
    // Phase 2: Production
    if (argc < 4) {
        printUsage(argv[0]);
        return 1;
    }
    
    std::string inputPath = argv[1];
    std::string selfiePath = argv[2];
    std::string outputPath = argv[3];
    int faceId = (argc > 4) ? std::stoi(argv[4]) : 1;
    int mode = (argc > 5) ? std::stoi(argv[5]) : 2;
    
    return processVideo(inputPath, selfiePath, outputPath, faceId, mode);
}
