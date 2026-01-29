/**
 * Face Replacer - Two-Phase Pipeline with JSON tracking
 * 
 * Phase 1 (Preprocess): Detect faces, output JSON + preview video
 *   ./face_replacer --preprocess video.mp4 tracking.json preview.mp4
 * 
 * Phase 2 (Production): Replace face using JSON (no re-detection)
 *   ./face_replacer video.mp4 selfie.jpg output.mp4 tracking.json [face_id]
 */

#include "face_replacer.hpp"
#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <algorithm>
#include <vector>
#include <map>
#include <cmath>

void printUsage(const char* programName) {
    std::cout << "Face Replacer - Two-Phase Pipeline\n\n";
    std::cout << "=== PHASE 1: PREPROCESS ===\n";
    std::cout << "  " << programName << " --preprocess <video> <tracking.json> [preview.mp4]\n\n";
    std::cout << "  Outputs:\n";
    std::cout << "    - tracking.json: Face positions for all frames\n";
    std::cout << "    - preview.mp4: Video with labeled faces (optional)\n\n";
    std::cout << "=== PHASE 2: PRODUCTION ===\n";
    std::cout << "  " << programName << " <video> <selfie.jpg> <output.mp4> <tracking.json> [face_id]\n\n";
    std::cout << "  face_id: Which face to replace (default: 1)\n\n";
    std::cout << "Examples:\n";
    std::cout << "  " << programName << " --preprocess input.mp4 tracking.json preview.mp4\n";
    std::cout << "  " << programName << " input.mp4 selfie.jpg output.mp4 tracking.json 1\n";
}

//------------------------------------------------------------------------------
// JSON helpers (simple, no external library)
//------------------------------------------------------------------------------
struct FaceData {
    int id;
    int x, y, w, h;
};

struct FrameData {
    int frameNum;
    std::vector<FaceData> faces;
};

void writeTrackingJSON(const std::string& path, const std::vector<FrameData>& frames) {
    std::ofstream out(path);
    out << "{\n  \"frames\": [\n";
    
    for (size_t f = 0; f < frames.size(); f++) {
        const auto& frame = frames[f];
        out << "    {\"frame\": " << frame.frameNum << ", \"faces\": [";
        
        for (size_t i = 0; i < frame.faces.size(); i++) {
            const auto& face = frame.faces[i];
            out << "{\"id\": " << face.id 
                << ", \"x\": " << face.x 
                << ", \"y\": " << face.y
                << ", \"w\": " << face.w 
                << ", \"h\": " << face.h << "}";
            if (i < frame.faces.size() - 1) out << ", ";
        }
        
        out << "]}";
        if (f < frames.size() - 1) out << ",";
        out << "\n";
    }
    
    out << "  ]\n}\n";
    out.close();
}

std::vector<FrameData> readTrackingJSON(const std::string& path) {
    std::vector<FrameData> result;
    std::ifstream in(path);
    if (!in.is_open()) {
        std::cerr << "Cannot open " << path << std::endl;
        return result;
    }
    
    std::string content((std::istreambuf_iterator<char>(in)),
                         std::istreambuf_iterator<char>());
    in.close();
    
    // Simple parser for our specific JSON format
    size_t pos = 0;
    while ((pos = content.find("\"frame\":", pos)) != std::string::npos) {
        FrameData fd;
        
        // Parse frame number
        pos += 8;
        fd.frameNum = std::stoi(content.substr(pos));
        
        // Find faces array
        size_t facesStart = content.find("[", pos);
        size_t facesEnd = content.find("]", facesStart);
        std::string facesStr = content.substr(facesStart, facesEnd - facesStart + 1);
        
        // Parse each face
        size_t facePos = 0;
        while ((facePos = facesStr.find("\"id\":", facePos)) != std::string::npos) {
            FaceData face;
            
            facePos += 5;
            face.id = std::stoi(facesStr.substr(facePos));
            
            size_t xPos = facesStr.find("\"x\":", facePos) + 4;
            face.x = std::stoi(facesStr.substr(xPos));
            
            size_t yPos = facesStr.find("\"y\":", facePos) + 4;
            face.y = std::stoi(facesStr.substr(yPos));
            
            size_t wPos = facesStr.find("\"w\":", facePos) + 4;
            face.w = std::stoi(facesStr.substr(wPos));
            
            size_t hPos = facesStr.find("\"h\":", facePos) + 4;
            face.h = std::stoi(facesStr.substr(hPos));
            
            fd.faces.push_back(face);
            facePos = hPos;
        }
        
        result.push_back(fd);
        pos = facesEnd;
    }
    
    std::cout << "Loaded " << result.size() << " frames from JSON" << std::endl;
    return result;
}

//------------------------------------------------------------------------------
// Face Tracker
//------------------------------------------------------------------------------
class FaceTracker {
public:
    struct Track {
        int id;
        cv::Rect pos;
        int lost;
        cv::Scalar color;
    };
    
    FaceTracker() : m_nextId(1) {
        m_colors = {
            cv::Scalar(0, 255, 0), cv::Scalar(255, 0, 0), cv::Scalar(0, 0, 255),
            cv::Scalar(255, 255, 0), cv::Scalar(255, 0, 255), cv::Scalar(0, 255, 255),
            cv::Scalar(128, 255, 0), cv::Scalar(255, 128, 0), cv::Scalar(128, 0, 255)
        };
    }
    
    std::vector<std::pair<int, cv::Rect>> update(const std::vector<facereplacer::FaceInfo>& detections) {
        std::vector<std::pair<int, cv::Rect>> result;
        std::vector<bool> matched(detections.size(), false);
        
        // Match existing tracks
        for (auto& track : m_tracks) {
            float bestDist = 200.0f;  // Max matching distance
            int bestIdx = -1;
            
            for (size_t i = 0; i < detections.size(); i++) {
                if (matched[i]) continue;
                float dist = distance(track.pos, detections[i].boundingBox);
                if (dist < bestDist) {
                    bestDist = dist;
                    bestIdx = static_cast<int>(i);
                }
            }
            
            if (bestIdx >= 0) {
                track.pos = detections[bestIdx].boundingBox;
                track.lost = 0;
                matched[bestIdx] = true;
                result.push_back({track.id, track.pos});
            } else {
                track.lost++;
            }
        }
        
        // New tracks for unmatched detections
        for (size_t i = 0; i < detections.size(); i++) {
            if (!matched[i]) {
                Track t;
                t.id = m_nextId++;
                t.pos = detections[i].boundingBox;
                t.lost = 0;
                t.color = m_colors[(t.id - 1) % m_colors.size()];
                m_tracks.push_back(t);
                result.push_back({t.id, t.pos});
            }
        }
        
        // Remove old tracks
        m_tracks.erase(
            std::remove_if(m_tracks.begin(), m_tracks.end(),
                [](const Track& t) { return t.lost > 30; }),
            m_tracks.end());
        
        return result;
    }
    
    cv::Scalar getColor(int id) {
        for (const auto& t : m_tracks) {
            if (t.id == id) return t.color;
        }
        return m_colors[(id - 1) % m_colors.size()];
    }
    
private:
    float distance(const cv::Rect& a, const cv::Rect& b) {
        float cx1 = a.x + a.width / 2.0f, cy1 = a.y + a.height / 2.0f;
        float cx2 = b.x + b.width / 2.0f, cy2 = b.y + b.height / 2.0f;
        return std::sqrt((cx1-cx2)*(cx1-cx2) + (cy1-cy2)*(cy1-cy2));
    }
    
    std::vector<Track> m_tracks;
    int m_nextId;
    std::vector<cv::Scalar> m_colors;
};

//------------------------------------------------------------------------------
// PHASE 1: Preprocess
//------------------------------------------------------------------------------
int preprocess(const std::string& videoPath, const std::string& jsonPath, 
               const std::string& previewPath) {
    std::cout << "\n=== PHASE 1: PREPROCESS ===" << std::endl;
    
    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        std::cerr << "Cannot open " << videoPath << std::endl;
        return 1;
    }
    
    int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);
    int total = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    
    std::cout << "Video: " << width << "x" << height << " @ " << fps << " FPS, " 
              << total << " frames" << std::endl;
    
    cv::VideoWriter writer;
    bool writePreview = !previewPath.empty();
    if (writePreview) {
        writer.open(previewPath, cv::VideoWriter::fourcc('m','p','4','v'), fps, cv::Size(width, height));
    }
    
    facereplacer::Config config;
    config.useGPU = false;
    facereplacer::FaceReplacer detector(config);
    FaceTracker tracker;
    
    std::vector<FrameData> allFrames;
    std::map<int, int> faceCounts;
    
    cv::Mat frame;
    int frameNum = 0;
    
    while (true) {
        cap >> frame;
        if (frame.empty()) break;
        
        auto faces = detector.detectFaces(frame);
        auto tracked = tracker.update(faces);
        
        FrameData fd;
        fd.frameNum = frameNum;
        
        for (const auto& [id, rect] : tracked) {
            FaceData face;
            face.id = id;
            face.x = rect.x;
            face.y = rect.y;
            face.w = rect.width;
            face.h = rect.height;
            fd.faces.push_back(face);
            faceCounts[id]++;
            
            if (writePreview) {
                cv::Scalar color = tracker.getColor(id);
                cv::rectangle(frame, rect, color, 3);
                
                std::string label = "#" + std::to_string(id);
                int baseline;
                cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 1.2, 3, &baseline);
                cv::rectangle(frame, 
                    cv::Point(rect.x, rect.y - textSize.height - 10),
                    cv::Point(rect.x + textSize.width + 10, rect.y),
                    color, -1);
                cv::putText(frame, label, cv::Point(rect.x + 5, rect.y - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(255,255,255), 3);
            }
        }
        
        allFrames.push_back(fd);
        
        if (writePreview) {
            writer.write(frame);
        }
        
        frameNum++;
        if (frameNum % 30 == 0) {
            std::cout << "\rFrame " << frameNum << "/" << total << std::flush;
        }
    }
    
    cap.release();
    if (writePreview) writer.release();
    
    // Write JSON
    writeTrackingJSON(jsonPath, allFrames);
    
    std::cout << "\n\n=== TRACKING SUMMARY ===" << std::endl;
    for (const auto& [id, count] : faceCounts) {
        std::cout << "  Face #" << id << ": " << count << " frames" << std::endl;
    }
    
    std::cout << "\nJSON saved: " << jsonPath << std::endl;
    if (writePreview) std::cout << "Preview saved: " << previewPath << std::endl;
    
    std::cout << "\nNext: Review preview, then run production with face ID" << std::endl;
    
    return 0;
}

//------------------------------------------------------------------------------
// PHASE 2: Production
//------------------------------------------------------------------------------
int production(const std::string& videoPath, const std::string& selfiePath,
               const std::string& outputPath, const std::string& jsonPath, int targetId) {
    
    std::cout << "\n=== PHASE 2: PRODUCTION ===" << std::endl;
    std::cout << "Replacing face #" << targetId << std::endl;
    
    // Load tracking data
    auto trackingData = readTrackingJSON(jsonPath);
    if (trackingData.empty()) {
        std::cerr << "Failed to load tracking data" << std::endl;
        return 1;
    }
    
    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        std::cerr << "Cannot open " << videoPath << std::endl;
        return 1;
    }
    
    int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);
    int total = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    
    std::cout << "Video: " << width << "x" << height << " @ " << fps << " FPS" << std::endl;
    
    cv::VideoWriter writer(outputPath, cv::VideoWriter::fourcc('m','p','4','v'), 
                           fps, cv::Size(width, height));
    if (!writer.isOpened()) {
        std::cerr << "Cannot create output video" << std::endl;
        return 1;
    }
    
    // Load selfie
    cv::Mat selfie = cv::imread(selfiePath);
    if (selfie.empty()) {
        std::cerr << "Cannot load selfie" << std::endl;
        return 1;
    }
    std::cout << "Selfie: " << selfie.cols << "x" << selfie.rows << std::endl;
    
    // Setup replacer
    facereplacer::Config config;
    config.mode = facereplacer::ReplacementMode::HEAD_SEGMENTED;
    config.useGPU = false;
    config.colorCorrection = true;
    config.featherRadius = 20;
    
    facereplacer::FaceReplacer replacer(config);
    replacer.setSourceImage(selfie);
    
    cv::Mat frame;
    int frameNum = 0;
    int replaced = 0;
    
    auto startTime = std::chrono::high_resolution_clock::now();
    
    while (true) {
        cap >> frame;
        if (frame.empty()) break;
        
        cv::Mat result = frame.clone();
        
        // Find target face in tracking data for this frame
        if (frameNum < static_cast<int>(trackingData.size())) {
            const auto& fd = trackingData[frameNum];
            
            for (const auto& face : fd.faces) {
                if (face.id == targetId) {
                    // Found target face - replace it
                    facereplacer::FaceInfo target;
                    target.boundingBox = cv::Rect(face.x, face.y, face.w, face.h);
                    
                    replacer.setTargetFace(target);
                    result = replacer.processFrame(frame);
                    replaced++;
                    break;
                }
            }
        }
        
        writer.write(result);
        frameNum++;
        
        if (frameNum % 30 == 0) {
            std::cout << "\rFrame " << frameNum << "/" << total 
                      << " (replaced: " << replaced << ")" << std::flush;
        }
    }
    
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(endTime - startTime);
    
    cap.release();
    writer.release();
    
    std::cout << "\n\nDone! " << frameNum << " frames in " << duration.count() << "s" << std::endl;
    std::cout << "Face #" << targetId << " replaced in " << replaced << " frames" << std::endl;
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
            std::cerr << "Usage: " << argv[0] << " --preprocess <video> <tracking.json> [preview.mp4]" << std::endl;
            return 1;
        }
        std::string preview = (argc > 4) ? argv[4] : "";
        return preprocess(argv[2], argv[3], preview);
    }
    
    // Phase 2: Production
    if (argc < 5) {
        printUsage(argv[0]);
        return 1;
    }
    
    std::string videoPath = argv[1];
    std::string selfiePath = argv[2];
    std::string outputPath = argv[3];
    std::string jsonPath = argv[4];
    int faceId = (argc > 5) ? std::stoi(argv[5]) : 1;
    
    return production(videoPath, selfiePath, outputPath, jsonPath, faceId);
}
