/**
 * Face Replacer - Two-Phase Pipeline
 * 
 * Phase 1 (Preprocess): Detect faces, output JSON + preview
 *   ./face_replacer --preprocess video.mp4 tracking.json preview.mp4
 * 
 * Phase 2 (Production): Replace using edited JSON (max 1 face per frame)
 *   ./face_replacer video.mp4 selfie.jpg output.mp4 edited.json
 * 
 * JSON format (preprocess output):
 * [
 *   {"frame":0,"id":1,"x":100,"y":50,"w":80,"h":100},
 *   {"frame":0,"id":2,"x":300,"y":60,"w":75,"h":95},
 *   {"frame":1,"id":1,"x":102,"y":51,"w":80,"h":100}
 * ]
 * 
 * After editing (keep 1 per frame, ID optional):
 * [
 *   {"frame":0,"x":100,"y":50,"w":80,"h":100},
 *   {"frame":1,"x":102,"y":51,"w":80,"h":100}
 * ]
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
    std::cout << "  Output JSON with all detected faces per frame.\n";
    std::cout << "  Review preview, then edit JSON to keep only target faces.\n\n";
    std::cout << "=== PHASE 2: PRODUCTION ===\n";
    std::cout << "  " << programName << " <video> <selfie.jpg> <output.mp4> <edited.json>\n\n";
    std::cout << "  Reads edited JSON (max 1 face per frame) and replaces.\n\n";
    std::cout << "Examples:\n";
    std::cout << "  " << programName << " --preprocess input.mp4 tracking.json preview.mp4\n";
    std::cout << "  # Edit tracking.json - keep only the face to replace per frame\n";
    std::cout << "  " << programName << " input.mp4 selfie.jpg output.mp4 tracking.json\n";
}

//------------------------------------------------------------------------------
// JSON helpers
//------------------------------------------------------------------------------
struct FaceEntry {
    int frame;
    int x, y, w, h;
};

// Write JSON array (preprocess output - includes ID for reference)
void writeJSON(const std::string& path, const std::vector<std::tuple<int,int,cv::Rect>>& entries) {
    std::ofstream out(path);
    out << "[\n";
    
    for (size_t i = 0; i < entries.size(); i++) {
        const auto& [frame, id, rect] = entries[i];
        out << "  {\"frame\":" << frame 
            << ",\"id\":" << id
            << ",\"x\":" << rect.x 
            << ",\"y\":" << rect.y
            << ",\"w\":" << rect.width 
            << ",\"h\":" << rect.height << "}";
        if (i < entries.size() - 1) out << ",";
        out << "\n";
    }
    
    out << "]\n";
    out.close();
}

// Read JSON - production only needs frame,x,y,w,h (ID ignored)
std::map<int, cv::Rect> readJSON(const std::string& path) {
    std::map<int, cv::Rect> result;
    
    std::ifstream in(path);
    if (!in.is_open()) {
        std::cerr << "Cannot open " << path << std::endl;
        return result;
    }
    
    std::string content((std::istreambuf_iterator<char>(in)),
                         std::istreambuf_iterator<char>());
    in.close();
    
    size_t pos = 0;
    int count = 0;
    
    while ((pos = content.find("\"frame\":", pos)) != std::string::npos) {
        try {
            int frame, x, y, w, h;
            
            size_t start = pos + 8;
            frame = std::stoi(content.substr(start));
            
            size_t xPos = content.find("\"x\":", pos) + 4;
            x = std::stoi(content.substr(xPos));
            
            size_t yPos = content.find("\"y\":", pos) + 4;
            y = std::stoi(content.substr(yPos));
            
            size_t wPos = content.find("\"w\":", pos) + 4;
            w = std::stoi(content.substr(wPos));
            
            size_t hPos = content.find("\"h\":", pos) + 4;
            h = std::stoi(content.substr(hPos));
            
            // Last entry per frame wins (if user left duplicates)
            result[frame] = cv::Rect(x, y, w, h);
            count++;
            
        } catch (...) {
            // Skip malformed
        }
        pos++;
    }
    
    std::cout << "Loaded " << count << " entries for " << result.size() << " frames" << std::endl;
    return result;
}

//------------------------------------------------------------------------------
// Face Tracker (preprocess only)
//------------------------------------------------------------------------------
class FaceTracker {
public:
    struct Track {
        int id;
        cv::Rect pos;
        int lost;
    };
    
    FaceTracker() : m_nextId(1) {}
    
    std::vector<std::pair<int, cv::Rect>> update(const std::vector<facereplacer::FaceInfo>& detections) {
        std::vector<std::pair<int, cv::Rect>> result;
        std::vector<bool> matched(detections.size(), false);
        
        for (auto& track : m_tracks) {
            float bestDist = 150.0f;
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
        
        for (size_t i = 0; i < detections.size(); i++) {
            if (!matched[i]) {
                Track t;
                t.id = m_nextId++;
                t.pos = detections[i].boundingBox;
                t.lost = 0;
                m_tracks.push_back(t);
                result.push_back({t.id, t.pos});
            }
        }
        
        m_tracks.erase(
            std::remove_if(m_tracks.begin(), m_tracks.end(),
                [](const Track& t) { return t.lost > 30; }),
            m_tracks.end());
        
        return result;
    }
    
private:
    float distance(const cv::Rect& a, const cv::Rect& b) {
        float cx1 = a.x + a.width / 2.0f, cy1 = a.y + a.height / 2.0f;
        float cx2 = b.x + b.width / 2.0f, cy2 = b.y + b.height / 2.0f;
        return std::sqrt((cx1-cx2)*(cx1-cx2) + (cy1-cy2)*(cy1-cy2));
    }
    
    std::vector<Track> m_tracks;
    int m_nextId;
};

cv::Scalar getColor(int id) {
    static std::vector<cv::Scalar> colors = {
        cv::Scalar(0, 255, 0), cv::Scalar(255, 0, 0), cv::Scalar(0, 0, 255),
        cv::Scalar(255, 255, 0), cv::Scalar(255, 0, 255), cv::Scalar(0, 255, 255),
        cv::Scalar(128, 255, 0), cv::Scalar(255, 128, 0), cv::Scalar(128, 0, 255)
    };
    return colors[(id - 1) % colors.size()];
}

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
    
    std::vector<std::tuple<int,int,cv::Rect>> allEntries;  // frame, id, rect
    std::map<int, int> faceCounts;
    cv::Mat frame;
    int frameNum = 0;
    
    while (true) {
        cap >> frame;
        if (frame.empty()) break;
        
        auto faces = detector.detectFaces(frame);
        auto tracked = tracker.update(faces);
        
        for (const auto& [id, rect] : tracked) {
            allEntries.push_back({frameNum, id, rect});
            faceCounts[id]++;
            
            if (writePreview) {
                cv::Scalar color = getColor(id);
                cv::rectangle(frame, rect, color, 3);
                
                std::string label = "#" + std::to_string(id);
                int baseline;
                cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 1.5, 3, &baseline);
                cv::rectangle(frame, 
                    cv::Point(rect.x, rect.y - textSize.height - 15),
                    cv::Point(rect.x + textSize.width + 10, rect.y),
                    color, -1);
                cv::putText(frame, label, cv::Point(rect.x + 5, rect.y - 8),
                    cv::FONT_HERSHEY_SIMPLEX, 1.5, cv::Scalar(255,255,255), 3);
            }
        }
        
        if (writePreview) {
            cv::putText(frame, "F:" + std::to_string(frameNum), cv::Point(10, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255,255,255), 2);
            writer.write(frame);
        }
        
        frameNum++;
        if (frameNum % 30 == 0) {
            std::cout << "\rFrame " << frameNum << "/" << total << std::flush;
        }
    }
    
    cap.release();
    if (writePreview) writer.release();
    
    writeJSON(jsonPath, allEntries);
    
    std::cout << "\n\n=== TRACKING SUMMARY ===" << std::endl;
    for (const auto& [id, count] : faceCounts) {
        std::cout << "  Face #" << id << ": " << count << " frames" << std::endl;
    }
    
    std::cout << "\nJSON saved: " << jsonPath << " (" << allEntries.size() << " entries)" << std::endl;
    if (writePreview) std::cout << "Preview saved: " << previewPath << std::endl;
    
    std::cout << "\n=== NEXT STEPS ===" << std::endl;
    std::cout << "1. Watch preview - note which #ID is your target at each part" << std::endl;
    std::cout << "2. Edit JSON - keep only ONE face per frame (the one to replace)" << std::endl;
    std::cout << "   e.g., frames 0-100 keep #1, frames 101-200 keep #2, etc." << std::endl;
    std::cout << "3. Run production: ./face_replacer " << videoPath << " selfie.jpg output.mp4 " << jsonPath << std::endl;
    
    return 0;
}

//------------------------------------------------------------------------------
// PHASE 2: Production (ID ignored - just frame -> bbox)
//------------------------------------------------------------------------------
int production(const std::string& videoPath, const std::string& selfiePath,
               const std::string& outputPath, const std::string& jsonPath) {
    
    std::cout << "\n=== PHASE 2: PRODUCTION ===" << std::endl;
    
    // Load curated tracking data (frame -> bbox, ignores ID)
    auto tracking = readJSON(jsonPath);
    if (tracking.empty()) {
        std::cerr << "No tracking data loaded" << std::endl;
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
    std::cout << "Frames to process: " << tracking.size() << std::endl;
    
    cv::VideoWriter writer(outputPath, cv::VideoWriter::fourcc('m','p','4','v'), 
                           fps, cv::Size(width, height));
    if (!writer.isOpened()) {
        std::cerr << "Cannot create output video" << std::endl;
        return 1;
    }
    
    cv::Mat selfie = cv::imread(selfiePath);
    if (selfie.empty()) {
        std::cerr << "Cannot load selfie" << std::endl;
        return 1;
    }
    std::cout << "Selfie: " << selfie.cols << "x" << selfie.rows << std::endl;
    
    facereplacer::Config config;
    config.mode = facereplacer::ReplacementMode::HEAD_SEGMENTED;
    config.useGPU = false;
    config.colorCorrection = true;
    config.featherRadius = 15;
    
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
        
        // Check if this frame has a target face
        auto it = tracking.find(frameNum);
        if (it != tracking.end()) {
            facereplacer::FaceInfo target;
            target.boundingBox = it->second;
            
            replacer.setTargetFace(target);
            result = replacer.processFrame(frame);
            replaced++;
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
    std::cout << "Replaced in " << replaced << " frames" << std::endl;
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
    
    if (arg1 == "--preprocess" || arg1 == "-p") {
        if (argc < 4) {
            std::cerr << "Usage: " << argv[0] << " --preprocess <video> <tracking.json> [preview.mp4]" << std::endl;
            return 1;
        }
        std::string preview = (argc > 4) ? argv[4] : "";
        return preprocess(argv[2], argv[3], preview);
    }
    
    if (argc < 5) {
        printUsage(argv[0]);
        return 1;
    }
    
    return production(argv[1], argv[2], argv[3], argv[4]);
}
