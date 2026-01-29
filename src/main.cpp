/**
 * Face Replacer - Three-Phase Pipeline
 * 
 * Phase 1 (Preprocess): Detect faces, output JSON + preview
 *   ./face_replacer --preprocess video.mp4 tracking.json preview.mp4
 * 
 * Phase 2 (Segment): Generate SAM masks for tracked faces
 *   ./face_replacer --segment video.mp4 tracking.json masks/
 * 
 * Phase 3 (Production): Replace using masks (fast, reusable for any selfie)
 *   ./face_replacer video.mp4 selfie.jpg output.mp4 tracking.json masks/
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
#include <sys/stat.h>
#include <unistd.h>

void printUsage(const char* programName) {
    std::cout << "Face Replacer - Three-Phase Pipeline\n\n";
    std::cout << "=== PHASE 1: PREPROCESS ===\n";
    std::cout << "  " << programName << " --preprocess <video> <tracking.json> [preview.mp4]\n";
    std::cout << "  Detect faces and create tracking data.\n\n";
    std::cout << "=== PHASE 2: SEGMENT ===\n";
    std::cout << "  " << programName << " --segment <video> <tracking.json> <masks_dir/>\n";
    std::cout << "  Run SAM segmentation on tracked faces. Do this ONCE.\n\n";
    std::cout << "=== PHASE 3: PRODUCTION ===\n";
    std::cout << "  " << programName << " <video> <selfie> <output> <tracking.json> <masks_dir/>\n";
    std::cout << "  Fast replacement using precomputed masks. Run for each selfie.\n\n";
    std::cout << "Example workflow:\n";
    std::cout << "  " << programName << " --preprocess input.mp4 tracking.json preview.mp4\n";
    std::cout << "  # Edit tracking.json to select target face\n";
    std::cout << "  " << programName << " --segment input.mp4 tracking.json masks/\n";
    std::cout << "  # Now replace with any selfie (fast!):\n";
    std::cout << "  " << programName << " input.mp4 alice.jpg output_alice.mp4 tracking.json masks/\n";
    std::cout << "  " << programName << " input.mp4 bob.jpg output_bob.mp4 tracking.json masks/\n";
}

//------------------------------------------------------------------------------
// JSON helpers
//------------------------------------------------------------------------------
struct FaceEntry {
    int frame;
    int x, y, w, h;
};

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
            result[frame] = cv::Rect(x, y, w, h);
        } catch (...) {}
        pos++;
    }
    return result;
}

//------------------------------------------------------------------------------
// Face Tracker
//------------------------------------------------------------------------------
class FaceTracker {
public:
    struct Track { int id; cv::Rect pos; int lost; };
    
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
                if (dist < bestDist) { bestDist = dist; bestIdx = static_cast<int>(i); }
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
                Track t; t.id = m_nextId++; t.pos = detections[i].boundingBox; t.lost = 0;
                m_tracks.push_back(t);
                result.push_back({t.id, t.pos});
            }
        }
        
        m_tracks.erase(std::remove_if(m_tracks.begin(), m_tracks.end(),
            [](const Track& t) { return t.lost > 30; }), m_tracks.end());
        
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
        {0,255,0}, {255,0,0}, {0,0,255}, {255,255,0}, {255,0,255},
        {0,255,255}, {128,255,0}, {255,128,0}, {128,0,255}
    };
    return colors[(id - 1) % colors.size()];
}

//------------------------------------------------------------------------------
// Python/SAM helpers
//------------------------------------------------------------------------------
std::string findPython() {
    std::vector<std::string> paths = {"./venv/bin/python3", "venv/bin/python3", "python3"};
    for (const auto& p : paths) {
        if (access(p.c_str(), X_OK) == 0) return p;
    }
    return "python3";
}

bool runSAM(const std::string& python, const std::string& input, const std::string& output, const cv::Rect& bbox) {
    std::string cmd = python + " scripts/sam_segment.py \"" + input + "\" \"" + output + "\" " +
        std::to_string(bbox.x) + " " + std::to_string(bbox.y) + " " +
        std::to_string(bbox.width) + " " + std::to_string(bbox.height) + " 2>/dev/null";
    return (std::system(cmd.c_str()) == 0);
}

void createDir(const std::string& path) {
    mkdir(path.c_str(), 0755);
}

std::string maskPath(const std::string& dir, int frame) {
    return dir + "/mask_" + std::to_string(frame) + ".png";
}

std::string cropInfoPath(const std::string& dir, int frame) {
    return dir + "/crop_" + std::to_string(frame) + ".txt";
}

void saveCropInfo(const std::string& path, const cv::Rect& crop) {
    std::ofstream out(path);
    out << crop.x << " " << crop.y << " " << crop.width << " " << crop.height << std::endl;
    out.close();
}

cv::Rect loadCropInfo(const std::string& path) {
    cv::Rect r(0, 0, 0, 0);
    std::ifstream in(path);
    if (in.is_open()) {
        in >> r.x >> r.y >> r.width >> r.height;
    }
    return r;
}

//------------------------------------------------------------------------------
// PHASE 1: Preprocess
//------------------------------------------------------------------------------
int preprocess(const std::string& videoPath, const std::string& jsonPath, const std::string& previewPath) {
    std::cout << "\n=== PHASE 1: PREPROCESS ===" << std::endl;
    
    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened()) { std::cerr << "Cannot open " << videoPath << std::endl; return 1; }
    
    int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);
    int total = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    
    std::cout << "Video: " << width << "x" << height << " @ " << fps << " FPS, " << total << " frames" << std::endl;
    
    cv::VideoWriter writer;
    if (!previewPath.empty()) {
        writer.open(previewPath, cv::VideoWriter::fourcc('m','p','4','v'), fps, cv::Size(width, height));
    }
    
    facereplacer::Config config; config.useGPU = false;
    facereplacer::FaceReplacer detector(config);
    FaceTracker tracker;
    
    std::vector<std::tuple<int,int,cv::Rect>> allEntries;
    std::map<int, int> faceCounts;
    cv::Mat frame;
    int frameNum = 0;
    
    while (cap.read(frame)) {
        auto faces = detector.detectFaces(frame);
        auto tracked = tracker.update(faces);
        
        for (const auto& [id, rect] : tracked) {
            allEntries.push_back({frameNum, id, rect});
            faceCounts[id]++;
            
            if (writer.isOpened()) {
                cv::rectangle(frame, rect, getColor(id), 3);
                std::string label = "#" + std::to_string(id);
                int baseline;
                auto sz = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 1.5, 3, &baseline);
                cv::rectangle(frame, cv::Point(rect.x, rect.y - sz.height - 15),
                    cv::Point(rect.x + sz.width + 10, rect.y), getColor(id), -1);
                cv::putText(frame, label, cv::Point(rect.x + 5, rect.y - 8),
                    cv::FONT_HERSHEY_SIMPLEX, 1.5, cv::Scalar(255,255,255), 3);
            }
        }
        
        if (writer.isOpened()) writer.write(frame);
        frameNum++;
        if (frameNum % 30 == 0) std::cout << "\rFrame " << frameNum << "/" << total << std::flush;
    }
    
    writeJSON(jsonPath, allEntries);
    
    std::cout << "\n\n=== TRACKING SUMMARY ===" << std::endl;
    for (const auto& [id, count] : faceCounts) {
        std::cout << "  Face #" << id << ": " << count << " frames" << std::endl;
    }
    std::cout << "\nSaved: " << jsonPath << std::endl;
    if (!previewPath.empty()) std::cout << "Preview: " << previewPath << std::endl;
    
    std::cout << "\nNext: Edit " << jsonPath << " to keep only target face, then run --segment" << std::endl;
    
    return 0;
}

//------------------------------------------------------------------------------
// PHASE 2: Segment
//------------------------------------------------------------------------------
int segment(const std::string& videoPath, const std::string& jsonPath, const std::string& masksDir) {
    std::cout << "\n=== PHASE 2: SEGMENT ===" << std::endl;
    
    auto tracking = readJSON(jsonPath);
    if (tracking.empty()) { std::cerr << "No tracking data" << std::endl; return 1; }
    
    std::cout << "Frames to segment: " << tracking.size() << std::endl;
    
    createDir(masksDir);
    
    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened()) { std::cerr << "Cannot open " << videoPath << std::endl; return 1; }
    
    std::string python = findPython();
    std::cout << "Using Python: " << python << std::endl;
    std::cout << "Masks output: " << masksDir << "/" << std::endl;
    
    cv::Mat frame;
    int frameNum = 0;
    int processed = 0;
    int total = static_cast<int>(tracking.size());
    
    auto startTime = std::chrono::high_resolution_clock::now();
    
    while (cap.read(frame)) {
        auto it = tracking.find(frameNum);
        if (it != tracking.end()) {
            cv::Rect faceRect = it->second;
            
            // Expand for head region
            int expandTop = static_cast<int>(faceRect.height * 0.6);
            int expandSide = static_cast<int>(faceRect.width * 0.4);
            int expandBottom = static_cast<int>(faceRect.height * 0.25);
            
            cv::Rect cropRect;
            cropRect.x = std::max(0, faceRect.x - expandSide);
            cropRect.y = std::max(0, faceRect.y - expandTop);
            cropRect.width = std::min(faceRect.width + expandSide * 2, frame.cols - cropRect.x);
            cropRect.height = std::min(faceRect.height + expandTop + expandBottom, frame.rows - cropRect.y);
            
            // Face in crop for SAM prompt
            cv::Rect faceInCrop(
                faceRect.x - cropRect.x,
                faceRect.y - cropRect.y,
                faceRect.width,
                faceRect.height
            );
            
            // Expand prompt slightly
            int expand = static_cast<int>(faceInCrop.width * 0.1);
            cv::Rect samPrompt = faceInCrop;
            samPrompt.x = std::max(0, samPrompt.x - expand);
            samPrompt.y = std::max(0, samPrompt.y - expand);
            samPrompt.width = std::min(samPrompt.width + expand*2, cropRect.width - samPrompt.x);
            samPrompt.height = std::min(samPrompt.height + expand*2, cropRect.height - samPrompt.y);
            
            // Save crop
            cv::Mat crop = frame(cropRect);
            std::string tempCrop = "/tmp/seg_crop.png";
            cv::imwrite(tempCrop, crop);
            
            // Run SAM
            std::string outMask = maskPath(masksDir, frameNum);
            if (runSAM(python, tempCrop, outMask, samPrompt)) {
                // Save crop coordinates for production phase
                saveCropInfo(cropInfoPath(masksDir, frameNum), cropRect);
                processed++;
            } else {
                std::cerr << "SAM failed for frame " << frameNum << std::endl;
            }
            
            std::remove(tempCrop.c_str());
            
            std::cout << "\rSegmented " << processed << "/" << total << std::flush;
        }
        frameNum++;
    }
    
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(endTime - startTime);
    
    std::cout << "\n\nDone! " << processed << " masks in " << duration.count() << "s" << std::endl;
    std::cout << "Masks saved to: " << masksDir << "/" << std::endl;
    std::cout << "\nNow run production with any selfie:\n";
    std::cout << "  ./face_replacer " << videoPath << " selfie.jpg output.mp4 " << jsonPath << " " << masksDir << "/" << std::endl;
    
    return 0;
}

//------------------------------------------------------------------------------
// PHASE 3: Production (fast - uses precomputed masks)
//------------------------------------------------------------------------------
int production(const std::string& videoPath, const std::string& selfiePath,
               const std::string& outputPath, const std::string& jsonPath,
               const std::string& masksDir) {
    
    std::cout << "\n=== PHASE 3: PRODUCTION ===" << std::endl;
    
    auto tracking = readJSON(jsonPath);
    if (tracking.empty()) { std::cerr << "No tracking data" << std::endl; return 1; }
    
    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened()) { std::cerr << "Cannot open " << videoPath << std::endl; return 1; }
    
    int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);
    int total = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    
    cv::VideoWriter writer(outputPath, cv::VideoWriter::fourcc('m','p','4','v'), fps, cv::Size(width, height));
    if (!writer.isOpened()) { std::cerr << "Cannot create output" << std::endl; return 1; }
    
    // Setup selfie
    cv::Mat selfie = cv::imread(selfiePath);
    if (selfie.empty()) { std::cerr << "Cannot load selfie" << std::endl; return 1; }
    
    facereplacer::Config config;
    config.useGPU = false;
    config.colorCorrection = true;
    facereplacer::FaceReplacer replacer(config);
    
    // Detect face in selfie and setup
    auto faces = replacer.detectFaces(selfie);
    if (faces.empty()) { std::cerr << "No face in selfie" << std::endl; return 1; }
    
    cv::Rect selfieFace = faces[0].boundingBox;
    std::cout << "Selfie face: " << selfieFace << std::endl;
    
    // Extract selfie head region
    int expandTop = static_cast<int>(selfieFace.height * 0.6);
    int expandSide = static_cast<int>(selfieFace.width * 0.4);
    int expandBottom = static_cast<int>(selfieFace.height * 0.25);
    
    cv::Rect selfieHeadRect;
    selfieHeadRect.x = std::max(0, selfieFace.x - expandSide);
    selfieHeadRect.y = std::max(0, selfieFace.y - expandTop);
    selfieHeadRect.width = std::min(selfieFace.width + expandSide * 2, selfie.cols - selfieHeadRect.x);
    selfieHeadRect.height = std::min(selfieFace.height + expandTop + expandBottom, selfie.rows - selfieHeadRect.y);
    
    cv::Mat selfieHead = selfie(selfieHeadRect).clone();
    
    // SAM on selfie
    std::string python = findPython();
    std::string tempIn = "/tmp/selfie_head.png";
    std::string tempOut = "/tmp/selfie_mask.png";
    cv::imwrite(tempIn, selfieHead);
    
    // SAM prompt: use nearly full crop (SAM will find the person)
    cv::Rect selfiePrompt(
        5, 5,
        selfieHead.cols - 10,
        selfieHead.rows - 10
    );
    
    std::cout << "Segmenting selfie..." << std::endl;
    cv::Mat selfieMask;
    if (runSAM(python, tempIn, tempOut, selfiePrompt)) {
        selfieMask = cv::imread(tempOut, cv::IMREAD_GRAYSCALE);
    }
    std::remove(tempIn.c_str());
    std::remove(tempOut.c_str());
    
    if (selfieMask.empty()) {
        std::cout << "SAM failed for selfie, using ellipse" << std::endl;
        selfieMask = cv::Mat::zeros(selfieHead.size(), CV_8UC1);
        cv::ellipse(selfieMask, cv::Point(selfieHead.cols/2, selfieHead.rows*0.45),
            cv::Size(selfieHead.cols*0.45, selfieHead.rows*0.48), 0, 0, 360, cv::Scalar(255), -1);
    }
    // Fill holes in mask (eyes, mouth)
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(15, 15));
    cv::morphologyEx(selfieMask, selfieMask, cv::MORPH_CLOSE, kernel, cv::Point(-1,-1), 3);
    
    cv::GaussianBlur(selfieMask, selfieMask, cv::Size(15, 15), 7);
    
    // Debug output
    cv::imwrite("debug_production_selfie.jpg", selfieHead);
    cv::imwrite("debug_production_mask.jpg", selfieMask);
    std::cout << "Debug: selfieHead " << selfieHead.cols << "x" << selfieHead.rows << std::endl;
    std::cout << "Debug: selfieFace " << selfieFace << std::endl;
    
    // Get selfie mask bounding rect (actual head size)
    // Need binary mask for findNonZero
    cv::Mat selfieMaskBinary;
    cv::threshold(selfieMask, selfieMaskBinary, 127, 255, cv::THRESH_BINARY);
    
    std::vector<cv::Point> selfiePoints;
    cv::findNonZero(selfieMaskBinary, selfiePoints);
    
    if (selfiePoints.empty()) {
        std::cerr << "ERROR: Selfie mask is empty!" << std::endl;
        return 1;
    }
    
    cv::Rect selfieMaskRect = cv::boundingRect(selfiePoints);
    cv::Point selfieMaskCenter(selfieMaskRect.x + selfieMaskRect.width/2,
                               selfieMaskRect.y + selfieMaskRect.height/2);
    
    std::cout << "Selfie mask rect: " << selfieMaskRect << std::endl;
    std::cout << "Selfie mask center: " << selfieMaskCenter << std::endl;
    
    std::cout << "Processing " << tracking.size() << " frames..." << std::endl;
    
    cv::Mat frame;
    int frameNum = 0;
    int replaced = 0;
    
    auto startTime = std::chrono::high_resolution_clock::now();
    
    while (cap.read(frame)) {
        cv::Mat result = frame.clone();
        
        auto it = tracking.find(frameNum);
        if (it != tracking.end()) {
            cv::Rect targetFace = it->second;
            
            // Load precomputed mask and crop coordinates
            cv::Mat targetMask = cv::imread(maskPath(masksDir, frameNum), cv::IMREAD_GRAYSCALE);
            cv::Rect cropRect = loadCropInfo(cropInfoPath(masksDir, frameNum));
            
            // Fallback: recalculate if crop info missing
            if (cropRect.width == 0) {
                int eTop = static_cast<int>(targetFace.height * 0.6);
                int eSide = static_cast<int>(targetFace.width * 0.4);
                int eBottom = static_cast<int>(targetFace.height * 0.25);
                
                cropRect.x = std::max(0, targetFace.x - eSide);
                cropRect.y = std::max(0, targetFace.y - eTop);
                cropRect.width = std::min(targetFace.width + eSide * 2, frame.cols - cropRect.x);
                cropRect.height = std::min(targetFace.height + eTop + eBottom, frame.rows - cropRect.y);
            }
            
            if (!targetMask.empty() && cropRect.width > 0 && cropRect.height > 0) {
                // Get target mask bounding rect (use binary threshold)
                cv::Mat targetMaskBinary;
                cv::threshold(targetMask, targetMaskBinary, 127, 255, cv::THRESH_BINARY);
                
                std::vector<cv::Point> targetPoints;
                cv::findNonZero(targetMaskBinary, targetPoints);
                
                // Debug first frame
                if (replaced == 0) {
                    std::cout << "Frame " << frameNum << " debug:" << std::endl;
                    std::cout << "  targetMask size: " << targetMask.cols << "x" << targetMask.rows << std::endl;
                    std::cout << "  targetPoints: " << targetPoints.size() << std::endl;
                    std::cout << "  cropRect: " << cropRect << std::endl;
                }
                
                if (!targetPoints.empty()) {
                    cv::Rect targetMaskRect = cv::boundingRect(targetPoints);
                    cv::Point targetMaskCenter(
                        cropRect.x + targetMaskRect.x + targetMaskRect.width/2,
                        cropRect.y + targetMaskRect.y + targetMaskRect.height/2
                    );
                    
                    // Debug first frame
                    if (replaced == 0) {
                        std::cout << "  targetMaskRect: " << targetMaskRect << std::endl;
                        std::cout << "  targetMaskCenter: " << targetMaskCenter << std::endl;
                    }
                    
                    // Scale selfie to match target mask size (use height as reference)
                    float scale = static_cast<float>(targetMaskRect.height) / selfieMaskRect.height;
                    
                    cv::Size newSize(static_cast<int>(selfieHead.cols * scale),
                                    static_cast<int>(selfieHead.rows * scale));
                    
                    if (newSize.width > 0 && newSize.height > 0) {
                        cv::Mat resizedHead, resizedMask;
                        cv::resize(selfieHead, resizedHead, newSize, 0, 0, cv::INTER_LINEAR);
                        cv::resize(selfieMask, resizedMask, newSize, 0, 0, cv::INTER_LINEAR);
                        
                        // Feather the mask border (gentle fade)
                        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(9, 9));
                        cv::Mat erodedMask;
                        cv::erode(resizedMask, erodedMask, kernel);
                        cv::GaussianBlur(erodedMask, resizedMask, cv::Size(15, 15), 7);
                        
                        // Debug: check mask isn't empty after feathering
                        if (replaced == 0) {
                            int nonZero = cv::countNonZero(resizedMask);
                            std::cout << "  resizedMask nonzero after feather: " << nonZero << std::endl;
                        }
                        
                        // Scaled selfie mask center
                        cv::Point scaledMaskCenter(
                            static_cast<int>(selfieMaskCenter.x * scale),
                            static_cast<int>(selfieMaskCenter.y * scale)
                        );
                        
                        // Align mask centers
                        int placeX = targetMaskCenter.x - scaledMaskCenter.x;
                        int placeY = targetMaskCenter.y - scaledMaskCenter.y;
                        
                        // Debug first frame
                        if (replaced == 0) {
                            std::cout << "  scale: " << scale << std::endl;
                            std::cout << "  newSize: " << newSize << std::endl;
                            std::cout << "  scaledMaskCenter: " << scaledMaskCenter << std::endl;
                            std::cout << "  placeX,Y: " << placeX << ", " << placeY << std::endl;
                        }
                        
                        // Feather target mask
                        cv::Mat targetMaskFeathered;
                        cv::GaussianBlur(targetMask, targetMaskFeathered, cv::Size(21, 21), 10);
                        
                        // DEBUG MODE: Black out target area
                        cv::Mat cropROI = result(cropRect);
                        for (int y = 0; y < cropROI.rows && y < targetMaskFeathered.rows; y++) {
                            for (int x = 0; x < cropROI.cols && x < targetMaskFeathered.cols; x++) {
                                float alpha = targetMaskFeathered.at<uchar>(y, x) / 255.0f;
                                if (alpha > 0.1f) {
                                    cropROI.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 0);  // Pure black
                                }
                            }
                        }
                        
                        // Place selfie (aligned by mask centers)
                        int srcX = 0, srcY = 0, dstX = placeX, dstY = placeY;
                        int copyW = resizedHead.cols, copyH = resizedHead.rows;
                        
                        if (dstX < 0) { srcX = -dstX; copyW += dstX; dstX = 0; }
                        if (dstY < 0) { srcY = -dstY; copyH += dstY; dstY = 0; }
                        if (dstX + copyW > result.cols) copyW = result.cols - dstX;
                        if (dstY + copyH > result.rows) copyH = result.rows - dstY;
                        
                        // Debug first frame
                        if (replaced == 0) {
                            std::cout << "  srcX,srcY: " << srcX << ", " << srcY << std::endl;
                            std::cout << "  dstX,dstY: " << dstX << ", " << dstY << std::endl;
                            std::cout << "  copyW,copyH: " << copyW << ", " << copyH << std::endl;
                        }
                        
                        if (copyW > 0 && copyH > 0) {
                            // Also check srcX+copyW doesn't exceed resizedHead bounds
                            if (srcX + copyW > resizedHead.cols) copyW = resizedHead.cols - srcX;
                            if (srcY + copyH > resizedHead.rows) copyH = resizedHead.rows - srcY;
                            
                            if (copyW > 0 && copyH > 0) {
                                cv::Mat srcRegion = resizedHead(cv::Rect(srcX, srcY, copyW, copyH));
                                cv::Mat maskRegion = resizedMask(cv::Rect(srcX, srcY, copyW, copyH));
                                cv::Mat dstRegion = result(cv::Rect(dstX, dstY, copyW, copyH));
                                
                                // DEBUG: Erode selfie mask more to show black border
                                cv::Mat borderKernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(15, 15));
                                cv::Mat innerMask;
                                cv::erode(maskRegion, innerMask, borderKernel);
                                
                                for (int y = 0; y < copyH; y++) {
                                    for (int x = 0; x < copyW; x++) {
                                        float alpha = innerMask.at<uchar>(y, x) / 255.0f;
                                        if (alpha > 0.1f) {
                                            // Place selfie (inside black border)
                                            dstRegion.at<cv::Vec3b>(y, x) = srcRegion.at<cv::Vec3b>(y, x);
                                        }
                                    }
                                }
                                
                                if (replaced == 0) {
                                    std::cout << "  Selfie placed successfully!" << std::endl;
                                }
                            }
                        }
                    }
                }
            }
            replaced++;
        }
        
        writer.write(result);
        frameNum++;
        if (frameNum % 30 == 0) {
            std::cout << "\rFrame " << frameNum << "/" << total << " (replaced: " << replaced << ")" << std::flush;
        }
    }
    
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(endTime - startTime);
    
    std::cout << "\n\nDone! " << frameNum << " frames in " << duration.count() << "s" << std::endl;
    std::cout << "Replaced: " << replaced << " frames" << std::endl;
    std::cout << "Output: " << outputPath << std::endl;
    
    return 0;
}

//------------------------------------------------------------------------------
// Main
//------------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    if (argc < 3) { printUsage(argv[0]); return 1; }
    
    std::string arg1 = argv[1];
    
    if (arg1 == "--preprocess" || arg1 == "-p") {
        if (argc < 4) { std::cerr << "Usage: --preprocess <video> <tracking.json> [preview.mp4]" << std::endl; return 1; }
        std::string preview = (argc > 4) ? argv[4] : "";
        return preprocess(argv[2], argv[3], preview);
    }
    
    if (arg1 == "--segment" || arg1 == "-s") {
        if (argc < 5) { std::cerr << "Usage: --segment <video> <tracking.json> <masks_dir/>" << std::endl; return 1; }
        return segment(argv[2], argv[3], argv[4]);
    }
    
    // Production: video selfie output tracking masks_dir
    if (argc < 6) { printUsage(argv[0]); return 1; }
    
    return production(argv[1], argv[2], argv[3], argv[4], argv[5]);
}
