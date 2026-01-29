/**
 * Face Replacer - Video/Photo Processing with Face Tracking
 * 
 * Usage:
 *   ./face_replacer <target_video> <selfie_image> <output_video> [mode] [face_index]
 */

#include "face_replacer.hpp"
#ifdef USE_CUDA
#include "cuda/gpu_blend.cuh"
#endif
#include <iostream>
#include <string>
#include <chrono>
#include <algorithm>

void printUsage(const char* programName) {
    std::cout << "Face Replacer - Video/Photo Processing with Face Tracking\n\n";
    std::cout << "Usage:\n";
    std::cout << "  " << programName << " <target> <selfie_image> <output> [mode] [face_index]\n\n";
    std::cout << "Arguments:\n";
    std::cout << "  target        - Video or image containing faces to replace\n";
    std::cout << "  selfie_image  - Your selfie (source face)\n";
    std::cout << "  output        - Output file path\n";
    std::cout << "  mode          - Replacement mode (optional, default: 2)\n";
    std::cout << "                  1 = RECT_TO_RECT (simple rectangle)\n";
    std::cout << "                  2 = HEAD_SEGMENTED (with background preservation)\n";
    std::cout << "                  3 = LIVE_ANIMATED (with temporal smoothing)\n";
    std::cout << "  face_index    - Which face to replace if multiple detected (default: 0)\n\n";
    std::cout << "Examples:\n";
    std::cout << "  " << programName << " video.mp4 my_selfie.jpg output.mp4\n";
    std::cout << "  " << programName << " video.mp4 my_selfie.jpg output.mp4 2 0\n";
}

bool isVideoFile(const std::string& path) {
    std::string ext = path.substr(path.find_last_of('.') + 1);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return (ext == "mp4" || ext == "avi" || ext == "mov" || 
            ext == "mkv" || ext == "webm" || ext == "flv");
}

// Find the face closest to a reference position
int findClosestFace(const std::vector<facereplacer::FaceInfo>& faces, 
                    const facereplacer::FaceInfo& reference) {
    if (faces.empty()) return -1;
    
    int bestIdx = 0;
    float minDist = std::numeric_limits<float>::max();
    
    // Reference center
    float refCx = reference.boundingBox.x + reference.boundingBox.width / 2.0f;
    float refCy = reference.boundingBox.y + reference.boundingBox.height / 2.0f;
    
    for (size_t i = 0; i < faces.size(); i++) {
        float cx = faces[i].boundingBox.x + faces[i].boundingBox.width / 2.0f;
        float cy = faces[i].boundingBox.y + faces[i].boundingBox.height / 2.0f;
        float dist = (cx - refCx) * (cx - refCx) + (cy - refCy) * (cy - refCy);
        
        if (dist < minDist) {
            minDist = dist;
            bestIdx = static_cast<int>(i);
        }
    }
    
    return bestIdx;
}

int processVideo(const std::string& targetPath, const std::string& selfiePath,
                 const std::string& outputPath, facereplacer::Config& config, int faceIndex) {
    
    // Open video
    cv::VideoCapture cap(targetPath);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video: " << targetPath << std::endl;
        return 1;
    }
    
    // Get video properties
    int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);
    int totalFrames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    
    std::cout << "Video: " << width << "x" << height << " @ " << fps << " FPS" << std::endl;
    std::cout << "Total frames: " << totalFrames << std::endl;
    
    // Create video writer - use H264 codec
    int fourcc = cv::VideoWriter::fourcc('a', 'v', 'c', '1');  // H.264
    cv::VideoWriter writer(outputPath, fourcc, fps, cv::Size(width, height));
    
    if (!writer.isOpened()) {
        // Fallback to mp4v
        fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
        writer.open(outputPath, fourcc, fps, cv::Size(width, height));
        if (!writer.isOpened()) {
            std::cerr << "Error: Could not create output video: " << outputPath << std::endl;
            return 1;
        }
    }
    
    // Load selfie
    cv::Mat selfieImage = cv::imread(selfiePath);
    if (selfieImage.empty()) {
        std::cerr << "Error: Could not load selfie image: " << selfiePath << std::endl;
        return 1;
    }
    std::cout << "Selfie: " << selfieImage.cols << "x" << selfieImage.rows << std::endl;
    
    // Create replacer
    facereplacer::FaceReplacer replacer(config);
    replacer.setSourceImage(selfieImage);
    
    // Read first frame to detect and select initial target face
    cv::Mat firstFrame;
    cap >> firstFrame;
    if (firstFrame.empty()) {
        std::cerr << "Error: Could not read first frame" << std::endl;
        return 1;
    }
    
    auto faces = replacer.detectFaces(firstFrame);
    if (faces.empty()) {
        std::cerr << "Error: No faces detected in first frame!" << std::endl;
        return 1;
    }
    
    std::cout << "Detected " << faces.size() << " face(s) in first frame" << std::endl;
    
    if (faceIndex >= static_cast<int>(faces.size())) {
        std::cerr << "Warning: Face index " << faceIndex << " out of range. Using face 0." << std::endl;
        faceIndex = 0;
    }
    
    // Initial target face (we'll track this throughout the video)
    facereplacer::FaceInfo trackedFace = faces[faceIndex];
    std::cout << "Initial face position: " << trackedFace.boundingBox << std::endl;
    
    // Save marked first frame
    cv::Mat markedFrame = replacer.markFace(firstFrame, faceIndex);
    std::string markedPath = outputPath.substr(0, outputPath.find_last_of('.')) + "_marked.jpg";
    cv::imwrite(markedPath, markedFrame);
    std::cout << "Marked frame saved: " << markedPath << std::endl;
    
    // Reset video to beginning
    cap.set(cv::CAP_PROP_POS_FRAMES, 0);
    
    // Process all frames with TRACKING
    std::cout << "\nProcessing video with face tracking..." << std::endl;
    auto startTime = std::chrono::high_resolution_clock::now();
    
    int frameCount = 0;
    int trackingLostCount = 0;
    cv::Mat frame, result;
    
    while (true) {
        cap >> frame;
        if (frame.empty()) break;
        
        // ALWAYS detect faces for tracking
        auto currentFaces = replacer.detectFaces(frame);
        
        if (!currentFaces.empty()) {
            // Find the face closest to our tracked face
            int closestIdx = findClosestFace(currentFaces, trackedFace);
            if (closestIdx >= 0) {
                trackedFace = currentFaces[closestIdx];
            }
        } else {
            // No face found this frame - use last known position
            trackingLostCount++;
        }
        
        // Set target and process
        replacer.setTargetFace(trackedFace);
        result = replacer.processFrame(frame);
        
        writer.write(result);
        
        frameCount++;
        if (frameCount % 30 == 0 || frameCount == totalFrames) {
            float progress = 100.0f * frameCount / totalFrames;
            std::cout << "\rProgress: " << frameCount << "/" << totalFrames 
                      << " (" << static_cast<int>(progress) << "%) "
                      << "Face: " << trackedFace.boundingBox.x << "," << trackedFace.boundingBox.y
                      << std::flush;
        }
    }
    
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(endTime - startTime);
    
    cap.release();
    writer.release();
    
    std::cout << "\n\nProcessed " << frameCount << " frames in " << duration.count() << " seconds" << std::endl;
    if (duration.count() > 0) {
        std::cout << "Average: " << (frameCount / duration.count()) << " FPS" << std::endl;
    }
    if (trackingLostCount > 0) {
        std::cout << "Tracking lost for " << trackingLostCount << " frames" << std::endl;
    }
    std::cout << "Output saved: " << outputPath << std::endl;
    
    return 0;
}

int processImage(const std::string& targetPath, const std::string& selfiePath,
                 const std::string& outputPath, facereplacer::Config& config, int faceIndex) {
    
    cv::Mat targetImage = cv::imread(targetPath);
    if (targetImage.empty()) {
        std::cerr << "Error: Could not load target image: " << targetPath << std::endl;
        return 1;
    }
    
    cv::Mat selfieImage = cv::imread(selfiePath);
    if (selfieImage.empty()) {
        std::cerr << "Error: Could not load selfie image: " << selfiePath << std::endl;
        return 1;
    }
    
    std::cout << "Target: " << targetImage.cols << "x" << targetImage.rows << std::endl;
    std::cout << "Selfie: " << selfieImage.cols << "x" << selfieImage.rows << std::endl;
    
    facereplacer::FaceReplacer replacer(config);
    replacer.setSourceImage(selfieImage);
    
    auto faces = replacer.detectFaces(targetImage);
    if (faces.empty()) {
        std::cerr << "Error: No faces detected!" << std::endl;
        return 1;
    }
    
    std::cout << "Detected " << faces.size() << " face(s)" << std::endl;
    
    if (faceIndex >= static_cast<int>(faces.size())) {
        faceIndex = 0;
    }
    
    // Save marked
    cv::Mat markedImage = replacer.markFace(targetImage, faceIndex);
    std::string markedPath = outputPath.substr(0, outputPath.find_last_of('.')) + "_marked.jpg";
    cv::imwrite(markedPath, markedImage);
    
    // Process
    replacer.setTargetFace(faces[faceIndex]);
    cv::Mat result = replacer.processFrame(targetImage);
    
    cv::imwrite(outputPath, result);
    std::cout << "Result saved: " << outputPath << std::endl;
    
    return 0;
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        printUsage(argv[0]);
        return 1;
    }
    
    std::string targetPath = argv[1];
    std::string selfiePath = argv[2];
    std::string outputPath = argv[3];
    int mode = (argc > 4) ? std::stoi(argv[4]) : 2;
    int faceIndex = (argc > 5) ? std::stoi(argv[5]) : 0;
    
    facereplacer::Config config;
    
    switch (mode) {
        case 1:
            config.mode = facereplacer::ReplacementMode::RECT_TO_RECT;
            std::cout << "Mode: RECT_TO_RECT" << std::endl;
            break;
        case 2:
            config.mode = facereplacer::ReplacementMode::HEAD_SEGMENTED;
            std::cout << "Mode: HEAD_SEGMENTED" << std::endl;
            break;
        case 3:
            config.mode = facereplacer::ReplacementMode::LIVE_ANIMATED;
            std::cout << "Mode: LIVE_ANIMATED" << std::endl;
            break;
        default:
            config.mode = facereplacer::ReplacementMode::HEAD_SEGMENTED;
    }
    
#ifdef USE_CUDA
    config.useGPU = facereplacer::cuda::isCudaAvailable();
#else
    config.useGPU = false;
#endif
    std::cout << "GPU: " << (config.useGPU ? "YES" : "NO (CPU)") << std::endl;
    
    config.colorCorrection = true;
    config.preserveLighting = true;
    config.featherRadius = 15;
    config.temporalSmoothing = 5;
    
    if (isVideoFile(targetPath)) {
        std::cout << "Processing VIDEO..." << std::endl;
        return processVideo(targetPath, selfiePath, outputPath, config, faceIndex);
    } else {
        std::cout << "Processing IMAGE..." << std::endl;
        return processImage(targetPath, selfiePath, outputPath, config, faceIndex);
    }
}
