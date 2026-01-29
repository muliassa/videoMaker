/**
 * Face Replacer - Video/Photo Processing
 * 
 * Usage:
 *   ./face_replacer <target_video> <selfie_image> <output_video> [mode] [face_index]
 * 
 * Modes:
 *   1 = RECT_TO_RECT (simple rectangle replacement)
 *   2 = HEAD_SEGMENTED (head segmentation with background preservation) [default]
 *   3 = LIVE_ANIMATED (advanced, with temporal smoothing)
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
    std::cout << "Face Replacer - Video/Photo Processing\n\n";
    std::cout << "Usage:\n";
    std::cout << "  " << programName << " <target> <selfie_image> <output> [mode] [face_index]\n\n";
    std::cout << "Arguments:\n";
    std::cout << "  target        - Video or image containing faces to replace\n";
    std::cout << "                  Supported: .mp4, .avi, .mov, .mkv, .jpg, .png\n";
    std::cout << "  selfie_image  - Your selfie (source face)\n";
    std::cout << "  output        - Output file path\n";
    std::cout << "  mode          - Replacement mode (optional, default: 2)\n";
    std::cout << "                  1 = RECT_TO_RECT (simple rectangle)\n";
    std::cout << "                  2 = HEAD_SEGMENTED (with background preservation)\n";
    std::cout << "                  3 = LIVE_ANIMATED (advanced, temporal smoothing)\n";
    std::cout << "  face_index    - Which face to replace if multiple detected (default: 0)\n\n";
    std::cout << "Examples:\n";
    std::cout << "  " << programName << " video.mp4 my_selfie.jpg output.mp4\n";
    std::cout << "  " << programName << " video.mp4 my_selfie.jpg output.mp4 3 0\n";
    std::cout << "  " << programName << " photo.jpg my_selfie.jpg output.jpg 2\n";
}

bool isVideoFile(const std::string& path) {
    std::string ext = path.substr(path.find_last_of('.') + 1);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return (ext == "mp4" || ext == "avi" || ext == "mov" || 
            ext == "mkv" || ext == "webm" || ext == "flv");
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
    
    // Create video writer
    int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
    cv::VideoWriter writer(outputPath, fourcc, fps, cv::Size(width, height));
    
    if (!writer.isOpened()) {
        std::cerr << "Error: Could not create output video: " << outputPath << std::endl;
        return 1;
    }
    
    // Load selfie
    cv::Mat selfieImage = cv::imread(selfiePath);
    if (selfieImage.empty()) {
        std::cerr << "Error: Could not load selfie image: " << selfiePath << std::endl;
        return 1;
    }
    
    // Create replacer
    facereplacer::FaceReplacer replacer(config);
    replacer.setSourceImage(selfieImage);
    
    // Read first frame to detect and select target face
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
    
    // Save marked first frame
    cv::Mat markedFrame = replacer.markFace(firstFrame, faceIndex);
    std::string markedPath = outputPath.substr(0, outputPath.find_last_of('.')) + "_marked.jpg";
    cv::imwrite(markedPath, markedFrame);
    std::cout << "Marked frame saved: " << markedPath << std::endl;
    
    // Store initial target face for static replacement
    facereplacer::FaceInfo initialTarget = faces[faceIndex];
    
    // Reset video to beginning
    cap.set(cv::CAP_PROP_POS_FRAMES, 0);
    
    // Process all frames
    std::cout << "\nProcessing video..." << std::endl;
    auto startTime = std::chrono::high_resolution_clock::now();
    
    int frameCount = 0;
    cv::Mat frame, result;
    
    while (true) {
        cap >> frame;
        if (frame.empty()) break;
        
        // For LIVE_ANIMATED mode, re-detect face each frame for tracking
        // For other modes, use initial position (assumes static person)
        if (config.mode == facereplacer::ReplacementMode::LIVE_ANIMATED) {
            auto currentFaces = replacer.detectFaces(frame);
            if (!currentFaces.empty()) {
                // Find closest face to initial position
                int bestIdx = 0;
                float minDist = std::numeric_limits<float>::max();
                for (size_t i = 0; i < currentFaces.size(); i++) {
                    float dx = currentFaces[i].boundingBox.x - initialTarget.boundingBox.x;
                    float dy = currentFaces[i].boundingBox.y - initialTarget.boundingBox.y;
                    float dist = dx*dx + dy*dy;
                    if (dist < minDist) {
                        minDist = dist;
                        bestIdx = static_cast<int>(i);
                    }
                }
                replacer.setTargetFace(currentFaces[bestIdx]);
            } else {
                // No face found, use last known position
                replacer.setTargetFace(initialTarget);
            }
        } else {
            // Static mode - use initial face position
            replacer.setTargetFace(initialTarget);
        }
        
        result = replacer.processFrame(frame);
        writer.write(result);
        
        frameCount++;
        if (frameCount % 30 == 0) {
            float progress = 100.0f * frameCount / totalFrames;
            std::cout << "\rProgress: " << frameCount << "/" << totalFrames 
                      << " (" << static_cast<int>(progress) << "%)" << std::flush;
        }
    }
    
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(endTime - startTime);
    
    cap.release();
    writer.release();
    
    std::cout << "\n\nProcessed " << frameCount << " frames in " << duration.count() << " seconds" << std::endl;
    std::cout << "Average: " << (frameCount / std::max(1L, duration.count())) << " FPS" << std::endl;
    std::cout << "Output saved: " << outputPath << std::endl;
    
    return 0;
}

int processImage(const std::string& targetPath, const std::string& selfiePath,
                 const std::string& outputPath, facereplacer::Config& config, int faceIndex) {
    
    // Load images
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
    
    std::cout << "Target image: " << targetImage.cols << "x" << targetImage.rows << std::endl;
    std::cout << "Selfie image: " << selfieImage.cols << "x" << selfieImage.rows << std::endl;
    
    // Create replacer
    facereplacer::FaceReplacer replacer(config);
    replacer.setSourceImage(selfieImage);
    
    // Detect faces
    auto faces = replacer.detectFaces(targetImage);
    if (faces.empty()) {
        std::cerr << "Error: No faces detected in target image!" << std::endl;
        return 1;
    }
    
    std::cout << "Detected " << faces.size() << " face(s)" << std::endl;
    
    if (faceIndex >= static_cast<int>(faces.size())) {
        std::cerr << "Warning: Face index " << faceIndex << " out of range. Using face 0." << std::endl;
        faceIndex = 0;
    }
    
    // Save marked image
    cv::Mat markedImage = replacer.markFace(targetImage, faceIndex);
    std::string markedPath = outputPath.substr(0, outputPath.find_last_of('.')) + "_marked.jpg";
    cv::imwrite(markedPath, markedImage);
    std::cout << "Marked image saved: " << markedPath << std::endl;
    
    // Process
    replacer.setTargetFace(faces[faceIndex]);
    
    auto startTime = std::chrono::high_resolution_clock::now();
    cv::Mat result = replacer.processFrame(targetImage);
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    
    std::cout << "Processing time: " << duration.count() << " ms" << std::endl;
    
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
    
    // Configure
    facereplacer::Config config;
    
    switch (mode) {
        case 1:
            config.mode = facereplacer::ReplacementMode::RECT_TO_RECT;
            std::cout << "Mode: RECT_TO_RECT (simple rectangle)" << std::endl;
            break;
        case 2:
            config.mode = facereplacer::ReplacementMode::HEAD_SEGMENTED;
            std::cout << "Mode: HEAD_SEGMENTED (with background preservation)" << std::endl;
            break;
        case 3:
            config.mode = facereplacer::ReplacementMode::LIVE_ANIMATED;
            std::cout << "Mode: LIVE_ANIMATED (with face tracking)" << std::endl;
            break;
        default:
            std::cerr << "Invalid mode: " << mode << ". Using HEAD_SEGMENTED." << std::endl;
            config.mode = facereplacer::ReplacementMode::HEAD_SEGMENTED;
    }
    
    // Check GPU availability
#ifdef USE_CUDA
    config.useGPU = facereplacer::cuda::isCudaAvailable();
#else
    config.useGPU = false;
#endif
    std::cout << "GPU acceleration: " << (config.useGPU ? "ENABLED" : "DISABLED (CPU mode)") << std::endl;
    
    config.colorCorrection = true;
    config.preserveLighting = true;
    config.featherRadius = 15;
    config.temporalSmoothing = 5;  // For Mode 3
    
    std::cout << std::endl;
    
    // Process based on input type
    if (isVideoFile(targetPath)) {
        std::cout << "Input type: VIDEO" << std::endl;
        return processVideo(targetPath, selfiePath, outputPath, config, faceIndex);
    } else {
        std::cout << "Input type: IMAGE" << std::endl;
        return processImage(targetPath, selfiePath, outputPath, config, faceIndex);
    }
}
