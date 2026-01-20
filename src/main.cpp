/**
 * Face Replacer - Static Photo Processing
 * 
 * Usage:
 *   ./face_replacer <target_image> <selfie_image> <output_image> [mode] [face_index]
 * 
 * Modes:
 *   1 = RECT_TO_RECT (simple rectangle replacement)
 *   2 = HEAD_SEGMENTED (head segmentation with background preservation) [default]
 *   3 = LIVE_ANIMATED (advanced, for video)
 */

#include "face_replacer.hpp"
#include "cuda/gpu_blend.cuh"
#include <iostream>
#include <string>

void printUsage(const char* programName) {
    std::cout << "Face Replacer - Static Photo Processing\n\n";
    std::cout << "Usage:\n";
    std::cout << "  " << programName << " <target_image> <selfie_image> <output_image> [mode] [face_index]\n\n";
    std::cout << "Arguments:\n";
    std::cout << "  target_image  - Image containing the face to replace\n";
    std::cout << "  selfie_image  - Your selfie (source face)\n";
    std::cout << "  output_image  - Output file path\n";
    std::cout << "  mode          - Replacement mode (optional, default: 2)\n";
    std::cout << "                  1 = RECT_TO_RECT (simple rectangle)\n";
    std::cout << "                  2 = HEAD_SEGMENTED (with background preservation)\n";
    std::cout << "                  3 = LIVE_ANIMATED (advanced)\n";
    std::cout << "  face_index    - Which face to replace if multiple detected (default: 0)\n\n";
    std::cout << "Examples:\n";
    std::cout << "  " << programName << " group_photo.jpg my_selfie.jpg output.jpg\n";
    std::cout << "  " << programName << " group_photo.jpg my_selfie.jpg output.jpg 2 1\n";
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
    
    // Load images
    std::cout << "Loading images..." << std::endl;
    
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
            std::cout << "Mode: LIVE_ANIMATED (advanced)" << std::endl;
            break;
        default:
            std::cerr << "Invalid mode: " << mode << ". Using HEAD_SEGMENTED." << std::endl;
            config.mode = facereplacer::ReplacementMode::HEAD_SEGMENTED;
    }
    
    // Check GPU availability
    config.useGPU = facereplacer::cuda::isCudaAvailable();
    std::cout << "GPU acceleration: " << (config.useGPU ? "ENABLED" : "DISABLED (CPU fallback)") << std::endl;
    
    config.colorCorrection = true;
    config.preserveLighting = true;
    config.featherRadius = 15;
    
    // Create replacer
    std::cout << "\nInitializing face replacer..." << std::endl;
    facereplacer::FaceReplacer replacer(config);
    
    // Prepare selfie
    std::cout << "Processing selfie..." << std::endl;
    replacer.setSourceImage(selfieImage);
    
    // Detect and mark face in target
    std::cout << "Detecting faces in target image..." << std::endl;
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
    
    // Save marked image for preview
    cv::Mat markedImage = replacer.markFace(targetImage, faceIndex);
    std::string markedPath = outputPath.substr(0, outputPath.find_last_of('.')) + "_marked.jpg";
    cv::imwrite(markedPath, markedImage);
    std::cout << "Marked image saved: " << markedPath << std::endl;
    
    // Set target face
    replacer.setTargetFace(faces[faceIndex]);
    
    // Process
    std::cout << "\nReplacing face..." << std::endl;
    auto startTime = std::chrono::high_resolution_clock::now();
    
    cv::Mat result = replacer.processFrame(targetImage);
    
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    std::cout << "Processing time: " << duration.count() << " ms" << std::endl;
    
    // Save result
    cv::imwrite(outputPath, result);
    std::cout << "\nResult saved: " << outputPath << std::endl;
    
    // Create comparison image
    std::string comparisonPath = outputPath.substr(0, outputPath.find_last_of('.')) + "_comparison.jpg";
    
    // Resize images for comparison
    int targetHeight = 400;
    cv::Mat targetResized, selfieResized, resultResized;
    
    float scale = static_cast<float>(targetHeight) / targetImage.rows;
    cv::resize(targetImage, targetResized, cv::Size(), scale, scale);
    cv::resize(selfieImage, selfieResized, cv::Size(), scale, scale);
    cv::resize(result, resultResized, cv::Size(), scale, scale);
    
    // Create comparison
    int totalWidth = targetResized.cols + selfieResized.cols + resultResized.cols + 20;
    cv::Mat comparison(targetHeight + 40, totalWidth, CV_8UC3, cv::Scalar(50, 50, 50));
    
    // Place images
    targetResized.copyTo(comparison(cv::Rect(0, 30, targetResized.cols, targetResized.rows)));
    selfieResized.copyTo(comparison(cv::Rect(targetResized.cols + 10, 30, selfieResized.cols, selfieResized.rows)));
    resultResized.copyTo(comparison(cv::Rect(targetResized.cols + selfieResized.cols + 20, 30, resultResized.cols, resultResized.rows)));
    
    // Add labels
    cv::putText(comparison, "Target", cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
    cv::putText(comparison, "Selfie", cv::Point(targetResized.cols + 15, 20), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
    cv::putText(comparison, "Result", cv::Point(targetResized.cols + selfieResized.cols + 25, 20), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
    
    cv::imwrite(comparisonPath, comparison);
    std::cout << "Comparison saved: " << comparisonPath << std::endl;
    
    std::cout << "\nDone!" << std::endl;
    
    return 0;
}
