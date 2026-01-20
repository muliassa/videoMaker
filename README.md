# Face Replacer (C++ with CUDA)

High-performance face replacement tool for Ubuntu with NVIDIA GPU acceleration.

## Features

- **Mode 1: RECT_TO_RECT** - Simple rectangle replacement with scaling
- **Mode 2: HEAD_SEGMENTED** - Head segmentation with background preservation (default)
- **Mode 3: LIVE_ANIMATED** - Advanced methods for live/animated faces

### Key Capabilities

- CUDA GPU acceleration for real-time processing
- Multiple face detection using OpenCV DNN or Haar Cascades
- Head segmentation with GrabCut and skin color detection
- Color matching and lighting adjustment
- Feathered blending for seamless results
- Poisson blending (seamless clone) fallback

## Requirements

### System
- Ubuntu 18.04+ (tested on 20.04, 22.04)
- NVIDIA GPU with CUDA support
- CMake 3.18+

### Dependencies

```bash
# Install OpenCV with CUDA support
sudo apt update
sudo apt install -y \
    build-essential cmake git \
    libopencv-dev \
    nvidia-cuda-toolkit \
    libcudnn8-dev

# For OpenCV with CUDA (if not available via apt)
# You may need to build OpenCV from source with CUDA enabled
```

### Building OpenCV with CUDA (if needed)

```bash
git clone https://github.com/opencv/opencv.git
git clone https://github.com/opencv/opencv_contrib.git

cd opencv && mkdir build && cd build

cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D WITH_CUDA=ON \
      -D WITH_CUDNN=ON \
      -D OPENCV_DNN_CUDA=ON \
      -D ENABLE_FAST_MATH=1 \
      -D CUDA_FAST_MATH=1 \
      -D CUDA_ARCH_BIN="5.0 6.0 7.0 7.5 8.0 8.6" \
      -D WITH_CUBLAS=1 \
      -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
      ..

make -j$(nproc)
sudo make install
```

## Building

```bash
cd face_replacer_cpp
mkdir build && cd build

# With CUDA (default)
cmake ..

# Without CUDA (CPU only)
cmake -DUSE_CUDA=OFF ..

# Build
make -j$(nproc)
```

## Usage

### Command Line

```bash
# Basic usage (Mode 2 - HEAD_SEGMENTED)
./face_replacer target_photo.jpg selfie.jpg output.jpg

# Specify mode
./face_replacer target_photo.jpg selfie.jpg output.jpg 1  # RECT_TO_RECT
./face_replacer target_photo.jpg selfie.jpg output.jpg 2  # HEAD_SEGMENTED
./face_replacer target_photo.jpg selfie.jpg output.jpg 3  # LIVE_ANIMATED

# Specify which face to replace (if multiple faces detected)
./face_replacer group_photo.jpg selfie.jpg output.jpg 2 1  # Replace face index 1
```

### Output Files

The program generates:
- `output.jpg` - The result image
- `output_marked.jpg` - Target image with detected faces marked (green = selected)
- `output_comparison.jpg` - Side-by-side comparison

### Library API

```cpp
#include "face_replacer.hpp"

// Configure
facereplacer::Config config;
config.mode = facereplacer::ReplacementMode::HEAD_SEGMENTED;
config.useGPU = true;
config.colorCorrection = true;
config.featherRadius = 15;

// Create replacer
facereplacer::FaceReplacer replacer(config);

// Load selfie
cv::Mat selfie = cv::imread("selfie.jpg");
replacer.setSourceImage(selfie);

// Detect faces in target
cv::Mat target = cv::imread("target.jpg");
auto faces = replacer.detectFaces(target);

// Select and replace
replacer.setTargetFace(faces[0]);
cv::Mat result = replacer.processFrame(target);

// Save
cv::imwrite("output.jpg", result);
```

## Project Structure

```
face_replacer_cpp/
├── CMakeLists.txt
├── README.md
├── include/
│   ├── face_replacer.hpp      # Main API
│   ├── face_detector.hpp      # Face detection
│   ├── segmentation.hpp       # Head/face segmentation
│   └── cuda/
│       └── gpu_blend.cuh      # CUDA kernels header
└── src/
    ├── main.cpp               # CLI application
    ├── face_replacer.cpp      # Core implementation
    ├── face_detector.cpp      # Detection implementation
    ├── segmentation.cpp       # Segmentation implementation
    └── cuda/
        └── gpu_blend.cu       # CUDA kernels
```

## Modes Explained

### Mode 1: RECT_TO_RECT
- Fastest, simplest approach
- Extracts face rectangle, scales to target size
- Elliptical mask with Gaussian blur for blending
- Best for: Quick tests, further processing

### Mode 2: HEAD_SEGMENTED (Recommended for static photos)
- Uses GrabCut + skin detection for head segmentation
- Creates accurate mask preserving background
- Feathered blending for seamless edges
- Color and lighting correction
- Best for: Static photos, natural look

### Mode 3: LIVE_ANIMATED
- Landmark-based face alignment
- Delaunay triangulation warping
- Temporal smoothing (for video)
- Best for: Video processing, moving faces

## Performance

| Mode | Resolution | GPU (RTX 3080) | CPU (i7-10700) |
|------|------------|----------------|----------------|
| RECT_TO_RECT | 1080p | ~5ms | ~25ms |
| HEAD_SEGMENTED | 1080p | ~15ms | ~80ms |
| LIVE_ANIMATED | 1080p | ~25ms | ~150ms |

## Known Limitations

1. Face detection requires frontal or near-frontal faces
2. Large pose differences between selfie and target may produce artifacts
3. Extreme lighting differences may not be fully corrected
4. Hair/glasses may not segment perfectly with basic methods

## Future Improvements

- [ ] Deep learning based segmentation (BiSeNet, etc.)
- [ ] 3DMM face fitting for pose estimation
- [ ] Expression transfer
- [ ] Real-time video processing pipeline
- [ ] Multi-GPU support

## License

MIT License
