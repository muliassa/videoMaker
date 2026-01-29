# Compiler
CXX = g++

# Directories
BUILD_DIR = build

# Source files - remove all NVIDIA SDK files
SOURCES = src/main.cpp \
          src/face_replacer.cpp \
          src/face_detector.cpp \
          src/segmentation.cpp

TARGET = $(BUILD_DIR)/vidmaker

BUILDS = BUILDS = $(BUILD_DIR)/main.o $(BUILD_DIR)/face_detector.cpp $(BUILD_DIR)/face_replacer.o $(BUILD_DIR)/segmentation.o

# Include directories - simplified
INCLUDES = -Iincludes \
           -Idownloads/onnxruntime-linux-x64-gpu-1.18.0/include \
           -I/usr/include/x86_64-linux-gnu \
           -I/usr/include/opencv4 

# Library directories - only standard system paths
LIB_DIRS = -L/usr/lib/x86_64-linux-gnu \
           -Ldownloads/onnxruntime-linux-x64-gpu-1.18.0/lib

# Libraries - FFmpeg + OpenCV only
# LIBS = -lavformat -lavcodec -lavutil -lavfilter -lswscale \
#        -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui \
#        -lssl -lcrypto

LIBS = -l:libavformat.so.60.16.100 \
       -l:libavcodec.so.60.31.102 \
       -l:libavutil.so.58.29.100 \
       -l:libavfilter.so.9.12.100 \
       -l:libswscale.so.7.5.100 \
       -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui \
       -lopencv_calib3d \
       -lonnxruntime -lcudart \
       -lssl -lcrypto

# Compiler flags
CXXFLAGS = -std=c++17 -O2 -Wall

# Force consistent FFmpeg library usage
export LD_LIBRARY_PATH = /usr/lib/x86_64-linux-gnu

# Default target
all: $(TARGET)

# Create build directory
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Main target
$(TARGET): $(SOURCES) $(BUILDS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(LIB_DIRS) $(BUILDS) $(SOURCES) $(LIBS) -o $@ 

$(BUILD_DIR)/main.o: main.cpp
	$(CXX) $(CXXFLAGS) -c $(INCLUDES) main.cpp -o $@ 

$(BUILD_DIR)/preview.o: face_detector.cpp
	$(CXX) $(CXXFLAGS) -c $(INCLUDES) face_detector.cpp -o $@ 

$(BUILD_DIR)/transcoder.o: face_replacer.cpp
	$(CXX) $(CXXFLAGS) -c $(INCLUDES) face_replacer.cpp -o $@ 

$(BUILD_DIR)/clipsMaker.o: segmentation.cpp
	$(CXX) $(CXXFLAGS) -c $(INCLUDES) segmentation.cpp -o $@ 

# Test with library path control
test: $(TARGET)
	LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu ./$(TARGET)

clean:
	rm -rf $(BUILD_DIR)

.PHONY: all clean test
