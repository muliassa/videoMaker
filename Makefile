# Compiler
CXX = g++

# Directories
BUILD_DIR = build
SRC_DIR = src

# Source files
SOURCES = $(SRC_DIR)/main.cpp \
          $(SRC_DIR)/face_replacer.cpp \
          $(SRC_DIR)/face_detector.cpp \
          $(SRC_DIR)/segmentation.cpp

# Object files (must match source files!)
OBJECTS = $(BUILD_DIR)/main.o \
          $(BUILD_DIR)/face_replacer.o \
          $(BUILD_DIR)/face_detector.o \
          $(BUILD_DIR)/segmentation.o

TARGET = $(BUILD_DIR)/face_replacer

# Include directories
INCLUDES = -Iinclude \
           -Idownloads/onnxruntime-linux-x64-gpu-1.18.0/include \
           -I/usr/include/x86_64-linux-gnu \
           -I/usr/include/opencv4

# Library directories
LIB_DIRS = -L/usr/lib/x86_64-linux-gnu \
           -Ldownloads/onnxruntime-linux-x64-gpu-1.18.0/lib

# Libraries - added opencv_dnn and opencv_objdetect for face detection
LIBS = -l:libavformat.so.60.16.100 \
       -l:libavcodec.so.60.31.102 \
       -l:libavutil.so.58.29.100 \
       -l:libavfilter.so.9.12.100 \
       -l:libswscale.so.7.5.100 \
       -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui \
       -lopencv_calib3d -lopencv_dnn -lopencv_objdetect -lopencv_photo \
       -lonnxruntime -lcudart \
       -lssl -lcrypto

# Compiler flags
CXXFLAGS = -std=c++17 -O2 -Wall

# Force consistent FFmpeg library usage
export LD_LIBRARY_PATH = /usr/lib/x86_64-linux-gnu

# Default target
all: $(BUILD_DIR) $(TARGET)

# Create build directory
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Main target - link object files only (not sources)
$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) $(OBJECTS) $(LIB_DIRS) $(LIBS) -o $@

# Pattern rule for object files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Explicit dependencies (optional but helps with header changes)
$(BUILD_DIR)/main.o: $(SRC_DIR)/main.cpp include/face_replacer.hpp
$(BUILD_DIR)/face_replacer.o: $(SRC_DIR)/face_replacer.cpp include/face_replacer.hpp include/face_detector.hpp include/segmentation.hpp
$(BUILD_DIR)/face_detector.o: $(SRC_DIR)/face_detector.cpp include/face_detector.hpp include/face_replacer.hpp
$(BUILD_DIR)/segmentation.o: $(SRC_DIR)/segmentation.cpp include/segmentation.hpp include/face_replacer.hpp

clean:
	rm -rf $(BUILD_DIR)

.PHONY: all clean
