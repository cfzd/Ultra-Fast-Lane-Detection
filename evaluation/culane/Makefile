PROJECT_NAME:= evaluate

# config ----------------------------------

INCLUDE_DIRS := include
LIBRARY_DIRS := lib

# You may switch different versions of opencv like this:
# export PKG_CONFIG_PATH=/usr/local/opencv-4.1.1/lib/pkgconfig:$PKG_CONFIG_PATH 
# then use `pkg-config opencv4 --cflags --libs` since `opencv4.pc` is found

COMMON_FLAGS := -DCPU_ONLY
CXXFLAGS := -std=c++11 -fopenmp #`pkg-config --cflags opencv`
LDFLAGS := -fopenmp -Wl,-rpath,./lib #`pkg-config --libs opencv`

BUILD_DIR := build

# make rules -------------------------------
CXX ?= g++
BUILD_DIR ?= ./build

LIBRARIES += opencv_core opencv_highgui opencv_imgproc opencv_imgcodecs

CXXFLAGS += $(COMMON_FLAGS) $(foreach includedir,$(INCLUDE_DIRS),-I$(includedir))
LDFLAGS +=  $(COMMON_FLAGS) $(foreach includedir,$(LIBRARY_DIRS),-L$(includedir)) $(foreach library,$(LIBRARIES),-l$(library))
SRC_DIRS += $(shell find * -type d -exec bash -c "find {} -maxdepth 1 \( -name '*.cpp' -o -name '*.proto' \) | grep -q ." \; -print)
CXX_SRCS += $(shell find src/ -name "*.cpp")
CXX_TARGETS:=$(patsubst %.cpp, $(BUILD_DIR)/%.o, $(CXX_SRCS))
ALL_BUILD_DIRS := $(sort $(BUILD_DIR) $(addprefix $(BUILD_DIR)/, $(SRC_DIRS)))

.PHONY: all
all: $(PROJECT_NAME)

.PHONY: $(ALL_BUILD_DIRS)
$(ALL_BUILD_DIRS):
	@mkdir -p $@

$(BUILD_DIR)/%.o: %.cpp | $(ALL_BUILD_DIRS)
	@echo "CXX" $<
	@$(CXX) $(CXXFLAGS) -c -o $@ $<

$(PROJECT_NAME): $(CXX_TARGETS)
	@echo "CXX/LD" $@
	@$(CXX) -o $@ $^ $(LDFLAGS)

.PHONY: clean
clean:
	@rm -rf $(CXX_TARGETS)
	@rm -rf $(PROJECT_NAME)
	@rm -rf $(BUILD_DIR)
