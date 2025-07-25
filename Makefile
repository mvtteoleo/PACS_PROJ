# Directories
BUILD_DIR := build
TESTS_DIR := tests
HEADER_DIR := header
SRC_DIR := src

# Tools and Flags
CXX := g++
COMMON_FLAGS := -Wall -std=c++23 -I$(HEADER_DIR) -I$(SRC_DIR) -g -fsanitize=address -O0
COMMON_LIBS := -fopenmp 

# Groups

Tests := laplacian packs
NEEDS_EIGEN := 
NEEDS_GINAC := 
NEEDS_FFTW  := 

# Utility Functions
# Check if a name is in a list
# Usage: $(call contains, list, item)
contains = $(filter $(2),$(1))

# All .cpp sources in tests
TEST_SRCS := $(wildcard $(TESTS_DIR)/*.cpp)
EXE_NAMES := $(patsubst $(TESTS_DIR)/%.cpp,%,$(TEST_SRCS))
EXES := $(addprefix ./,$(EXE_NAMES))
OBJS := $(addprefix $(BUILD_DIR)/,$(addsuffix .o,$(EXE_NAMES)))

# Default target
all: $(Tests)

# Ensure build dir exists
$(BUILD_DIR):
	mkdir -p $@

# Rule to compile any .cpp into a .o
$(BUILD_DIR)/%.o: $(TESTS_DIR)/%.cpp | $(BUILD_DIR)
	$(CXX) $(COMMON_FLAGS) -c $< -o $@

# Rule to link .o to executable
./%: $(BUILD_DIR)/%.o
	@echo "Linking $@..."
	$(CXX) $< -o $@ \
	$(COMMON_FLAGS) \
	$(COMMON_LIBS) \
	$(if $(call contains,$(NEEDS_FFTW),$*),-Ifftw3,) \
	$(if $(call contains,$(NEEDS_EIGEN),$*),-leigen3,) \
	$(if $(call contains,$(NEEDS_GINAC),$*),-lginac -lcln,) \
	# Add specific group logic as needed here

.PHONY: all clean $(Tests)

clean:
	rm -rf $(BUILD_DIR) $(EXES)

