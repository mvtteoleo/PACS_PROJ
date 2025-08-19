  # ==============================
# Compiler and Flags
CXX       := g++
CXXFLAGS  := -std=c++23 -O3 -Wall -Wextra -pedantic -fopenmp
CPPFLAGS  := -Iheader -Isrc -I.

# Directories
SRC_DIR   := src
TEST_DIR  := tests
BUILD_DIR := build

# Main target
EXEC      := main

# ==============================
# Sources & Objects
SRCS      := $(wildcard $(SRC_DIR)/*.cpp) 
OBJS      := $(patsubst %.cpp,$(BUILD_DIR)/%.o,$(SRCS))

# Tests
TEST_SRCS := $(wildcard $(TEST_DIR)/*.cpp)
TESTS     := $(patsubst $(TEST_DIR)/%.cpp,$(BUILD_DIR)/%,$(TEST_SRCS))
TEST_OBJS := $(patsubst %.cpp,$(BUILD_DIR)/%.o,$(TEST_SRCS))

# ==============================
# Default goal
.DEFAULT_GOAL := all
.PHONY: all clean distclean tests run_tests

all: $(EXEC)

tests: $(TESTS)

run_tests: tests
	@for t in $(TESTS); do \
		echo "Running $$t..."; \
		./$$t || exit 1; \
	done

# ==============================
# Build rules

$(EXEC): $(OBJS)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $^ -o $@

# Compile sources
$(BUILD_DIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $< -o $@

# Link test executables
$(BUILD_DIR)/%: $(BUILD_DIR)/tests/%.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $^ -o $@

# ==============================
# Cleaning
clean:
	$(RM) -r $(BUILD_DIR)/* *.o

distclean: clean
	$(RM) $(EXEC) $(TESTS)

