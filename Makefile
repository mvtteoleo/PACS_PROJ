# ==============================
# Compiler and Flags
CXX       := g++
MPICXX    := mpicxx
CXXFLAGS  := -std=c++23 -O0 -g  -Wall -Wextra -pedantic -fopenmp
CPPFLAGS := -Iheader -Isrc -I. -I${FFTW_INCLUDE}
LDFLAGS  := -L${FFTW_LIB}
LDLIBS   := -lfftw3 -lm #-lfftw3_mpi 

# Directories
SRC_DIR      := src
TEST_DIR     := tests
SERIAL_DIR   := $(TEST_DIR)/serial
PARALLEL_DIR := $(TEST_DIR)/parallel
BUILD_DIR    := build
C2DECOMP_DIR := deps/2Decomp_C

# Main target
EXEC      := main

# ==============================
# Sources & Objects
SRCS      := $(wildcard $(SRC_DIR)/*.cpp) 
OBJS      := $(patsubst %.cpp,$(BUILD_DIR)/%.o,$(SRCS))

# Serial tests
SERIAL_SRCS  := $(wildcard $(SERIAL_DIR)/*.cpp)
SERIAL_TESTS := $(patsubst $(SERIAL_DIR)/%.cpp,$(BUILD_DIR)/serial/%,$(SERIAL_SRCS))
SERIAL_OBJS  := $(patsubst %.cpp,$(BUILD_DIR)/%.o,$(SERIAL_SRCS))

# Parallel tests
PARALLEL_SRCS  := $(wildcard $(PARALLEL_DIR)/*.cpp)
PARALLEL_TESTS := $(patsubst $(PARALLEL_DIR)/%.cpp,$(BUILD_DIR)/parallel/%,$(PARALLEL_SRCS))
PARALLEL_OBJS  := $(patsubst %.cpp,$(BUILD_DIR)/%.o,$(PARALLEL_SRCS))
C2DECOMP_SRCS  := $(wildcard $(C2DECOMP_DIR)/*.cpp)

# Default MPI processes (can be overridden: make run_parallel NP=8)
NP ?= 4

# ==============================
# Default goal
.DEFAULT_GOAL := all
.PHONY: all clean distclean tests serial parallel \
        SERIAL_TESTS PARALLEL_TESTS run_tests run_serial run_parallel

all: $(EXEC)

tests: serial parallel

serial: SERIAL_TESTS
parallel: PARALLEL_TESTS

SERIAL_TESTS: $(SERIAL_TESTS)
PARALLEL_TESTS: $(PARALLEL_TESTS)

run_tests: run_serial run_parallel

run_serial: SERIAL_TESTS
	@for t in $(SERIAL_TESTS); do \
		echo "Running $$t..."; \
		./$$t || exit 1; \
	done

run_parallel: PARALLEL_TESTS
	@for t in $(PARALLEL_TESTS); do \
		echo "Running $$t with mpirun -np $(NP)..."; \
		mpirun -np $(NP) ./$$t || exit 1; \
	done

# ==============================
# Build rules

$(EXEC): $(OBJS)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $^ -o $@ $(LDFLAGS) $(LDLIBS)

# Compile sources
$(BUILD_DIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $< -o $@ $(LDFLAGS) $(LDLIBS)

# Link serial test executables
$(BUILD_DIR)/serial/%: $(BUILD_DIR)/tests/serial/%.o $(OBJS)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $^ -o $@ $(LDFLAGS) $(LDLIBS)

# Link parallel test executables (with C2Decomp sources)
$(BUILD_DIR)/parallel/%: $(BUILD_DIR)/tests/parallel/%.o $(OBJS) $(C2DECOMP_SRCS)
	@mkdir -p $(dir $@)
	$(MPICXX) $(CXXFLAGS) $(CPPFLAGS) -I$(C2DECOMP_DIR) $^ -o $@ $(LDFLAGS) $(LDLIBS)

# ==============================
# Cleaning
clean:
	$(RM) -r $(BUILD_DIR)/* *.o

distclean: clean
	$(RM) $(EXEC) $(SERIAL_TESTS) $(PARALLEL_TESTS)

