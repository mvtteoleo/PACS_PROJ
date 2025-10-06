# ==============================
# Compiler and Flags
CXX       := g++
MPICXX    := mpic++
CPPFLAGS  := -Iheader -Isrc -I. 
LDLIBS   := -lfftw3 -lm -lboost_iostreams -lboost_system #-lfftw3_mpi 

NIX_CPPFLAGS := -I$(EIGEN_INCLUDE_DIR) -I$(FFTW_INCLUDE_DIR) -I$(PETSC_DIR)/include
NIX_LDFLAGS  := -L$(patsubst %/include,%/lib,$(FFTW_INCLUDE_DIR)) -L$(PETSC_DIR)/lib
NIX_LDLIBS   := -lpetsc

# Append Nix flags to the project's default flags
CPPFLAGS += $(NIX_CPPFLAGS)
LDFLAGS  += $(NIX_LDFLAGS)
LDLIBS   += $(NIX_LDLIBS) 


# Optimization flags
OPT_O3   := -O3  -Wall -Wextra -pedantic -fopenmp -std=c++23
OPT_O0   := -O0 -g -Wall -Wextra -pedantic -fopenmp  -std=c++23

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

C2DECOMP_SRCS := $(wildcard $(C2DECOMP_DIR)/*.cpp)
C2DECOMP_OBJS := $(patsubst $(C2DECOMP_DIR)/%.cpp,$(BUILD_DIR)/2Decomp_C/%.o,$(C2DECOMP_SRCS))

# Serial & parallel tests
SERIAL_SRCS  := $(wildcard $(SERIAL_DIR)/*.cpp)
SERIAL_TESTS := $(patsubst $(SERIAL_DIR)/%.cpp,$(BUILD_DIR)/serial/%,$(SERIAL_SRCS))

PARALLEL_SRCS  := $(wildcard $(PARALLEL_DIR)/*.cpp)
PARALLEL_TESTS := $(patsubst $(PARALLEL_DIR)/%.cpp,$(BUILD_DIR)/parallel/%,$(PARALLEL_SRCS))

# Default MPI processes
NP ?= 4

# ==============================
# Default goal
.DEFAULT_GOAL := all
.PHONY: all clean distclean tests serial parallel \
        SERIAL_TESTS PARALLEL_TESTS run_tests run_serial run_parallel

all: $(EXEC)

tests: serial parallel
serial: $(SERIAL_TESTS)
parallel: $(PARALLEL_TESTS)

# ==============================
# Compile main program (everything O3)
$(EXEC): $(OBJS)
	$(CXX) $(OPT_O3) $(CPPFLAGS) $^ -o $@ $(LDLIBS)

# Compile src files (O3)
$(BUILD_DIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(OPT_O3) $(CPPFLAGS) -c $< -o $@

# Compile 2Decomp_C library (O0 with MPI compiler)
$(BUILD_DIR)/2Decomp_C/%.o: $(C2DECOMP_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(MPICXX) $(OPT_O0) $(CPPFLAGS) -I$(C2DECOMP_DIR) -c $< -o $@

# Link parallel test executables (use mpicxx and include O0 objects)
$(BUILD_DIR)/parallel/%: $(BUILD_DIR)/tests/parallel/%.o $(OBJS) $(C2DECOMP_OBJS)
	@mkdir -p $(dir $@)
	$(MPICXX) $(OPT_O3) $(CPPFLAGS) -I$(C2DECOMP_DIR) $^ -o $@ $(LDLIBS)

# Link serial test executables (O3)
$(BUILD_DIR)/serial/%: $(BUILD_DIR)/tests/serial/%.o $(OBJS)
	@mkdir -p $(dir $@)
	$(CXX) $(OPT_O3) $(CPPFLAGS) $^ -o $@ $(LDLIBS)

# ==============================
# Cleaning
clean:
	$(RM) -r $(BUILD_DIR)/* *.o

distclean: clean
	$(RM) $(EXEC) $(SERIAL_TESTS) $(PARALLEL_TESTS)

