#CC = g++
CC = mpic++

CCFLAGS = -std=c++14
#CCFLAGS = -std=c++0x

INCLUDE_DIR = src

## Libraries: External tools, csv-parser, Eigen etc
#EIGEN_DIR = lib/eigen-3.3.8
#EIGEN_DIR = ../peri-ice/lib/eigen-3.3.9
EIGEN_DIR = lib/eigen-3.3.9
#INIH_DIR = lib/inih
#EIGEN_DIR = lib/eigen-3.2.10
#CSV_DIR = lib/fast-cpp-csv-parser

BUILD_DIR = bin
OBJ_DIR = obj
RUN_DIR = run
DATA_DIR = data
OUTPUT_DIR = output

CPPFLAGS += -O3
CPPFLAGS += -Wall -Wextra # -Wshadow # -Wcast-align # -Wold-style-cast 

CPPFLAGS += -I$(INCLUDE_DIR)
CPPFLAGS += -I$(EIGEN_DIR)
#CPPFLAGS += -I$(INIH_DIR)
#CPPFLAGS += -I /usr/include/hdf5/serial/ -lhdf5_serial -lhdf5_cpp

## ubuntu
CPPFLAGS += -I/usr/include/hdf5/serial/ -L/usr/lib/x86_64-linux-gnu/hdf5/serial -lhdf5_serial -lhdf5_cpp

## smic
#CPPFLAGS += -I/usr/local/packages/hdf5/1.10.6/vklossna/include -L/usr/local/packages/hdf5/1.10.6/vklossna/lib -lhdf5 -lhdf5_cpp
#CPPFLAGS += -I$(CSV_DIR)

#CPPFLAGS += -lpthread


# All subdirectories of includes
SUB_DIRS = $(wildcard $(INCLUDE_DIR)/*)
# Substitute 
SUB_DIRS_BUILD = $(patsubst $(INCLUDE_DIR)/%, $(OBJ_DIR)/%, $(SUB_DIRS))
# Data dirs
# All header files
T_h = $(wildcard  $(INCLUDE_DIR)/**/*.h)
# All cpp files
T_cpp = $(wildcard  $(INCLUDE_DIR)/**/*.cpp)
# .cpp replaced by .o
T_o = $(T_cpp:%.cpp=%.o)
# Replace the include path structure by object path structure: pattern-substitute
T_o_Build = $(patsubst $(INCLUDE_DIR)/%, $(OBJ_DIR)/%, $(T_o))

#all: dir simulate2d
all: dir simulate2d simulate3d

simulate2d:
	@echo "Building in 2D"
	$(CC) $(CCFLAGS) $(RUN_DIR)/simulate2d.cpp $? -o $(BUILD_DIR)/$@ $(CPPFLAGS)  -fopenmp

simulate3d:
	@echo "Building in 3D"
	$(CC) $(CCFLAGS) $(RUN_DIR)/simulate3d.cpp $? -o $(BUILD_DIR)/$@ $(CPPFLAGS)  -fopenmp

debug: $(T_o_Build)
	@echo "Building final target in 2D for debug:"
	$(CC) $(CCFLAGS) $(RUN_DIR)/simulate2d.cpp $? -o $(BUILD_DIR)/$@ $(CPPFLAGS) -g  -fopenmp
	gdb $(BUILD_DIR)/simulate2d

# Maintain the same directory structure
$(OBJ_DIR)/%.o: $(INCLUDE_DIR)/%.cpp
	@echo "Building:"
	$(CC) $(CCFLAGS) -c $? -o $@ $(CPPFLAGS)

ex2:
	@echo "Running executable 2D:"
	@echo "-----------------------------------------------------------------------"
	@time -p $(BUILD_DIR)/simulate2d

ex3:
	@echo "Running executable 3D:"
	@echo "-----------------------------------------------------------------------"
	@time -p $(BUILD_DIR)/simulate3d

clean:
	@echo "Deleting:"
	rm -rf $(BUILD_DIR)/* $(OBJ_DIR)/*

genplot:
	@echo "Generating plots:"
	python3 plot_timestep.py
	sxiv $(OUTPUT_DIR)/img/* &

showplot:
	sxiv $(OUTPUT_DIR)/img/*.png &

clout:
	@echo "Deleting output files"
	rm -rf $(OUTPUT_DIR)/csv/*.csv $(OUTPUT_DIR)/hdf5/*.h5 $(OUTPUT_DIR)/img/*.png
	rm -rf $(OUTPUT_DIR)/img/*.mp4
	rm -r $(OUTPUT_DIR)/*.npy

getfresh_py:
	@echo "Delete exiting data"
	rm -rf $(DATA_DIR)/*
	@echo "Getting new data"
	mkdir -p $(DATA_DIR)/csv $(DATA_DIR)/hdf5
	cp meshdata/all.h5 $(DATA_DIR)/hdf5/

setup:
	@echo "Generating new setup"
	python3 gen_setup.py
	make getfresh_py

dir:
	@echo "Creating directories:"
	mkdir -p $(BUILD_DIR) $(OBJ_DIR) $(SUB_DIRS_BUILD)
	mkdir -p $(DATA_DIR)/csv $(DATA_DIR)/hdf5
	mkdir -p $(OUTPUT_DIR)/csv $(OUTPUT_DIR)/hdf5 $(OUTPUT_DIR)/img $(OUTPUT_DIR)/vid

again: clean $(TARGETS)  

