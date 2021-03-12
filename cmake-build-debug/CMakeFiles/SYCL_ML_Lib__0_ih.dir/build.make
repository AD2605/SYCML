# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.17

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Disable VCS-based implicit rules.
% : %,v


# Disable VCS-based implicit rules.
% : RCS/%


# Disable VCS-based implicit rules.
% : RCS/%,v


# Disable VCS-based implicit rules.
% : SCCS/s.%


# Disable VCS-based implicit rules.
% : s.%


.SUFFIXES: .hpux_make_needs_suffix_list


# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /home/atharva/Downloads/CLion-2020.3/clion-2020.3/bin/cmake/linux/bin/cmake

# The command to remove a file.
RM = /home/atharva/Downloads/CLion-2020.3/clion-2020.3/bin/cmake/linux/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/atharva/CLionProjects/SYCL_ML_Lib

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/atharva/CLionProjects/SYCL_ML_Lib/cmake-build-debug

# Utility rule file for SYCL_ML_Lib__0_ih.

# Include the progress variables for this target.
include CMakeFiles/SYCL_ML_Lib__0_ih.dir/progress.make

CMakeFiles/SYCL_ML_Lib__0_ih: SYCL_ML_Lib_.bc
CMakeFiles/SYCL_ML_Lib__0_ih: SYCL_ML_Lib_.sycl


SYCL_ML_Lib_.bc: ../Kernels
SYCL_ML_Lib_.bc: ../Kernels
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/atharva/CLionProjects/SYCL_ML_Lib/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building ComputeCpp integration header file /home/atharva/CLionProjects/SYCL_ML_Lib/cmake-build-debug/SYCL_ML_Lib_.sycl"
	/home/atharva/ComputeCPP/computeCPP/bin/compute++ -sycl -O2 -mllvm -inline-threshold=1000 -intelspirmetadata -sycl-target spir64 -DSYCL_LANGUAGE_VERSION=2017 -std=c++14 -I"/home/atharva/CLionProjects/SYCL_ML_Lib/$(COMPUTECPP_INCLUDE_DIRECTORY)"	-I"/home/atharva/ComputeCPP/computeCPP/include"	-I"/usr/include"  -sycl-ih /home/atharva/CLionProjects/SYCL_ML_Lib/cmake-build-debug/SYCL_ML_Lib_.sycl -o /home/atharva/CLionProjects/SYCL_ML_Lib/cmake-build-debug/SYCL_ML_Lib_.bc -c /home/atharva/CLionProjects/SYCL_ML_Lib/Kernels/

SYCL_ML_Lib_.sycl: SYCL_ML_Lib_.bc
	@$(CMAKE_COMMAND) -E touch_nocreate SYCL_ML_Lib_.sycl

SYCL_ML_Lib__0_ih: CMakeFiles/SYCL_ML_Lib__0_ih
SYCL_ML_Lib__0_ih: SYCL_ML_Lib_.bc
SYCL_ML_Lib__0_ih: SYCL_ML_Lib_.sycl
SYCL_ML_Lib__0_ih: CMakeFiles/SYCL_ML_Lib__0_ih.dir/build.make

.PHONY : SYCL_ML_Lib__0_ih

# Rule to build all files generated by this target.
CMakeFiles/SYCL_ML_Lib__0_ih.dir/build: SYCL_ML_Lib__0_ih

.PHONY : CMakeFiles/SYCL_ML_Lib__0_ih.dir/build

CMakeFiles/SYCL_ML_Lib__0_ih.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/SYCL_ML_Lib__0_ih.dir/cmake_clean.cmake
.PHONY : CMakeFiles/SYCL_ML_Lib__0_ih.dir/clean

CMakeFiles/SYCL_ML_Lib__0_ih.dir/depend:
	cd /home/atharva/CLionProjects/SYCL_ML_Lib/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/atharva/CLionProjects/SYCL_ML_Lib /home/atharva/CLionProjects/SYCL_ML_Lib /home/atharva/CLionProjects/SYCL_ML_Lib/cmake-build-debug /home/atharva/CLionProjects/SYCL_ML_Lib/cmake-build-debug /home/atharva/CLionProjects/SYCL_ML_Lib/cmake-build-debug/CMakeFiles/SYCL_ML_Lib__0_ih.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/SYCL_ML_Lib__0_ih.dir/depend

