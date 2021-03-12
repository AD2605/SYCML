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

# Include any dependencies generated for this target.
include Linalg/Kernels/CMakeFiles/Linalg_Kernels.dir/depend.make

# Include the progress variables for this target.
include Linalg/Kernels/CMakeFiles/Linalg_Kernels.dir/progress.make

# Include the compile flags for this target's objects.
include Linalg/Kernels/CMakeFiles/Linalg_Kernels.dir/flags.make

Linalg/Kernels/CMakeFiles/Linalg_Kernels.dir/BLAS1.cpp.o: Linalg/Kernels/CMakeFiles/Linalg_Kernels.dir/flags.make
Linalg/Kernels/CMakeFiles/Linalg_Kernels.dir/BLAS1.cpp.o: ../Linalg/Kernels/BLAS1.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/atharva/CLionProjects/SYCL_ML_Lib/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object Linalg/Kernels/CMakeFiles/Linalg_Kernels.dir/BLAS1.cpp.o"
	cd /home/atharva/CLionProjects/SYCL_ML_Lib/cmake-build-debug/Linalg/Kernels && /home/atharva/ComputeCPP/computeCPP/bin/compute++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Linalg_Kernels.dir/BLAS1.cpp.o -c /home/atharva/CLionProjects/SYCL_ML_Lib/Linalg/Kernels/BLAS1.cpp

Linalg/Kernels/CMakeFiles/Linalg_Kernels.dir/BLAS1.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Linalg_Kernels.dir/BLAS1.cpp.i"
	cd /home/atharva/CLionProjects/SYCL_ML_Lib/cmake-build-debug/Linalg/Kernels && /home/atharva/ComputeCPP/computeCPP/bin/compute++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/atharva/CLionProjects/SYCL_ML_Lib/Linalg/Kernels/BLAS1.cpp > CMakeFiles/Linalg_Kernels.dir/BLAS1.cpp.i

Linalg/Kernels/CMakeFiles/Linalg_Kernels.dir/BLAS1.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Linalg_Kernels.dir/BLAS1.cpp.s"
	cd /home/atharva/CLionProjects/SYCL_ML_Lib/cmake-build-debug/Linalg/Kernels && /home/atharva/ComputeCPP/computeCPP/bin/compute++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/atharva/CLionProjects/SYCL_ML_Lib/Linalg/Kernels/BLAS1.cpp -o CMakeFiles/Linalg_Kernels.dir/BLAS1.cpp.s

# Object files for target Linalg_Kernels
Linalg_Kernels_OBJECTS = \
"CMakeFiles/Linalg_Kernels.dir/BLAS1.cpp.o"

# External object files for target Linalg_Kernels
Linalg_Kernels_EXTERNAL_OBJECTS =

Linalg/Kernels/libLinalg_Kernels.a: Linalg/Kernels/CMakeFiles/Linalg_Kernels.dir/BLAS1.cpp.o
Linalg/Kernels/libLinalg_Kernels.a: Linalg/Kernels/CMakeFiles/Linalg_Kernels.dir/build.make
Linalg/Kernels/libLinalg_Kernels.a: Linalg/Kernels/CMakeFiles/Linalg_Kernels.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/atharva/CLionProjects/SYCL_ML_Lib/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library libLinalg_Kernels.a"
	cd /home/atharva/CLionProjects/SYCL_ML_Lib/cmake-build-debug/Linalg/Kernels && $(CMAKE_COMMAND) -P CMakeFiles/Linalg_Kernels.dir/cmake_clean_target.cmake
	cd /home/atharva/CLionProjects/SYCL_ML_Lib/cmake-build-debug/Linalg/Kernels && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/Linalg_Kernels.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
Linalg/Kernels/CMakeFiles/Linalg_Kernels.dir/build: Linalg/Kernels/libLinalg_Kernels.a

.PHONY : Linalg/Kernels/CMakeFiles/Linalg_Kernels.dir/build

Linalg/Kernels/CMakeFiles/Linalg_Kernels.dir/clean:
	cd /home/atharva/CLionProjects/SYCL_ML_Lib/cmake-build-debug/Linalg/Kernels && $(CMAKE_COMMAND) -P CMakeFiles/Linalg_Kernels.dir/cmake_clean.cmake
.PHONY : Linalg/Kernels/CMakeFiles/Linalg_Kernels.dir/clean

Linalg/Kernels/CMakeFiles/Linalg_Kernels.dir/depend:
	cd /home/atharva/CLionProjects/SYCL_ML_Lib/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/atharva/CLionProjects/SYCL_ML_Lib /home/atharva/CLionProjects/SYCL_ML_Lib/Linalg/Kernels /home/atharva/CLionProjects/SYCL_ML_Lib/cmake-build-debug /home/atharva/CLionProjects/SYCL_ML_Lib/cmake-build-debug/Linalg/Kernels /home/atharva/CLionProjects/SYCL_ML_Lib/cmake-build-debug/Linalg/Kernels/CMakeFiles/Linalg_Kernels.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : Linalg/Kernels/CMakeFiles/Linalg_Kernels.dir/depend

