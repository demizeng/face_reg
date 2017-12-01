# Name of this module
set(MODULE_NAME saccuda)

# Find cula include dirs
FIND_PATH( CULA_INCLUDE_DIR "cula.h"
           PATH_SUFFIXES "cula/include" )

# Find cula library dirs
FIND_PATH( CULA_LIBRARY_DIRS "libcula_lapack_basic.so"
           PATH_SUFFIXES "cula/lib64" )

# Append cmake variables1
if(CULA_INCLUDE_DIR AND CULA_LIBRARY_DIRS)
  message("* CULA was found")
  include_directories(${CULA_INCLUDE_DIR})
  link_directories(${CULA_LIBRARY_DIRS})
endif()

# Remove VTK defs for nvcc
remove_vtk_definitions(${MODULE_NAME})

# General settings
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
file(GLOB_RECURSE SOURCES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} FOLLOW_SYMLINKS
     src/*.cpp src/*.hpp src/*.cu include/*.h include/*.hpp include/*.cuh)

# Add CUDA library
cuda_add_library(${MODULE_NAME} STATIC ${SOURCES})
add_dependencies(${MODULE_NAME} interface)
target_link_libraries(${MODULE_NAME} ${LINK_TO_${MODULE_NAME}})
