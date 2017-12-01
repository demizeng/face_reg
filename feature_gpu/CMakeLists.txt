cmake_minimum_required(VERSION 2.8)
project(feature_gpu)
find_package(PCL 1.6 REQUIRED)

#cuda
find_package(CUDA REQUIRED)
INCLUDE(FindCUDA)
include_directories(${CUDA_INCLUDE_DIRS})
message("CUDA INCLUDE DIRS: ${CUDA_INCLUDE_DIRS}")

#pcl
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARIES_DIRS})
add_definitions(${PCL_DEFINITIONS})

INCLUDE_DIRECTORIES(./saccuda/include)
macro(REMOVE_VTK_DEFINITIONS module_name)
     get_directory_property(_dir_defs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} COMPILE_DEFINITIONS)
     set(_vtk_definitions)
     foreach(_item ${_dir_defs})
         if(_item MATCHES "vtk*")
             list(APPEND _vtk_definitions -D${_item})
         endif()
     endforeach()
     remove_definitions(${_vtk_definitions})
endmacro(REMOVE_VTK_DEFINITIONS)
add_subdirectory(saccuda)
add_executable(ransac_cuda "ransac_gpu.cpp")
target_link_libraries(ransac_cuda ${PCL_LIBRARIES} saccuda)

#add_executable(normal_nogpu "compute_nogpu.cpp")
#target_link_libraries(normal_nogpu ${PCL_LIBRARIES})
#add_executable(cudanormal "main.cpp")
#target_link_libraries(cudanormal ${PCL_LIBRARIES})
