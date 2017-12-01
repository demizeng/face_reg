#ifndef REGISTRATION_UTILS_GPU_H
#define REGISTRATION_UTILS_GPU_H


#include <cstdlib>
#include <iostream>
#include <stddef.h>
#include <pcl/point_types.h>
#include <pcl/gpu/features/features.hpp>
#include <pcl/gpu/octree/octree.hpp>


#define BLOCKS              16              // have to be power of 2
#define THREADS             1024            // maximum!


namespace cudaransac
{

class RegistrationUtilsGPU
{
protected:
    static float ransac_threshold;
    static float ransac_radius;
    static float ransac_poly;
    static int   ransac_converged;
    static int   ransac_inliers;
public:

    /**
     *  @brief      claculateNeighborIndex      For every point in source calculate index of the nearest point in target in the features
     *                                          space. The result save in d_neighbor_index vector.
     *  @param[in]  d_source_features           Features of source pointcloud (gpu data exported from DeviceArray2D<FPFHSignature33>)
     *  @param[in]  source_rows                 Number of source features (value of DeviceArray2D<FPFHSignature33>::rows())
     *  @param[in]  source_steps                Source feature step in bytes (value of DeviceArray2D<FPFHSignature33>::step())
     *  @param[in]  d_target_features           Features of target pointcloud (gpu data exported from DeviceArray2D<FPFHSignature33>.
     *  @param[in]  target_rows                 Number of target features (value of DeviceArray2D<FPFHSignature33>::rows())
     *  @param[in]  target_steps                Source feature step in bytes (value of DeviceArray2D<FPFHSignature33>::step())
     *  @param[out] d_neighbor_index            Index of the nearest neighbor of corresponding point in target cloud. There is gpu memory
     *                                          allocation inside.
     */
    static void calculateNeighborIndex(float* d_source_features, int source_rows, size_t source_step,
                                       float* d_target_features, int target_rows, size_t target_step,
                                       int** d_neighbor_index);

    /**
     * @brief download                          Download neighbor index data from gpu memory.
     * @param d_neighbor_index[in]              Pointer to gpu neighbor index data.
     * @param h_neighbor_index[out]             Pointer to host memory where to data copying should be performed.
     * @param points_number[in]                 Number of bytes to copy.
     */
    static void downloadNeighborsIndex(int *d_neighbor_index, int *h_neighbor_index, size_t bytes);


    static void performRanSaC(pcl::gpu::Feature::PointType* d_source_point_cloud, int source_points, int* d_source_neighbors,
                              pcl::gpu::Feature::PointType* d_target_point_cloud, pcl::gpu::Octree &d_target_octree,
                               float* transformation_matrix);

    static void setThreshold(float ransac_threshold_)
    {
        if (ransac_threshold_<0 || ransac_threshold_>1)
        {
            std::cout << "\x1B[32m" << "[Heuros::RegistrationUtilsGPU] Ransac threshold must be in range (0,1)" << "\x1B[0m" << std::endl;
            ransac_threshold = -1.0;
        }
        else
            ransac_threshold = ransac_threshold_;
    }

    static float getThreshold()
    {
        return ransac_threshold;
    }

    static void setRadius(float ransac_radius_)
    {
        if (ransac_radius_<0)
        {
            std::cout << "\x1B[32m" << "[Heuros::RegistrationUtilsGPU] Ransac threshold must be in range (0,1)" << "\x1B[0m" << std::endl;
            ransac_radius = -1.0;
        }
        else
            ransac_radius = ransac_radius_;
    }

    static float getRadius()
    {
        return ransac_radius;
    }

    static void setPoly(float ransac_poly_)
    {
        if (ransac_poly_<0 || ransac_poly_>1)
        {
            ransac_poly = -1.0;
            std::cout << "\x1B[32m" << "[Heuros::RegistrationUtilsGPU] Ransac threshold must be in range (0,1)" << "\x1B[0m" << std::endl;
        }
        else
            ransac_poly = ransac_poly_;
    }

    static float getPoly()
    {
        return ransac_poly;
    }

    static int getConverged()
    {
        return ransac_converged;
    }

    static int getInliers()
    {
        return ransac_inliers;
    }
};

}

#endif //REGISTRATION_UTILS_GPU_H
