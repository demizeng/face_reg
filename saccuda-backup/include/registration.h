#ifndef REGISTRATION_H
#define REGISTRATION_H


#include <iostream>
#include <pcl/features/fpfh.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/registration/sample_consensus_prerejective.h>

#include "registration_utils_gpu.h"

#include <boost/chrono.hpp>
#include <pcl/gpu/octree/octree.hpp>
#include <pcl/gpu/features/features.hpp>
#include <pcl/gpu/containers/device_array.h>

namespace cudaransac
{

class Registration
{
public:

    //====================================================
    //================= PUBLIC VARIABLES =================
    //====================================================

    //====================================================
    //================== PUBLIC METHODS ==================
    //====================================================

    /** \brief Constructor */
    Registration();

    /** \brief Main routine method (GPU version) */
    bool alignGPU(cudaPointCloud &source_cloud,cudaPointCloud &target_cloud,cudaFPFHarrar &source_fpfh,cudaFPFHarrar &target_fpfh);
    Eigen::Matrix<float, 4, 4> getfinaltransform();
    int getconverged();
    int getinliers();

protected:

    //====================================================
    //================ PROTECTED VARIABLES ===============
    //====================================================
    Eigen::Matrix<float, 4, 4> transformation;

    //====================================================
    //================= PROTECTED METHODS ================
    //====================================================



    //====================================================
    //================= PROTECTED DATATYPES ==============
    //====================================================

    /** \brief Point type. */
    typedef pcl::PointXYZ PointT;

    /** \brief Point cloud type. */
    typedef pcl::PointCloud<PointT> PointCloudT;

    typedef pcl::search::KdTree<PointT> KDTreeT;

    /** \brief Feature type. */

    typedef pcl::gpu::Feature::PointCloud cudaPointCloud;
    typedef boost::shared_ptr<cudaPointCloud> cudaPointCloudPtr;
    typedef pcl::FPFHSignature33 FeatureT;
    typedef pcl::gpu::DeviceArray2D<FeatureT> cudaFPFHarrar;


};

}

#endif //REGISTRATION_H
