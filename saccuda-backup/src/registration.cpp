#include "../include/registration.h"
#include "../include/registration_utils_gpu.h"
#include <boost/chrono.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/common/transforms.h>
#include <pcl/gpu/features/features.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>


using namespace std;
using namespace pcl;
using namespace pcl::gpu;


namespace cudaransac
{

Registration::Registration(){}

bool Registration::alignGPU(cudaPointCloud &source_cloud,cudaPointCloud &target_cloud,cudaFPFHarrar &source_fpfh,cudaFPFHarrar &target_fpfh)
{
    // Remember variables
    assert( source_cloud );
    assert( target_cloud );

    //Building octree for target cloud
    pcl::gpu::Octree target_octree_gpu;
    target_octree_gpu.setCloud(target_cloud);
    target_octree_gpu.build();

    //Calculate nearest neighbors in features space
    int* d_neighbors;
    boost::chrono::high_resolution_clock::time_point start_n = boost::chrono::high_resolution_clock::now();
    RegistrationUtilsGPU::calculateNeighborIndex((float*)(&(*source_fpfh)), source_fpfh.rows(), source_fpfh.step(),
                                                 (float*)(&(*target_fpfh)), target_fpfh.rows(), target_fpfh.step(),
                                                 &d_neighbors);
    boost::chrono::nanoseconds sec_n = boost::chrono::high_resolution_clock::now() - start_n;

    //Perform RanSaC
    //三个参数设置可以考虑写成registration类的方法
    RegistrationUtilsGPU::setRadius(1.5*0.005);                                                             //1.5 voxel size
    RegistrationUtilsGPU::setThreshold((float)target_cloud.size() * 0.75 / source_cloud.size());    //80% of maximm inliers ratio
    RegistrationUtilsGPU::setPoly(0.20);                                                                     //Set poly threshold to 0.2 - the lower the stronger is the rejection


    float transformation_matrix[16];
    boost::chrono::high_resolution_clock::time_point start = boost::chrono::high_resolution_clock::now();
    RegistrationUtilsGPU::performRanSaC(&(*source_cloud), source_cloud.size(), d_neighbors,
                                        &(*target_cloud), target_octree_gpu,
                                        transformation_matrix);
    boost::chrono::nanoseconds sec = boost::chrono::high_resolution_clock::now() - start;

    //测试的时候调用
    cout << " GPU Inliers ratio: " << (float)inliers / target_cloud_gpu.size()  << endl;
    cout << " GPU NEIGHBORS time: " << sec_n.count()/1000000 << " ms" << endl;
    cout << " GPU RANSAC time: " << sec.count()/1000000 << " ms"  << endl;

    //Convert to eigen
    memcpy(transformation.data(), transformation_matrix, 16*sizeof(float));

    return true;
}

int Registration::getconverged()
{
    return RegistrationUtilsGPU::getConverged();
}
Eigen::Matrix<float, 4, 4> Registration::getfinaltransform()
{
    return transformation;
}
int Registration::getinliers()
{
    return RegistrationUtilsGPU::getInliers();
}

}
