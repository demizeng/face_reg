#include <iostream>
#include <boost/chrono.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/common/transforms.h>
#include <pcl/gpu/features/features.hpp>
#include <pcl/point_types.h>
#include <pcl/impl/point_types.hpp>
#include <pcl/point_cloud.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/filter.h>
#include <pcl/gpu/containers/device_array.h>
#include <pcl/gpu/octree/device_format.hpp>
#include <pcl/gpu/octree/octree.hpp>
#include <pcl/pcl_macros.h>
#include <pcl/registration/sample_consensus_prerejective.h>
#include <pcl/filters/voxel_grid.h>
#include <time.h>
//#include <include/registration_utils_gpu.h>
#include "registration_utils_gpu.h"

using namespace std;
using namespace pcl;
using namespace pcl::gpu;
using namespace cudaransac;

typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloudT;
typedef pcl::gpu::Feature::PointCloud cudaPointCloud;
typedef boost::shared_ptr<cudaPointCloud> cudaPointCloudPtr;
typedef pcl::gpu::Feature::Normals cudaPointNormal;
typedef pcl::search::KdTree<PointT> KDTreeT;
typedef pcl::FPFHSignature33 FeatureT;
typedef DeviceArray2D<FeatureT> cudaFPFHarrar;

void visualize_pcd(PointCloudT::Ptr pcd_src,
   PointCloudT::Ptr pcd_tgt,PointCloudT::Ptr pcd_final)
{
   pcl::visualization::PCLVisualizer viewer("registration Viewer");
   pcl::visualization::PointCloudColorHandlerCustom<PointT> src_h (pcd_src, 0, 255, 0);
   pcl::visualization::PointCloudColorHandlerCustom<PointT> tgt_h (pcd_tgt, 255, 0, 0);
   pcl::visualization::PointCloudColorHandlerCustom<PointT> final_h (pcd_final, 0, 0, 255);
   viewer.addPointCloud (pcd_src, src_h, "source cloud");
   viewer.addPointCloud (pcd_tgt, tgt_h, "tgt cloud");
   viewer.addPointCloud (pcd_final, final_h, "final cloud");
   //viewer.addPointCloudNormals(pcd_src,cloud_normals);
   //viewer.addCoordinateSystem(1.0);
   while (!viewer.wasStopped())
   {
       viewer.spinOnce(100);
       boost::this_thread::sleep(boost::posix_time::microseconds(100000));
   }
}

void matrix2angle (Eigen::Matrix4f &result_trans,Eigen::Vector3f &result_angle)
{
  double ax,ay,az;
  if (result_trans(2,0)==1 || result_trans(2,0)==-1)
  {
      az=0;
      double dlta;
      dlta=atan2(result_trans(0,1),result_trans(0,2));
      if (result_trans(2,0)==-1)
      {
          ay=M_PI/2;
          ax=az+dlta;
      }
      else
      {
          ay=-M_PI/2;
          ax=-az+dlta;
      }
  }
  else
  {
      ay=-asin(result_trans(2,0));
      ax=atan2(result_trans(2,1)/cos(ay),result_trans(2,2)/cos(ay));
      az=atan2(result_trans(1,0)/cos(ay),result_trans(0,0)/cos(ay));
  }
  result_angle<<ax,ay,az;
}


int main()
{
    //加载点云文件(原点云，待配准)
    PointCloudT::Ptr cloud_src (new PointCloudT);//将长的转成短的//读取点云数据
    pcl::io::loadPCDFile ("../data/zjw_face0_filtered.pcd",*cloud_src);
    PointCloudT::Ptr cloud_tgt (new PointCloudT);
    pcl::io::loadPCDFile ("../data/zjw_face1_filtered.pcd",*cloud_tgt);
    PointCloudT::Ptr cloud_result (new PointCloudT);
    const float leaf = 0.01f;
    PointCloudT::Ptr cloud_src_o (new PointCloudT);
    PointCloudT::Ptr cloud_tgt_o (new PointCloudT);

    cudaPointCloud prepare(2);//排除掉CUDA预热时间（在第一次申请显存时会有一段预热时间，无法避免）

    boost::chrono::high_resolution_clock::time_point start = boost::chrono::high_resolution_clock::now();
    //下采样
    std::cout<<"before down :"<<"src:"<<cloud_src->size()<<" points;tgt "<<cloud_tgt->size()<<" points."<<endl;
    pcl::VoxelGrid<PointT> grid;

    grid.setLeafSize (leaf, leaf, leaf);
    grid.setInputCloud (cloud_src);
    grid.filter (*cloud_src_o);
    //grid.setLeafSize (leaf, leaf, leaf);
    grid.setInputCloud (cloud_tgt);
    grid.filter (*cloud_tgt_o);
    boost::chrono::high_resolution_clock::time_point downcloud = boost::chrono::high_resolution_clock::now();
    cout<<"after down : src_cloud size: "<<cloud_src_o->size()<<",tgt_cloud size: "<<cloud_tgt_o->size()<<endl;
/*
    //去除NAN点
    std::vector<int> indices; //保存去除的点的索引
    pcl::removeNaNFromPointCloud(*cloud_src_o,*cloud_src_o, indices); //去除点云中的NaN点
    pcl::removeNaNFromPointCloud(*cloud_tgt_o,*cloud_tgt_o, indices); //去除点云中的NaN点
    std::cout<<"remove *cloud_tgt_o nan"<<endl;
    std::cout<<"remove *cloud_src_o nan"<<endl;

*/
    //数据CPU->GPU
    //clock_t begin=clock();
    cudaPointCloud ccloud_src (cloud_src_o->size());
    cudaPointCloud ccloud_tgt (cloud_tgt_o->size());
    //pcl::gpu::Feature::PointCloud ccloud_tgt1 (cloud_tgt_o->size());
    //clock_t end=clock();
    //std::cout<<"cudacloud time: "<<(double)(end-begin)/(double)CLOCKS_PER_SEC<<" s"<<endl;
    //clock_t beginnormal=clock();
    cudaPointNormal cnormal_src (cloud_src_o->size());
    cudaPointNormal cnormal_tgt (cloud_tgt_o->size());
    //clock_t endnormal=clock();
    //std::cout<<"cudanormal time: "<<(double)(endnormal-beginnormal)/(double)CLOCKS_PER_SEC<<" s"<<endl;

    ccloud_src.upload(cloud_src_o->points);
    ccloud_tgt.upload(cloud_tgt_o->points);
    boost::chrono::high_resolution_clock::time_point data_time = boost::chrono::high_resolution_clock::now();

    //计算法线（CUDA）
    NormalEstimation cudanormal;
    cudanormal.setInputCloud(ccloud_src);
    cudanormal.setRadiusSearch(0.02,4);
    cudanormal.setViewPoint(0.0,0.0,0.0);
    cudanormal.compute(cnormal_src);
    cudanormal.setInputCloud(ccloud_tgt);
    cudanormal.compute(cnormal_tgt);
    boost::chrono::high_resolution_clock::time_point normal_time = boost::chrono::high_resolution_clock::now();

    //计算FPFH（CUDA）
    cudaFPFHarrar tgt_fpfh;
    cudaFPFHarrar src_fpfh;
    FPFHEstimation fpfh_estimation;
    fpfh_estimation.setInputCloud(ccloud_src);
    fpfh_estimation.setInputNormals(cnormal_src);
    //第二个参数代表octree每次搜索返回的点数
    fpfh_estimation.setRadiusSearch(0.04,4);
    fpfh_estimation.compute(src_fpfh);
    fpfh_estimation.setInputCloud(ccloud_tgt);
    fpfh_estimation.setInputNormals(cnormal_tgt);
    fpfh_estimation.compute(tgt_fpfh);
    boost::chrono::high_resolution_clock::time_point fpfh_time = boost::chrono::high_resolution_clock::now();

    std::cout<<"prepare down!"<<endl;
/*
    //将FPFH从GPU转回CPU
    pcl::PointCloud<FeatureT>::Ptr tgt_fpfh_ptr (new pcl::PointCloud<FeatureT>);
    tgt_fpfh_ptr->width=tgt_fpfh.rows();
    tgt_fpfh_ptr->height=1;
    pcl::PointCloud<FeatureT>::Ptr src_fpfh_ptr (new pcl::PointCloud<FeatureT>);
    src_fpfh_ptr->width=src_fpfh.rows();
    src_fpfh_ptr->height=1;
    int cols;
    tgt_fpfh.download(tgt_fpfh_ptr->points,cols);
    src_fpfh.download(src_fpfh_ptr->points,cols);
    boost::chrono::high_resolution_clock::time_point download_fpfh_time = boost::chrono::high_resolution_clock::now();
*/
    //RANSAC姿态估计
//    pcl::SampleConsensusPrerejective<PointT,PointT,pcl::FPFHSignature33> align;

//    align.setInputSource (cloud_src_o);
//    align.setSourceFeatures (src_fpfh_ptr);
//    align.setInputTarget (cloud_tgt_o);
//    align.setTargetFeatures (tgt_fpfh_ptr);
//    align.setMaximumIterations (3000); // Number of RANSAC iterations
//    align.setNumberOfSamples (3); // Number of points to sample for generating/prerejecting a pose
//    align.setCorrespondenceRandomness (5); // Number of nearest features to use
//    /*The alignment class uses the CorrespondenceRejectorPoly class for early elimination of bad poses based on pose-invariant
//     * geometric consistencies of the inter-distances between sampled points on the object and the scene. The closer this value
//     * is set to 1, the more greedy and thereby fast the algorithm becomes. However, this also increases the risk of eliminating
//     * good poses when noise is present.
//     * */
//    align.setSimilarityThreshold (0.9); // Polygonal edge length similarity threshold 数值越小时间越长（接近线性），效果非线性
//    /*This is the Euclidean distance threshold used for determining whether a transformed object point is correctly aligned to
//    the nearest scene point or not. In this example, we have used a heuristic value of 1.5 times the point cloud resolution.*/
//    align.setMaxCorrespondenceDistance (3.0f*leaf); // Inlier threshold 会影响inliers参数， 值越大inliers越大
//    align.setInlierFraction (0.3f); // Required inlier fraction for accepting a pose hypothesis,当配准的总点数/待配准点云总点数超过了inliers fraction比例，则认为配准有效。
//    align.align (*cloud_result);

    //Building octree for target cloud
    pcl::gpu::Octree target_octree_gpu;
    target_octree_gpu.setCloud(ccloud_tgt);
    target_octree_gpu.build();

    //Calculate nearest neighbors in features space
    int* d_neighbors;
    boost::chrono::high_resolution_clock::time_point start_n = boost::chrono::high_resolution_clock::now();
    cudaransac::RegistrationUtilsGPU::calculateNeighborIndex((float*)(&(src_fpfh)), src_fpfh.rows(), src_fpfh.step(),
                                                 (float*)(&(tgt_fpfh)), tgt_fpfh.rows(), tgt_fpfh.step(),
                                                 &d_neighbors);
    boost::chrono::nanoseconds sec_n = boost::chrono::high_resolution_clock::now() - start_n;
    std::cout<<"neighbor down!"<<endl;

    //Perform RanSaC
    cudaransac::RegistrationUtilsGPU::setRadius(1.5*0.01);                                                             //1.5 voxel size
    cudaransac::RegistrationUtilsGPU::setThreshold((float)ccloud_tgt.size() * 0.75 / ccloud_src.size());    //80% of maximm inliers ratio
    cudaransac::RegistrationUtilsGPU::setPoly(0.20);                                                                     //Set poly threshold to 0.2 - the lower the stronger is the rejection
    std::cout<<"set parameters down!"<<endl;

    float transformation_matrix[16];
    boost::chrono::high_resolution_clock::time_point start_r = boost::chrono::high_resolution_clock::now();
    cudaransac::RegistrationUtilsGPU::performRanSaC(&(*ccloud_src), ccloud_src.size(), d_neighbors,
                                        &(*ccloud_tgt), target_octree_gpu,
                                        transformation_matrix);
    boost::chrono::nanoseconds sec = boost::chrono::high_resolution_clock::now() - start_r;
    std::cout<<"ransac down!"<<endl;

     boost::chrono::high_resolution_clock::time_point ransac_time = boost::chrono::high_resolution_clock::now();
    //测试的时候调用
    int inliers=cudaransac::RegistrationUtilsGPU::getInliers();
    cout << " GPU Inliers ratio: " << (float)inliers / ccloud_tgt.size()  << endl;
    cout << " GPU NEIGHBORS time: " << sec_n.count()/1000000 << " ms" << endl;
    cout << " GPU RANSAC time: " << sec.count()/1000000 << " ms"  << endl;

    //Convert to eigen
    Eigen::Matrix4f transformation;
    memcpy(transformation.data(), transformation_matrix, 16*sizeof(float));


    boost::chrono::nanoseconds fpfh_sec = fpfh_time - normal_time;
    boost::chrono::nanoseconds normal_sec = normal_time - data_time;
    boost::chrono::nanoseconds total_sec = ransac_time - start;
    //boost::chrono::nanoseconds downfpfh_sec = download_fpfh_time - fpfh_time;
    boost::chrono::nanoseconds data_sec = data_time - downcloud;
    boost::chrono::nanoseconds ransac_sec = ransac_time - fpfh_time;
    boost::chrono::nanoseconds down_sec = downcloud - start;
    cout<<"data time: "<<data_sec<<",normal time: "<<normal_sec<<",fpfh time: "<<fpfh_sec<<",total time: "<<total_sec<<endl;
    cout<<"ransac time: "<<ransac_sec<<",down cloud: "<<down_sec<<endl;

    std::cout<<"ransac hans converged: "<<cudaransac::RegistrationUtilsGPU::getConverged()<<endl;
    std::cout<<transformation;
    Eigen::Vector3f ANGLE_result;
    matrix2angle(transformation,ANGLE_result);
    cout<<" angle in x y z:\n"<<ANGLE_result(0)*180/M_PI<<","<<ANGLE_result(1)*180/M_PI<<","<<ANGLE_result(2)*180/M_PI<<endl;
    cout<<" offset in x y z:\n"<<transformation(0,3)<<","<<transformation(1,3)<<","<<transformation(2,3)<<endl;

    pcl::transformPointCloud(*cloud_src,*cloud_result,transformation);
    visualize_pcd(cloud_src,cloud_tgt,cloud_result);


}
