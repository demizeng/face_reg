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

using namespace std;
using namespace pcl;
using namespace pcl::gpu;

typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloudT;
typedef pcl::gpu::Feature::PointCloud cudaPointCloud;
typedef boost::shared_ptr<cudaPointCloud> cudaPointCloudPtr;
typedef pcl::gpu::Feature::Normals cudaPointNormal;

void visualize_pcd(PointCloudT::Ptr pcd_src,
   PointCloudT::Ptr pcd_tgt)
{
   pcl::visualization::PCLVisualizer viewer("registration Viewer");
   pcl::visualization::PointCloudColorHandlerCustom<PointT> src_h (pcd_src, 0, 255, 0);
   pcl::visualization::PointCloudColorHandlerCustom<PointT> tgt_h (pcd_tgt, 255, 0, 0);
   //pcl::visualization::PointCloudColorHandlerCustom<PointT> final_h (pcd_final, 0, 0, 255);
   viewer.addPointCloud (pcd_src, src_h, "source cloud");
   viewer.addPointCloud (pcd_tgt, tgt_h, "tgt cloud");
   //viewer.addPointCloud (pcd_final, final_h, "final cloud");
   //viewer.addPointCloudNormals(pcd_src,cloud_normals);
   while (!viewer.wasStopped())
   {
       viewer.spinOnce(100);
       boost::this_thread::sleep(boost::posix_time::microseconds(100000));
   }
}

int main()
{
    //加载点云文件(原点云，待配准)
    PointCloudT::Ptr cloud_src_o (new PointCloudT);//将长的转成短的//读取点云数据
    pcl::io::loadPCDFile ("../r30_face1_filtered.pcd",*cloud_src_o);
    PointCloudT::Ptr cloud_tgt_o (new PointCloudT);
    pcl::io::loadPCDFile ("../face1_filtered.pcd",*cloud_tgt_o);
    PointCloudT::Ptr cloud_result (new PointCloudT);

    //去除NAN点
    std::vector<int> indices; //保存去除的点的索引
    pcl::removeNaNFromPointCloud(*cloud_src_o,*cloud_src_o, indices); //去除点云中的NaN点
    pcl::removeNaNFromPointCloud(*cloud_tgt_o,*cloud_tgt_o, indices); //去除点云中的NaN点
    std::cout<<"remove *cloud_tgt_o nan"<<endl;
    std::cout<<"remove *cloud_src_o nan"<<endl;
    cout<<"src_cloud size: "<<cloud_src_o->size()<<",tgt_cloud size: "<<cloud_tgt_o->size()<<endl;

    boost::chrono::high_resolution_clock::time_point start = boost::chrono::high_resolution_clock::now();
    cudaPointCloud ccloud_src ((cloud_src_o->width)*(cloud_src_o->height));
    cudaPointCloud ccloud_tgt ((cloud_tgt_o->width)*(cloud_tgt_o->height));
    cudaPointNormal cnormal_src ((cloud_src_o->width)*(cloud_src_o->height));
    cudaPointNormal cnormal_tgt ((cloud_src_o->width)*(cloud_src_o->height));
    ccloud_src.upload(cloud_src_o->points);
    ccloud_tgt.upload(cloud_tgt_o->points);
    boost::chrono::high_resolution_clock::time_point data_time = boost::chrono::high_resolution_clock::now();
    NormalEstimation cudanormal;
    cudanormal.setInputCloud(ccloud_src);
    cudanormal.setRadiusSearch(0.02,4);
    cudanormal.setViewPoint(0.0,0.0,0.0);
    cudanormal.compute(cnormal_src);
    cudanormal.setInputCloud(ccloud_tgt);
    cudanormal.compute(cnormal_tgt);
    boost::chrono::high_resolution_clock::time_point normal_time = boost::chrono::high_resolution_clock::now();

    DeviceArray2D<FPFHSignature33> tgt_fpfh;
    DeviceArray2D<FPFHSignature33> src_fpfh;
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
    boost::chrono::nanoseconds fpfh_sec = fpfh_time - normal_time;
    boost::chrono::nanoseconds normal_sec = normal_time - data_time;
    boost::chrono::nanoseconds total_sec = fpfh_time - start;
    boost::chrono::nanoseconds data_sec = data_time - start;
    cout<<"data time: "<<data_sec<<",normal time: "<<normal_sec<<",fpfh time: "<<fpfh_sec<<",total time: "<<total_sec<<endl;


    visualize_pcd(cloud_src_o,cloud_tgt_o);


}
