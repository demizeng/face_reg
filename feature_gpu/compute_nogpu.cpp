#include <iostream>
#include <boost/chrono.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/common/transforms.h>
#include <pcl/common/norms.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/filter.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/search/kdtree.h>
#include <pcl/features/fpfh_omp.h>

using namespace std;

typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloud;

void visualize_pcd(PointCloud::Ptr pcd_src,
   PointCloud::Ptr pcd_tgt
   )
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
    PointCloud::Ptr cloud_src_o (new PointCloud);//将长的转成短的//读取点云数据
    pcl::io::loadPCDFile ("../r30_face1_filtered.pcd",*cloud_src_o);
    PointCloud::Ptr cloud_tgt_o (new PointCloud);
    pcl::io::loadPCDFile ("../face1_filtered.pcd",*cloud_tgt_o);
    PointCloud::Ptr cloud_result (new PointCloud);

    //去除NAN点
    std::vector<int> indices; //保存去除的点的索引
    pcl::removeNaNFromPointCloud(*cloud_src_o,*cloud_src_o, indices); //去除点云中的NaN点
    pcl::removeNaNFromPointCloud(*cloud_tgt_o,*cloud_tgt_o, indices); //去除点云中的NaN点
    std::cout<<"remove *cloud_tgt_o nan"<<endl;
    std::cout<<"remove *cloud_src_o nan"<<endl;

    boost::chrono::high_resolution_clock::time_point start = boost::chrono::high_resolution_clock::now();

  //compute with cpu,multi threads
    pcl::NormalEstimationOMP<PointT,pcl::Normal> nomalOMP;
    pcl::search::KdTree< PointT>::Ptr tree_nomalOMP(new pcl::search::KdTree< PointT>());
    nomalOMP.setNumberOfThreads(4);
    nomalOMP.setRadiusSearch(0.02);
    nomalOMP.setInputCloud(cloud_src_o);
    nomalOMP.setSearchMethod(tree_nomalOMP);
    pcl::PointCloud<pcl::Normal>::Ptr cloud_src_normalsOMP(new pcl::PointCloud< pcl::Normal>);
    nomalOMP.compute(*cloud_src_normalsOMP);
    nomalOMP.setInputCloud(cloud_tgt_o);
    pcl::PointCloud<pcl::Normal>::Ptr cloud_tgt_normalsOMP(new pcl::PointCloud< pcl::Normal>);
    nomalOMP.compute(*cloud_tgt_normalsOMP);
/*
    pcl::NormalEstimation<PointT,pcl::Normal> nomal;
    pcl::search::KdTree< PointT>::Ptr tree_nomal(new pcl::search::KdTree< PointT>());
    nomal.setRadiusSearch(0.02);
    nomal.setInputCloud(cloud_src_o);
    nomal.setSearchMethod(tree_nomal);
    pcl::PointCloud<pcl::Normal>::Ptr cloud_src_normals(new pcl::PointCloud< pcl::Normal>);
    nomal.compute(*cloud_src_normals);
*/
    boost::chrono::high_resolution_clock::time_point normal_time = boost::chrono::high_resolution_clock::now();
    pcl::FPFHEstimationOMP<PointT,pcl::Normal,pcl::FPFHSignature33> fpfh_est;
    pcl::search::KdTree<PointT>::Ptr tree_fpfh (new pcl::search::KdTree<PointT>);
    fpfh_est.setNumberOfThreads(4);
    fpfh_est.setRadiusSearch(0.04);
    fpfh_est.setSearchMethod(tree_fpfh);
    fpfh_est.setInputCloud(cloud_src_o);
    fpfh_est.setInputNormals(cloud_src_normalsOMP);
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfhs_src(new pcl::PointCloud<pcl::FPFHSignature33>());
    fpfh_est.compute(*fpfhs_src);
    fpfh_est.setInputCloud(cloud_tgt_o);
    fpfh_est.setInputNormals(cloud_tgt_normalsOMP);
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfhs_tgt(new pcl::PointCloud<pcl::FPFHSignature33>());
    fpfh_est.compute(*fpfhs_tgt);

    boost::chrono::high_resolution_clock::time_point fpfh_time = boost::chrono::high_resolution_clock::now();
    boost::chrono::nanoseconds fpfh_sec = fpfh_time - normal_time;
    boost::chrono::nanoseconds normal_sec = normal_time - start;
    boost::chrono::nanoseconds total_sec = fpfh_time - start;
    cout<<"normal time: "<<normal_sec<<",fpfh time: "<<fpfh_sec<<",total time: "<<total_sec<<endl;

    visualize_pcd(cloud_src_o,cloud_tgt_o);


}
