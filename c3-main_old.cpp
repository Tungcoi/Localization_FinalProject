
#include <carla/client/Client.h>
#include <carla/client/ActorBlueprint.h>
#include <carla/client/BlueprintLibrary.h>
#include <carla/client/Map.h>
#include <carla/geom/Location.h>
#include <carla/geom/Transform.h>
#include <carla/client/Sensor.h>
#include <carla/sensor/data/LidarMeasurement.h>
#include <thread>

#include <carla/client/Vehicle.h>

//pcl code
//#include "render/render.h"

namespace cc = carla::client;
namespace cg = carla::geom;
namespace csd = carla::sensor::data;

using namespace std::chrono_literals;
using namespace std::string_literals;

using namespace std;

#include <string>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/voxel_grid.h>
#include "helper.h"
#include <sstream>
#include <chrono>
#include <ctime>
#include <pcl/registration/icp.h>
#include <pcl/registration/ndt.h>
#include <pcl/console/time.h>   // TicToc

PointCloudT pclCloud;
cc::Vehicle::Control control;
std::chrono::time_point<std::chrono::system_clock> currentTime;
vector<ControlState> cs;

bool refresh_view = false;
const bool USE_ICP = true;
const bool USE_SAMPLE = false;
void keyboardEventOccurred(const pcl::visualization::KeyboardEvent &event, void* viewer)
{

      //boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer = *static_cast<boost::shared_ptr<pcl::visualization::PCLVisualizer> *>(viewer_void);
    if (event.getKeySym() == "Right" && event.keyDown()){
        cs.push_back(ControlState(0, -0.02, 0));
      }
    else if (event.getKeySym() == "Left" && event.keyDown()){
        cs.push_back(ControlState(0, 0.02, 0));
      }
      if (event.getKeySym() == "Up" && event.keyDown()){
        cs.push_back(ControlState(0.1, 0, 0));
      }
    else if (event.getKeySym() == "Down" && event.keyDown()){
        cs.push_back(ControlState(-0.1, 0, 0));
      }
    if(event.getKeySym() == "a" && event.keyDown()){
        refresh_view = true;
    }
}

void Accuate(ControlState response, cc::Vehicle::Control& state){

    if(response.t > 0){
        if(!state.reverse){
            state.throttle = min(state.throttle+response.t, 1.0f);
        }
        else{
            state.reverse = false;
            state.throttle = min(response.t, 1.0f);
        }
    }
    else if(response.t < 0){
        response.t = -response.t;
        if(state.reverse){
            state.throttle = min(state.throttle+response.t, 1.0f);
        }
        else{
            state.reverse = true;
            state.throttle = min(response.t, 1.0f);

        }
    }
    state.steer = min( max(state.steer+response.s, -1.0f), 1.0f);
    state.brake = response.b;
}


// Hàm xử lý ICP
Eigen::Matrix4d ICP(PointCloudT::Ptr target, PointCloudT::Ptr source, Pose startingPose, int iterations) {
    pcl::console::TicToc time;
    time.tic();

     // Log Pose khởi tạo
    std::cout << "Initial Pose (startingPose): " << std::endl;
    std::cout << "x: " << startingPose.position.x << " y: " << startingPose.position.y << " z: " << startingPose.position.z << std::endl;
    std::cout << "yaw: " << startingPose.rotation.yaw << " pitch: " << startingPose.rotation.pitch << " roll: " << startingPose.rotation.roll << std::endl;


    Eigen::Matrix4d transformation_matrix = Eigen::Matrix4d::Identity();
    Eigen::Matrix4d starting_pose_transform = transform3D(
        startingPose.rotation.yaw, startingPose.rotation.pitch, startingPose.rotation.roll,
        startingPose.position.x, startingPose.position.y, startingPose.position.z
    );
    PointCloudT::Ptr source_transformed(new PointCloudT);
    pcl::transformPointCloud(*source, *source_transformed, starting_pose_transform);

    pcl::IterativeClosestPoint<PointT, PointT> icp;
    icp.setInputSource(source_transformed);
    icp.setInputTarget(target);
    icp.setMaxCorrespondenceDistance(2.0);
    icp.setMaximumIterations(iterations);
    // icp.setTransformationEpsilon(1e-4);
    
    // icp.setEuclideanFitnessEpsilon(0.05); //2
    // icp.setRANSACOutlierRejectionThreshold(0.1); //0.2


    // Log các tham số ICP
    std::cout << "Max Correspondence Distance: " << icp.getMaxCorrespondenceDistance() << std::endl;
    std::cout << "Maximum Iterations: " << icp.getMaximumIterations() << std::endl;
    std::cout << "Transformation Epsilon: " << icp.getTransformationEpsilon() << std::endl;
    std::cout << "Euclidean Fitness Epsilon: " << icp.getEuclideanFitnessEpsilon() << std::endl;
    std::cout << "RANSAC Outlier Rejection Threshold: 0.1" << std::endl;


    PointCloudT::Ptr cloud_icp(new PointCloudT);
    icp.align(*cloud_icp);
    std::cout << "Finished ICP alignment in " << time.toc() << " ms" << "\n";
    std::cout << "ICP converged: " << std::boolalpha << icp.hasConverged();
    std::cout << ", Fitness score: " << icp.getFitnessScore() << "\n";
    // Log số lượng điểm bị loại bỏ bởi RANSAC (nếu có)
    std::cout << "Number of inliers (points used in alignment): " << icp.getFinalTransformation().rows() << std::endl;
    if (icp.hasConverged()) {
        transformation_matrix = icp.getFinalTransformation().cast<double>();
        transformation_matrix = transformation_matrix * starting_pose_transform;

        std::cout << "Transformation Matrix after ICP: " << std::endl;
        std::cout << transformation_matrix << std::endl;

        std::cout << "Number of points in source cloud: " << source->points.size() << std::endl;
        std::cout << "Number of points in target cloud: " << target->points.size() << std::endl;

        // Lấy số lượng điểm trùng khớp (correspondences)
        std::vector<int> correspondences;
        icp.getCorrespondences(correspondences);
        std::cout << "Number of correspondences (matched points): " << correspondences.size() << std::endl;

        Pose finalPose = getPose(transformation_matrix);
        std::cout << "Final Pose after ICP: " << std::endl;
        std::cout << "x: " << finalPose.position.x << " y: " << finalPose.position.y << " z: " << finalPose.position.z << std::endl;
        std::cout << "yaw: " << finalPose.rotation.yaw << " pitch: " << finalPose.rotation.pitch << " roll: " << finalPose.rotation.roll << std::endl;
    
    } else {
        std::cout << "WARNING: ICP did not converge" << "\n";
    }

    std::cout << "------- END OF ICP FUNC -------" << "\n";
    return transformation_matrix;
}

// Hàm xử lý NDT
Eigen::Matrix4d NDT(pcl::NormalDistributionsTransform<PointT, PointT>& ndt, PointCloudT::Ptr source, Pose startingPose, int iterations) {
    pcl::console::TicToc time;
    time.tic();

    Eigen::Matrix4d transformation_matrix = Eigen::Matrix4d::Identity();
    Eigen::Matrix4f init_guess = transform3D(
        startingPose.rotation.yaw, startingPose.rotation.pitch, startingPose.rotation.roll,
        startingPose.position.x, startingPose.position.y, startingPose.position.z
    ).cast<float>();

    ndt.setMaximumIterations(iterations);
    ndt.setInputSource(source);
    PointCloudT::Ptr output_cloud(new PointCloudT);
    ndt.align(*output_cloud, init_guess);

    std::cout << "Finished NDT alignment in " << time.toc() << " ms" << "\n";
    std::cout << "NDT converged: " << std::boolalpha << ndt.hasConverged();
    std::cout << ", Fitness score: " << ndt.getFitnessScore() << "\n";
    if (ndt.hasConverged()) {
        transformation_matrix = ndt.getFinalTransformation().cast<double>();
    } else {
        std::cout << "WARNING: NDT did not converge" << "\n";
    }
    return transformation_matrix;
}

void drawCar(Pose pose, int num, Color color, double alpha, pcl::visualization::PCLVisualizer::Ptr& viewer){

    BoxQ box;
    box.bboxTransform = Eigen::Vector3f(pose.position.x, pose.position.y, 0);
    box.bboxQuaternion = getQuaternion(pose.rotation.yaw);
    box.cube_length = 4;
    box.cube_width = 2;
    box.cube_height = 2;
    renderBox(viewer, box, num, color, alpha);
}

int main(){

    auto client = cc::Client("localhost", 2000);
    client.SetTimeout(2s);
    auto world = client.GetWorld();

    auto blueprint_library = world.GetBlueprintLibrary();
    auto vehicles = blueprint_library->Filter("vehicle");

    auto map = world.GetMap();
    auto transform = map->GetRecommendedSpawnPoints()[1];
    auto ego_actor = world.SpawnActor((*vehicles)[12], transform);

    //Create lidar
    auto lidar_bp = *(blueprint_library->Find("sensor.lidar.ray_cast"));
    // CANDO: Can modify lidar values to get different scan resolutions
    lidar_bp.SetAttribute("upper_fov", "15");
    lidar_bp.SetAttribute("lower_fov", "-25");
    lidar_bp.SetAttribute("channels", "32");
    lidar_bp.SetAttribute("range", "30");
    lidar_bp.SetAttribute("rotation_frequency", "60");
    lidar_bp.SetAttribute("points_per_second", "500000");

    auto user_offset = cg::Location(0, 0, 0);
    auto lidar_transform = cg::Transform(cg::Location(-0.5, 0, 1.8) + user_offset);
    auto lidar_actor = world.SpawnActor(lidar_bp, lidar_transform, ego_actor.get());
    auto lidar = boost::static_pointer_cast<cc::Sensor>(lidar_actor);
    bool new_scan = true;
    std::chrono::time_point<std::chrono::system_clock> lastScanTime, startTime;

    pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
    viewer->setBackgroundColor (0, 0, 0);
    viewer->registerKeyboardCallback(keyboardEventOccurred, (void*)&viewer);

    auto vehicle = boost::static_pointer_cast<cc::Vehicle>(ego_actor);
    Pose pose(Point(0,0,0), Rotate(0,0,0));
    //Pose pose(Point(vehicle->GetTransform().location.x, vehicle->GetTransform().location.y, vehicle->GetTransform().location.z), Rotate(vehicle->GetTransform().rotation.yaw * pi/180, vehicle->GetTransform().rotation.pitch * pi/180, vehicle->GetTransform().rotation.roll * pi/180));

    // Load map
    PointCloudT::Ptr mapCloud(new PointCloudT);
      pcl::io::loadPCDFile("map.pcd", *mapCloud);
      cout << "Loaded " << mapCloud->points.size() << " data points from map.pcd" << endl;
    renderPointCloud(viewer, mapCloud, "map", Color(0,0,1));

    typename pcl::PointCloud<PointT>::Ptr cloudFiltered (new pcl::PointCloud<PointT>);
    typename pcl::PointCloud<PointT>::Ptr scanCloud (new pcl::PointCloud<PointT>);

    pcl::NormalDistributionsTransform<PointT, PointT> ndt;
    if (USE_SAMPLE)
        ndt.setTransformationEpsilon(1e-4);
    else
        ndt.setTransformationEpsilon(0.01);
    ndt.setStepSize(0.1);
    ndt.setResolution(1.0);
    ndt.setInputTarget(mapCloud);


    lidar->Listen([&new_scan, &lastScanTime, &scanCloud](auto data){

        if(new_scan){
            auto scan = boost::static_pointer_cast<csd::LidarMeasurement>(data);
            for (auto detection : *scan){
                if((detection.x*detection.x + detection.y*detection.y + detection.z*detection.z) > 8.0){
                    pclCloud.points.push_back(PointT(detection.x, detection.y, detection.z));
                }
            }
            if(pclCloud.points.size() > 5000){ // CANDO: Can modify this value to get different scan resolutions
                lastScanTime = std::chrono::system_clock::now();
                *scanCloud = pclCloud;
                new_scan = false;
            }
        }
    });

    Pose poseRef(Point(vehicle->GetTransform().location.x, vehicle->GetTransform().location.y, vehicle->GetTransform().location.z), Rotate(vehicle->GetTransform().rotation.yaw * pi/180, vehicle->GetTransform().rotation.pitch * pi/180, vehicle->GetTransform().rotation.roll * pi/180));
    double maxError = 0;

    while (!viewer->wasStopped())
      {
        while(new_scan){
            std::this_thread::sleep_for(0.1s);
            world.Tick(1s);
        }
        if(refresh_view){
            viewer->setCameraPosition(pose.position.x, pose.position.y, 60, pose.position.x+1, pose.position.y+1, 0, 0, 0, 1);
            refresh_view = false;
        }

        viewer->removeShape("box0");
        viewer->removeShape("boxFill0");
        Pose truePose = Pose(Point(vehicle->GetTransform().location.x, vehicle->GetTransform().location.y, vehicle->GetTransform().location.z), Rotate(vehicle->GetTransform().rotation.yaw * pi/180, vehicle->GetTransform().rotation.pitch * pi/180, vehicle->GetTransform().rotation.roll * pi/180)) - poseRef;
        drawCar(truePose, 0,  Color(1,0,0), 0.7, viewer);
        double theta = truePose.rotation.yaw;
        double stheta = control.steer * pi/4 + theta;
        viewer->removeShape("steer");
        renderRay(viewer, Point(truePose.position.x+2*cos(theta), truePose.position.y+2*sin(theta),truePose.position.z),  Point(truePose.position.x+4*cos(stheta), truePose.position.y+4*sin(stheta),truePose.position.z), "steer", Color(0,1,0));


        ControlState accuate(0, 0, 1);
        if(cs.size() > 0){
            accuate = cs.back();
            cs.clear();

            Accuate(accuate, control);
            vehicle->ApplyControl(control);
        }

          viewer->spinOnce();

        if(!new_scan){

            // Tính toán độ chênh lệch ban đầu giữa pose khởi tạo và pose thực tế
            double delta_x = pose.position.x - truePose.position.x;
            double delta_y = pose.position.y - truePose.position.y;
            double delta_z = pose.position.z - truePose.position.z;

            // Tính khoảng cách Euclidean
            double distance = sqrt(delta_x * delta_x + delta_y * delta_y + delta_z * delta_z);

            // Tính sự chênh lệch về góc (yaw, pitch, roll)
            double delta_yaw = pose.rotation.yaw - truePose.rotation.yaw;
            double delta_pitch = pose.rotation.pitch - truePose.rotation.pitch;
            double delta_roll = pose.rotation.roll - truePose.rotation.roll;

            // Log các giá trị này để xem độ chênh lệch ban đầu
            std::cout << "Initial pose vs Actual pose differences:" << std::endl;
            std::cout << "Position difference: (" << delta_x << ", " << delta_y << ", " << delta_z << "), Euclidean distance: " << distance << std::endl;
            std::cout << "Rotation difference: yaw = " << delta_yaw << ", pitch = " << delta_pitch << ", roll = " << delta_roll << std::endl;
            
            
            new_scan = true;
            // TODO: (Filter scan using voxel filter)
            std::cout << "Number of points before voxel filter: " << scanCloud->points.size() << std::endl;
            std::cout << "Voxel Grid Leaf Size: 1.0f" << std::endl;  // Hoặc giá trị mà bạn đang sử dụng
            pcl::VoxelGrid<PointT> voxelGrid;
            voxelGrid.setInputCloud(scanCloud);
            double filterRes = 1.0f; //resoultion
            voxelGrid.setLeafSize(filterRes, filterRes, filterRes);
            voxelGrid.filter(*cloudFiltered);
            std::cout << "Number of points after voxel filter: " << cloudFiltered->points.size() << std::endl;
            // Log số lượng điểm bị loại bỏ trong quá trình lọc
            int removedPoints = scanCloud->points.size() - cloudFiltered->points.size();
            std::cout << "Number of points removed by voxel filter: " << removedPoints << std::endl;

            // TODO: Find pose transform by using ICP or NDT matching
            //pose = ....
            // Tính toán ma trận biến đổi giữa đám mây điểm đã lọc và bản đồ
            Eigen::Matrix4d transformMatrix;
            int maxIteration = 100;

            //cout<<"$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$d"<<endl;
            //pose.Print();
            if (USE_ICP) {
                if (USE_SAMPLE)
                    maxIteration = 120;
                transformMatrix = ICP(mapCloud, cloudFiltered, pose, maxIteration);
            } else {
                if (USE_SAMPLE)
                    maxIteration = 95;
                transformMatrix = NDT(ndt, cloudFiltered, pose, maxIteration);
            }
            pose = getPose(transformMatrix); // Cập nhật vị trí của xe
            //pose.Print();
            truePose.Print();
            cout<<"$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$d"<<endl;
            // TODO: Transform scan so it aligns with ego's actual pose and render that scan

            // TODO: Change `scanCloud` below to your transformed scan
            // Biến đổi và hiển thị đám mây điểm căn chỉnh
            PointCloudT::Ptr alignedCloud(new PointCloudT);
            pcl::transformPointCloud(*cloudFiltered, *alignedCloud, transformMatrix);
            viewer->removePointCloud("aligned");
            renderPointCloud(viewer, alignedCloud, "aligned", Color(1, 0, 0));
            viewer->removePointCloud("scan");
            renderPointCloud(viewer, scanCloud, "scan", Color(0, 1, 0));

            viewer->removeAllShapes();
            drawCar(pose, 1,  Color(0,1,0), 0.35, viewer);

              double poseError = sqrt( (truePose.position.x - pose.position.x) * (truePose.position.x - pose.position.x) + (truePose.position.y - pose.position.y) * (truePose.position.y - pose.position.y) );
            if(poseError > maxError)
                maxError = poseError;
            double distDriven = sqrt( (truePose.position.x) * (truePose.position.x) + (truePose.position.y) * (truePose.position.y) );
            viewer->removeShape("maxE");
            viewer->addText("Max Error: "+to_string(maxError)+" m", 200, 100, 32, 1.0, 1.0, 1.0, "maxE",0);
            viewer->removeShape("derror");
            viewer->addText("Pose error: "+to_string(poseError)+" m", 200, 150, 32, 1.0, 1.0, 1.0, "derror",0);
            viewer->removeShape("dist");
            viewer->addText("Distance: "+to_string(distDriven)+" m", 200, 200, 32, 1.0, 1.0, 1.0, "dist",0);

            if(maxError > 1.2 || distDriven >= 170.0 ){
                viewer->removeShape("eval");
            if(maxError > 1.2){
                viewer->addText("Try Again", 200, 50, 32, 1.0, 0.0, 0.0, "eval",0);
            }
            else{
                viewer->addText("Passed!", 200, 50, 32, 0.0, 1.0, 0.0, "eval",0);
            }
        }

            pclCloud.points.clear();
        }
      }
    return 0;
}