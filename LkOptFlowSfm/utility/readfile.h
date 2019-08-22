#pragma once
#include <vector>
using namespace std;

// extern Eigen::Vector3d g_map_offset(39.86966052, 116.17656364, 62.41314815);
typedef struct imudata 
{
  Eigen::Vector3d acc;
  Eigen::Vector3d gyro;
  double timestamp;
} IMUDATA;


typedef struct gpsdata 
{
  Eigen::Vector3d pos_wb_;
  Eigen::Vector3d vel_wb_;
  double timestamp;
} GPSDATA;


typedef struct wheeldata 
{
  double wheeldata;
  double timestamp;
} WHEELDATA;


typedef struct posedata 
{
  double timestamp;
  Eigen::Vector3d vel;
  // cv::Mat pose;
  Eigen::Matrix4d pose;
} POSEDATA;

typedef struct error_data 
{
  Eigen::Vector3d errdata;
  double timestamp;
} ERRORDATA;

vector<string> ListDir(string src_dir); 
void LoadImages(const string &image_path, vector<string> &images,vector<double> &timestamps); 
void read_imu(const string &imu_path_);
void read_pose(const string &pose_path_); 
void read_wheel(const string &wheel_path_); 

vector<string> Split(string &s, string &delim);
// void read_map(const string &map_path_,const string &traffic_path_); 
void read_gps(const string &gps_path_) ;
void read_error(const string &error_path_) ;
void ReadMapEcef(const string map_path);
void read_groundtruth_data(const string &groundtruth_path);

extern vector<IMUDATA> imu_vectors;
extern vector<GPSDATA> gps_vectors;
extern vector<WHEELDATA> wheelodo_vectors;
extern vector<POSEDATA> posedata_vectors;
extern vector<ERRORDATA> err_vectors;

extern map<int,TrafficSign3d> all_sign_t3d;
extern map<int,Line3d> all_line_t3d;
extern map<int,LaneStage3d> temp_lane_t3d ;
extern map<int,Lane3d> all_lane_t3d;
extern Eigen::Vector3d g_ecef_offset;
extern map<int,Lane3dPts> all_lane_pts;
extern vector<POSEDATA> groundtruth_vectors;