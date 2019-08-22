// #include "Converter.h"
#include "Viewer.h"
#include "timesystem.h"
#include "utility.h"
#include "readfile.h"
#include "DataType.h"
// #include <vector>
using namespace std;
vector<IMUDATA> imu_vectors;
vector<GPSDATA> gps_vectors;
vector<WHEELDATA> wheelodo_vectors;
vector<POSEDATA> posedata_vectors;

vector<POSEDATA> groundtruth_vectors;
vector<ERRORDATA> err_vectors;
map<int,TrafficSign3d> all_sign_t3d;
map<int,Line3d> all_line_t3d;
map<int,LaneStage3d> temp_lane_t3d ;
map<int,Lane3d> all_lane_t3d;
Eigen::Vector3d g_ecef_offset;
map<int,Lane3dPts> all_lane_pts;
// map<int, LaneShow> all_lane_pts;
// std::vector<LaneShow> all_lane_pts;
Eigen::Vector3d g_map_offset(39.86966052, 116.17656364, 62.41314815);
vector<string> ListDir(string src_dir) 
{
  vector<string> files;
  DIR *dir;
  struct dirent *ptr;

  if ((dir = opendir(src_dir.c_str())) == NULL) 
  {
    std::cout << " Open dir error ...: " << src_dir << endl;
    exit(1);
  }
  while ((ptr = readdir(dir)) != NULL) 
  {
    if (strcmp(ptr->d_name, ".") == 0 ||
        strcmp(ptr->d_name, "..") == 0) 
    { // cur  or parent
      continue;
    }
    string temp_name = ptr->d_name;
    int jpg_pos;
    int png_pos;
    int jpeg_pos;
    // int str_length = temp_name.size();
    jpg_pos = temp_name.rfind(".jpg");
    png_pos = temp_name.rfind(".png");
    jpeg_pos = temp_name.rfind(".jpeg");
    // std::cout << "jpg_pos " << jpg_pos << " png_pos " << png_pos  << " jpeg_pos " << jpeg_pos << std::endl;
    if(jpg_pos > 0 || png_pos > 0 || jpeg_pos > 0)
    {
      files.push_back(temp_name);
    }
    else
    {
      continue;
    } 
    // if() 
    // else if (ptr->d_type == 8) 
    // { // file
    //   files.push_back(ptr->d_name);
    // } 
    // else if (ptr->d_type == 10) 
    // { // link file
    //   continue;
    // } 
    // else if (ptr->d_type == 4) 
    // { // dir
    //   // files.push_back(ptr->d_name);
    //   continue;
    // }
  }

  closedir(dir);
  sort(files.begin(), files.end());
  return files;
}
void LoadImages(const string &image_path, vector<string> &images,
                vector<double> &timestamps) 
{
  vector<string> image_names = ListDir(image_path);
  // std::cout << " read success size " << image_names.size() << std::endl;
  for (size_t i = 0; i < image_names.size(); i++) 
  {
    string name = image_names[i];
    images.push_back(image_path + image_names[i]);
    std::stringstream line_stream(name);
    string cell;
    vector<string> result;
    while (std::getline(line_stream, cell, '.')) 
    {
      result.push_back(cell);
    }
    string str_timestamp = result[0];
    stringstream ss;
    double dtime;
    ss << str_timestamp;
    ss >> dtime;
    timestamps.push_back(dtime * 1e-6);
  }
}
void read_imu(const string &imu_path_) 
{
  string imu_path = imu_path_;//"/media/caijuan/cjFile/Work/Data/0930/standard_IMU.simu";

  std::ifstream f_imus;
  f_imus.open(imu_path.c_str());
  if (!f_imus.is_open()) 
  {
    std::cout << "not found " << imu_path << std::endl;
  }
  while (!f_imus.eof()) 
  {
    std::string s;
    std::getline(f_imus, s);
    if (!s.empty()) 
    {
      std::stringstream ss;
      ss << s;
      string str;
      int cnt = 0;
      double data[30];
      while (ss >> str) 
      {
        data[cnt] = std::stod(str);
        if (ss.peek() == ',' || ss.peek() == ' ') 
        {
          ss.ignore();
        }
        cnt++;
      }
      double time_stamp = data[0] * 1e-6;
      double acc_x = data[1] * 9.81012;
      double acc_y = data[2] * 9.81012;
      double acc_z = data[3] * 9.81012;
      double gyr_x = data[4] * M_PI / 180;
      double gyr_y = data[5] * M_PI / 180;
      double gyr_z = data[6] * M_PI / 180;
      Eigen::Vector3d acc, gyr;
      acc = Eigen::Vector3d(acc_x, acc_y, acc_z);
      gyr = Eigen::Vector3d(gyr_x, gyr_y, gyr_z);
      IMUDATA current_imu;
      current_imu.acc = acc;
      current_imu.gyro = gyr;
      current_imu.timestamp = time_stamp;
      // aviewer.SetImuData(acc,gyr);
      imu_vectors.push_back(current_imu);
    }
  }
}
#if 0
void read_pose(const string &pose_path_) 
{
  string pose_path = pose_path_;//"/media/caijuan/cjFile/Work/Data/0930/camera_pose.txt";

  std::ifstream f_pos;
  f_pos.open(pose_path.c_str());
  if (!f_pos.is_open()) 
  {
    std::cout << "not found " << pose_path << std::endl;
  }
  while (!f_pos.eof()) 
  {
    std::string s;
    std::getline(f_pos, s);
    if (!s.empty()) 
    {
      std::stringstream ss;
      ss << s;
      string str;
      int cnt = 0;
      double data[9];
      while (ss >> str) 
      {
        data[cnt] = std::stod(str);
        if (ss.peek() == ',' || ss.peek() == ' ') 
        {
          ss.ignore();
        }
        cnt++;
      }
      //         << time_stamp << " " << std::setprecision(9) << t(0) << " " <<
      //         t(1) << " " << t(2) << " "
      //   << q.w() << " " << q.x() << " " << q.y() << " " << q.z() << "\n";
      double time_stamp = data[0];
      double tx = data[1];
      double ty = data[2];
      double tz = data[3];
      double qw = data[4];
      double qx = data[5];
      double qy = data[6];
      double qz = data[7];
      POSEDATA currentpose;
      currentpose.timestamp = time_stamp;

      cv::Mat current_Twc = cv::Mat::eye(4, 4, CV_32F); /// (4,4);
      Eigen::Quaterniond q(qw, qx, qy, qz);
      Eigen::Matrix3d currentR=q.toRotationMatrix();
      current_Twc = Converter::toCvSE3(currentR, Eigen::Vector3d(tx, ty, tz));
      currentpose.pose = current_Twc.clone();
      posedata_vectors.push_back(currentpose);
    }
  }
}
#else
void read_pose(const string &pose_path_) 
{
  string pose_path = pose_path_;//"/media/caijuan/cjFile/Work/Data/0930/camera_pose.txt";

  std::ifstream f_pos;
  f_pos.open(pose_path.c_str());
  if (!f_pos.is_open()) 
  {
    std::cout << "not found " << pose_path << std::endl;
  }
  while (!f_pos.eof()) 
  {
    std::string s;
    std::getline(f_pos, s);
    if (!s.empty()) 
    {
      std::stringstream ss;
      ss << s;
      string str;
      int cnt = 0;
      double data[15];
      while (ss >> str) 
      {
        data[cnt] = std::stod(str);
        if (ss.peek() == ',' || ss.peek() == ' ') 
        {
          ss.ignore();
        }
        cnt++;
      }
      //         << time_stamp << " " << std::setprecision(9) << t(0) << " " <<
      //         t(1) << " " << t(2) << " "
      //   << q.w() << " " << q.x() << " " << q.y() << " " << q.z() << "\n";
      // double time_stamp = data[0];
      // double tx = data[1];
      // double ty = data[2];
      // double tz = data[3];
      // double qw = data[4];
      // double qx = data[5];
      // double qy = data[6];
      // double qz = data[7];
      double r = data[9];
      double p = data[10];
      double y = data[11];

      y = -y;
      if(y < -180)
      {
        y = y + 360;
      }
      // std::cout << " ypr " << y << " " << p << " " << r << std::endl;

      double tx = data[2];
      double ty = data[3];
      double tz = data[4];

      double time_stamp = data[12] * 1e-6;
      POSEDATA currentpose;
      currentpose.timestamp = time_stamp;

      // cv::Mat current_Twc = cv::Mat::eye(4, 4, CV_64F); /// (4,4);
      Eigen::Matrix4d current_Twc = Eigen::Matrix4d::Identity();
      // Eigen::Quaterniond q(qw, qx, qy, qz);
      // Eigen::Matrix3d currentR=q.toRotationMatrix();
      Eigen::Matrix3d currentR = Utility::ypr2R(Eigen::Vector3d(y,p,r));
      // current_Twc = Converter::toCvSE3(currentR, Eigen::Vector3d(tx, ty, tz));
      current_Twc.block<3,3>(0,0) = currentR;
      current_Twc.block<3,1>(0,3) = Eigen::Vector3d(tx, ty, tz);
      // std::cout << "read file pose " << tx << " " << ty << " " << tz << " ypr:" << Eigen::Vector3d(y,p,r).transpose()<< std::endl;
      currentpose.pose = current_Twc;
      posedata_vectors.push_back(currentpose);
    }
  }
}
#endif
void read_wheel(const string &wheel_path_) 
{
  string wheel_path = wheel_path_;//"/media/caijuan/cjFile/Work/Data/0930/standard_ODO.sodo";
  std::ifstream file;
  file.open(wheel_path.c_str());
  if (!file.is_open()) 
  {
    std::cout << wheel_path << " not found!!";
  }
  size_t last_t = 0;
//   double last_left_wheel = 0.0, last_right_wheel = 0.0;
  while (!file.eof()) 
  {
    std::string s;
    std::getline(file, s);
    if (!s.empty()) 
    {
      std::stringstream ss;
      ss << s;
      string str;
      int cnt = 0;
      double data[5];
      while (ss >> str) 
      {
        data[cnt] = std::stod(str);
        if (ss.peek() == ',' || ss.peek() == ' ')
          ss.ignore();
        cnt++;
      }

      double left_wheel = data[3];
    //   double right_wheel = data[4];
      last_t = (size_t)(data[0] * 1e-6);
    //   last_left_wheel = left_wheel;
    //   last_right_wheel = right_wheel;
      WHEELDATA wheeldata;
      wheeldata.wheeldata = left_wheel;
      wheeldata.timestamp = last_t;
      wheelodo_vectors.push_back(wheeldata);
    }
  }
}

vector<string> Split(string &s, string &delim) 
{
  vector<string> ret;
  size_t last = 0;
  size_t index = s.find_first_of(delim, last);
  while (index != std::string::npos) {
    ret.push_back(s.substr(last, index - last));
    last = index + 1;
    index = s.find_first_of(delim, last);
  }
  if (index - last > 0) {
    ret.push_back(s.substr(last, index - last));
  }
  return ret;
}

void read_gps(const string &gps_path_) 
{
  string gps_path =gps_path_;
    //   "/media/caijuan/cjFile/Work/Data/0930/standard_UM4B0_recorder.sgnss";
  std::ifstream file;
//   int gps_cnt = 0;
  file.open(gps_path.c_str());

  if (!file.is_open()) 
  {
    std::cout << gps_path << " not found!!" << std::endl;
  }

  while (!file.eof()) 
  {
    std::string s;
    std::getline(file, s);
    if (!s.empty()) 
    {
    //   char c = s.at(0);
      // if (c < '0' || c > '9') continue;

      std::stringstream ss;
      ss << s;
      string str;
      int cnt = 0;
      double data[30];
      while (ss >> str) 
      {
        data[cnt] = std::stod(str);
        if (ss.peek() == ',' || ss.peek() == ' ')
          ss.ignore();
        cnt++;
      }

      double lattitude = data[2];
      double longitude = data[3];
      double altitude = data[4];
    //   int fix_type = data[5];
      // double time_stamp = data[7] * 1e-6 + g_image_delay_for_imu;
      double time_stamp = GPSWS2Unix(data[0], data[1]) * 1e-6;
    //   int satelnum = data[8];
    //   double confidence = data[9];
      double velE = data[14];
      double velN = data[15];
      double velU = data[16];
      double std_lat = data[17];
      double std_lon = data[18];
      double std_alt = data[19];
      double std_vE = data[20];
      double std_vN = data[21];
      double std_vU = data[22];
      // LOG(ERROR) << std::fixed << time_stamp << " " << lattitude << " " <<
      // longitude << " " << altitude << " " << satelnum << " " << confidence;

      double gps_pos_noise =
          sqrt(std_lat * std_lat + std_lon * std_lon + std_alt * std_alt) *
          0.577;
      double gps_vel_noise =
          sqrt(std_vE * std_vE + std_vN * std_vN + std_vU * std_vU) * 0.577;

      Eigen::Vector3d pos_wb{
          lattitude, longitude,
          altitude}; 
      Eigen::Vector3d vel_wb{velE, velN, velU};
      if (gps_pos_noise < 1e-5)
        gps_pos_noise = 1e5;
      if (gps_vel_noise < 1e-5)
        gps_vel_noise = 1e5;
      GPSDATA current_gps;
      current_gps.pos_wb_ = pos_wb;
      current_gps.vel_wb_ = vel_wb;
      current_gps.timestamp = time_stamp;
      gps_vectors.push_back(current_gps);
    }
  }
  // }
}


void read_error(const string &error_path_) 
{
  string err_path = error_path_;//"/media/caijuan/cjFile/Work/Data/0930/window_error.txt";
  std::ifstream file;
//   int err_cnt = 0;
  file.open(err_path.c_str());

  if (!file.is_open()) 
  {
    std::cout << err_path << " not found!!" << std::endl;
  }

  while (!file.eof()) {

    std::string s;
    std::getline(file, s);
    if (!s.empty()) {
      std::stringstream ss;
      ss << s;
      string str;
      int cnt = 0;
      double data[12];
      while (ss >> str) {
        data[cnt] = std::stod(str);
        if (ss.peek() == ',' || ss.peek() == ' ')
          ss.ignore();
        cnt++;
      }

      double timestamp = data[0] * 1e-3;
      double err_x = data[3];
      double err_y = data[4];
      double err_z = data[5];
      ERRORDATA current_error;
      current_error.errdata = Eigen::Vector3d(err_x, err_y, err_z);
      current_error.timestamp = timestamp;
      err_vectors.push_back(current_error);
    }
  }
}

void ReadMapEcef(const string map_path)
{
    string map_name = map_path;// + "osm_map.json";
    HDMap HdMap(map_name,"json");
    all_sign_t3d = HdMap.GetMapIntTrafficsign3ds();
    all_line_t3d = HdMap.GetMapIntLine3ds();
    temp_lane_t3d = HdMap.GetMapIntLanestage3d();
    g_ecef_offset = HdMap.GetOffsetUtm();
    // all_lane_t3d = temp_lane_t3d.begin().lane3ds;

    for(std::map<int,LaneStage3d> ::iterator iter = temp_lane_t3d.begin(); iter!=temp_lane_t3d.end();iter++)
    {
      LaneStage3d temp = iter->second;
      std::map<int, Lane3d> lane3ds = temp.lane3ds;
      for(std::map<int , Lane3d>:: iterator iterlane = lane3ds.begin();iterlane != lane3ds.end(); iterlane++)
      {
        Pt3ds temp_lane_pts;
        for(int i = 0,iend = iterlane->second.seg3ds.size();i < iend; i++)
        {
          Seg3d seg3d = iterlane->second.seg3ds[i];
          double t_step = 1.0;
          for (int t = 0; t < seg3d.h; t += t_step)
          {
            Eigen::Matrix<double, 3, 1> pt;
            pt(0) = seg3d.parameters(0, 3) * pow(t, 3) +
                    seg3d.parameters(0, 2) * pow(t, 2) + seg3d.parameters(0, 1) * t +
                    seg3d.parameters(0, 0);
            pt(1) = seg3d.parameters(1, 3) * pow(t, 3) +
                    seg3d.parameters(1, 2) * pow(t, 2) + seg3d.parameters(1, 1) * t +
                    seg3d.parameters(1, 0);
            pt(2) = seg3d.parameters(2, 3) * pow(t, 3) +
                    seg3d.parameters(2, 2) * pow(t, 2) + seg3d.parameters(2, 1) * t +
                    seg3d.parameters(2, 0);
            temp_lane_pts.push_back(pt);
          }      
        }
        // LaneShow temp_lane;
        // temp_lane.type = iterlane->second.type;
        // temp_lane.lane_id = iterlane->second.id;
        // temp_lane.lan3pts = temp_lane_pts;
        // all_lane_pts.push_back(temp_lane);
        all_lane_pts[iterlane->second.id].pts = temp_lane_pts;
      }
    }
    
}

void read_groundtruth_data(const string &groundtruth_path)
{
  string pose_path = groundtruth_path;//"/media/caijuan/cjFile/Work/Data/0930/camera_pose.txt";

  std::ifstream f_pos;
  f_pos.open(pose_path.c_str());
  if (!f_pos.is_open()) 
  {
    std::cout << "not found " << pose_path << std::endl;
  }
  while (!f_pos.eof()) 
  {
    std::string s;
    std::getline(f_pos, s);
    if (!s.empty()) 
    {
      std::stringstream ss;
      ss << s;
      string str;
      int cnt = 0;
      double data[15];
      while (ss >> str) 
      {
        data[cnt] = std::stod(str);
        if (ss.peek() == ',' || ss.peek() == ' ') 
        {
          ss.ignore();
        }
        cnt++;
      }
      //         << time_stamp << " " << std::setprecision(9) << t(0) << " " <<
      //         t(1) << " " << t(2) << " "
      //   << q.w() << " " << q.x() << " " << q.y() << " " << q.z() << "\n";
      // double time_stamp = data[0];
      // double tx = data[1];
      // double ty = data[2];
      // double tz = data[3];
      // double qw = data[4];
      // double qx = data[5];
      // double qy = data[6];
      // double qz = data[7];
      double r = data[9];
      double p = data[10];
      double y = data[11];

      y = -y;
      if(y < -180)
      {
        y = y + 360;
      }
      // std::cout << " ypr " << y << " " << p << " " << r << std::endl;

      double tx = data[2];
      double ty = data[3];
      double tz = data[4];

      double time_stamp = data[12] * 1e-6;
      POSEDATA currentpose;
      currentpose.timestamp = time_stamp;

      // cv::Mat current_Twc = cv::Mat::eye(4, 4, CV_64F); /// (4,4);
      Eigen::Matrix4d current_Twc = Eigen::Matrix4d::Identity();
      // Eigen::Quaterniond q(qw, qx, qy, qz);
      // Eigen::Matrix3d currentR=q.toRotationMatrix();
      Eigen::Matrix3d currentR = Utility::ypr2R(Eigen::Vector3d(y,p,r));
      // current_Twc = Converter::toCvSE3(currentR, Eigen::Vector3d(tx, ty, tz));
      current_Twc.block<3,3>(0,0) = currentR;
      current_Twc.block<3,1>(0,3) = Eigen::Vector3d(tx, ty, tz);
      // std::cout << "read file pose " << tx << " " << ty << " " << tz << " ypr:" << Eigen::Vector3d(y,p,r).transpose()<< std::endl;
      currentpose.pose = current_Twc;
      groundtruth_vectors.push_back(currentpose);
    }
  }
}