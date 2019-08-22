#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <pangolin/pangolin.h>
#include <dirent.h>
#include <iostream>
#include <cstdio>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <iterator>
#include <unordered_map>
#include "utility/timesystem.h"
#include <boost/container/flat_map.hpp>
//#include "utility/readfile.h"
#include "utility/tic_toc.h"

#include <glog/logging.h>
using namespace std;
using namespace cv;
using namespace Eigen;
//euroc
//int WIDTH = 752;
//int HEIGHT = 480;
//int MAX_CNT = 300;
//int MIN_DIST = 30;
//double fx = 4.616e+02;
//double fy = 4.603e+02;
//double cx = 3.630e+02;
//double cy = 2.481e+02;

//kitti
//int WIDTH = 1241;
//int HEIGHT = 376;
//int MAX_CNT = 300;
//int MIN_DIST = 30;
//double fx = 7.070912e+02;
//double fy = 7.070912e+02;
//double cx = 6.018873e+02;
//double cy = 1.831104e+02;

//real data beijing
//int WIDTH = 1920;
//int HEIGHT = 1080;
//int MAX_CNT = 300;
//int MIN_DIST = 30;
//double fx = 1113.7270515822293;
//double fy = 1120.0745528247721;
//double cx = 958.278236856668;
//double cy = 547.8965570492111;

//suzhou
//int WIDTH = 1280;
//int HEIGHT = 960;
//int MAX_CNT = 3000;
//int MIN_DIST = 5;
//double fx = 798.134;
//double fy = 799.432;
//double cx = 689.455;
//double cy = 513.364;

//ts
int WIDTH = 1280;
int HEIGHT = 720;
int MAX_CNT = 500;
int MIN_DIST = 10;
double fx = 638.985945;
double fy = 638.453925;
double cx = 639.290225;
double cy = 353.586505;

int nImgs = 20;
vector<string> vStrImageFileNames;
vector<double> vTimeStamps;

template<typename derived_t>
auto SkewSymmMatrix(const Eigen::MatrixBase<derived_t>& phi)
-> Eigen::Matrix<typename derived_t::RealScalar, 3, 3> {
    return (Eigen::Matrix<typename derived_t::RealScalar, 3, 3>()
            <<      0.0, -phi.z(),  phi.y(),
            phi.z(),      0.0, -phi.x(),
            -phi.y(),  phi.x(),      0.0).finished();
}

template<typename derived_t>
Eigen::Matrix3d RightJacobianInverse(
        const Eigen::MatrixBase<derived_t>& phi, double double_epsilon) {
    using Eigen::Matrix3d;
    Matrix3d I = Matrix3d::Identity();
    double phi_norm2 = phi.squaredNorm();
    if (phi_norm2 <= double_epsilon)
        return I + 0.5 * SkewSymmMatrix(phi);
    double phi_norm = sqrt(phi_norm2);
    double cos_phi = cos(phi_norm);
    double sin_phi = sin(phi_norm);
    Matrix3d phi_hat = SkewSymmMatrix(phi);
    Matrix3d R = I + 0.5*SkewSymmMatrix(phi) +
                      (1.0/phi_norm2+0.5*(1.0+cos_phi)/(phi_norm*sin_phi)) *
                      phi_hat*phi_hat;
    return R;
}

struct SFMFeature{
	bool state;
	int id;
	vector<pair<int,Vector2d>> observation;
	double position[3];
	double depth;
	SFMFeature(){
		state = false;
		id = -1;
		depth = -1;
	}
};

struct ReprojectionError3D
{
    ReprojectionError3D(double observed_u, double observed_v)
            :observed_u(observed_u), observed_v(observed_v)
    {}

    template <typename T>
    bool operator()(const T* const camera_R, const T* const camera_T, const T* point, T* residuals) const
    {
        T p[3];
        ceres::QuaternionRotatePoint(camera_R, point, p);
        p[0] += camera_T[0]; p[1] += camera_T[1]; p[2] += camera_T[2];
        T xp = p[0] / p[2];
        T yp = p[1] / p[2];
        residuals[0] = xp - T(observed_u);
        residuals[1] = yp - T(observed_v);
        return true;
    }

    static ceres::CostFunction* Create(const double observed_x,
                                       const double observed_y)
    {
        return (new ceres::AutoDiffCostFunction<
                ReprojectionError3D, 2, 4, 3, 3>(
                new ReprojectionError3D(observed_x,observed_y)));
    }

    double observed_u;
    double observed_v;
};

typedef struct gpsdata
{
    Eigen::Vector3d pos_wb_;
    Eigen::Vector3d vel_wb_;
    double timestamp;
} GPSDATA;

vector<GPSDATA> gps_vectors;

bool inBorder(const cv::Point2f &pt)
{
    const int BORDER_SIZE = 1;
    int img_x = cvRound(pt.x);
    int img_y = cvRound(pt.y);
    return BORDER_SIZE <= img_x && img_x < WIDTH - BORDER_SIZE && BORDER_SIZE <= img_y && img_y < HEIGHT - BORDER_SIZE;
}

void setMask(Mat& mask,vector<Point2f>& pts)
{
	for(auto& pt:pts)
	{
		// if(mask.at<uchar>(pt) == 255)
		// {
		circle(mask, pt, MIN_DIST, 0, -1);
		//}
	}
}

void reduceVector(vector<cv::Point2f> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

void reduceVector(vector<int> &v, vector<uchar> status){
	int j = 0;
	for(int i=0;i<int(v.size());i++)
		if(status[i])
			v[j++] = v[i];
	v.resize(j);
}

int nCorres(int l, int r, vector<SFMFeature>& sfm)
{
	int c = 0;
	for(auto& s:sfm)
	{
		int ll = s.observation[0].first;
		int rr = s.observation[s.observation.size() - 1].first;
		if(ll <= l && rr >= r)
			c++;
	}
	return c;
}

Point2d pixel2cam ( const Point2d& p )
{
    return Point2d
           (
               ( p.x - cx ) / fx,
               ( p.y - cy ) / fy
           );
}
Point2f cam2pixel (const Point2d& p){
    return Point2f(
            p.x * fx + cx,
            p.y * fy + cy
    );
}

vector<pair<Vector2d, Vector2d>> getCorrespondings(int l, int r, vector<SFMFeature>& sfm)
{
	vector<pair<Vector2d, Vector2d>> corres;
	for(auto& s:sfm)
	{
		int ll = s.observation[0].first;
		int rr = s.observation[s.observation.size() - 1].first;
		if(ll <= l && rr >= r)
		{
			corres.push_back(make_pair(s.observation[l-ll].second, s.observation[r-ll].second));
		}
	}
	return corres;
}

bool solveRelativeRT(const vector<pair<Vector2d, Vector2d>> &corres, Matrix3d &Rotation, Vector3d &Translation)
{
	if (corres.size() >= 15)
    {
        vector<cv::Point2f> ll, rr;
        for (int i = 0; i < int(corres.size()); i++)
        {
            ll.push_back(cv::Point2f(corres[i].first(0), corres[i].first(1)));
            rr.push_back(cv::Point2f(corres[i].second(0), corres[i].second(1)));
        }
        cv::Mat mask;
        cv::Mat E = cv::findFundamentalMat(ll, rr, cv::FM_RANSAC, 0.3 / 460, 0.99, mask);
        cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
        cv::Mat rot, trans;
        int inlier_cnt = cv::recoverPose(E, ll, rr, cameraMatrix, rot, trans, mask);
        //cout << "inlier_cnt " << inlier_cnt << endl;

        Eigen::Matrix3d R;
        Eigen::Vector3d T;
        for (int i = 0; i < 3; i++)
        {   
            T(i) = trans.at<double>(i, 0);
            for (int j = 0; j < 3; j++)
                R(i, j) = rot.at<double>(i, j);
        }

        Rotation = R.transpose();
        Translation = -R.transpose() * T;
        if(inlier_cnt > 12)
            return true;
        else
            return false;
    }
    return false;
}

void triangulatePoint(Eigen::Matrix<double, 3, 4> &Pose0, Eigen::Matrix<double, 3, 4> &Pose1,
						Vector2d &point0, Vector2d &point1, Vector3d &point_3d)
{
	Matrix4d design_matrix = Matrix4d::Zero();
	design_matrix.row(0) = point0[0] * Pose0.row(2) - Pose0.row(0);
	design_matrix.row(1) = point0[1] * Pose0.row(2) - Pose0.row(1);
	design_matrix.row(2) = point1[0] * Pose1.row(2) - Pose1.row(0);
	design_matrix.row(3) = point1[1] * Pose1.row(2) - Pose1.row(1);
	Vector4d triangulated_point;
	triangulated_point =
		      design_matrix.jacobiSvd(Eigen::ComputeFullV).matrixV().rightCols<1>();
	point_3d(0) = triangulated_point(0) / triangulated_point(3);
	point_3d(1) = triangulated_point(1) / triangulated_point(3);
	point_3d(2) = triangulated_point(2) / triangulated_point(3);
}

void triangulateTwoFrames(int frame0, Eigen::Matrix<double, 3, 4> &Pose0, 
						  int frame1, Eigen::Matrix<double, 3, 4> &Pose1,
						  vector<SFMFeature> &sfm)
{
	assert(frame0 != frame1);
	for (int j = 0; j < sfm.size(); j++)
	{
		if (sfm[j].state == true)
			continue;
		bool has_0 = false, has_1 = false;
		Vector2d point0;
		Vector2d point1;
		for (int k = 0; k < (int)sfm[j].observation.size(); k++)
		{
			if (sfm[j].observation[k].first == frame0)
			{
				point0 = sfm[j].observation[k].second;
				has_0 = true;
			}
			if (sfm[j].observation[k].first == frame1)
			{
				point1 = sfm[j].observation[k].second;
				has_1 = true;
			}
		}
		if (has_0 && has_1)
		{
			Vector3d point_3d;
			triangulatePoint(Pose0, Pose1, point0, point1, point_3d);
			sfm[j].state = true;
			sfm[j].position[0] = point_3d(0);
			sfm[j].position[1] = point_3d(1);
			sfm[j].position[2] = point_3d(2);
			//cout << "trangulated : " << frame1 << "  3d point : "  << j << "  " << point_3d.transpose() << endl;
		}							  
	}
}

bool solveFrameByPnP(Matrix3d &R_initial, Vector3d &P_initial, int i,
								vector<SFMFeature> &sfm)
{
	vector<cv::Point2f> pts_2_vector;
	vector<cv::Point3f> pts_3_vector;
	for (int j = 0; j < sfm.size(); j++)
	{
		if (sfm[j].state != true)
			continue;
		Vector2d point2d;
		for (int k = 0; k < (int)sfm[j].observation.size(); k++)
		{
			if (sfm[j].observation[k].first == i)
			{
				Vector2d img_pts = sfm[j].observation[k].second;
				cv::Point2f pts_2(img_pts(0), img_pts(1));
				pts_2_vector.push_back(pts_2);
				cv::Point3f pts_3(sfm[j].position[0], sfm[j].position[1], sfm[j].position[2]);
				pts_3_vector.push_back(pts_3);
				break;
			}
		}
	}
	if (int(pts_2_vector.size()) < 15)
	{
		printf("unstable features tracking, please slowly move you device!\n");
		if (int(pts_2_vector.size()) < 10)
			return false;
	}
	cv::Mat r, rvec, t, D, tmp_r;
	cv::eigen2cv(R_initial, tmp_r);
	cv::Rodrigues(tmp_r, rvec);
	cv::eigen2cv(P_initial, t);
	cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
	bool pnp_succ;
	pnp_succ = cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, 1);
	if(!pnp_succ)
	{
		return false;
	}
	cv::Rodrigues(rvec, r);
	//cout << "r " << endl << r << endl;
	MatrixXd R_pnp;
	cv::cv2eigen(r, R_pnp);
	MatrixXd T_pnp;
	cv::cv2eigen(t, T_pnp);
	R_initial = R_pnp;
	P_initial = T_pnp;
	return true;

}

void Draw(Eigen::Matrix<double, 3, 4> poses[], vector<SFMFeature>& sfm) 
{
    if (sfm.empty()) {
        cerr << "parameter is empty!" << endl;
        return;
    }
    float fx = 277.34;
    float fy = 291.402;
    float cx = 312.234;
    float cy = 239.777;
    // create pangolin window and plot the trajectory
    pangolin::CreateWindowAndBind("Trajectory Viewer", 1024, 768);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pangolin::OpenGlRenderState s_cam(
            pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
            pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0)
    );

    pangolin::View &d_cam = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
            .SetHandler(new pangolin::Handler3D(s_cam));

    while (pangolin::ShouldQuit() == false) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        d_cam.Activate(s_cam);
        glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

        // draw poses
        float sz = 0.1;
        int width = 640, height = 480;
        // for (auto &Tcw: poses) {
        for(int i = 0; i < nImgs; i++)
        {
        	glPushMatrix();
	        //Sophus::Matrix4f m = Tcw.inverse().matrix().cast<float>();
	        Matrix4d current_pos = Matrix4d::Identity();
	        current_pos.block<3,4>(0,0) = poses[i];
	        current_pos = current_pos.inverse();
	        glMultMatrixd((GLdouble *) current_pos.data());
	        glColor3f(1, i/12.0, 0);
	        glLineWidth(2);
	        glBegin(GL_LINES);
	        glVertex3f(0, 0, 0);
	        glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);
	        glVertex3f(0, 0, 0);
	        glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
	        glVertex3f(0, 0, 0);
	        glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
	        glVertex3f(0, 0, 0);
	        glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);
	        glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);
	        glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
	        glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
	        glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
	        glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
	        glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);
	        glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);
	        glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);
	        glEnd();
	        glPopMatrix();
        }
        
        // }

        // points
        glPointSize(1);
        glBegin(GL_POINTS);
        for (size_t i = 0; i < sfm.size(); i++) 
        {
        	if(sfm[i].state == true)
        	{
        		glColor3f(0.7, 0.9, 0.7);
            	glVertex3d(sfm[i].position[0], sfm[i].position[1], sfm[i].position[2]);
        	}
            
        }
        glEnd();
        pangolin::FinishFrame();
        //usleep(5000);   // sleep 5 ms
    }
}

//find l,and solve R,t
bool relativePose(Matrix3d &relative_R, Vector3d &relative_T, int &l, vector<SFMFeature>& sfm)
{
    // find previous frame which contains enough correspondance and parallex with newest frame
    for (int i = 0; i < nImgs - 1; i++)
    {
        vector<pair<Vector2d, Vector2d>> corres;
        corres = getCorrespondings(i, nImgs-1, sfm);
        //std::cout << "corres = " << corres.size() <<std::endl;
        if (corres.size() > 30)
        {
            double sum_parallax = 0;
            double average_parallax;
            for (int j = 0; j < int(corres.size()); j++)
            {
                Vector2d pts_0(corres[j].first(0), corres[j].first(1));
                Vector2d pts_1(corres[j].second(0), corres[j].second(1));
                double parallax = (pts_0 - pts_1).norm();
                sum_parallax = sum_parallax + parallax;

            }
            average_parallax = 1.0 * sum_parallax / int(corres.size());
			std::cout << "average_parallax = " << average_parallax *460 << std::endl;
            if(average_parallax * 460 > 25 && solveRelativeRT(corres, relative_R, relative_T))//视差大于阈值且能够成功计算相对变换，令l为i
            {
                l = i;
                printf("average_parallax %f choose l %d and newest frame to triangulate the whole structure\n", average_parallax * 460, l);
                return true;
            }
        }
    }
    return false;
}

void read_gps(const string &gps_path_)
{
	string gps_path =gps_path_;
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

void LoadImages(const string &strImagePath, const string &image_list, vector<string> &strImagesFileNames,
                vector<double> &timeStamps) {

    ifstream fimage(image_list);
    assert(fimage.is_open() && "can't load image message");
    timeStamps.clear();
    strImagesFileNames.clear();
    static double TIME_SHIFT = 0.;
    int i = 0;
    while(!fimage.eof()){
        ++i;
        string s;
        getline(fimage, s);
        if(!s.empty() && i>3500){
            stringstream stime(s.substr(0,16));
            double time;
            stime >> time;
            time = time / 1000.;
            strImagesFileNames.push_back(strImagePath + s);
            timeStamps.push_back(time);
        }
    }
}

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
struct Point2ffHash{
    size_t operator()(const Point2f& per) const{
        return hash<float>()(per.x) ^ hash<float>()(per.y);
    }
};
struct Point2ffCmp{
    bool operator()(const Point2f p0, const Point2f p1) const{
        return p0.x == p1.x && p0.y == p1.y;
    }
};

int main(int argc, char **argv)
{
    if(argc != 3){
        cerr << "Usage: ./sfm path_to_image_file path_to_gps" << endl;
        return 1;
    }

	::google::InitGoogleLogging(argv[0]);
	::google::SetStderrLogging(::google::INFO);
	::google::SetLogDestination(google::INFO, "../log/INFO_");

	::google::InstallFailureSignalHandler();
	::google::ParseCommandLineFlags(&argc, &argv, true);
	// ::google::SetStderrLogging(FLAGS_logLevel); // 确认log信息输出等级

	FLAGS_stderrthreshold = 0;     // INFO: 0, WARNING: 1, ERROR: 2, FATAL: 3
	FLAGS_colorlogtostderr = true; //设置输出到屏幕的日志显示相应颜色

	Mat cur_frame,frw_frame;
	vector<Point2f> cur_pts,frw_pts;
	vector<uchar> status;
    vector<float> err;

	//Mat msk;
    //Mat msk0 = Mat::zeros(WIDTH,HEIGHT,CV_8UC1);
    cv::Rect rect(0, 50, WIDTH,HEIGHT-150);  //(x,y) = (180,200)  size = (200,200)
    //cv::Rect rect(0, 50, WIDTH-50,HEIGHT-150);
	Mat msk0 = Mat::zeros(HEIGHT, WIDTH, CV_8UC1);
	msk0(rect).setTo(255);
//	Mat msk0(HEIGHT/2,WIDTH/2, CV_8UC1, cv::Scalar(255));
//	Mat msk0(msk,rect);

 	vector<SFMFeature> sfm;
 	vector<int> ids;
    vector<int> flow_ids;
 	unsigned long id = 0;
    unsigned long flow_id = 0;

    unordered_map<Point2f,int,Point2ffHash,Point2ffCmp> flow_pts;

//    boost::container::flat_map<Point2f,int> flow_pts;
	LoadImages(string(argv[1]),vStrImageFileNames,vTimeStamps);
    //nImgs = vStrImageFileNames.size();
	nImgs = 110;
	double startTime = vTimeStamps[7100];
	std::cout << "startTime = " << startTime;
	double endTime = vTimeStamps[6849];
	Quaterniond q[nImgs];
	Vector3d T[nImgs];

    TicToc begin;
	for(int i = 0;i<nImgs;i++)
	{
		Mat mask = msk0.clone();
		//Mat mask(msk0,rect);
        Mat frw_frame = imread(vStrImageFileNames[i+2970],1);//ts 转弯
//        Mat frw_frame = imread(vStrImageFileNames[i+1170],1);//ts 转弯
        //Mat mask(frw_frame,rect);
//        Mat frw_frame = imread(vStrImageFileNames[i+3000],0);
        //resize(frw_frame,frw_frame,Size(frw_frame.cols/2,frw_frame.rows/2),0,0,INTER_LINEAR);
		cv::cvtColor(frw_frame,frw_frame, cv::COLOR_BGR2GRAY);
		//GaussianBlur(frw_frame,frw_frame,Size(5,5),0);
		if(i == 0)
		{
			goodFeaturesToTrack(frw_frame,cur_pts,MAX_CNT,0.01,MIN_DIST);
			for(auto &c:cur_pts)
				ids.push_back(-1);
			//frw_pts = cur_pts;
			cur_frame = frw_frame;
			// for(auto& pt:cur_pts)
			// 	circle(img, pt, 2, cv::Scalar(0, 0, 255), 2);
			continue;
		}

		//KLT optical flow
		calcOpticalFlowPyrLK(cur_frame, frw_frame, cur_pts, frw_pts, status, err, cv::Size(40, 40), 7);
		
		for(int j = 0;j < cur_pts.size();j++)
		{
			if(ids[j] < 0 && status[j])
				ids[j] = id++;
		}
		//cout<<ids.size()<<" "<<cur_pts.size()<<endl;

		reduceVector(ids, status);
		reduceVector(frw_pts, status);
		reduceVector(cur_pts, status);
		vector<uchar> status2;
		findFundamentalMat(cur_pts, frw_pts, cv::FM_RANSAC, 1.0, 0.99, status2);
		reduceVector(frw_pts,status2);
		reduceVector(cur_pts,status2);
		reduceVector(ids, status2);

        for(int k = 0;k<ids.size();k++){
            flow_pts.insert(make_pair(Point2f(frw_pts[k].x,frw_pts[k].y),ids[k]));
        }
        LOG(INFO) << "ids.size)( = " << ids.size();
		for(int j = 0; j < ids.size(); j++)
		{
			vector<SFMFeature>::iterator it;
			it = find_if(sfm.begin(), sfm.end(), [&](SFMFeature s){return s.id == ids[j];});
			if(it == sfm.end())
			{
				//cout<<"create new sfm feature."<<endl;
				SFMFeature sfmeat;
				sfmeat.id = ids[j];
				Point2d p0,p1;
				p0 = pixel2cam(Point2d(cur_pts[j].x, cur_pts[j].y));
				p1 = pixel2cam(Point2d(frw_pts[j].x, frw_pts[j].y));
				sfmeat.observation.push_back(make_pair(i-1, Vector2d(p0.x, p0.y)));
				sfmeat.observation.push_back(make_pair(i, Vector2d(p1.x, p1.y)));
				sfm.push_back(sfmeat);
                flow_ids.push_back(ids[j]);
			}
			else
			{
				//cout<<"append measurement."<<endl;
				Point2d p1 = pixel2cam(Point2d(frw_pts[j].x, frw_pts[j].y));
				it->observation.push_back(make_pair(i, Vector2d(p1.x, p1.y)));
			}
		}
		LOG(INFO) << "i am here";

		setMask(mask,frw_pts);
		imshow("mask",mask);
		vector<Point2f> newFeat;

		goodFeaturesToTrack(frw_frame,newFeat,MAX_CNT,0.1,MIN_DIST,mask);

		Mat flow;
		cvtColor(frw_frame,flow,CV_GRAY2BGR);
//        for (int i = 0;i<sfm.size();++i){
//            circle(flow,cam2pixel(Point2d(sfm[i].observation[0].second[0],sfm[i].observation[0].second[1])),1,cv::Scalar(30,150,255),2);
//            putText(flow,to_string(sfm[i].id),cam2pixel(Point2d(sfm[i].observation[0].second[0],sfm[i].observation[0].second[1])),FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1);
//        }
		LOG(INFO) << "i am here = " << flow_pts.size();
        LOG(INFO) << " frw size = " << frw_pts.size();
        LOG(INFO) << " cursize = " << cur_pts.size();

        for (int k = 0; k <cur_pts.size() ; ++k) {
			circle(flow,frw_pts[k],1,cv::Scalar(30,150,255),2);
			putText(flow,to_string(flow_pts.find(Point2f(frw_pts[k].x,frw_pts[k].y))->second),frw_pts[k],FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1);
//            putText(flow,to_string(flow_pts.find(Point2f(frw_pts[k].x,frw_pts[k].y))->second),frw_pts[k],FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1);

		}
		LOG(INFO) << "i am here";
        LOG(INFO) << "flow size =  "<< flow_ids.size();
		flow_pts.clear();
        flow_ids.clear();
		imwrite("/home/zhouchang/Pictures/image/" + to_string(i) +".jpg",flow);
		imshow("flow",flow);
        LOG(INFO) << "i am here = " << flow_pts.size();

		//display image with feature
		Mat disp;
		cvtColor(frw_frame,disp,CV_GRAY2BGR);

		for(int j = 0;j < cur_pts.size();j++)
		{
			if(inBorder(frw_pts[j]) && msk0.at<uchar>(frw_pts[j].y,frw_pts[j].x) == 255)
			{
//				circle(disp, frw_pts[j], 2, cv::Scalar(30, 180, 255), 2);
				circle(disp,frw_pts[j],1,cv::Scalar(30,150,255),2);
				line(disp,cur_pts[j],frw_pts[j],Scalar(0,255,255));
				//circle(msk, frw_pts[j], 2, cv::Scalar(0, 0, 255), 2);
			}
			else
			{
				//circle(msk, frw_pts[j], 2, cv::Scalar(255, 0, 0), 2);
				circle(disp, frw_pts[j], 2, cv::Scalar(255, 0, 0), 2);
			}
		}

		cur_frame = frw_frame;
		cur_pts = frw_pts;
		for(auto& pt:newFeat)
		{
			cur_pts.push_back(pt);
			ids.push_back(-1);
		}
		frw_pts.clear();
		//imshow("msk",msk);
		imshow("frame",disp);


		waitKey(200);
	}

	int l = -1;

	Matrix3d R;
	Vector3d P;
	// solveRelativeRT(corres, R, P);//投影矩阵·

	if(!relativePose(R, P, l, sfm))
	{
		cout<<"failed to find appropriate reference frame."<<endl;
		return -1;
	}

    q[l].w() = 1;
    q[l].x() = 0;
    q[l].y() = 0;
    q[l].z() = 0;
    T[l].setZero();
    q[nImgs - 1] = q[l] * Quaterniond(R);
    T[nImgs - 1] = P;

    Matrix3d c_Rotation[nImgs];
    Vector3d c_Translation[nImgs];
    Quaterniond c_Quat[nImgs];
    //l frame to current frame
    double c_rotation[nImgs][4];
    double c_translation[nImgs][3];
    Eigen::Matrix<double, 3, 4> Pose[nImgs];

    c_Quat[l] = q[l].inverse();
    c_Rotation[l] = c_Quat[l].toRotationMatrix();
    c_Translation[l] = -1 * (c_Rotation[l] * T[l]);
    Pose[l].block<3, 3>(0, 0) = c_Rotation[l];
    Pose[l].block<3, 1>(0, 3) = c_Translation[l];

    c_Quat[nImgs - 1] = q[nImgs - 1].inverse();
    c_Rotation[nImgs - 1] = c_Quat[nImgs - 1].toRotationMatrix();
    c_Translation[nImgs - 1] = -1 * (c_Rotation[nImgs - 1] * T[nImgs - 1]);
    Pose[nImgs - 1].block<3, 3>(0, 0) = c_Rotation[nImgs - 1];
    Pose[nImgs - 1].block<3, 1>(0, 3) = c_Translation[nImgs - 1];

    //1: trangulate between l ----- nImgs - 1
	//2: solve pnp l + 1; trangulate l + 1 ------- nImgs - 1; 
	for (int i = l; i < nImgs - 1 ; i++)
	{
		//solve pnp
		if (i > l)
		{
			Matrix3d R_initial = c_Rotation[i - 1];
			Vector3d P_initial = c_Translation[i - 1];
			if(!solveFrameByPnP(R_initial, P_initial, i, sfm))
			{
				cout<<"failed to solvePnP."<<endl;
				return -1;
			}
			//cout<<"solvePnP success"<<endl;
			c_Rotation[i] = R_initial;
			c_Translation[i] = P_initial;
			c_Quat[i] = c_Rotation[i];
			Pose[i].block<3, 3>(0, 0) = c_Rotation[i];
			Pose[i].block<3, 1>(0, 3) = c_Translation[i];
		}

		// triangulate point based on the solve pnp result
		triangulateTwoFrames(i, Pose[i], nImgs - 1, Pose[nImgs - 1], sfm);
	}
    //3: triangulate l-----l+1 l+2 ... nImgs -2 ;l is wordFrame now
	for (int i = l + 1; i < nImgs - 1; i++)
		triangulateTwoFrames(l, Pose[l], i, Pose[i], sfm);
	//4: solve pnp l-1; triangulate l-1 ----- l
	//             l-2              l-2 ----- l
	for (int i = l - 1; i >= 0; i--)
	{
		//solve pnp
		Matrix3d R_initial = c_Rotation[i + 1];
		Vector3d P_initial = c_Translation[i + 1];
		if(!solveFrameByPnP(R_initial, P_initial, i, sfm))
		{
			cout<<"failed to solvePnP."<<endl;
			return -1;
		}
		c_Rotation[i] = R_initial;
		c_Translation[i] = P_initial;
		c_Quat[i] = c_Rotation[i];
		Pose[i].block<3, 3>(0, 0) = c_Rotation[i];
		Pose[i].block<3, 1>(0, 3) = c_Translation[i];
		//triangulate
		triangulateTwoFrames(i, Pose[i], l, Pose[l], sfm);
	}
	//5: triangulate all other points
	for (int j = 0; j < sfm.size(); j++)
	{
		if (sfm[j].state == true)
			continue;
		if ((int)sfm[j].observation.size() >= 2)
		{
			Vector2d point0, point1;
			int frame_0 = sfm[j].observation[0].first;
			point0 = sfm[j].observation[0].second;
			int frame_1 = sfm[j].observation.back().first;
			point1 = sfm[j].observation.back().second;
			Vector3d point_3d;
			triangulatePoint(Pose[frame_0], Pose[frame_1], point0, point1, point_3d);
			sfm[j].state = true;
			sfm[j].position[0] = point_3d(0);
			sfm[j].position[1] = point_3d(1);
			sfm[j].position[2] = point_3d(2);
			//cout << "trangulated : " << frame_0 << " " << frame_1 << "  3d point : "  << j << "  " << point_3d.transpose() << endl;
		}		
	}
	//output keyframe pose
//	for (int i = 0; i < nImgs; i++)
//	{
//		q[i] = c_Rotation[i].transpose();
//		//cout << "solvePnP  q" << " i " << i <<"  " <<q[i].w() << "  " << q[i].vec().transpose() << endl;
//	}
//	for (int i = 0; i < nImgs; i++)
//	{
//		Vector3d t_tmp;
//		t_tmp = -1 * (q[i] * c_Translation[i]);
//		//cout << "solvePnP t" << " i " << i <<"  " << t_tmp.x() <<"  "<< t_tmp.y() <<"  "<< t_tmp.z() << endl;
//	}
	//full BA
	ceres::Problem problem;
	ceres::LocalParameterization* local_parameterization = new ceres::QuaternionParameterization();
	//cout << " begin full BA " << endl;
	for (int i = 0; i < nImgs; i++)
	{
		//double array for ceres
		c_translation[i][0] = c_Translation[i].x();
		c_translation[i][1] = c_Translation[i].y();
		c_translation[i][2] = c_Translation[i].z();
		c_rotation[i][0] = c_Quat[i].w();
		c_rotation[i][1] = c_Quat[i].x();
		c_rotation[i][2] = c_Quat[i].y();
		c_rotation[i][3] = c_Quat[i].z();
		problem.AddParameterBlock(c_rotation[i], 4, local_parameterization);
		problem.AddParameterBlock(c_translation[i], 3);
		//cout << "l = " << l << endl;
		if (i == l)
		{
			problem.SetParameterBlockConstant(c_rotation[i]);
		}
		if (i == l || i == nImgs - 1)
		{
			problem.SetParameterBlockConstant(c_translation[i]);
		}
	}

    LOG(INFO) << sfm.size() << endl;
	for (int i = 0; i < sfm.size(); i++)
	{
		//cout << "sfm[i].observation.size() =  " << sfm[i].observation.size() << endl;
		if (sfm[i].state != true)
			continue;
		//第i个点，优化观察到i的帧
//        cout << sfm[i].observation.size() << endl;
		for (int j = 0; j < int(sfm[i].observation.size()); j++)
		{
			int l = sfm[i].observation[j].first;
//			cout << "l =  " << l << endl;
//			cout << "sfm[].position = " << sfm[i].position[0] << sfm[i].position[1] << sfm[i].position[2] << endl;
//			cout << "sfm[i].observation[j].second.x() = " << sfm[i].observation[j].second.x()  << " sfm[i].observation[j].second.y()" <<  sfm[i].observation[j].second.y() << endl;
			ceres::CostFunction* cost_function = ReprojectionError3D::Create(
					sfm[i].observation[j].second.x(),
					sfm[i].observation[j].second.y());

			problem.AddResidualBlock(cost_function, NULL, c_rotation[l], c_translation[l], sfm[i].position);
            //LOG(INFO) << "sss " << sfm[i].position[0] << " " << sfm[i].position[1] << " " << sfm[i].position[2];
            //LOG(INFO) << "sfm[i].position position = " << sfm[i].position[0] << " " << sfm[i].position[1] << " " << sfm[i].position <<endl;
		}

	}
	ceres::Solver::Options options;
	options.linear_solver_type = ceres::DENSE_SCHUR;
	//options.minimizer_progress_to_stdout = true;
	//options.max_solver_time_in_seconds = 0.2;
	options.max_num_iterations = 70;
	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
	std::cout << summary.BriefReport() << "\n";
	LOG(INFO) << summary.BriefReport();
	if (summary.termination_type == ceres::CONVERGENCE || summary.final_cost < 5e-03)
	{
		cout << "vision only BA converge" << endl;
	}
	else
	{
		cout << "vision only BA not converge " << endl;
		//return false;
	}
    //LOG(INFO) << "time = " << begin.toc();
//	for (int i = 0; i < nImgs; i++)
//	{
//		q[i].w() = c_rotation[i][0];
//		q[i].x() = c_rotation[i][1];
//		q[i].y() = c_rotation[i][2];
//		q[i].z() = c_rotation[i][3];
//		q[i] = q[i].inverse();
//		cout << "final  q" << " i " << i <<"  " <<q[i].w() << "  " << q[i].vec().transpose() << endl;
//	}
//	for (int i = 0; i < nImgs; i++)
//	{
//
//		T[i] = -1 * (q[i] * Vector3d(c_translation[i][0], c_translation[i][1], c_translation[i][2]));
//		cout << "final  t" << " i " << i <<"  " << T[i](0) <<"  "<< T[i](1) <<"  "<< T[i](2) << endl;
//	}
////	for (int i = 0; i < (int)sfm.size(); i++)
////	{
////		if(sfm[i].state)
////			sfm_tracked_points[sfm[i].id] = Vector3d(sfm[i].position[0], sfm[i].position[1], sfm[i].position[2]);
////	}
//
//	for(int i = 0; i < nImgs; i++)
//	{
//		cout<<"Pose "<<i<<endl;
//		cout<<Pose[i]<<endl<<endl;
//	}
	Draw(Pose, sfm);
	return 0;
}
