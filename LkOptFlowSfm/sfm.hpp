#pragma once
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <cstdlib>
#include <deque>
#include <map>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <glog/logging.h>

using namespace Eigen;
using namespace std;
using namespace cv;

struct SFMFeature
{
    bool state;
    int id;   //corner id
    vector<pair<int,Vector2d>> observation;
    double position[3]; //corner 3D position
    double depth;
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
//        LOG(INFO) << "ceres camera_R = " << camera_R[0] << " " << camera_R[1] << " " << camera_R[2] << " " << camera_R[3];
//        LOG(INFO) << "ceres point = " << point[0] << " " << point[1] << " " << point[2];
//        LOG(INFO) << "ceres p = " << p[0] << " " << p[1] << " " << p[2];

        ceres::QuaternionRotatePoint(camera_R, point, p);
//        LOG(INFO) << "after ceres camera_R = " << camera_R[0] << " " << camera_R[1] << " " << camera_R[2] << " " << camera_R[3];
//        LOG(INFO) << "after ceres point = " << point[0] << " " << point[1] << " " << point[2];
//        LOG(INFO) << "after ceres p = " << p[0] << " " << p[1] << " " << p[2];

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

bool solveFrameByPnP(Matrix3d &R_initial, Vector3d &P_initial, int i, vector<SFMFeature> &sfm);

void triangulatePoint(Eigen::Matrix<double, 3, 4> &Pose0, Eigen::Matrix<double, 3, 4> &Pose1,
                      Vector2d &point0, Vector2d &point1, Vector3d &point_3d);
void triangulateTwoFrames(int frame0, Eigen::Matrix<double, 3, 4> &Pose0,
                          int frame1, Eigen::Matrix<double, 3, 4> &Pose1,
                          vector<SFMFeature> &sfm);

int feature_num;