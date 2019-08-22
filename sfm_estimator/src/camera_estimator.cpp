#include <stdio.h>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <ros/ros.h>

#include "camera_estimator.h"
#include "parameters.h"
#include "utility/visualization.h"
#include <glog/logging.h>

Estimator estimator;

std::condition_variable con;
double current_time = -1;

std::queue<sensor_msgs::PointCloudConstPtr> feature_buf;
std::queue<geometry_msgs::PointStampedConstPtr> gt_buf;

std::mutex m_buf;
std::mutex m_state;
std::mutex m_estimator;

double latest_time;
Eigen::Vector3d tmp_P;
Eigen::Quaterniond tmp_Q;
Eigen::Vector3d tmp_V;
Eigen::Vector3d tmp_Ba;
Eigen::Vector3d tmp_Bg;
Eigen::Vector3d acc_0;
Eigen::Vector3d gyr_0;
bool init_feature = 0;
bool init_imu = 1;
double last_imu_t = 0;

int flag = 0;

void update()
{
    TicToc t_predict;
    latest_time = current_time;
    tmp_P = estimator.Ps[WINDOW_SIZE];
    tmp_Q = estimator.Rs[WINDOW_SIZE];
}

std::vector<sensor_msgs::PointCloudConstPtr> getMeasurements()
{
    std::vector<sensor_msgs::PointCloudConstPtr> measurements;

    while (true)
    {
        if (feature_buf.empty())
            return measurements;

        sensor_msgs::PointCloudConstPtr img_msg = feature_buf.front();
        feature_buf.pop();

        measurements.emplace_back(img_msg);
    }
    return measurements;
}

void feature_callback(const sensor_msgs::PointCloudConstPtr &feature_msg)
{
    if (!init_feature)
    {
        //skip the first detected feature, which doesn't contain optical flow speed
        init_feature = 1;
        return;
    }
    m_buf.lock();
    feature_buf.push(feature_msg);
    m_buf.unlock();
    con.notify_one();
}
void gt_callback(const geometry_msgs::PointStampedConstPtr &gt_msg){
    m_buf.lock();
    gt_buf.push(gt_msg);
    m_buf.unlock();
    con.notify_one();
}

void process(){
    while (true){
        std::vector<sensor_msgs::PointCloudConstPtr> measurements;
        std::unique_lock<std::mutex> lk(m_buf);
        con.wait(lk, [&]
        {
            return (measurements = getMeasurements()).size() != 0;
        });
        lk.unlock();
        m_estimator.lock();

        flag++;
        if( flag == 20)
            estimator.solver_flag = Estimator::SFM;
        std::cout << " flag = " << flag << std::endl;
        for (auto &measurement : measurements)
        {
            auto img_msg = measurement;
            //double dx = 0, dy = 0, dz = 0, rx = 0, ry = 0, rz = 0;
            TicToc t_s;
            std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>> image;
            ROS_INFO("img_msg->points.size() = %d", img_msg->points.size());
            for (unsigned int i = 0; i < img_msg->points.size();i++){
                int v = img_msg->channels[0].values[i] + 0.5;
                int feature_id = v / 1;
                int camera_id = v % 1;
                double x = img_msg->points[i].x;
                double y = img_msg->points[i].y;
                double z = img_msg->points[i].z;
                double p_u = img_msg->channels[1].values[i];
                double p_v = img_msg->channels[2].values[i];
                double velocity_x = img_msg->channels[3].values[i];
                double velocity_y = img_msg->channels[4].values[i];
                ROS_ASSERT(z == 1);
                Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
                xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
                image[feature_id].emplace_back(camera_id, xyz_uv_velocity);
            }
            //estimator.processImage(image, img_msg->header);

            estimator.visualIntegration(image, img_msg->header);

            double whole_t = t_s.toc();
            printStatistics(estimator, whole_t);
            std_msgs::Header header = img_msg->header;
            header.frame_id = "world";

            pubOdometry(estimator, header);
            pubKeyPoses(estimator, header);
            pubCameraPose(estimator, header);
            pubPointCloud(estimator, header);
            pubTF(estimator, header);
            pubKeyframe(estimator);
            //ROS_ERROR("end: %f, at %f", img_msg->header.stamp.toSec(), ros::Time::now().toSec());
        }
        m_estimator.unlock();
        m_buf.lock();
        m_state.lock();
       // if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
            update();
        m_state.unlock();
        m_buf.unlock();
    }
}

int main(int argc, char **argv){

    ::google::InitGoogleLogging(argv[0]);
    ::google::SetStderrLogging(::google::INFO);
    ::google::SetLogDestination(google::INFO, "log/INFO_");

    ::google::InstallFailureSignalHandler();
    ::google::ParseCommandLineFlags(&argc, &argv, true);
    // ::google::SetStderrLogging(FLAGS_logLevel); // 确认log信息输出等级

    FLAGS_stderrthreshold = 0;     // INFO: 0, WARNING: 1, ERROR: 2, FATAL: 3
    FLAGS_colorlogtostderr = true; //设置输出到屏幕的日志显示相应颜色

    ros::init(argc, argv, "sfm_estimator");
    ros::NodeHandle n("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Debug);
    readParameters(n);
    estimator.setParameter();
    registerPub(n);

    ros::Subscriber sub_image = n.subscribe("/sfm_feature_tracker/feature", 2000, feature_callback);
    ros::Subscriber sub_gt = n.subscribe("/leica/posion", 100, gt_callback);

    std::thread measurement_process{process};
    ros::spin();

    return 0;
}