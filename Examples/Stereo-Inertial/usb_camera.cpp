#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <glob.h>
#include <unistd.h>
#include <dirent.h>
#include <stdlib.h>
#include <string>
#include <stdio.h>
#include <map>
#include <Eigen/Dense>
#include <System.h>

using namespace std;


cv::Mat FastGuidedfilter(cv::Mat &I, int r, float eps, int size) {
    r = r / size;
    int wsize = 2 * r + 1;
    I.convertTo(I, CV_32FC1, 1/255.0);

    cv::Mat small_I, small_p;
    cv::resize(I, small_I, I.size()/size, 0, 0, cv::INTER_AREA);
    small_p = small_I;

    cv::Mat mean_I, mean_p;
    cv::boxFilter(small_I, mean_I, -1, cv::Size(wsize, wsize), cv::Point(-1, -1), true, cv::BORDER_REFLECT);
    mean_p = mean_I;

    cv::Mat mean_II, mean_Ip;
    mean_II = small_I.mul(small_I);
    cv::boxFilter(mean_II, mean_II, -1, cv::Size(wsize, wsize), cv::Point(-1, -1), true, cv::BORDER_REFLECT);
    mean_Ip = mean_II;

    cv::Mat var_I, cov_Ip, mean_mul_I;
    mean_mul_I=mean_I.mul(mean_I);
    cv::subtract(mean_II, mean_mul_I, var_I);
    cov_Ip = var_I;
    
    cv::Mat a, b;
    cv::divide(cov_Ip, (var_I+eps),a);
    cv::subtract(mean_p, a.mul(mean_I), b);

    cv::Mat mean_a, mean_b;
    cv::boxFilter(a, mean_a, -1, cv::Size(wsize, wsize), cv::Point(-1, -1), true, cv::BORDER_REFLECT);
    cv::boxFilter(b, mean_b, -1, cv::Size(wsize, wsize), cv::Point(-1, -1), true, cv::BORDER_REFLECT);

    cv::resize(mean_a, mean_a, I.size());
    cv::resize(mean_b, mean_b, I.size());

    cv::Mat out = mean_a.mul(I) + mean_b;
    out.convertTo(out, CV_8UC1, 255);
    return out;
}


int get_int_from_string(string& str);
//get num in file name in sort from small to big
vector<int> getFiles(char* dirc){
    vector<string> files;
    struct dirent *ptr;
    DIR *dir;
    dir = opendir(dirc);
    
    if(dir == NULL)
    {
        perror("open dir error ...");
        exit(1);
    }

    while((ptr = readdir(dir)) != NULL){
        if(strcmp(ptr->d_name,".")==0 || strcmp(ptr->d_name,"..")==0)    ///current dir OR parrent dir  
            continue;  
        if(ptr->d_type == 8)//it;s file
        {
            files.push_back(ptr->d_name);
        }

        else if(ptr->d_type == 10)//link file
            continue;
        else if(ptr->d_type == 4) //dir
        {
            files.push_back(ptr->d_name);
        }
    }
    closedir(dir);
    
    vector<int> result;
    for(int i=0;i < files.size();i++)
    {
        result.push_back(get_int_from_string(files[i]));
    }
    sort(result.begin(),result.end());

    for(size_t i = 0; i < result.size();++i){
        //cout << result[i] << endl;
    }
    return result;
}

int get_int_from_string(string& str)
{
    int result = 0;
    for(int i = 0; i < str.size(); i++)
    {
        if (str[i] >= '0'&& str[i] <= '9')  
        {  
            result = result * 10 + str[i] - 48;  
        }  
    }
    return result;
}


int main(int argc, char** argv)
{   
    if (argc != 6)
    {
        cout << "usage: image_publisher left_image_folder right_image_folder imu_file " << endl;
        return -1;
    }
    ORB_SLAM3::System SLAM(argv[1],argv[2],ORB_SLAM3::System::IMU_STEREO, true);
    
    std::cout << argv[1] << std::endl;
    std::cout << argv[2] << std::endl;
    std::cout << argv[3] << std::endl;
    std::cout << argv[4] << std::endl;
    std::cout << argv[5] << std::endl;
    vector<int> left_image_count = getFiles(argv[3]);// 
    vector<int> right_image_count = getFiles(argv[4]);
    FILE *fp;
    fp = fopen(argv[5],"r");
    double imu_time,last_imu_time;
    float acceleration[3],angular_v[3];
    float last_acceleration[3], last_angular_v[3];
    int time_count_left,time_count_right;
    int imu_seq = 1;
    std::map<int,int> imu_big_interval;
    int fscanf_return;
    fscanf_return = fscanf(fp,"%lf,%f,%f,%f,%f,%f,%f",
                &imu_time,angular_v,angular_v+1,angular_v+2,acceleration,acceleration+1,acceleration+2);
    last_acceleration[0] = acceleration[0];
    last_acceleration[1] = acceleration[1];
    last_acceleration[2] = acceleration[2];
    last_angular_v[0] = angular_v[0];
    last_angular_v[1] = angular_v[1];
    last_angular_v[2] = angular_v[2];
    last_imu_time = imu_time;
    if (fscanf_return != 7)
    {
        std::cout << "imu format error " << last_imu_time <<std::endl;
        fclose(fp);
        return -1;
    }
    for(size_t i = 10;(i < right_image_count.size() - 10) ;++i)
    {
        std::vector<ORB_SLAM3::IMU::Point> vImuMeas;
        vImuMeas.clear();    

        if (feof(fp))
            break;
        
        ostringstream stringStream;
        //转换左图
        std::string left_filename = string(argv[3]) + "/" + to_string(left_image_count[i]) + ".png";
        cv::Mat left_image = cv::imread(left_filename);
        if(left_image.empty())
        {
            std::cout << "left image empty" << std::endl;
            return 0;
        }
        time_count_left = left_image_count[i];
    
        //转换右图
        std::string right_filename = string(argv[4])+"/"+ to_string(right_image_count[i]) + ".png";
        cv::Mat right_image = cv::imread(right_filename);
        if(right_image.empty())
        {
            std::cout << "right image empty" << std::endl;
            return 0;
        }
        time_count_right = right_image_count[i];
        if(time_count_left != time_count_right)
        {
           std::cout << "left image time != right image time" <<  time_count_left << std::endl;
           return -1; 
        }
        // std::cout << "imu_time: " << imu_time << " " << last_angular_v[0] << " " << last_angular_v[1] << " " << last_angular_v[2] << " " << last_acceleration[0] <<  " " << last_acceleration[1] <<  " " << last_acceleration[2] << std::endl;                                         
        while (imu_time < left_image_count[i]*0.0001 - 0.005)
        {
            if(last_imu_time > imu_time)
            {
                std::cout << "imu time disorder" << last_imu_time << std::endl;
            }
            if ( imu_time - last_imu_time > 0.5)
            {
                std::cout << "large interval in imu" << last_imu_time << std::endl;
            }
            // vImuMeas.push_back(ORB_SLAM3::IMU::Point(acceleration[0] - 0.032011,acceleration[1] - 0.103757,acceleration[2] - 0.122642,
            //                                          angular_v[0] + 0.00316916,angular_v[1] - 0.00239944,angular_v[2] + 0.000110356,imu_time));
            vImuMeas.push_back(ORB_SLAM3::IMU::Point(acceleration[0],acceleration[1],acceleration[2],
                                         angular_v[0],angular_v[1],angular_v[2],imu_time));
                        
            last_imu_time = imu_time;
            fscanf_return = fscanf(fp,"%lf,%f,%f,%f,%f,%f,%f",
                &imu_time,angular_v,angular_v+1,angular_v+2,acceleration,acceleration+1,acceleration+2);
            last_acceleration[0] = acceleration[0] ;
            last_acceleration[1] = acceleration[1] ;
            last_acceleration[2] = acceleration[2] ;
            last_angular_v[0] = angular_v[0] ;
            last_angular_v[1] = angular_v[1] ;
            last_angular_v[2] = angular_v[2] ;
            if (fscanf_return != 7)
            {
                std::cout << "imu format error " << last_imu_time <<std::endl;
                fclose(fp);
                return -1;
            }
            std::cout.setf(std::ios::fixed, std::ios::floatfield);
	        std::cout.precision(6);
            // std::cout << "imu_time: " << imu_time << " " << angular_v[0] << " " << angular_v[1] << " " << angular_v[2] << " " << acceleration[0] <<  " " << acceleration[1] <<  " " << acceleration[2] << std::endl;
        }
        std::cout << "image time: " << time_count_left * 0.0001 << std::endl;
        // cv::imshow("left", left_image_res);
        // cv::imshow("right", right_image_res);
        // cv::waitKey(1);
        // std::cout << "vImuMeas " << vImuMeas.size() << std::endl;
        //cv::Rect roi(0, 0, 752, 200); // x, y, width, height
        //eft_image(roi).setTo(cv::Scalar(0, 0, 0)); // 对于彩色图片，黑色是(0,0,0)
        //right_image(roi).setTo(cv::Scalar(0, 0, 0)); // 对于彩色图片，黑色是(0,0,0)
         SLAM.TrackStereo(left_image, right_image, time_count_left * 0.0001, vImuMeas);
        //SLAM.TrackStereo(left_image, right_image, time_count_left * 0.0001f);

        //SLAM.TrackMonocular(left_image, time_count_left * 0.0001, vImuMeas);
        usleep(50000);
    }
    
    SLAM.Shutdown();
    SLAM.SaveTrajectoryEuRoC("CameraTrajectory.txt");
    SLAM.SaveKeyFrameTrajectoryEuRoC("KeyFrameTrajectory.txt");
    fclose(fp);
    return 0;

}
