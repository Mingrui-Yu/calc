/**
 *  测试在 KITTI 数据集上的 PR 曲线
 * 以 位姿 ground truth 作为判断是否在同一地点的标准
 */

#include "deeplcd.h"

#include <iostream>
#include <fstream>
#include <chrono>
#include <numeric>
#include <iomanip>
#include <cmath>

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace std;

typedef Eigen::Matrix<double, 3, 3> Mat33;
typedef Eigen::Matrix<double, 3, 1> Vec3;

void LoadImages(const string &strSequence, vector<string> &vstrImageFilenames,
                vector<double> &vTimestamps);

void LoadGroundtruthPose(const string &strPose, 
    std::vector<Mat33, Eigen::aligned_allocator<Mat33> > &vPoses_R,
    std::vector<Vec3, Eigen::aligned_allocator<Vec3> > &vPoses_t);

std::pair<double, double> VectorMeanAndVariance(const std::vector<double> v);

bool PoseIsVeryNear(std::vector<Mat33, Eigen::aligned_allocator<Mat33> > &vPoses_R,
    std::vector<Vec3, Eigen::aligned_allocator<Vec3> > &vPoses_t,
    int current_id, int loop_id);

bool PoseIsAcceptablyNear(std::vector<Mat33, Eigen::aligned_allocator<Mat33> > &vPoses_R,
    std::vector<Vec3, Eigen::aligned_allocator<Vec3> > &vPoses_t,
    int current_id, int loop_id);


// ----------------------------------------------------------------------------------------------------

int main(int argc, char** argv)
{
	if(argc != 4)
    {
        std::cerr << std::endl << "Usage: ./mytest/test2   PATH_TO_SEQUENCE  PATH_TO_GROUNDTRUTH_POSE  PATH_TO_RESULT" << std::endl;
        return 1;
    }

    

    // ---------------------------------- Load the images' path -----------------------------------------
    std::string strPathToSequence = argv[1];
    std::vector<string> vstrImageFilenames;
    std::vector<double> vTimestamps;
    LoadImages(strPathToSequence,  vstrImageFilenames, vTimestamps);

    // ---------------------------------- Load the groud_truth file -----------------------------------------
    std::string strPathToPose = argv[2];
    std::vector<Mat33, Eigen::aligned_allocator<Mat33>> vPoses_R;
    std::vector<Vec3, Eigen::aligned_allocator<Vec3>> vPoses_t;
    LoadGroundtruthPose(strPathToPose,  vPoses_R, vPoses_t);

    // ---------------------------------- create the file to store test results ----------------------------------
    ofstream f;
    std::string filename = argv[3];
    f.open(filename.c_str());
    f << fixed;

    // ----------------------------------------- 一些数据初始化 ----------------------------------------------
    int totalNumGroundTruthP = 0;
    // 存储不同 threshold 下的结果
    std::vector<double> vThreshold;
    std::vector<int> vTotalNumTP;
    std::vector<int> vTotalNumFP;
    std::vector<int> vTotalNumFN;
    for(double thres = 1.00; thres >= 0.70; thres -= 0.01){
        vThreshold.push_back(thres);
        vTotalNumTP.push_back(0);
        vTotalNumFP.push_back(0);
        vTotalNumFN.push_back(0);
    }



    // ---------------------------------- load & create KF Database ----------------------------------
    std::cout << "create the database ..." << std::endl;
    // 记录 存储在 Database 中的 frame 对应的 frame id
    std::vector<unsigned long> vFrameIdInDatabase; 

    deeplcd::DeepLCD lcd; // Using default constructor, takes net from model directory downloaded on compilation
    // lcd.SetNumExclude(0);
    cv::Size sz(160, 120);

    for(size_t i = 0,  nImgs = vstrImageFilenames.size(); i < nImgs; i+=10)
    {
        cv::Mat originalImg = cv::imread(vstrImageFilenames[i], CV_LOAD_IMAGE_UNCHANGED);
        cv::Mat imgGray, imgResize;
        if (originalImg.channels() == 3){
            cv::cvtColor(originalImg, imgGray, cv::COLOR_BGR2GRAY); 
        }else{
            imgGray = originalImg.clone();
        }
        
        cv::GaussianBlur(imgGray, imgGray, cv::Size(7, 7), 0);
        cv::resize(imgGray, imgResize, sz);
        // const deeplcd::descriptor descr = lcd.calcDescr(imgResize);

        lcd.add(imgResize);
        vFrameIdInDatabase.push_back((int)i);
    }
    std::cout << "has done." << endl;


    // ---------------------------------- query the specified image in the database ----------------------------------

    for(size_t nImgId = 0,  nImgs = vstrImageFilenames.size(); nImgId < nImgs; nImgId++){
        std::cout << "current frame: " << nImgId << std::endl;
        int current_id = nImgId;
        cv::Mat originalImg = cv::imread(vstrImageFilenames[nImgId], CV_LOAD_IMAGE_UNCHANGED);
        cv::Mat imgGray, imgResize;
        if (originalImg.channels() == 3){
            cv::cvtColor(originalImg, imgGray, cv::COLOR_BGR2GRAY); 
        }else{
            imgGray = originalImg.clone();
        }
        cv::GaussianBlur(imgGray, imgGray, cv::Size(7, 7), 0);
        cv::resize(imgGray, imgResize, sz);
        const deeplcd::descriptor descr = lcd.calcDescr(imgResize);

        
        // ------------------   根据 groundtruth 位姿找出数据集中总共存在的正确匹配的个数
        int numGroundTruthP = 0;
        for(auto &db_id: vFrameIdInDatabase){

            if( std::abs((int)current_id - (int)db_id) < 100){ // 时间上临近的就不算 LOOP 了
                continue;
            }
            bool bGroundTruthCorrect = PoseIsVeryNear(vPoses_R, vPoses_t, current_id, db_id);
            
            if (bGroundTruthCorrect){
                std::cout << std::setprecision(3) << "GroundTruth for Image " 
                    << current_id << ":  db id: " << db_id  << std::endl;
                numGroundTruthP++;
            }
        }
        totalNumGroundTruthP += numGroundTruthP;


        // DeepLCD 查询
        deeplcd::QueryResults queryResults;
        lcd.query(descr, queryResults, vFrameIdInDatabase.size(), false); // add_after = false

        // ----------------------------------  不同 threshold 的结果, 查询阶段
        for(size_t index = 0; index < vThreshold.size(); index++){
            int numTP = 0;
            int numFP = 0;
                
            // 如果查询结果满足位姿要求，视为TP，否则，视为FP
            for(auto it = queryResults.begin(); it != queryResults.end(); it++){
                auto qr = *it;
                int loop_id = vFrameIdInDatabase[qr.id];

                if(qr.score < vThreshold[index]){
                    continue;
                } 

                if( std::abs((int)current_id - (int)loop_id) < 100){ // 时间上临近的就不算 LOOP 了
                    continue;
                }
                bool bCorrect = PoseIsVeryNear(vPoses_R, vPoses_t, current_id, loop_id);         
                bool bAcceptable = PoseIsAcceptablyNear(vPoses_R, vPoses_t, current_id, loop_id);
    
                if (bCorrect){
                    numTP++;
                } else if ( ! bAcceptable) {
                    numFP++;
                }

                std::cout << std::setprecision(3) << "Searching for Image " << current_id << ":  loop id: " << loop_id
                        << ", score: " << qr.score  << ", bAcceptable: " << bAcceptable << std::endl;
                // if(! bCorrect){
                //     cv::Mat current_img = cv::imread(vstrImageFilenames[current_id], CV_LOAD_IMAGE_UNCHANGED);
                //     cv::Mat loop_img = cv::imread(vstrImageFilenames[loop_id], CV_LOAD_IMAGE_UNCHANGED);
                //     cv::imshow("current_img", current_img);
                //     cv::imshow("loop_img", loop_img);
                //     cv::waitKey(0);
                // }
            }
            vTotalNumTP[index] += numTP;
            vTotalNumFP[index] += numFP;
        }

    }



    // 结果输出 保存至 txt 文件
    for(size_t i = 0; i < vThreshold.size(); i++){
        vTotalNumFN[i] = totalNumGroundTruthP - vTotalNumTP[i];

        double precision  = ((double)vTotalNumTP[i]) / (double)(vTotalNumTP[i] + vTotalNumFP[i]);
        double recall = (double)vTotalNumTP[i] / (double)(vTotalNumTP[i] + vTotalNumFN[i]);

        std::cout << "threshold:" << vThreshold[i] <<", precision: " << precision << ", recall: " << recall << std::endl;
        f << std::setprecision(6) << vThreshold[i] << " " << precision << " " << recall << std::endl;
    }

    f.close();
    
    return 0;
} 


// --------------------------------------------------------------------------------------------------------------------------

bool PoseIsVeryNear(std::vector<Mat33, Eigen::aligned_allocator<Mat33> > &vPoses_R,
    std::vector<Vec3, Eigen::aligned_allocator<Vec3> > &vPoses_t,
    int current_id, int loop_id){

    Eigen::AngleAxisd rotation_vector;
    rotation_vector.fromRotationMatrix(vPoses_R[current_id].inverse() * vPoses_R[loop_id]);

    bool bCorrect = ((vPoses_t[current_id] - vPoses_t[loop_id]).norm() < 5)
                                        && ((std::abs(rotation_vector.angle()) < 3.14 / 12));
                                    
    return bCorrect;
}

// --------------------------------------------------------------------------------------------------------------------------

bool PoseIsAcceptablyNear(std::vector<Mat33, Eigen::aligned_allocator<Mat33> > &vPoses_R,
    std::vector<Vec3, Eigen::aligned_allocator<Vec3> > &vPoses_t,
    int current_id, int loop_id){

    Eigen::AngleAxisd rotation_vector;
    rotation_vector.fromRotationMatrix(vPoses_R[current_id].inverse() * vPoses_R[loop_id]);

    bool bCorrect = ((vPoses_t[current_id] - vPoses_t[loop_id]).norm() < 20)
                                        && ((std::abs(rotation_vector.angle()) < 3.14 / 6));
                                    
    return bCorrect;
}

// --------------------------------------------------------------------------------------------------------------------------
std::pair<double, double> VectorMeanAndVariance(const std::vector<double> v){
    double sum = std::accumulate(std::begin(v), std::end(v), 0.0);
    double m =  sum / v.size();
    double accum = 0.0;
    std::for_each (std::begin(v), std::end(v), [&](const double d) {
        accum += (d - m) * (d - m);
    });
    double stdev = sqrt(accum / (v.size()-1));

    std::pair<double, double> pairMeanAndVariance;
    pairMeanAndVariance.first = m;
    pairMeanAndVariance.second = stdev;

    return pairMeanAndVariance;
}


// --------------------------------------------------------------------------------------------------------------------------
void LoadImages(const string &strPathToSequence, vector<string> &vstrImageFilenames, vector<double> &vTimestamps)
{
    ifstream fTimes;
    string strPathTimeFile = strPathToSequence + "/times.txt";
    fTimes.open(strPathTimeFile.c_str());
    while(!fTimes.eof())
    {
        string s;
        getline(fTimes,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            double t;
            ss >> t;
            vTimestamps.push_back(t);
        }
    }
    fTimes.close();

    string strPrefixLeft = strPathToSequence + "/image_0/";  // 使用 gray 图

    const int nTimes = vTimestamps.size();
    vstrImageFilenames.resize(nTimes);

    for(int i=0; i<nTimes; i++)
    {
        stringstream ss;
        ss << setfill('0') << setw(6) << i;
        vstrImageFilenames[i] = strPrefixLeft + ss.str() + ".png";
    }
}

// ---------------------------------------------------------------------------------------

void LoadGroundtruthPose(const string &strPose, 
    std::vector<Mat33, Eigen::aligned_allocator<Mat33> > &vPoses_R,
    std::vector<Vec3, Eigen::aligned_allocator<Vec3> > &vPoses_t){

    vPoses_R.clear();
    vPoses_t.clear();

    ifstream fPoses;
    std::string strPathPoseFile = strPose;
    fPoses.open(strPathPoseFile.c_str());
    while(!fPoses.eof())
    {
        double R0, R1, R2, R3, R4, R5, R6, R7, R8;
        double t0, t1, t2;
        fPoses >> R0 >> R1 >> R2 >> t0
                      >> R3 >> R4 >> R5 >> t1
                      >> R6 >> R7 >> R8 >> t2;

        Mat33 R; 
        R << R0, R1, R2,
                  R3, R4, R5,
                  R6, R7, R8;
        Vec3 t; 
        t << t0, t1, t2;

        vPoses_R.push_back(R);
        vPoses_t.push_back(t);
    }

    fPoses.close();
}