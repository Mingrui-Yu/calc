/**
 *  测试在 Alderley 数据集上的 PR 曲线
 * 
 */

#include "deeplcd.h"

#include <iostream>
#include <fstream>
#include <chrono>
#include <numeric>
#include <iomanip>
#include <cmath>
#include <boost/format.hpp>

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace std;

typedef Eigen::Matrix<double, 3, 3> Mat33;
typedef Eigen::Matrix<double, 3, 1> Vec3;


void LoadCSV(const std::string &strCSVpath, std::vector<int> &vnImageIdxNight, std::vector<int> &vnImageIdxDay);


// ----------------------------------------------------------------------------------------------------

int main(int argc, char** argv)
{
    std::string dir = argv[1];


    // ---------------------------------------------  加载 CSV 文件 -----------------------------------------------------
    std::string csvpath = dir + "/framematches.csv";
    std::vector<int>vnImageNightIdx;
    std::vector<int>vnImageDayIdx;
    LoadCSV(csvpath, vnImageNightIdx, vnImageDayIdx);
    assert(vnImageNightIdx.size() == vnImageDayIdx.size());
    std::vector<std::pair<int, int>> vpairImageMatchIdx;
    for(size_t i = 0; i < vnImageNightIdx.size(); i++){
        vpairImageMatchIdx.push_back({vnImageNightIdx[i], vnImageDayIdx[i]});
    }

    // ----------------------------------------- 一些数据初始化 ----------------------------------------------
    int totalNumGroundTruthP = 0;
    // 存储不同 threshold 下的结果
    std::vector<double> vThreshold;
    std::vector<int> vTotalNumTP;
    std::vector<int> vTotalNumFP;
    std::vector<int> vTotalNumFN;
    for(double thres = 0.70; thres >= 0.70; thres -= 0.01){
        vThreshold.push_back(thres);
        vTotalNumTP.push_back(0);
        vTotalNumFP.push_back(0);
        vTotalNumFN.push_back(0);
    }

    int imageBegin = vpairImageMatchIdx.size() - 200;
    int imageEnd = vpairImageMatchIdx.size();

    // -------------------------------------------- 创建 Database -------------------------------------------------------
    std::string imageNightDir = dir + "/FRAMESA";
    std::string imageDayDir = dir + "/FRAMESB";
    boost::format fmt("%s/Image%05d.jpg");

     std::vector<int> vFrameIdInDatabase; 

    deeplcd::DeepLCD lcd; // Using default constructor, takes net from model directory downloaded on compilation
    cv::Size sz(160, 120);

    for(size_t i =  imageBegin,  nImgs = imageEnd; i < nImgs; i+=1)
    {
        std::string imageName = (fmt % imageNightDir % vpairImageMatchIdx[i].first).str();
        // cout << "imageName: " << imageName << std::endl;
        cv::Mat imgGray = cv::imread(imageName, cv::IMREAD_GRAYSCALE);
        cv::Mat imgResize;
        // cv::GaussianBlur(imgGray, imgGray, cv::Size(7, 7), 0);
        cv::resize(imgGray, imgResize, sz);
        deeplcd::descriptor descr = lcd.calcDescr(imgResize);

        lcd.add(imgResize);
        vFrameIdInDatabase.push_back(vpairImageMatchIdx[i].first);


        std::string imageName_day = (fmt % imageDayDir % vpairImageMatchIdx[i].second).str();
        // cout << "imageName: " << imageName << std::endl;
        cv::Mat imgGray_day = cv::imread(imageName_day, cv::IMREAD_GRAYSCALE);
        cv::Mat imgResize_day;
        // cv::GaussianBlur(imgGray_day, imgGray_day, cv::Size(7, 7), 0);
        cv::resize(imgGray_day, imgResize_day, sz);
        deeplcd::descriptor descr_day = lcd.calcDescr(imgResize_day);

        std::cout << "night id: " << vpairImageMatchIdx[i].first << ", day id: " << vpairImageMatchIdx[i].second 
                << ", score: " << lcd.score(descr.descr, descr_day.descr) << std::endl;
    }
    std::cout << "has done." << endl;


    // ---------------------------------- query the specified image in the database ----------------------------------
    std::cout << "start querying ..." << std::endl;
    for(size_t i = imageBegin,  N = imageEnd; i < N; i++){
        std::cout << "current image id: " << vpairImageMatchIdx[i].second << std::endl;
        std::string currentImageName = (fmt % imageDayDir % vpairImageMatchIdx[i].second).str();
        cv::Mat imgGray = cv::imread(currentImageName, cv::IMREAD_GRAYSCALE);
        cv::Mat imgResize;
        // cv::GaussianBlur(imgGray, imgGray, cv::Size(7, 7), 0);
        cv::resize(imgGray, imgResize, sz);

        const deeplcd::descriptor descr = lcd.calcDescr(imgResize);
        // DeepLCD 查询
        deeplcd::QueryResults queryResults;
        lcd.query(descr, queryResults, 10, false); // add_after = false

        // ----------------------------------  不同 threshold 的结果, 查询阶段
        for(size_t index = 0; index < vThreshold.size(); index++){
            for(auto it = queryResults.begin(); it != queryResults.end(); it++){
                auto qr = *it;
                int groundtruthId = vpairImageMatchIdx[i].first;
                int queryId = vFrameIdInDatabase[qr.id];

                // if(qr.score < vThreshold[index]){
                //     continue;
                // } 

                bool bCorrect = std::abs(queryId - groundtruthId) <= 20;
                std::cout << "groundtruthId: " << groundtruthId << ", queryId: " << queryId << ", score: " << qr.score
                    << ", bCorrect: " << bCorrect << std::endl;

                cv::Mat current_img = cv::imread(currentImageName, CV_LOAD_IMAGE_UNCHANGED);
                cv::Mat loop_img = cv::imread((fmt % imageNightDir % queryId).str(), CV_LOAD_IMAGE_UNCHANGED);
                cv::imshow("current_img", current_img);
                cv::imshow("loop_img", loop_img);
                cv::waitKey(1);
            }
        }
        
    }

    // f.close();
	
    return 0;
} 





// ---------------------------------------------------------------------------------

void LoadCSV(const std::string &strCSVpath, std::vector<int> &vnImageNightIdx, std::vector<int> &vnImageDayIdx){

    std::fstream fin;
    fin.open(strCSVpath, ios::in);

    if(!fin){
        std::cerr << "error: cannot open the csv file " << strCSVpath << std::endl;
        return;
    }

    std::string line, word, idxA, idxB;

    while(!fin.eof()){
        getline(fin, line);
        if(!line.empty()){
            std::stringstream s(line);
            getline(s, idxA, ',');
            getline(s, idxB, ',');
            vnImageNightIdx.push_back(std::stoi(idxA));
            vnImageDayIdx.push_back(std::stoi(idxB));
        }
    }
}