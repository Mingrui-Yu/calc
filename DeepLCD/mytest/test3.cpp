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
    for(double thres = 0.90; thres >= 0.60; thres -= 0.01){
        vThreshold.push_back(thres);
        vTotalNumTP.push_back(0);
        vTotalNumFP.push_back(0);
        vTotalNumFN.push_back(0);
    }

    int imageBegin = 10000;
    int imageEnd = vpairImageMatchIdx.size();
    int imageInterval = 50;
    // -------------------------------------------- 创建 Database -------------------------------------------------------
    std::string imageNightDir = dir + "/FRAMESA";
    std::string imageDayDir = dir + "/FRAMESB";
    boost::format fmt("%s/Image%05d.jpg");

     std::vector<int> vFrameIdInDatabase; 

    deeplcd::DeepLCD lcd; // Using default constructor, takes net from model directory downloaded on compilation
    cv::Size sz(160, 120);

    for(size_t i =  imageBegin,  nImgs = imageEnd; i < nImgs; i+= imageInterval)
    {
        std::string imageName = (fmt % imageDayDir % vpairImageMatchIdx[i].second).str();
        // cout << "imageName: " << imageName << std::endl;
        cv::Mat imgGray = cv::imread(imageName, cv::IMREAD_GRAYSCALE);
        cv::Mat imgResize;
        // cv::GaussianBlur(imgGray, imgGray, cv::Size(7, 7), 0);
        cv::resize(imgGray, imgResize, sz);
        deeplcd::descriptor descr = lcd.calcDescr(imgResize);

        lcd.add(imgResize);
        vFrameIdInDatabase.push_back(vpairImageMatchIdx[i].second);


        std::string imageName_night = (fmt % imageNightDir % vpairImageMatchIdx[i].first).str();
        // cout << "imageName: " << imageName << std::endl;
        cv::Mat imgGray_night = cv::imread(imageName_night, cv::IMREAD_GRAYSCALE);
        cv::Mat imgResize_night;
        // cv::GaussianBlur(imgGray_night, imgGray_night, cv::Size(7, 7), 0);
        cv::resize(imgGray_night, imgResize_night, sz);
        deeplcd::descriptor descr_night = lcd.calcDescr(imgResize_night);

        // cv::imshow("nightgray", imgGray_night);
        // cv::imshow("daygray", imgGray);
        // cv::waitKey(1);

        std::cout << "night id: " << vpairImageMatchIdx[i].first << ", day id: " << vpairImageMatchIdx[i].second 
                << ", score: " << lcd.score(descr.descr, descr_night.descr) << std::endl;
    }
    std::cout << "has done." << endl;

    totalNumGroundTruthP = vFrameIdInDatabase.size();

    // ---------------------------------- query the specified image in the database ----------------------------------
    std::cout << "start querying ..." << std::endl;
    for(size_t i = imageBegin,  N = imageEnd; i < N; i += imageInterval){
        std::cout << "current image id: " << vpairImageMatchIdx[i].first << std::endl;
        std::string currentImageName = (fmt % imageNightDir % vpairImageMatchIdx[i].first).str();
        cv::Mat imgGray = cv::imread(currentImageName, cv::IMREAD_GRAYSCALE);
        cv::Mat imgResize;
        // cv::GaussianBlur(imgGray, imgGray, cv::Size(7, 7), 0);
        cv::resize(imgGray, imgResize, sz);

        const deeplcd::descriptor descr = lcd.calcDescr(imgResize);
        // DeepLCD 查询
        deeplcd::QueryResults queryResults;
        lcd.query(descr, queryResults, vFrameIdInDatabase.size(), false); // add_after = false

        // ----------------------------------  不同 threshold 的结果, 查询阶段
        for(size_t index = 0; index < vThreshold.size(); index++){
            int numTP = 0;
            int numFP = 0;

            for(auto it = queryResults.begin(); it != queryResults.end(); it++){
                auto qr = *it;
                int groundtruthId = vpairImageMatchIdx[i].second;
                int queryId = vFrameIdInDatabase[qr.id];

                if(qr.score < vThreshold[index]){
                    continue;
                } 

                bool bCorrect = std::abs(queryId - groundtruthId) <= 20;

                if(bCorrect){
                    numTP++;
                } else {
                    numFP++;
                }
                std::cout << "groundtruthId: " << groundtruthId << ", queryId: " << queryId << ", score: " << qr.score
                    << ", bCorrect: " << bCorrect << std::endl;

            }
            vTotalNumTP[index] += numTP;
            vTotalNumFP[index] += numFP;
            // if(numTP)
            
        }
    }

        // 结果输出 保存至 txt 文件
    for(size_t i = 0; i < vThreshold.size(); i++){
        vTotalNumFN[i] = totalNumGroundTruthP - vTotalNumTP[i];

        double precision;
        if (vTotalNumTP[i] + vTotalNumFP[i] == 0){
            precision = 1.0;
        }else{
            precision  = ((double)vTotalNumTP[i]) / (double)(vTotalNumTP[i] + vTotalNumFP[i]);
        }
        double recall = (double)vTotalNumTP[i] / (double)(vTotalNumTP[i] + vTotalNumFN[i]);

        
        std::cout << "threshold: " << vThreshold[i] <<", precision: " << precision << ", recall: " << recall << std::endl;
        std::cout << "TP: " << vTotalNumTP[i] << ", FP: " << vTotalNumFP[i] << ", FN: " << vTotalNumFN[i] << std::endl;
        // f << std::setprecision(6) << vThreshold[i] << " " << precision << " " << recall << std::endl;
    }

    // std::cout << "vTotalNumTP: " << vTotalNumTP[0] << std::endl;
    // std::cout << "vTotalNumFP: " << vTotalNumFP[0] << std::endl;
    // std::cout << "vTotalNumFN: " << vFrameIdInDatabase.size() - vTotalNumTP[0] << std::endl;

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