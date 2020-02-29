#include "deeplcd.h"

#include <iostream>
#include <fstream>
#include <chrono>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace std;


void LoadImages(const string &strSequence, vector<string> &vstrImageFilenames,
                vector<double> &vTimestamps);



int main(int argc, char** argv)
{
	std::cout <<  std::endl << "Deep Loop Closure Detection Demo For KITTI !\n";

    if(argc != 2)
    {
        std::cerr << std::endl << "Usage: ./KITTI_demo   path_to_sequence" << std::endl;
        return 1;
    }

    // Retrieve paths to images and the timestamps
    std::vector<std::string> vstrImageFilenames;
    std::vector<double> vTimestamps;
    std::string sequenceDir = std::string(argv[1]);
    LoadImages(sequenceDir, vstrImageFilenames, vTimestamps);

    cv::Size sz(160, 120);
    int nImages =  vstrImageFilenames.size();
    int nKFs = 0;

    deeplcd::DeepLCD test_lcd; // Using default constructor, takes net from rvbaa_model directory downloaded on compilation
    deeplcd::query_result q;

    std::cout << "\n---------------------  Start loading frames ... ---------------------\n";

    cv::Mat im;
    cv::Mat imResize;
    std::vector<cv::Mat> vKFs; 
    cv::Mat loopKF;
    std::vector<bool> vbLoopCandidateDetected;
    int niKF = 0;
    int nCoolingTime = 0;

    for(int ni = 0; ni < nImages; ni += 7){
        im = cv::imread(vstrImageFilenames[ni],CV_LOAD_IMAGE_UNCHANGED);
        cv::cvtColor(im, im, cv::COLOR_BGR2GRAY); 
        cv::resize(im, imResize, sz);

        vbLoopCandidateDetected.push_back(false);

        if (niKF > 50 && nCoolingTime <= 0){
            q = test_lcd.query(imResize, 0);
            if (q.score > 0.92){
                vbLoopCandidateDetected[niKF] = true;
                bool bConsecutiveLoopDetected = true;
                for (int i = 0; i < 5; i++){
                    if (vbLoopCandidateDetected[niKF - i] == false)
                    bConsecutiveLoopDetected = false;
                    break;
                }
                if(bConsecutiveLoopDetected){
                    std::cout <<  "Current KF ID: " << niKF << ", Query result: " << q << "\n";
                    loopKF = vKFs[q.id];
                    cv::imshow("Current KF", im);
                    cv::imshow("Loop KF", loopKF);
                    cv::waitKey(0);
                    nCoolingTime = 10; 
                }
            }
        }

        test_lcd.add(imResize);
        vKFs.push_back(im);

        nKFs++; 
        niKF++; 
        nCoolingTime--;

        if( nKFs % 100 == 99)  std::cout << "Has processed " << nKFs + 1 << "KFs" << std::endl;
    }
    std::cout << "\n--------------------- "  << nKFs <<   " KFs are processed and stored in database.---------------------\n";
    
    return 0;
} 






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

    string strPrefixLeft = strPathToSequence + "/image_2/";  // 使用 rgb 图

    const int nTimes = vTimestamps.size();
    vstrImageFilenames.resize(nTimes);

    for(int i=0; i<nTimes; i++)
    {
        stringstream ss;
        ss << setfill('0') << setw(6) << i;
        vstrImageFilenames[i] = strPrefixLeft + ss.str() + ".png";
    }
}