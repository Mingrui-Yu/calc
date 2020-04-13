/**
 * This file is to test its performance.
 * Test 1: simply query a image with highest similarity score compared to the specified image in KITTI color sequence 00.
 *  With no other constraint.
 * calculate the mean and standard variance of the time cost
 *  -- time cost for extracting feature vectors
 *  -- time cost for adding one image's features to database
 *  -- time cost for querying
 */

#include "deeplcd.h"

#include <iostream>
#include <fstream>
#include <chrono>
#include <numeric>
// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"


using namespace std;

void LoadImages(const string &strSequence, vector<string> &vstrImageFilenames,
                vector<double> &vTimestamps);

std::pair<double, double> VectorMeanAndVariance(const std::vector<double> v);



int main(int argc, char** argv)
{
	if(argc != 2)
    {
        std::cerr << std::endl << "Usage: ./mytest/test1   path_to_sequence" << std::endl;
        return 1;
    }

    cv::Size sz(160, 120);
    std::chrono::steady_clock::time_point t1;
    std::chrono::steady_clock::time_point t2;
    std::chrono::duration<double> time_used;

    // ---------------------------------- Load the images' path -----------------------------------------
    std::string strPathToSequence = argv[1];
    std::vector<string> vstrImageFilenames;
    std::vector<double> vTimestamps;
    LoadImages(strPathToSequence,  vstrImageFilenames, vTimestamps);

    // ---------------------------------- create the file to store test results ----------------------------------
    ofstream f;
    std::string filename = "mytest/result/test1_time_cost_result.txt";
    f.open(filename.c_str());
    f << fixed;

    // ---------------------------------- load & create KF Database ----------------------------------
    std::cout << "create the database ..." << std::endl;
    t1 = std::chrono::steady_clock::now(); // time cost for loading the network model
    deeplcd::DeepLCD lcd; // Using default constructor, takes net from model directory downloaded on compilation
    t2 = std::chrono::steady_clock::now();
    time_used = std::chrono::duration_cast <std::chrono::duration<double>> (t2 - t1);
    f  <<  setprecision(3) << " -- time cost for loading the network model " << time_used.count() << "s" << std::endl << std::endl; 

    std::vector<double> vTimeAddToDatabase;
    vTimeAddToDatabase.reserve(vstrImageFilenames.size());
    std::vector<double> vTimeExtractFeatureVectors;
    vTimeExtractFeatureVectors.reserve(vstrImageFilenames.size());

    for(size_t i = 0,  nImgs = vstrImageFilenames.size(); i < nImgs; i+=10)
    {
        cv::Mat originalImg = cv::imread(vstrImageFilenames[i], CV_LOAD_IMAGE_UNCHANGED);
        cv::Mat imgGray, imgResize;
        if (originalImg.channels() == 3){
            cv::cvtColor(originalImg, imgGray, cv::COLOR_BGR2GRAY); 
        }else{
            imgGray = originalImg.clone();
        }
        
        t1 = std::chrono::steady_clock::now();  // time cost for extracting feature vector 
        cv::resize(imgGray, imgResize, sz);
        const deeplcd::descriptor descr = lcd.calcDescr(imgResize);
        t2 = std::chrono::steady_clock::now();
        time_used = std::chrono::duration_cast <std::chrono::duration<double>> (t2 - t1);
        vTimeExtractFeatureVectors.push_back(time_used.count());

        t1 = std::chrono::steady_clock::now();  // time cost for add one image's features to databse
        lcd.add(descr);
        t2 = std::chrono::steady_clock::now();
        time_used = std::chrono::duration_cast <std::chrono::duration<double>> (t2 - t1);
        vTimeAddToDatabase.push_back(time_used.count());
    }
    std::cout << "has done." << endl;


    // ---------------------------------- query the specified image in the database & record the time cost ----------------------------------
    std::cout << "start test: time cost of querying a single image ... " << endl;
    std::vector<double> vTimeQuerying(vstrImageFilenames.size());

    // do the test for every image in the sequence, and calculate the mean and standard variance of the time cost
    for(size_t nImgId = 0,  nImgs = vstrImageFilenames.size(); nImgId < nImgs; nImgId++){
        cv::Mat originalImg = cv::imread(vstrImageFilenames[nImgId], CV_LOAD_IMAGE_UNCHANGED);
        cv::Mat imgGray, imgResize;
        if (originalImg.channels() == 3){
            cv::cvtColor(originalImg, imgGray, cv::COLOR_BGR2GRAY); 
        }else{
            imgGray = originalImg.clone();
        }
        cv::resize(imgGray, imgResize, sz);

        const deeplcd::descriptor descr = lcd.calcDescr(imgResize);

        t1 = std::chrono::steady_clock::now(); // time cost for querying
        deeplcd::query_result queryResult = lcd.query(descr, 0); // query(descriptor, false) means return 1 result in q, and don't add im's descriptor to database afterwards
        t2 = std::chrono::steady_clock::now();
        time_used = std::chrono::duration_cast <std::chrono::duration<double>> (t2 - t1);
        // std::cout << " -- time cost for querying: " << time_used.count() << "s" << std::endl; 
        std::cout << "Searching for Image " << nImgId << ": " << queryResult << std::endl;
        vTimeQuerying[nImgId] = time_used.count();
    }

    // calculate the mean and standard variance of time cost, then show them
    std::pair<double, double>  pairMeanAndVariance;
    pairMeanAndVariance = VectorMeanAndVariance(vTimeExtractFeatureVectors);
    f <<  setprecision(6) << " -- time cost for extracting features: "<< std::endl << " mean: " << pairMeanAndVariance.first 
        << "s, stdev: " <<  pairMeanAndVariance.second << "s" << std::endl << std::endl; 

    pairMeanAndVariance = VectorMeanAndVariance(vTimeAddToDatabase);
    f <<  setprecision(6) << " -- time cost for adding one image's features to database: "<< std::endl << " mean: " << pairMeanAndVariance.first 
        << "s, stdev: " <<  pairMeanAndVariance.second << "s" << std::endl << std::endl; 

    pairMeanAndVariance = VectorMeanAndVariance(vTimeQuerying);
    f <<  setprecision(6) << " -- time cost for querying: "<< std::endl << " mean: " << pairMeanAndVariance.first 
        << "s, stdev: " <<  pairMeanAndVariance.second << "s" << std::endl << std::endl; 
    
    f.close();
    std::cout << "time cost test result has been saved to  "  << filename << "." << std::endl;


    
    return 0;
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

    string strPrefixLeft = strPathToSequence + "/image_0/";  // 使用 rgb 图

    const int nTimes = vTimestamps.size();
    vstrImageFilenames.resize(nTimes);

    for(int i=0; i<nTimes; i++)
    {
        stringstream ss;
        ss << setfill('0') << setw(6) << i;
        vstrImageFilenames[i] = strPrefixLeft + ss.str() + ".png";
    }
}