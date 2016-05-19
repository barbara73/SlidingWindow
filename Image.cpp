
//
//  Image.cpp
//  SlidingWindow
//
//  Created by otl on 26/04/16.
//  Copyright © 2016 otl. All rights reserved.
//

#include "Image.hpp"
#include <vector>
#include <array>
#include <iostream>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;


// constructor initialising number of reduction, max stride and reduction by param
//--------------------------------------------------------------------------------
Image::Image(): nbReductions{3}, stride{15}, reductionParameter{0.85}, thresholdMagnitude{3}, partialRectangleNB{1111} {}
Image::Image(int red, int str, float redParam, int th, int nbPartRect):nbReductions{red}, stride{str}, reductionParameter{redParam}, thresholdMagnitude{th}, partialRectangleNB{nbPartRect} {}


// convert vector to cv::mat
//--------------------------
Mat Image::vectorToMat(vector<vector<float>> vec) {
    
    Mat mat(0, (int)vec[0].size(), DataType<float>::type);      //create a new, empty Mat with the row size of vec
    
    for (auto i = 0; i < vec.size(); ++i) {
        
        Mat sample(1, (int)vec[0].size(), DataType<float>::type, vec[i].data());    //make temporary Mat row and add
                                                                                    //to mat without data copy
        mat.push_back(sample);
    }
    return mat;
}


// downscale image
//----------------
Mat Image::downscale_image(Mat img) {
    Mat paddedImg, downscaledImg, resizedMap, map, tempMap, tempImg;
    int w = (blockWidth-1)/2;
    int h = (blockHeight-1)/2;
    int borderType = BORDER_CONSTANT;
    newMap = Mat::zeros(imgHeight, imgWidth, CV_32F);
    tempMap = Mat::zeros(imgHeight, imgWidth, CV_32F);
    vector<vector<float>> vectorMap;
    
    downscaledImg = img;
    
    //auto t0 = chrono::high_resolution_clock::now();
    for (int i=0; i!=nbReductions; ++i) {
        
        copyMakeBorder(downscaledImg, paddedImg, h, h, w, w, borderType, 0);    //padd image with zeros
        
        vectorMap = slide_window_over_image(i, paddedImg);  // slide window over downscaled image
        
        map.release();
        map = vectorToMat(vectorMap);                       //make Mat image which is smaller than before
        vector<vector<float>> newVector;
        vectorMap.swap(newVector);
        
        const Size size(imgWidth, imgHeight);
        resize(map, resizedMap, size);                      // resize map to original size WHY NOT WORKING???????
        cout << "map: " << map.size() << "resized Map: " << resizedMap.size() << endl;
    
        max(resizedMap, newMap, tempMap);                   // take maximum pixel of each map
        newMap = tempMap;
        
        resize(downscaledImg, tempImg, Size(), reductionParameter, reductionParameter, CV_INTER_AREA);          //ok
        downscaledImg = tempImg;
    }
    
    //auto t1 = chrono::high_resolution_clock::now();
    //cout << chrono::duration_cast<chrono::seconds>(t1-t0).count() << " sec for all images\n";
    
    return newMap;
}


// slide window over image
//------------------------
vector<vector<float>> Image::slide_window_over_image(int param, Mat img) {
    stride -= param;
    
    vector<float> oFeatures(img.cols-blockWidth+1);         //orientation features
    vector<vector<float>> totalFeaturePerPatch(img.rows-blockHeight+1, vector<float>(img.cols-blockWidth+1));
    vector<float> orientationFeatures;
    /*for (auto v : totalFeaturePerPatch) {                   //initialise vector with zeros
        for (auto s : v) {
            s = 0.;
        }
    }*/
    
    Mat image_grey;
    // change colour image to greyscale image
    cvtColor(img, image_grey, CV_BGR2GRAY);
    
    for (int r=0; r<image_grey.rows-blockHeight; r=r+stride) {
        for (int c=0; c<image_grey.cols-blockWidth; c=c+stride) {
            
            Rect roi(c, r, blockWidth, blockHeight);
            Mat imageRoi = image_grey(roi);
            
            Scalar meanValue = mean(imageRoi);
            
            float score;
            if (meanValue[0]>20) {
                
                
                orientationFeatures = make_orientationHistogramFeatures(imageRoi);
                score = testTheDataXGBoost(classifier, orientationFeatures, 1, partialRectangleNB*9);
                
            }
            else score = 0.;

            oFeatures.push_back(score);
        }
        totalFeaturePerPatch.push_back(oFeatures);
    }
    
    return totalFeaturePerPatch;
}




// Test data with xgboost
//-----------------------
float Image::testTheDataXGBoost(BoosterHandle handle, vector<float> test, int r, int c) {
    DMatrixHandle h_test;
    XGDMatrixCreateFromMat((float *) &test[0], r, c, -1, &h_test);
    bst_ulong out_len;
    const float *f;
    XGBoosterPredict(handle, h_test, 0, 0, &out_len, &f);
    
    for (unsigned int i=0;i<out_len;i++)
        std::cout << "prediction[" << i << "]=" << f[i] << std::endl;
    XGDMatrixFree(h_test);
    return f[0];
}


// calculate the orientation histogram features
//---------------------------------------------
std::vector<float> Image::make_orientationHistogramFeatures(Mat& image_grey) {
    
    Mat magnitude, direction, grad_y, grad_x;//, image_grey;
    int scale = 1;
    int delta = 0;
    int ddepth = CV_32F;
    
    // change colour image to greyscale image
    //cvtColor(image, image_grey, CV_BGR2GRAY);
    
    // apply sobel to get gradient image
    Sobel(image_grey, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);
    Sobel(image_grey, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
    
    bool angleInDegrees = true;
    
    // determine the gradient magnitude image and the gradient direction image
    cartToPolar(grad_x, grad_y, magnitude, direction, angleInDegrees);
    
    int const max_BINARY_value = 1;
    
    Mat orient0;                                            //mask
    threshold(magnitude, orient0, thresholdMagnitude, max_BINARY_value, THRESH_BINARY);
    
    return group_to_orientations(direction, orient0);       //regionSum;
}


// group gradients into 8 orientations
//------------------------------------
std::vector<float> Image::group_to_orientations(Mat dir, Mat edge0) {
    
    Mat orient1 = Mat::zeros(dir.rows, dir.cols, CV_32F);
    Mat orient2 = Mat::zeros(dir.rows, dir.cols, CV_32F);
    Mat orient3 = Mat::zeros(dir.rows, dir.cols, CV_32F);
    Mat orient4 = Mat::zeros(dir.rows, dir.cols, CV_32F);
    Mat orient5 = Mat::zeros(dir.rows, dir.cols, CV_32F);
    Mat orient6 = Mat::zeros(dir.rows, dir.cols, CV_32F);
    Mat orient7 = Mat::zeros(dir.rows, dir.cols, CV_32F);
    Mat orient8 = Mat::zeros(dir.rows, dir.cols, CV_32F);
    
    for(int i=0; i<dir.rows; i++) {
        
        float* d = dir.ptr<float>(i);
        float* row1 = orient1.ptr<float>(i);
        float* row2 = orient2.ptr<float>(i);
        float* row3 = orient3.ptr<float>(i);
        float* row4 = orient4.ptr<float>(i);
        float* row5 = orient5.ptr<float>(i);
        float* row6 = orient6.ptr<float>(i);
        float* row7 = orient7.ptr<float>(i);
        float* row8 = orient8.ptr<float>(i);
        
        for(int j=0; j<dir.cols; j++) {
            
            float value = d[j];
            
            if (value >= 0 && value < 45)
                row1[j] = 1;
            else if (value >= 45 && value < 90)
                row2[j] = 1;
            else if (value >= 90 && value < 135)
                row3[j] = 1;
            else if (value >= 135 && value < 180)
                row4[j] = 1;
            else if (value >= 180 && value < 225)
                row5[j] = 1;
            else if (value >= 225 && value < 270)
                row6[j] = 1;
            else if (value >= 270 && value < 315)
                row7[j] = 1;
            else if (value >= 315 && value < 360)
                row8[j] = 1;
            else
                std::cout << "there is a mistake! degrees from [0, 360)" << std::endl;
        }
    }
    
    Mat integralImg0, integralImg1, integralImg2, integralImg3, integralImg4, integralImg5, integralImg6, integralImg7, integralImg8;
    Mat edge1, edge2, edge3, edge4, edge5, edge6, edge7, edge8;
    
    integral(edge0, integralImg0);
    multiply(edge0, orient1, edge1, 1, -1);
    integral(edge1, integralImg1);
    multiply(edge0, orient2, edge2, 1, -1);
    integral(edge2, integralImg2);
    multiply(edge0, orient3, edge3, 1, -1);
    integral(edge3, integralImg3);
    multiply(edge0, orient4, edge4, 1, -1);
    integral(edge4, integralImg4);
    multiply(edge0, orient5, edge5, 1, -1);
    integral(edge5, integralImg5);
    multiply(edge0, orient6, edge6, 1, -1);
    integral(edge6, integralImg6);
    multiply(edge0, orient7, edge7, 1, -1);
    integral(edge7, integralImg7);
    multiply(edge0, orient8, edge8, 1, -1);
    integral(edge8, integralImg8);
    
    
    // evaluate the sum of the patch for normalisation
    int max0, max1, max2, max3, max4, max5, max6, max7, max8;
    int x = integralImg0.cols;
    int y = integralImg0.rows;
    max0 = integralImg0.at<double>(Point(x,y));
    max1 = integralImg1.at<double>(Point(x,y));
    max2 = integralImg2.at<double>(Point(x,y));
    max3 = integralImg3.at<double>(Point(x,y));
    max4 = integralImg4.at<double>(Point(x,y));
    max5 = integralImg5.at<double>(Point(x,y));
    max6 = integralImg6.at<double>(Point(x,y));
    max7 = integralImg7.at<double>(Point(x,y));
    max8 = integralImg8.at<double>(Point(x,y));
    
    // calculate the region sum in a rectangle of each map
    std::vector<float> newSum {};
    
    for (int i = 0; i != partialRectangleNB; ++i) {
        
        Rect bb = boundingBox[i];
        float sum0, sum1, sum2, sum3, sum4, sum5, sum6, sum7, sum8;
        
        sum0 = calculateImageIntegral(integralImg0, bb, max0);
        sum1 = calculateImageIntegral(integralImg1, bb, max1);
        sum2 = calculateImageIntegral(integralImg2, bb, max2);
        sum3 = calculateImageIntegral(integralImg3, bb, max3);
        sum4 = calculateImageIntegral(integralImg4, bb, max4);
        sum5 = calculateImageIntegral(integralImg5, bb, max5);
        sum6 = calculateImageIntegral(integralImg6, bb, max6);
        sum7 = calculateImageIntegral(integralImg7, bb, max7);
        sum8 = calculateImageIntegral(integralImg8, bb, max8);
        
        // histogram of ith rectangle
        std::vector<float> totalSum = {sum0, sum1, sum2, sum3, sum4, sum5, sum6, sum7, sum8};
        newSum.insert (newSum.end(), totalSum.begin(), totalSum.end());
        
    }
    return newSum;
}

//calculate rectangle inside image integral
//-----------------------------------------
float Image::calculateImageIntegral(Mat integralImg, Rect rect, int max) {
    float a = (integralImg.at<double>(Point(rect.x+rect.width, rect.y+rect.height)) -
               integralImg.at<double>(Point(rect.x+rect.width, rect.y)) -
               integralImg.at<double>(Point(rect.x, rect.y+rect.height)) +
               integralImg.at<double>(Point(rect.x, rect.y))) / (max+0.000001);
    return a;
  
}


// destructor
//-----------
Image::~Image() {}