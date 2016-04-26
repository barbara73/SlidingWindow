//
//  Image.hpp
//  SlidingWindow
//
//  Created by otl on 26/04/16.
//  Copyright © 2016 otl. All rights reserved.
//

#ifndef Image_hpp
#define Image_hpp

#include <stdio.h>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "xgboost/c_api.h"
#include <dmlc/timer.h>

class Image {
    const int thresholdMagnitude;
    const int partialRectangleNB;
    int stride;
    const int nbReductions;
    const float reductionParameter;
    int blockWidth, blockHeight;
    int imgWidth, imgHeight;
    cv::vector<cv::Rect> boundingBox;
    BoosterHandle classifier;
    cv::Mat newMap = cv::Mat::zeros(imgHeight, imgWidth, CV_32F);
    
public:
    // methods used by ImagePatch and Image
    //-------------------------------------
    std::vector<float> make_orientationHistogramFeatures(cv::Mat&);
    std::vector<float> group_to_orientations(cv::Mat, cv::Mat);
    
    // methods only used by Image
    //---------------------------
    cv::Mat downscale_image(cv::Mat);
    std::vector<std::vector<float>> slide_window_over_image(int, cv::Mat);
    float testTheDataXGBoost(BoosterHandle, std::vector<float>, int, int);
    cv::Mat vectorToMat(std::vector<std::vector<float>>);
    
    void set_imageHeight(const int h) {
        imgHeight = h;
    }
    void set_imageWidth(const int w) {
        imgWidth = w;
    }
    void set_windowHeight(const int h) {
        blockHeight = h;
    }
    void set_windowWidth(const int w) {
        blockWidth = w;
    }
    void set_hBooster(const BoosterHandle bh) {
        classifier = bh;
    }
    void set_bBox(const cv::vector<cv::Rect> bb) {
        boundingBox = bb;
    }
    
    
    // to view the matrix
    //-------------------
    template <class T>
    void print(T & t, size_t rows, size_t columns)
    {
        for(size_t i = 0;i < rows; ++i)
        {
            for(size_t j = 0;j < columns; ++j)
                printf("%f ", t[i][j]);
            
            printf("\n");
        }
        printf("\n");
    }
    
    
    // to view the vector
    //-------------------
    template <class T>
    void printVector(T & t, size_t rows)
    {
        for(size_t i = 0;i < rows; ++i)
        {
            printf("%d ", t[i]);
        }
        printf("\n");
    }

    
    // constructor/destructor
    //-----------------------
    Image();
    Image(int, int, float, int, int);
    ~Image();
};

#endif /* Image_hpp */
