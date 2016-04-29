//
//  ImagePatch.cpp
//  SlidingWindow
//
//  Created by otl on 26/04/16.
//  Copyright © 2016 otl. All rights reserved.
//

#include "ImagePatch.hpp"
#include <iostream>

using namespace cv;


// extract the features from the image patches
//--------------------------------------------
std::vector<std::vector<float>> ImagePatch::extract_features_of_patches(std::vector<cv::Rect>& bb, vector<String> fileName) {
    //bbRectangle = bb;
    fileNameSize = fileName.size();
    std::vector<float> orientationFeatures;
    std::vector<std::vector<float>> totalFeaturePerPatch;
    
    
    //for(size_t i = 0; i < fileNameSize; ++i)
    for(size_t i = 0; i < 1000; ++i)
    {
        Mat img = imread(fileName[i]);
        
        if(!img.data)
            std::cerr << "Problem loading image!!!" << std::endl;
        
        orientationFeatures = Image::make_orientationHistogramFeatures(img);
        totalFeaturePerPatch.push_back(orientationFeatures);
        
    }
    //print(totalFeaturePerPatch, 10, partialRectangleNB*9);
    
    return totalFeaturePerPatch;
}


// make labels per patch
//----------------------
std::vector<float> ImagePatch::extract_label_of_patches(int label) {
    bLabel = label;
    //std::vector<int> labelPerPatch(static_cast<int>(fileNameSize), bLabel);
    std::vector<float> labelPerPatch(static_cast<int>(1000), bLabel);
    
    //printVector(labelPerPatch, 10);
    
    return labelPerPatch;
}

