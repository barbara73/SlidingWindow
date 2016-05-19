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
std::vector<std::vector<float>> ImagePatch::extract_features_of_patches(vector<String>& fileName) {
    fileNameSize = fileName.size();
    std::vector<float> orientationFeatures;
    std::vector<std::vector<float>> totalFeaturePerPatch;
    Mat img, img_grey;
    
    //for(size_t i = 0; i < fileNameSize; ++i)
    for(size_t i = 0; i < 1000; ++i)
    {
        img = imread(fileName[i]);
        
        if(!img.data)
            std::cerr << "Problem loading image!!!" << std::endl;
        
        // change colour image to greyscale image
        cvtColor(img, img_grey, CV_BGR2GRAY);
        orientationFeatures = Image::make_orientationHistogramFeatures(img_grey);
        totalFeaturePerPatch.push_back(orientationFeatures);
        //orientationFeatures.clear();
        
    }
    
    return totalFeaturePerPatch;
}


// make labels per patch
//----------------------
std::vector<float> ImagePatch::extract_label_of_patches(int label) {
    bLabel = label;
    //std::vector<int> labelPerPatch(static_cast<int>(fileNameSize), bLabel);
    std::vector<float> labelPerPatch(static_cast<int>(1000), bLabel);
    
    return labelPerPatch;
}



// Train with xgBoost
//-------------------
BoosterHandle ImagePatch::trainTheDataXGBoost(std::vector<std::vector<float>> train, std::vector<float>label, int r, int c, int it) {
    BoosterHandle handle;
    DMatrixHandle h_train[1];
    XGDMatrixCreateFromMat((float *) &train[0], r, c, -1, &h_train[0]);     //take from vector &positives[0]
    
    // load the labels
    XGDMatrixSetFloatInfo(h_train[0], "label", &label[0], r);
    
    // create the booster and load some parameters
    XGBoosterCreate(h_train, 1, &handle);
    XGBoosterSetParam(handle, "booster", "gbtree");
    XGBoosterSetParam(handle, "objective", "reg:linear");
    XGBoosterSetParam(handle, "max_depth", "2");
    XGBoosterSetParam(handle, "eta", "0.1");
    XGBoosterSetParam(handle, "min_child_weight", "1");
    XGBoosterSetParam(handle, "subsample", "0.5");
    XGBoosterSetParam(handle, "colsample_bytree", "1");
    XGBoosterSetParam(handle, "num_parallel_tree", "1");
    
    // perform learning iterations
    for (int iter=0; iter<it; iter++)
        XGBoosterUpdateOneIter(handle, iter, h_train[0]);
    
    XGDMatrixFree(h_train[0]);
    return handle;
}

