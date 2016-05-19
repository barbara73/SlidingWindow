//
//  ImagePatch.hpp
//  SlidingWindow
//
//  Created by otl on 26/04/16.
//  Copyright © 2016 otl. All rights reserved.
//

#ifndef ImagePatch_hpp
#define ImagePatch_hpp

#include "Image.hpp"
#include <stdio.h>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "xgboost/c_api.h"
#include <dmlc/timer.h>


class ImagePatch: public Image {
    unsigned long fileNameSize;
    int bLabel;
public:
    std::vector<std::vector<float>> extract_features_of_patches(cv::vector<cv::String>&);
    std::vector<float> extract_label_of_patches(int);
    BoosterHandle trainTheDataXGBoost(std::vector<std::vector<float>>, std::vector<float>, int, int, int);
};

#endif /* ImagePatch_hpp */
