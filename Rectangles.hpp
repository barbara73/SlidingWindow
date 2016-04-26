//
//  Rectangles.hpp
//  SlidingWindow
//
//  Created by otl on 26/04/16.
//  Copyright © 2016 otl. All rights reserved.
//

#ifndef Rectangles_hpp
#define Rectangles_hpp

#include <stdio.h>
#include <vector>
#include <array>
#include <random>
#include "opencv2/imgproc/imgproc.hpp"


class Rectangles {
    const int patchWidth;
    const int patchHeight;
    const int numberOfRectangles;
public:
    cv::vector<cv::Rect> generate_rectangles();
    
    int get_patchWidth()const {
        return this->patchWidth;
    }
    int get_patchHeight()const {
        return this->patchHeight;
    }
    
    Rectangles();
    Rectangles(int, int);
    Rectangles(int, int, int);
    ~Rectangles();
};



#endif /* Rectangles_hpp */
