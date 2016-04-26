//
//  main.cpp
//  SlidingWindow
//
//  Created by otl on 26/04/16.
//  Copyright © 2016 otl. All rights reserved.
//

#include "Rectangles.hpp"
#include <iostream>

int main(int argc, const char * argv[]) {
    
    // call constructor of Rectangles to initialise the position of the rectangles as boundingBox(sR, sC, width, height), initialised in constructor to patchWidth=35, patchHeight=35 and numberOfRectangles=10000
    //----------------------------------------------------------------------------------
    Rectangles rect;
    cv::vector<cv::Rect> bBox = rect.generate_rectangles();
    return 0;
}
