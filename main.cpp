//
//  main.cpp
//  SlidingWindow
//
//  Created by otl on 26/04/16.
//  Copyright © 2016 otl. All rights reserved.
//

#include "Rectangles.hpp"
#include "Image.hpp"
#include <iostream>

using namespace std;
using namespace cv;

int main(int argc, const char * argv[]) {
    
    // call constructor of Rectangles to initialise the position of the rectangles as boundingBox(sR, sC, width, height), default: patchWidth{35}, patchHeight{35}, numberOfRectangles{10000}
    //----------------------------------------------------------------------------------
    Rectangles rect;
    cv::vector<cv::Rect> bBox = rect.generate_rectangles();
    
    
    
    // call constructor of Image to initialise the stride and the reduction parameter needed for the sliding window, default: nbReductions{3}, stride{15}, reductionParameter{0.85},  thresholdMagnitude{3}, partialRectangleNB{1111}, 
    //-----------------------------------------------------------------------------------------
    Image slidingWindow;
    slidingWindow.set_bBox(bBox);
    int windowWidth = rect.get_patchWidth();
    int windowHeight = rect.get_patchHeight();
    
    
    
    
    
    
    // read whole images from folder and make trainData and trainLabel
    String f3 = "/Users/otl/Documents/MATLAB/Patient1/NewImages/";
    vector<String> filenamesImg;
    glob(f3, filenamesImg);
    cout << filenamesImg.size() << endl;
    Mat img = imread(filenamesImg[0]);
    slidingWindow.set_imageHeight(img.rows);
    slidingWindow.set_imageWidth(img.cols);
    slidingWindow.set_windowHeight(windowHeight);
    slidingWindow.set_windowWidth(windowWidth);
    
    for(size_t i = 0; i < 1; ++i)
    {
        Mat img = imread(filenamesImg[i]);
        
        Mat map = slidingWindow.downscale_image(img);
        
        /*
         ss << folderName << "/" << name << (i + 1) << type;
         string filename = ss.str();
         ss.str("");
         
         //imwrite( "/Users/otl/Documents/MATLAB/Patient1/NewImages/Gray_Image.png", map);
         //imwrite( format("folder/image%d.png", i ), img);
         namedWindow( "MAP", CV_WINDOW_AUTOSIZE );
         imshow( "MAP", map);
         waitKey(0);
         
         */
        
    }
    
    
    //slidingWindow.set_hBooster(h_booster);


    
    
    return 0;
}
