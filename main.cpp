//
//  main.cpp
//  SlidingWindow
//
//  Created by otl on 26/04/16.
//  Copyright © 2016 otl. All rights reserved.
//

#include "Rectangles.hpp"
#include "Image.hpp"
#include "ImagePatch.hpp"
#include <iostream>

using namespace std;
using namespace cv;

int main(int argc, const char * argv[]) {
    
    // call constructor of Rectangles to initialise the position of the rectangles as
    // boundingBox(sR, sC, width, height), default: patchWidth{35}, patchHeight{35},
    // numberOfRectangles{10000}
    //--------------------------------------------------------------------------------  RECTANGLES
    Rectangles rect;
    cv::vector<cv::Rect> bBox = rect.generate_rectangles();
    
    
    // call constructor of Image to initialise the stride and the reduction parameter
    // needed for the sliding window, default: nbReductions{3}, stride{15},
    // reductionParameter{0.85}, thresholdMagnitude{3}, partialRectangleNB{1111},
    //--------------------------------------------------------------------------------
    Image slidingWindow;
    //slidingWindow.set_bBox(bBox);
    int windowWidth = rect.get_patchWidth();
    int windowHeight = rect.get_patchHeight();
    
    
    
    // call constructor of ImagePatch
    //-------------------------------                                                   FEATURE EXTRACTION
    ImagePatch feature;
    feature.set_bBox(bBox);
    // do this for all patients and for all classes!!!
    // read data from folder and make trainData and trainLabel - NEGATIVES
   
    String f0 = "/Users/otl/Documents/MATLAB/Patient1/newImagePatches/0/";
    vector<String> filenames0;
    glob(f0, filenames0);
    
    
    vector<vector<float>> negatives;
    vector<float> labelNegative;
    int label0 = 0;
    //auto t2 = chrono::high_resolution_clock::now();
    
    negatives = feature.extract_features_of_patches(filenames0);
    //filenames0.clear();
    
    //auto t3 = chrono::high_resolution_clock::now();
    //cout << chrono::duration_cast<chrono::seconds>(t3-t2).count() << " sec for extracting features 0\n";
    
    labelNegative = feature.extract_label_of_patches(label0);
    //int nbRowNeg = (int)labelNegative.size();
    
    
    // read data - POSITIVES
    String f1 = "/Users/otl/Documents/MATLAB/Patient1/newImagePatches/1/";
    vector<String> filenames1;
    glob(f1, filenames1);
    //cout << filenames1.size() << endl;
    
    vector<vector<float>> positives;
    vector<float> labelPositive;
    int label1 = 1;
    positives = feature.extract_features_of_patches(filenames1);
    //filenames1.clear();
    labelPositive = feature.extract_label_of_patches(label1);
    
    
    // make feature matrix and corresponding label vector
    positives.insert(positives.end(), negatives.begin(), negatives.end());
    //negatives.clear();
    labelPositive.insert(labelPositive.end(), labelNegative.begin(), labelNegative.end());
    //labelNegative.clear();
    
    //int nbRectPart = feature.get_NbRectangles();
    size_t nbCols = positives[0].size();
    
    cout << "nbCols: " << nbCols << endl;
    //int nbRowPos = (int)feature.get_SizeFileName();
    //int nbRow = nbRowNeg + nbRowPos;
    size_t nbRows = labelPositive.size();
    //int nbCols = nbRectPart*9;
    cout << "nbRows: " << nbRows << endl;
    
    // prepare for xgboost - TRAINING (set
    BoosterHandle h_booster;
    int iterations = 200;
    //h_booster = feature.trainTheDataXGBoost(positives, labelPositive, nbRows, nbCols, iterations);
    
    
    // free xgboost internal structures
    //XGBoosterFree(h_booster);
    

    //f0 = "/Users/otl/Documents/MATLAB/Patient1/newImagePatches/4/";
    
    

    
    // read whole images from folder - SLIDING WINDOW
    //-----------------------------------------------                                       SLIDING WINDOW
    
    String f3 = "/Users/otl/Dropbox/ARTORG/Image/Patient1/";
    vector<String> filenamesImg;
    glob(f3, filenamesImg);
    cout << "size filename: " << filenamesImg.size() << endl;
    
    for(size_t i = 0; i < 1; ++i)
    {
        Mat img = imread(filenamesImg[i]);
        slidingWindow.set_imageHeight(img.rows);
        slidingWindow.set_imageWidth(img.cols);
        slidingWindow.set_windowHeight(windowHeight);
        slidingWindow.set_windowWidth(windowWidth);

        
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
