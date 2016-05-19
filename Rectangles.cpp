//
//  Rectangles.cpp
//  SlidingWindow
//
//  Created by otl on 26/04/16.
//  Copyright © 2016 otl. All rights reserved.
//

#include "Rectangles.hpp"
#include <iostream>


using namespace std;

// default constructor
Rectangles::Rectangles():patchWidth{41}, patchHeight{41}, numberOfRectangles{10000} {}


// constructor called if patch is square
Rectangles::Rectangles(int x, int nbRectangles):patchWidth{x}, patchHeight{x}, numberOfRectangles{nbRectangles} {}


// constructor called if patch is not square
Rectangles::Rectangles(int x, int y, int nbRectangles):patchWidth{x}, patchHeight{y}, numberOfRectangles{nbRectangles} {}


// to view the matrix
template <class T>
void print(T & t, size_t rows, size_t columns)
{
    for(size_t i = 0;i < rows; ++i)
    {
        for(size_t j = 0;j < columns; ++j)
            printf("%d ", t[i][j]);
        
        printf("\n");
    }
    printf("\n");
}

// generate rectangles needed as features
//---------------------------------------
std::vector<cv::Rect> Rectangles::generate_rectangles() {
    
    random_device rdev{};
    default_random_engine e{1};//rdev()};
    uniform_int_distribution<int>  d1{0, patchWidth-1};
    uniform_int_distribution<int>  d2{0, patchHeight-1};
    
    
    int numberRect = numberOfRectangles+1000;
    vector<array<int, 4>> bBox(numberRect);
    
    
    int i = 0;
    
    while (i < bBox.size()) {
        int x1 = d1(e);
        int x2 = d1(e);
        int left; int top;
        
        int width = abs(x1 - x2);
        
        int y1 = d2(e);
        int y2 = d2(e);
        
        int height = abs(y1 - y2);
        
        if (!((width < 3) || (height < 3))) {
            
            if (x1 < x2)
                left = x1;
            else
                left = x2;
            
            if (y1 < y2)
                top = y1;
            else
                top = y2;
            
            bBox[i] = {left, top, width, height};
            
            i += 1;
        }
    }
    
    sort(bBox.begin(), bBox.end());
    auto last = unique(bBox.begin(), bBox.end());
    bBox.erase(last, bBox.end());
    
    random_shuffle(bBox.begin(), bBox.end());
    
    if (bBox.size() < numberOfRectangles) {
        cerr << "number of Rectangles smaller" << endl;
    }
    
    std::vector<array<int, 4>>::const_iterator firstNB = bBox.begin();
    std::vector<array<int, 4>>::const_iterator lastNB = bBox.begin() + numberOfRectangles;
    std::vector<array<int, 4>> boundingBoxArray(firstNB, lastNB);
    //print(boundingBoxArray, 10, 4);
    
    std::vector<cv::Rect> boundingBox(numberOfRectangles);
    
    for (int i=0; i!=numberOfRectangles; ++i) {
        boundingBox[i] = cv::Rect(boundingBoxArray[i][0], boundingBoxArray[i][1], boundingBoxArray[i][2], boundingBoxArray[i][3]);
    }
    
    return boundingBox;
}

// destructor
Rectangles::~Rectangles() {}