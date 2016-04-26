# SlidingWindow
This code is written in c++ in Xcode and uses opencv2 and xgBoost. Below are some installation details.

- install macports (https://www.macports.org)
- install cmake (https://cmake.org)
- install xcode
- install opencv2 (http://opencv.org/downloads.html)
  - http://blogs.wcode.org/2014/10/howto-install-build-and-use-opencv-macosx-10-10/ 
    - at this point: Add an SDK path to CMAKE_OSX_SYSROOT, it will look something like this “/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.9.sdk”. -> use MacOSX10.10.sdk instead (maybe works with MacOSX10.11.sdk as well?), older SDKs can be found on github
    - this tutorial can also be used for installation on El Capitan
  - http://blogs.wcode.org/2014/11/howto-setup-xcode-6-1-to-work-with-opencv-libraries/
- install homebrew
- install xgboost (http://xgboost.readthedocs.org/en/latest/build.html)
- install openmpi (does not work for me with xgboost by now)
- add in build settings:
	- Header search paths: /usr/local/include/xgboost/include /usr/local/include /usr/local/include/xgboost/rabit/include /usr/local/include/xgboost/dmlc-core/include
	- Library search paths: /usr/local/lib /usr/local/include/xgboost/lib /usr/local/include/xgboost/rabit/lib /usr/local/include/xgboost/dmlc-core
	- Other linker flags: -lopencv_calib3d -lopencv_core -lopencv_features2d -lopencv_flann -lopencv_highgui -lopencv_imgproc -lopencv_ml -lopencv_objdetect -lopencv_photo -lopencv_stitching -lopencv_superres -lopencv_ts -lopencv_video -lopencv_videostab -lrabit -lxgboost -ldmlc
