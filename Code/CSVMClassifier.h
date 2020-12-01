#pragma once

#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/objdetect.hpp"
#include <opencv2/ml.hpp>

using namespace cv::ml;
using namespace cv;
using namespace std;

class CSVMClassifier
{
public:
	Mat SVMtrain(Mat& trainMat, vector<int>& trainLabels, Mat& testResponse, Mat& testMat);
	void getSVMParams(SVM* svm);
	float SVMevaluate(Mat& testResponse, float& count, float& accuracy, vector<int>& testLabels);
};

