#pragma once
//We are going to use the above image as our dataset that comes with OpenCV samples.
//It contains 5000 images in all — 500 images of each digit. Each image is 20×20 grayscale 
//with a black background. 4500 of these digits will be used for training and the remaining 500
//will be used for testing the performance of the algorithm

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
//#include "opencv2/objdetect.hpp"
//#include <opencv2/ml.hpp>

using namespace cv;
using namespace std;

struct mystruct {
	vector<Mat> trainCells;
	vector<Mat> testCells;
};

class CDigitDataLoading
{
public:
	CDigitDataLoading(); // Constructor
	~CDigitDataLoading(); // destructor
	Mat readMyData();
	Mat deskew(Mat& img);
	mystruct organizingData(Mat& img);
	vector<Mat> labelAssigning(vector<Mat>& traincells, vector<Mat>& testcells);
	int SZ = 20; //Each image is 20×20 grayscale
	float affineFlags = WARP_INVERSE_MAP | INTER_LINEAR;
	vector<Mat> trainCells;
	vector<Mat> testCells;
	vector<int> trainLabels;
	vector<int> testLabels;
	vector<Mat> deskewedTrainCells;
	vector<Mat> deskewedTestCells;
	Mat deskewedImg;

};

