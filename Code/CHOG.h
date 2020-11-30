#pragma once
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/objdetect.hpp"
#include <opencv2/ml.hpp>

using namespace cv::ml;
using namespace cv;
using namespace std;

struct mystructhog {
	vector<vector<float> > hogTr;
	vector<vector<float> > hogTs;
};

class CHOG
{
public:
	void CreateTrainTestHOG(vector<Mat> deskewedTrainCells, vector<Mat> deskewedTestCells);
	void ConvertVectortoMatrix(vector<vector<float> >& trainHOG, vector<vector<float> >& testHOG);
	vector<vector<float> > trainHOG;
	vector<vector<float> > testHOG;
};

