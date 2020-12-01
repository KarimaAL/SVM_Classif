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

struct mystructhogT {
	vector<vector<float> > trainHOGV;
	vector<vector<float> > testHOGV;
};

class CHOG
{
public:
	mystructhogT CreateTrainTestHOG(vector<Mat> deskewedTrainCells, vector<Mat> deskewedTestCells, vector<int> m_trainLabels, vector<int> m_testLabels);
	Mat ConvertVectortoMatrix(vector<vector<float> >& trainHOG, vector<vector<float> >& testHOG, vector<int> m_trainLabels, vector<int> m_testLabels);
	vector<vector<float> > trainHOG;
	vector<vector<float> > testHOG;
};

