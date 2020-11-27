#pragma once
#include <opencv2/highgui.hpp>

using namespace cv;
class CData
{
public:
	Mat m_trainingDataMat;
	Mat m_labelsMat;
	Mat m_image;
	Mat createData();
};

