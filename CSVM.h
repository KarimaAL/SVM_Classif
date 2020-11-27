#pragma once

#include "CData.h"

class CSVM
{
public:
	void svmTraining(Mat trainingDataMat, Mat labelsMat, Mat image);
	CData m_data;
};

