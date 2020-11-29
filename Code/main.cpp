#include "CData.h"
#include "CSVM.h"
#include "CDigitDataLoading.h"
#include<iostream>

using namespace std;

int main(int, char**)
{
	cout << "OpenCV version : " << CV_VERSION << endl;
	CData m_data;
	CSVM m_svm;
   // m_data.createData();
	//m_svm.svmTraining(m_data.m_image);

	CDigitDataLoading image, organize,g, dImg;
	CDigitDataLoading deskewImage;
	Mat img = image.readMyData();
	auto f = organize.organizingData(img);
	g.labelAssigning(f.trainCells,f.testCells);
	//deskewImage.deskew(img);
	return 0;
}