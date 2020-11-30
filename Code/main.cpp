#include "CData.h"
#include "CSVM.h"
#include "CDigitDataLoading.h"
#include "CHOG.h"
#include<iostream>

using namespace std;

int main(int, char**)
{
	cout << "OpenCV version : " << CV_VERSION << endl;
	CData m_data;
	CSVM m_svm;
   // m_data.createData();
	//m_svm.svmTraining(m_data.m_image);

	CDigitDataLoading image, organize,labels;
	CHOG createHog;
	Mat img = image.readMyData();
	mystruct dataDiv = organize.organizingData(img);
	mystructdeskew deskewData = labels.labelAssigning(dataDiv.trainCells, dataDiv.testCells);
	createHog.CreateTrainTestHOG(deskewData.deskewedTrCells, deskewData.deskewedTsCells);
	return 0;
}