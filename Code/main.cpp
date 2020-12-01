#include "CData.h"
#include "CSVM.h"
#include "CDigitDataLoading.h"
#include "CHOG.h"
#include "CSVMClassifier.h"
#include<iostream>

using namespace std;

int main(int, char**)
{
	cout << "OpenCV version : " << CV_VERSION << endl;
	//CData m_data;
	//CSVM m_svm;
	CDigitDataLoading image, organize,labels;
	CHOG createHog;
	string pathName = "digits.png";  // Path to the image
	Mat data = image.readMyData(pathName);

	//mystruct dataDiv = organize.organizingData(data);

	//mystructdeskew deskewData = labels.deskewData(dataDiv.trainCells, dataDiv.testCells);
	//createHog.CreateTrainTestHOG(deskewData.deskewedTrCells, deskewData.deskewedTsCells, dataDiv.trainLabels, dataDiv.testLabels);
	//auto m_mat = m_getMat.ConvertVectortoMatrix(Thog.trainHOG, Thog.testHOG);
	return 0;
}