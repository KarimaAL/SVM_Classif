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
	CData m_data;
	CSVM m_svm;
   // m_data.createData();
	//m_svm.svmTraining(m_data.m_image);

	CDigitDataLoading image, organize,labels;
	CHOG createHog, m_getMat, Thog, g;
	//CSVMClassifier m_SVM_train;
	Mat img = image.readMyData();

	mystruct dataDiv = organize.organizingData(img);
	dataDiv.trainLabels;
	dataDiv.testLabels;

	mystructdeskew deskewData = labels.deskewData(dataDiv.trainCells, dataDiv.testCells);

	createHog.CreateTrainTestHOG(deskewData.deskewedTrCells, deskewData.deskewedTsCells, dataDiv.trainLabels, dataDiv.testLabels);

	//auto m_mat = m_getMat.ConvertVectortoMatrix(Thog.trainHOG, Thog.testHOG);

	return 0;
}