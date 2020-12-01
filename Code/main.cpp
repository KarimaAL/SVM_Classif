#include "CData.h"
#include "CSVM.h"
#include "CDigitDataLoading.h"
#include "CHOG.h"
#include "CSVMClassifier.h"
#include "CSVMPrediction.h"
#include<iostream>

using namespace std;

int main(int, char**)
{
	cout << "OpenCV version : " << CV_VERSION << endl;
	//CData m_data;
	//CSVM m_svm;

	CDigitDataLoading image;
	string pathName = "digits.png";  // Path to the image
	//Mat data = image.readMyData(pathName); //Reading the data in the given path
	CSVMPrediction TestImage;
	string TestpathName = "img_26.jpg";  // Path to the image
	TestImage.makePrediction(TestpathName);
	return 0;
}