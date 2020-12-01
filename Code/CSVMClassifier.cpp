#include "CSVMClassifier.h"
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/objdetect.hpp"
#include <opencv2/ml.hpp>

using namespace cv::ml;
using namespace cv;
using namespace std;

Mat CSVMClassifier :: SVMtrain(Mat& trainMat, vector<int>& trainLabels, Mat& testResponse, Mat& testMat) 
{
    cout << "train label size: " << trainLabels.size() << endl;
    Ptr<SVM> svm = SVM::create();
    svm->setGamma(0.50625);
    svm->setC(12.5);
    svm->setKernel(SVM::RBF);
    svm->setType(SVM::C_SVC);
    Ptr<TrainData> td = TrainData::create(trainMat, ROW_SAMPLE, trainLabels);
    cout << "get td" << endl;
    svm->train(td);
    //svm->trainAuto(td);
    if (td)
    {
        svm->save("model4.yml");
        svm->predict(testMat, testResponse);
        getSVMParams(svm);
    }
    return testResponse;
}
void CSVMClassifier:: getSVMParams(SVM* svm)
{
    cout << "Kernel type     : " << svm->getKernelType() << endl;
    cout << "Type            : " << svm->getType() << endl;
    cout << "C               : " << svm->getC() << endl;
    cout << "Degree          : " << svm->getDegree() << endl;
    cout << "Nu              : " << svm->getNu() << endl;
    cout << "Gamma           : " << svm->getGamma() << endl;
}

float CSVMClassifier:: SVMevaluate(Mat& testResponse, float& count, float& accuracy, vector<int>& testLabels) {

    for (int i = 0; i < testResponse.rows; i++)
    {
        //cout << testResponse.at<float>(i,0) << " " << testLabels[i] << endl;
        if (testResponse.at<float>(i, 0) == testLabels[i]) {
            count = count + 1;
        }
    }
    accuracy = (count / testResponse.rows) * 100;
    return accuracy;
}