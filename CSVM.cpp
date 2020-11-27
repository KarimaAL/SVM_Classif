#include "CSVM.h"
#include <opencv2/ml.hpp>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include<iostream>

using namespace cv::ml;
using namespace cv;
using namespace std;

void CSVM::svmTraining(Mat img)
{

    Mat image = img;
    // Set up training data
    int labels[10] = { 1, -1, -1, 1, 1, -1, 1, -1, -1, 1 };
    float trainingData[10][2] = { {255, 10}, {480, 400}, {350, 450}, {255, 255}, {200, 401}, {400, 50}, {100, 80}, {450, 250}, {501, 10}, {50, 300} };
    Mat trainingDataMat(10, 2, CV_32F, trainingData);
    Mat labelsMat(10, 1, CV_32SC1, labels);

    // Train the SVM
    Ptr<SVM> svm = SVM::create();
    svm->setType(SVM::C_SVC);
    svm->setKernel(SVM::LINEAR);
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
    svm->train(trainingDataMat, ROW_SAMPLE, labelsMat);

    //svm->save("svm_params.xml");
    int width = 512, height = 512;
    Mat imageTest = Mat::zeros(height, width, CV_8UC3);

    // Show the decision regions given by the SVM
    Vec3b green(0, 255, 0), blue(255, 0, 0), red(0, 0, 255);
    for (int i = 0; i < imageTest.rows; i++)
    {
        for (int j = 0; j < imageTest.cols; j++)
        {
            Mat sampleMat = (Mat_<float>(1, 2) << j, i);
            float response = svm->predict(sampleMat);
            if (response == 1)
                imageTest.at<Vec3b>(i, j) = green;
            else if (response == -1)
                imageTest.at<Vec3b>(i, j) = blue;
            else
                imageTest.at<Vec3b>(i, j) = red;
        }

    }
    Mat Result = imageTest + image;
    String windowName = "svm"; //Name of the window
    namedWindow(windowName); // Create a window
    imshow(windowName, Result); // Show our image inside the created window.
    waitKey(0); // Wait for any keystroke in the window
    destroyWindow(windowName); //destroy the created window

    //   // Show support vectors
    //int thickness = 2;
    //Mat sv = svm->getUncompressedSupportVectors();
    //for (int i = 0; i < sv.rows; i++)
    //{
    //    const float* v = sv.ptr<float>(i);
    //    circle(image, Point((int)v[0], (int)v[1]), 6, Scalar(128, 128, 128), thickness);
    //}
    //imwrite("result.png", image);        // save the image
    //imshow("SVM Simple Example", image); // show it to the user
    //waitKey();

}
