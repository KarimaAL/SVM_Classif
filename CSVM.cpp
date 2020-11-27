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
void CSVM::svmTraining(Mat trainingDataMat, Mat labelsMat, Mat img)
{
    Mat trainingData = trainingDataMat;
    Mat labels = labelsMat;
    Mat image = img;

    // Train the SVM
    Ptr<SVM> svm = SVM::create();
    svm->setType(SVM::C_SVC);
    svm->setKernel(SVM::LINEAR);
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
    svm->train(trainingData, ROW_SAMPLE, labels);
    //svm->save("svm_params.xml");

    // Show the decision regions given by the SVM
    Vec3b green(0, 255, 0), blue(255, 0, 0), red(0, 0, 255);
    for (int i = 0; i < image.rows; i++)
    {
        for (int j = 0; j < image.cols; j++)
        {
            Mat sampleMat = (Mat_<float>(1, 2) << j, i);
            float response = svm->predict(sampleMat);
            if (response == 1)
                image.at<Vec3b>(i, j) = green;
            else if (response == -1)
                image.at<Vec3b>(i, j) = blue;
            else
                image.at<Vec3b>(i, j) = red;
        }

    }
    String windowName = "svm"; //Name of the window
    namedWindow(windowName); // Create a window
    imshow(windowName, image); // Show our image inside the created window.
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
