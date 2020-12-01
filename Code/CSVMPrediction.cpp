#include "CSVMPrediction.h"
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/objdetect.hpp"
#include <opencv2/ml.hpp>

using namespace cv::ml;
using namespace cv;
using namespace std;

void CSVMPrediction :: makePrediction(string Testpathname)
{
    Mat image = imread(Testpathname, IMREAD_GRAYSCALE); //Read the image in grayscale
    cout << "input size: " << image.size() << endl; // Print the size of the image

    // Resize the image to the desired format
    Mat Resized;
    int ColumnOfNewImage = 20;
    int RowsOfNewImage = 20;
    resize(image, Resized, Size(ColumnOfNewImage, RowsOfNewImage));
    cout << "output size: " << Resized.size() << endl; // Print the size of the image

     //Hog features
     //Defining the parameters of the HOG descriptor
    HOGDescriptor hog(
        Size(20, 20), //winSize
        Size(10, 10), //blocksize
        Size(5, 5), //blockStride,
        Size(10, 10), //cellSize,
        9, //nbins,
        1, //derivAper,
        -1, //winSigma,
        HOGDescriptor::L2Hys, //histogramNormType,
        0.2, //L2HysThresh,
        0,//gammal correction,
        64,//nlevels=64
        1);
    vector<Point> locations;
    vector< float >  descriptor;
    vector<vector<float> > HOGFeatures;
    hog.compute(Resized, descriptor, Size(8, 8), Size(0, 0), locations);
    HOGFeatures.push_back(descriptor);

    // convert vector to matrix
    int descriptor_size = HOGFeatures[0].size();
    Mat predMat(HOGFeatures.size(), descriptor_size, CV_32FC1);
    for (int i = 0; i < HOGFeatures.size(); i++) {
        for (int j = 0; j < descriptor_size; j++) {
            predMat.at<float>(i, j) = HOGFeatures[i][j];
        }
    }
    // Make prediction
    Ptr<SVM> svm = Algorithm::load<SVM>("model4.yml");
    Mat testResponse;
    float label = svm->predict(predMat, testResponse);
    String windowName = "Test Image"; //Name of the window
    namedWindow(windowName, WINDOW_NORMAL); // Create a window
    imshow(windowName, Resized); // Show our image inside the created window.
    waitKey(0); // Wait for any keystroke in the window
    destroyWindow(windowName); //destroy the created window
    cout << "The predicted label is: " << testResponse << endl;
}
