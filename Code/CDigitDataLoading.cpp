#include "CDigitDataLoading.h"
#include <iostream>

using namespace cv;
using namespace std;

CDigitDataLoading::CDigitDataLoading()
{
    cout << "Object is being created" << endl;
}

CDigitDataLoading::~CDigitDataLoading()
{
    cout << "Object is being deleted" << endl;
}

Mat CDigitDataLoading::readMyData()
{
	string pathName = "digits.png";  // Path to the image
	Mat image = imread(pathName, IMREAD_GRAYSCALE); //Read the image in grayscale
    cout << image.rows << endl; // Print the size of the image

    String windowName = "digits"; //Name of the window
    namedWindow(windowName, WINDOW_NORMAL); // Create a window
    imshow(windowName, image); // Show our image inside the created window.
    waitKey(0); // Wait for any keystroke in the window
    destroyWindow(windowName); //destroy the created window
    return image;
}

Mat CDigitDataLoading::deskew(Mat& img) 
{
    Moments m = moments(img);
    if (abs(m.mu02) < 1e-2) {
        return img.clone();
    }
    float skew = m.mu11 / m.mu02;
    Mat warpMat = (Mat_<float>(2, 3) << 1, skew, -0.5 * SZ * skew, 0, 1, 0);
    Mat imgOut = Mat::zeros(img.rows, img.cols, img.type());
    warpAffine(img, imgOut, warpMat, imgOut.size(), affineFlags);
    return imgOut;
}

mystruct CDigitDataLoading::organizingData(Mat& img)
{
    // Divide the images into training ans testing classes
    Mat image = img;
    int ImgCount = 0;
    for (int i = 0; i < image.rows; i = i + SZ)
    {
        for (int j = 0; j < image.cols; j = j + SZ)
        {
            Mat digitImg = (image.colRange(j, j + SZ).rowRange(i, i + SZ)).clone();
            if (j < int(0.9 * image.cols))
            {
                trainCells.push_back(digitImg);
            }
            else
            {
                testCells.push_back(digitImg);
            }
            ImgCount++;
        }
    }
    cout << "Image Count : " << ImgCount << endl;

    //Assigne the lables to the data
    float digitClassNumber = 0;

    for (int z = 0; z<int(0.9 * ImgCount); z++) {
        if (z % 450 == 0 && z != 0) {
            digitClassNumber = digitClassNumber + 1;
        }
        trainLabels.push_back(digitClassNumber);
    }
    digitClassNumber = 0;
    for (int z = 0; z<int(0.1 * ImgCount); z++) {
        if (z % 50 == 0 && z != 0) {
            digitClassNumber = digitClassNumber + 1;
        }
        testLabels.push_back(digitClassNumber);
    }
   // return {trainCells, testCells };
    return mystruct{ trainCells, testCells };
}

vector<Mat> CDigitDataLoading::labelAssigning(vector<Mat>& traincells, vector<Mat>& testcells)
{

    for (int i = 0; i < traincells.size(); i++) {

        deskewedImg = deskew(traincells[i]);
        deskewedTrainCells.push_back(deskewedImg);
    }



    for (int i = 0; i < testcells.size(); i++) {

        deskewedImg = deskew(testcells[i]);
        deskewedTestCells.push_back(deskewedImg);
    }

    ////create Mat
    Mat M = Mat(10, 450, CV_32FC1); //Mat M=Mat(r,c,CV_32FC1);
    //copy vector to mat
    memcpy(M.data, deskewedTrainCells.data(), deskewedTrainCells.size() * sizeof(float));
    //print Mat
    //cout << "matrix: " << M.rows << endl;


    //String windowName = "deskewed digits"; //Name of the window
    //namedWindow(windowName, WINDOW_NORMAL); // Create a window
    //imshow(windowName, M); // Show our image inside the created window.
    //waitKey(0); // Wait for any keystroke in the window
    //destroyWindow(windowName); //destroy the created window

    return deskewedTrainCells, deskewedTestCells;
}

