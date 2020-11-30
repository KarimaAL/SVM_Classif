#include "CHOG.h"
#include "CDigitDataLoading.h"

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

void CHOG::CreateTrainTestHOG(vector<Mat> deskewedTrainCells, vector<Mat> deskewedTestCells)
{
    vector< float >  descriptorstrain; //variable to store the HOG features
    vector< float > descriptorstest;
    for (int y = 0; y < deskewedTrainCells.size(); y++) {
       
        vector<Point> locations;
        hog.compute(deskewedTrainCells[y], descriptorstrain, Size(8, 8), Size(0, 0), locations);
        trainHOG.push_back(descriptorstrain);
    }

    for (int y = 0; y < deskewedTestCells.size(); y++) {

        ;
        vector<Point> locations;
        hog.compute(deskewedTestCells[y], descriptorstest, Size(8, 8), Size(0, 0), locations);
        testHOG.push_back(descriptorstest);
    }

    ConvertVectortoMatrix(trainHOG, testHOG);
}

void CHOG::ConvertVectortoMatrix(vector<vector<float> >& trainHOG, vector<vector<float> >& testHOG)
{
    int descriptor_size = trainHOG[0].size();
    Mat trainMat(trainHOG.size(), descriptor_size, CV_32FC1);
    Mat testMat(testHOG.size(), descriptor_size, CV_32FC1);

    for (int i = 0; i < trainHOG.size(); i++) {
        for (int j = 0; j < descriptor_size; j++) {
            trainMat.at<float>(i, j) = trainHOG[i][j];
        }
    }
    for (int i = 0; i < testHOG.size(); i++) {
        for (int j = 0; j < descriptor_size; j++) {
            testMat.at<float>(i, j) = testHOG[i][j];
        }
    }

    String windowNametrain = "HOG Features training"; //Name of the window
    namedWindow(windowNametrain, WINDOW_NORMAL); // Create a window
    imshow(windowNametrain, trainMat); // Show our image inside the created window.
    waitKey(0); // Wait for any keystroke in the window
    destroyWindow(windowNametrain); //destroy the created window

    String windowName = "HOG Features testing"; //Name of the window
    namedWindow(windowName, WINDOW_NORMAL); // Create a window
    imshow(windowName, testMat); // Show our image inside the created window.
    waitKey(0); // Wait for any keystroke in the window
    destroyWindow(windowName); //destroy the created window
}