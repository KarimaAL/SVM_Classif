#include "CData.h"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include<iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

Mat CData::createData()
{
	// Set up training data
	int labels[10] = { 1, -1, -1, -1, 1, 1, -1, 1, -1, 1 };
	float trainingData[10][2] = { {501, 10}, {255, 10}, {501, 255}, {10, 501}, {12, 80}, {5, 500}, {10, 400}, {80, 50}, {100, 450}, {20, 250} };
	Mat trainingDataMat(10, 2, CV_32F, trainingData);
	Mat labelsMat(10, 1, CV_32SC1, labels);
	
	// Data for visual representation
	int width = 512, height = 512;
	Mat image = Mat::zeros(height, width, CV_8UC3);
	// Show the training data
	int thickness = -1;
	circle(image, Point(501, 10), 5, Scalar(255, 255, 0), thickness);
	circle(image, Point(255, 10), 5, Scalar(255, 0, 255), thickness);
	circle(image, Point(255, 255), 5, Scalar(255, 0, 255), thickness);
	circle(image, Point(200, 401), 5, Scalar(255, 0, 255), thickness);
	circle(image, Point(100, 80), 5, Scalar(255, 0, 255), thickness);
	circle(image, Point(50, 300), 5, Scalar(255, 0, 255), thickness);
	circle(image, Point(480, 400), 5, Scalar(255, 255, 0), thickness);
	circle(image, Point(350, 450), 5, Scalar(255, 255, 0), thickness);
	circle(image, Point(400, 50), 5, Scalar(255, 255, 0), thickness);
	circle(image, Point(450, 250), 5, Scalar(255, 255, 0), thickness);

	String windowName = "Data"; //Name of the window
	namedWindow(windowName); // Create a window
	imshow(windowName, image); // Show our image inside the created window.
	waitKey(0); // Wait for any keystroke in the window
	destroyWindow(windowName); //destroy the created window

	return trainingDataMat;
}