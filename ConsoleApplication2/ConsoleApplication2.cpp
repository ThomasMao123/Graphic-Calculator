// ConsoleApplication1.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "pch.h"
#include <iostream>
#include <stdlib.h>


#include<opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>

#include <fstream>
#include <string>


using namespace std;
using namespace cv;
using namespace cv::ml;

const int FEATURE_SIZE = 324;

HOGDescriptor hog(Size(28, 28), //winSize
	Size(8, 8), //blocksize
	Size(4, 4), //blockStride,
	Size(8, 8), //cellSize,
	9, //nbins,
	1, //derivAper,
	-1, //winSigma,
	0, //histogramNormType,
	0.2, //L2HysThresh,
	1,//gammal correction,
	64,//nlevels=64
	1);

Mat deskew(Mat& img)
{
	Moments m = moments(img);
	if (abs(m.mu02) < 1e-2)
	{
		// No deskewing needed. 
		return img.clone();
	}
	// Calculate skew based on central momemts. 
	double skew = m.mu11 / m.mu02;
	// Calculate affine transform to correct skewness. 
	Mat warpMat = (Mat_<double>(2, 3) << 1, skew, -0.5 * 1.5 * skew, 0, 1, 0);

	Mat imgOut = Mat::zeros(img.rows, img.cols, img.type());
	warpAffine(img, imgOut, warpMat, imgOut.size());

	return imgOut;
}


int ReverseInt(int i)
{
	unsigned char ch1, ch2, ch3, ch4;
	ch1 = i & 255;
	ch2 = (i >> 8) & 255;
	ch3 = (i >> 16) & 255;
	ch4 = (i >> 24) & 255;
	return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

void read_Mnist(string filename, vector<cv::Mat> &vec) {
	ifstream file(filename, ios::binary);
	if (file.is_open()) {
		int magic_number = 0;
		int number_of_images = 0;
		int n_rows = 0;
		int n_cols = 0;

		file.read((char*)&magic_number, sizeof(magic_number));
		magic_number = ReverseInt(magic_number);

		file.read((char*)&number_of_images, sizeof(number_of_images));
		number_of_images = ReverseInt(number_of_images);

		file.read((char*)&n_rows, sizeof(n_rows));
		n_rows = ReverseInt(n_rows);

		file.read((char*)&n_cols, sizeof(n_cols));
		n_cols = ReverseInt(n_cols);

		for (int i = 0; i < number_of_images; ++i) {
			cv::Mat tp = Mat::zeros(n_rows, n_cols, CV_8UC1);
			for (int r = 0; r < n_rows; ++r) {
				for (int c = 0; c < n_cols; ++c) {
					unsigned char temp = 0;
					file.read((char*)&temp, sizeof(temp));
					tp.at<uchar>(r, c) = (int)temp;
				}
			}
			vec.push_back(tp);
		}
	}
}

void read_Mnist_Label(string filename, vector<double> &vec) {
	ifstream file(filename, ios::binary);
	if (file.is_open()) {
		int magic_number = 0;
		int number_of_images = 0;
		int n_rows = 0;
		int n_cols = 0;
		file.read((char*)&magic_number, sizeof(magic_number));
		magic_number = ReverseInt(magic_number);
		file.read((char*)&number_of_images, sizeof(number_of_images));
		number_of_images = ReverseInt(number_of_images);
		for (int i = 0; i < number_of_images; ++i) {
			unsigned char temp = 0;
			file.read((char*)&temp, sizeof(temp));
			vec.push_back((double)temp);
		}
	}
}

Mat ComputeHog(Mat& source) {
	vector<float> hog_feature;
	hog.compute(source, hog_feature);
	Mat m(1, FEATURE_SIZE, CV_32F);

	for (size_t i = 0; i < hog_feature.size(); i++) {
		m.at<float>(0, i) = hog_feature[i];
	}
	return m;
}


int main()
{	
	
	vector<cv::Mat> pictures;
	vector<double> labels;
	vector<vector<float>> train_data;

	std::cout << "=======Loading training images========" << std::endl;
	read_Mnist_Label("C:/Users/13015/Desktop/cs126 final project/train-labels/train-labels.idx1-ubyte", labels);
	//read_Mnist("C:/Users/13015/Desktop/cs126 final project/train-images/train-images.idx3-ubyte", pictures);
	
	for (int i = 0; i < 60000; i++) {
		Mat temp = imread("C:/Users/13015/Desktop/cs126 final project/train-images-png/" + to_string(i) + ".png", IMREAD_GRAYSCALE);
		pictures.push_back(temp);
	}
	
	std::cout << "=======Finished loading training images========" << std::endl;

	Mat trainingDataMat(500, FEATURE_SIZE, CV_32F);
	Mat labelsMat(500, 1, CV_32SC1);

	std::cout << "========Computing Hog features...==========" << std::endl;
	for (size_t i = 0; i <500; i++) {
		vector<float> hog_feature;
		hog.compute(pictures[i], hog_feature);

		for (size_t j = 0; j < FEATURE_SIZE; j++) {
			trainingDataMat.at<float>((int) i, (int) j) = hog_feature[j];
		}

		labelsMat.at<int>(i, 0) = (int)labels[i];
	}
	std::cout << "==========Finished computing Hog features==========" << std::endl;

	Ptr<TrainData> td = TrainData::create(trainingDataMat, ROW_SAMPLE, labelsMat);
	Ptr<SVM> svm = SVM::create();
	svm->setType(SVM::C_SVC);
	svm->setKernel(SVM::RBF);
	//svm->setC(12.5);
	//svm->setGamma(0.50625);
	//svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 1000, FLT_EPSILON));
	
	std::cout << "========Training model...=========" << std::endl;
	svm->trainAuto(td);
	svm->save("digits_svm_model_auto_train.yml");
	std::cout << "========Finsihed training model=========" << std::endl;
	
	
	
	while (true) {
		Mat test = imread("C:/Users/13015/Desktop/cs126 final project/source/test3.png", IMREAD_GRAYSCALE);
		resize(test, test, Size(1920, 1080));
		
		///threshold(test, test, 135, 255, THRESH_BINARY_INV);
		
		
		Rect2d region = selectROI(test);
		Mat crop = test(region);

		resize(crop, crop, Size(28, 28));
		medianBlur(crop, crop, 1);
		cout << svm->predict(ComputeHog(crop));
		imshow("test", crop);
		waitKey(0);
		destroyAllWindows();
	}
	
	
	
	/**
	Ptr<SVM> svm = SVM::load("digits_svm_model.yml");
	vector<Mat> test_pictures;
	for (int i = 0; i < 10000; i++) {
		Mat temp = imread("C:/Users/13015/Desktop/cs126 final project/test-images-png/" + to_string(i) + ".png", IMREAD_GRAYSCALE);
		test_pictures.push_back(temp);
	}

	for (int i = 0; i < 10000; i++) {
		cout << svm->predict(ComputeHog(test_pictures[i])) << endl;
		imshow("test", test_pictures[i]);
		waitKey(1000);
		destroyAllWindows();
	}
	*/

	Mat test = imread("C:/Users/13015/Desktop/cs126 final project/test-images-png/" + to_string(1) + ".png", IMREAD_GRAYSCALE);
	vector<float> feature;
	hog.compute(test, feature);
	cout << feature.size() << endl;
}

