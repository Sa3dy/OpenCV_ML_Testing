#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>

using namespace std;
using namespace cv;

Mat extractWRTErodeDilate(Mat srcImage, Mat erodedilateResult)
{
	Mat extractedMat;
	Mat srcImageCopy = srcImage.clone();
	Mat erodedilateResultCopy = erodedilateResult.clone();

	for(int i = 0; i < srcImageCopy.rows; i++) {
		for(int j = 0; j < srcImageCopy.cols; j++) {

			if (!(erodedilateResultCopy.at<unsigned char>(i, j) == 255))
			{
				srcImageCopy.at<Vec3b>(i, j)[0] = 255;
				srcImageCopy.at<Vec3b>(i, j)[1] = 255;
				srcImageCopy.at<Vec3b>(i, j)[2] = 255;
			}

		}
	}
	extractedMat = srcImageCopy;
	return extractedMat;
}

int main(){

	char ch[30];

	Mat img;
	int i,j;
	for(j=1;j<=2;j++){
		for(i=1;i<=700;i++){
			sprintf( ch,"%s%d%s%d%s","trainsigns/",j,"(",i,").jpg");
			const char* imageName = ch;
			img = cvLoadImage(imageName,1);


			Mat srcImage = img.clone();
			Mat srcImageCopy = img.clone();

			Mat blurred;
			blur( srcImage, blurred, Size(3,3) );
			Mat hsv;
			cvtColor(blurred, hsv, CV_BGR2HSV);

			Mat erodedilateResult;

			inRange(hsv, Scalar(0, 40, 60), Scalar(20, 150, 255), erodedilateResult);

			int dilation_size = 20;
			int erosion_size = 20;

			dilate(erodedilateResult, erodedilateResult, getStructuringElement(
				MORPH_ELLIPSE,
				Size(2*dilation_size+1, 2*dilation_size+1),
				Point(dilation_size, dilation_size)));

			erode(erodedilateResult, erodedilateResult, getStructuringElement(
				MORPH_ELLIPSE,
				Size(2*erosion_size+1, 2*erosion_size+1),
				Point(erosion_size, erosion_size)));

			Mat extractedMat = extractWRTErodeDilate(srcImage, erodedilateResult);

			imshow("srcImage", srcImage);

			imshow("hsv", hsv);

			imshow("erodedilateResult", erodedilateResult);

			imshow("extractedMat", extractedMat);

			waitKey(0);
		}
	}


	return 0;
}
