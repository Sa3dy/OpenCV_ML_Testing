#include "opencv/cv.h" 
#include "opencv/highgui.h"
#include "opencv/ml.h"
#include <stdio.h>
#include <iostream>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>
#include <opencv2/nonfree/features2d.hpp>

using namespace cv; 
using namespace std;

using std::cout;
using std::cerr;
using std::endl;
using std::vector;

char ch[30];

//--------Using SURF as feature extractor and FlannBased for assigning a new point to the nearest one in the dictionary
Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("FlannBased");
Ptr<DescriptorExtractor> extractor = new SurfDescriptorExtractor();
SurfFeatureDetector detector(500);
//---dictionary size=number of cluster's centroids
int dictionarySize = 1500;
TermCriteria tc(CV_TERMCRIT_ITER, 10, 0.001);
int retries = 1;
int flags = KMEANS_PP_CENTERS;
BOWKMeansTrainer bowTrainer(dictionarySize, tc, retries, flags);
BOWImgDescriptorExtractor bowDE(extractor, matcher);

map<float, string> classesNames;

int main()
{

	int i,j;
	IplImage *img2;
	int k=0;
	Mat dictionary;

	classesNames[1] = "plane";
	classesNames[2] = "car";
	classesNames[3] = "tiger";
	classesNames[4] = "motorbike";

	FileStorage fs_reader("vocabulary.xml", FileStorage::READ);

	read(fs_reader["vocabulary"], dictionary);

	cout << "dictionary.rows * dictionary.cols: " << dictionary.rows * dictionary.cols << endl;

	fs_reader.release();

	bowDE.setVocabulary(dictionary);

	//Setting up SVM parameters
	CvSVMParams params;
	params.kernel_type=CvSVM::RBF;
	params.svm_type=CvSVM::C_SVC;
	params.gamma=0.50625000000000009;
	params.C=312.50000000000000;
	params.term_crit=cvTermCriteria(CV_TERMCRIT_ITER,100,0.000001);
	CvSVM svm;

	printf("%s\n","Training SVM classifier");

	svm.load("SVM_Classifier.xml");

	cout << "svm.get_support_vector_count(): " << svm.get_support_vector_count() << endl;

	cout<<"Processing evaluation data..."<<endl;

	Mat groundTruth(0, 1, CV_32FC1);
	Mat evalData(0, dictionarySize, CV_32FC1);
	k=0;
	vector<KeyPoint> keypoint2;
	Mat bowDescriptor2;


	Mat results(0, 1, CV_32FC1);
	for(j=1;j<=4;j++)
		for(i=1;i<=5;i++){


			sprintf( ch,"%s%d%s%d%s","eval/",j," (",i,").jpg");
			const char* imageName = ch;
			img2 = cvLoadImage(imageName,0);

			detector.detect(img2, keypoint2);
			bowDE.compute(img2, keypoint2, bowDescriptor2);

			evalData.push_back(bowDescriptor2);
			groundTruth.push_back((float) j);

			cout << "True Class: " << classesNames[(float) j] << endl;

			float response = svm.predict(bowDescriptor2);

			cout << "Predicted Class: " << classesNames[(float) response] << endl;

			results.push_back(response);
		}



		//calculate the number of unmatched classes 
		double errorRate = (double) countNonZero(groundTruth - results) / evalData.rows;
		cout << "Error rate is= " << errorRate << endl;

		system("PAUSE");
		return 0;

}