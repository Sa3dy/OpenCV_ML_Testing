#include <vector>
#include <boost/filesystem.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>

using namespace std;
using namespace boost::filesystem;
using namespace cv;

//location of the training data
#define TRAINING_DATA_DIR "data/train/"
//location of the evaluation data
#define EVAL_DATA_DIR "data/eval/"

//See article on BoW model for details
Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("FlannBased");
Ptr<DescriptorExtractor> extractor = DescriptorExtractor::create("ORB");
Ptr<FeatureDetector> detector = FeatureDetector::create("ORB");

//See article on BoW model for details
int dictionarySize = 1000;
TermCriteria tc(CV_TERMCRIT_ITER, 10, 0.001);
int retries = 1;
int flags = KMEANS_PP_CENTERS;

//See article on BoW model for details
BOWKMeansTrainer bowTrainer(dictionarySize, tc, retries, flags);
//See article on BoW model for details
BOWImgDescriptorExtractor bowDE(extractor, matcher);

/**
 * \brief Recursively traverses a folder hierarchy. Extracts features from the training images and adds them to the bowTrainer.
 */
void extractTrainingVocabulary(const path& basepath) {
	for (directory_iterator iter = directory_iterator(basepath); iter
			!= directory_iterator(); iter++) {
		directory_entry entry = *iter;

		if (is_directory(entry.path())) {

			cout << "Processing directory " << entry.path().string() << endl;
			extractTrainingVocabulary(entry.path());

		} else {

			path entryPath = entry.path();
			if (entryPath.extension() == ".jpg") {

				cout << "Processing file " << entryPath.string() << endl;
				Mat img = imread(entryPath.string());
				if (!img.empty()) {
					vector<KeyPoint> keypoints;
					detector->detect(img, keypoints);
					if (keypoints.empty()) {
						cerr << "Warning: Could not find key points in image: "
								<< entryPath.string() << endl;
					} else {
						Mat features;
						extractor->compute(img, keypoints, features);
						bowTrainer.add(features);
					}
				} else {
					cerr << "Warning: Could not read image: "
							<< entryPath.string() << endl;
				}

			}
		}
	}
}

/**
 * \brief Recursively traverses a folder hierarchy. Creates a BoW descriptor for each image encountered.
 */
void extractBOWDescriptor(const path& basepath, Mat& descriptors, Mat& labels) {
	for (directory_iterator iter = directory_iterator(basepath); iter
			!= directory_iterator(); iter++) {
		directory_entry entry = *iter;
		if (is_directory(entry.path())) {
			cout << "Processing directory " << entry.path().string() << endl;
			extractBOWDescriptor(entry.path(), descriptors, labels);
		} else {
			path entryPath = entry.path();
			if (entryPath.extension() == ".jpg") {
				cout << "Processing file " << entryPath.string() << endl;
				Mat img = imread(entryPath.string());
				if (!img.empty()) {
					vector<KeyPoint> keypoints;
					detector->detect(img, keypoints);
					if (keypoints.empty()) {
						cerr << "Warning: Could not find key points in image: "
								<< entryPath.string() << endl;
					} else {
						Mat bowDescriptor;
						bowDE.compute(img, keypoints, bowDescriptor);
						descriptors.push_back(bowDescriptor);
						float label=atof(entryPath.filename().string().c_str());
						labels.push_back(label);
					}
				} else {
					cerr << "Warning: Could not read image: "
							<< entryPath.string() << endl;
				}
			}
		}
	}
}

int main(int argc, char ** argv) {
	//cv::initModule_nonfree();
	cout<<"Creating dictionary..."<<endl;
	extractTrainingVocabulary(path(TRAINING_DATA_DIR));
	vector<Mat> descriptors = bowTrainer.getDescriptors();
	int count=0;
	for(vector<Mat>::iterator iter=descriptors.begin();iter!=descriptors.end();iter++)
	{
		count+=iter->rows;
	}
	cout<<"Clustering "<<count<<" features"<<endl;
	Mat dictionary = bowTrainer.cluster();
	bowDE.setVocabulary(dictionary);

	FileStorage fs_writer("dictionary.xml", FileStorage::WRITE);

	write(fs_writer, "dictionary", dictionary);

	fs_writer.release();

	cout<<"Processing training data..."<<endl;
	Mat trainingData(0, dictionarySize, CV_32FC1);
	Mat labels(0, 1, CV_32FC1);
	extractBOWDescriptor(path(TRAINING_DATA_DIR), trainingData, labels);

	NormalBayesClassifier classifier;
	cout<<"Training classifier..."<<endl;

	classifier.train(trainingData, labels);

	classifier.save("SVM_newClassifier.xml");

	cout<<"Processing evaluation data..."<<endl;
	Mat evalData(0, dictionarySize, CV_32FC1);
	Mat groundTruth(0, 1, CV_32FC1);
	extractBOWDescriptor(path(EVAL_DATA_DIR), evalData, groundTruth);

	cout<<"Evaluating classifier..."<<endl;
	Mat results;
	classifier.predict(evalData, &results);

	double errorRate = (double) countNonZero(groundTruth - results) / evalData.rows;
			;
	cout << "Error rate: " << errorRate << endl;

}
