#include <boost/filesystem.hpp>
#include <iterator>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cassert>
#include <string>
#include <map>
#include <random>

#include "scopedtimer.h"
#include "sift.h"

using Util::ScopedTimer;
using namespace boost::filesystem;
using std::cout;
using std::endl;
using std::vector;
using std::string;
using std::cerr;
using std::cin;
using std::random_device;
using std::uniform_int_distribution;
using std::vector;
using std::random_shuffle;

using cv::normalize;
using cv::NORM_MINMAX;

struct DataPoint {
    string fileName;
    bool isPositive;
    Mat descriptors;
    int startRange;
    int endRange;
};

void shuffleTrainingData(cv::Mat& m, cv::Mat& responses) {
    random_device rd;
    uniform_int_distribution<int> dist(0, responses.rows -1);

    for(int i =0; i< responses.rows; i ++) {
        auto rand1 = dist(rd); auto rand2 = dist(rd);
        auto matTmp = m.row(rand1);
        auto responseTemp = responses.row(rand1);
        m.row(rand1) = (m.row(rand2) + 0);
        m.row(rand2) = matTmp + 0;
        responses.row(rand1) = (responses.row(rand2) + 0);
        responses.row(rand2) = responseTemp + 0;
    }
}

vector<DataPoint> getDescriptorsAndKeypoints(vector<directory_entry> v) {
    auto start = 0;
    vector <DataPoint> dataPoints;
    SIFTDetector sift;
    vector<KeyPoint> keypoints;
    Mat descriptors;

    for (auto it = v.begin(); it != v.end(); ++it) {
        auto result = it->path().string().find("nosmoke");
        cv::Mat img = cv::imread(it->path().string());
        if (img.empty())
            continue;
        DataPoint dataPoint;
        dataPoint.fileName = it->path().string();
        dataPoint.isPositive = false;
        sift(img, Mat(), keypoints, dataPoint.descriptors);
        if (result == -1) {
        //this is a positive data training example
            dataPoint.isPositive = true;
        }
        dataPoint.startRange = start;
        dataPoint.endRange = start + keypoints.size();
        start = dataPoint.endRange;
        dataPoints.push_back(dataPoint);
    }
    return dataPoints;
}

std::vector<directory_entry> getTrainingImages(string path) {
    std::vector<directory_entry> v;
    assert (is_directory(path));
    copy(directory_iterator(path), directory_iterator(), back_inserter(v));
    return v;
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        cout << "Program usage : <training directory> <network input size>" << endl;
        exit(1);
    }

    path trainingDirectory(argv[1]);
    int networkInputSize = atoi(argv[2]);
    cv::Mat labels, vocabulary, descriptorSet, trainingData;
    auto data = getTrainingImages(trainingDirectory.string());
    vector<DataPoint> dataPoints;
    {
        ScopedTimer scopedTimer{"Retrieved descriptor for all images"};
        dataPoints = getDescriptorsAndKeypoints(data);
    }

    for(auto dp : dataPoints) {
        descriptorSet.push_back(dp.descriptors);
        cout << "Start range is "<<dp.startRange << "End range is "<<dp.endRange<<endl;
    }

    {

        ScopedTimer scopedTimer{"Finished running kmeans to cluster bag of words on dataset"};
        cv::kmeans(descriptorSet, networkInputSize, labels, 
            cv::TermCriteria(cv::TermCriteria::EPS +
      cv::TermCriteria::MAX_ITER, 10, 0.01), 1, cv::KMEANS_PP_CENTERS, vocabulary);
    }

    
    //generate bag of words for our data set

    cv::Mat responses;

    cv::Mat positiveCode, negativeCode;
    negativeCode = cv::Mat::zeros(cv::Size(2,1), CV_32F);
    positiveCode = negativeCode.clone();

    positiveCode.at<float>(0,0) = 1; negativeCode.at<float>(0,1) = 1;

    random_shuffle(dataPoints.begin(), dataPoints.end());
    
    for (auto dp: dataPoints) {
        //do nothing see if this compiles
        Mat hist = Mat::zeros(cv::Size(networkInputSize, 1), CV_32F);
        for (int j=dp.startRange; j< dp.endRange; j++) {
            hist.at<float>(labels.at<float>(j))++;
        }
        Mat normHist;
        normalize(hist, normHist, 0, hist.rows, NORM_MINMAX, -1, Mat());
        trainingData.push_back(normHist);
        if (dp.isPositive) {
            responses.push_back(positiveCode);
        } else {
            responses.push_back(negativeCode);
        }
    }


    cout << "Responses number of rows are "<<responses.rows<<endl;

    cout << "Size of training data is " << trainingData.size() << endl;


   // shuffleTrainingData(trainingData, responses);

    auto dataSet = cv::ml::TrainData::create(trainingData, cv::ml::ROW_SAMPLE, responses,
            cv::noArray(), cv::noArray(), cv::noArray(), cv::noArray());


    //We will only use 80% of our data set for training.
    dataSet->setTrainTestSplitRatio(0.8);
    cv::Ptr<cv::ml::ANN_MLP> nn = cv::ml::ANN_MLP::create();
    nn->setActivationFunction(cv::ml::ANN_MLP::GAUSSIAN);

    //Neural network with 3 hidden layers
    std::vector<int> colorHistogramLayerSizes {networkInputSize, 1000, 1000, 2};
    nn->setLayerSizes(colorHistogramLayerSizes);
    {
        ScopedTimer scopedTimer{"Trained neural network with 3 layers with single channel histogram features"};
        nn->train(dataSet);
    }
    cout << "Calcuating error for single channel color histogram neural network" << endl;
    auto error = nn->calcError(dataSet, true, cv::noArray());
    auto trainError = nn->calcError(dataSet, false, cv::noArray());
    cout << "Percentage error over the test set was " << error << " percent" << endl;
    cout << "Percentage error over the training set was " << trainError << " percent" << endl;

    auto testSamples =  dataSet->getTrainSamples();

    for(int i=0; i< testSamples.rows; ++i) {
        cout << "Size is "<< testSamples.row(i).size();
        auto prediction = nn->predict(testSamples.row(i));
        cout << "prediction was " << prediction << endl;
    }

    string answer;

    while (1) {
        cout << "Enter the path to an image to detect if it contains smoke or enter quit to exit" << endl;
        cin >> answer;
        if (answer == "quit") {
            break;
        }
        cv::Mat img = cv::imread(answer);
        if (img.empty()) {
            cerr << "WARNING: Could not read image." << std::endl;
            continue;
        } else {
            //to be implemented
        }
    }
    return (0);
}