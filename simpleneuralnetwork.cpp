#include <boost/filesystem.hpp>
#include <iterator>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cassert>
#include <string>

#include "scopedtimer.h"
using Util::ScopedTimer;
using namespace boost::filesystem;
using std::cout;
using std::endl;
using std::vector;
using std::string;
using std::cerr;
using std::cin;

/*
Adds training data images from directory. Images in training directory are expected to be JPG 
300 * 300. Returns number of images added
 */
int addTrainingData(std::vector<directory_entry> v, cv::Mat& m) {
    auto count = 0;
    for (auto it = v.begin(); it != v.end(); ++it) {
        cv::Mat img = cv::imread(it->path().string());
        if (img.empty())
            continue;

        if (img.rows == 300 && img.cols == 300) {
            cv::Mat cnv;
            img.convertTo(cnv, CV_32F);
            cv::Mat m_flat = cnv.reshape(1, 1); // unroll image into single channel vector
            m.push_back(m_flat);
            count++;
        }
    }
    return count;
}

cv::Mat getColorHistorgramDescriptorsSingleChannel(const cv::Mat& tData) {
    bool uniform = true; bool accumulate = false;
    int histSize = 256;
    /// Set the ranges ( for B,G,R) )
    float range[] = { 0, 256 } ;
    const float* histRange = { range };
    cv::Mat colorHist(tData.rows, 256, CV_32F);
    for(int i=0; i< tData.rows; ++i) {
        auto r = tData.row(i);
        vector<cv::Mat> bgr_planes;
        cv::Mat temp;
        cv::split( r , bgr_planes );
        cv::Mat hist;
        /// Compute the histograms:
        calcHist( &r, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange, uniform, accumulate);
        cv::Mat transpose = hist.t();
        transpose.copyTo(colorHist.row(i));
    }
    return colorHist;
}

cv::Mat getColorHistorgramDescriptorsMultipleChannels(std::vector<directory_entry> v) {
    bool uniform = true; bool accumulate = false;
    int histSize = 256;
    /// Set the ranges ( for B,G,R) )
    float range[] = { 0, 256 } ;
    const float* histRange = { range };
    cv::Mat colorHist(v.size(), 768, CV_32F);
    //hconcat(bgr_planes[1], bgr_planes[2], temp);
    //hconcat(bgr_planes[0], temp, colorHist.row(i));
}

std::vector<directory_entry> getTrainingImages(string path) {
    std::vector<directory_entry> v;
    assert (is_directory(path));
    copy(directory_iterator(path), directory_iterator(), back_inserter(v));
    return v;
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        cout << "Program usage : <training directory>   <test directory>" << endl;
        exit(1);
    }

    path positivePath(argc > 1 ? argv[1] : "."), negativePath(argc > 1 ? argv[2] : ".");
    cv::Mat trainingData;
    auto pData = getTrainingImages(positivePath.string());
    auto nData = getTrainingImages(negativePath.string());
    int pCount = addTrainingData(pData, trainingData);
    int nCount = addTrainingData(nData, trainingData);
    int dataCount = pCount + nCount;

    vector<float> responses(dataCount, 1);

    for (int i = pCount - 1; i < responses.size(); ++i) {
        responses[i] = 0;
    }

    cout << "Size of training data is " << responses.size() << endl;
    auto histFeatures = getColorHistorgramDescriptorsSingleChannel(trainingData);

    auto trainDataSet = cv::ml::TrainData::create(trainingData, cv::ml::ROW_SAMPLE, cv::Mat(responses, true),
            cv::noArray(), cv::noArray(), cv::noArray(), cv::noArray());

    auto colorHistDataSet = cv::ml::TrainData::create(histFeatures, cv::ml::ROW_SAMPLE, cv::Mat(responses, true),
            cv::noArray(), cv::noArray(), cv::noArray(), cv::noArray());

    //We will only use 80% of our data set for training.
    trainDataSet->setTrainTestSplitRatio(0.8); colorHistDataSet ->setTrainTestSplitRatio(0.8);
    cv::Ptr<cv::ml::ANN_MLP> nn = cv::ml::ANN_MLP::create();

    //Neural network with three layers and 512 nodes
    std::vector<int> layerSizes = {270000, 1012, 512, 1};
    nn->setLayerSizes(layerSizes);
    {
        ScopedTimer scopedTimer{"Trained neural network with 3 layers with varying descriptor set"};
        nn->train(trainDataSet);
    }
    cout << "Calcuating error for varying descriptor neural network" << endl;
    auto error = nn->calcError(trainDataSet, true, cv::noArray());
    auto trainError = nn->calcError(trainDataSet, false, cv::noArray());
    cout << "Percentage error over the test set was " << error << " percent" << endl;
    cout << "Percentage error over the training set was " << trainError << " percent" << endl;


  
    std::vector<int> colorHistogramLayerSizes {256, 200, 150, 100, 1};
    nn->setLayerSizes(layerSizes);
    {
        ScopedTimer scopedTimer{"Trained neural network with 3 layers with single channel histogram features"};
        nn->train(colorHistDataSet);
    }
    cout << "Calcuating error for single channel color histogram neural network" << endl;
    error = nn->calcError(trainDataSet, true, cv::noArray());
    trainError = nn->calcError(trainDataSet, false, cv::noArray());
    cout << "Percentage error over the test set was " << error << " percent" << endl;
    cout << "Percentage error over the training set was " << trainError << " percent" << endl;

    string answer;

    while (1) {
        cout << "Enter the path to an image to detect if it contains smoke or enter quit to exit" << endl;
        cin >> answer;
        if (answer == "quit") {
            break;
        }
        path inputImage(answer);
        cv::Mat img = cv::imread(inputImage.string());
        if (img.empty()) {
            cerr << "WARNING: Could not read image." << std::endl;
            continue;
        } else {
            cv::Mat cnv;
            img.convertTo(cnv, CV_32FC1);
            cv::Mat m_flat = cnv.reshape(1, 1); // unroll image into single channel vector
            auto prediction = nn->predict(m_flat);
            cout << "prediction was " << prediction << endl;
        }
    }
    return (0);
}