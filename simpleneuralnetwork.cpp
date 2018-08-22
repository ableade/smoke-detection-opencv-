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

/*
Adds training data images from directory. Images in training directory are expected to be JPG 
300 * 300. Returns number of images added
 */
int addTrainingData(std::vector<directory_entry> v, cv::Mat& m) {
    auto count = 0;
    for (auto it = v.begin(); it != v.end(); ++it) {
        cv::Mat img = cv::imread(it->path().string());
        cv:: Mat normImg(img.size(), img.type());

        if (img.empty())
            continue;

        cv::normalize(img, normImg, 0, 255, cv::NORM_MINMAX);
        if (img.rows == 300 && img.cols == 300) {
            cv::Mat cnv;
            normImg.convertTo(cnv, CV_32F);
            cv::Mat m_flat = cnv.reshape(1, 1); // unroll image into single channel vector
            m.push_back(m_flat);
            count++;
        }
    }
    return count;
}

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


Mat getDescriptors(vector<directory_entry> v) {
    for (auto it = v.begin(); it != v.end(); ++it) {
        cv::Mat img = cv::imread(it->path().string());
        if (! img.data)
            continue;
        cv::normalize(img, img, 0, 255, cv::NORM_MINMAX);
    }
}
int addColorHistTrainingData(std::vector<directory_entry> v, cv::Mat& tData) {
    auto count =0;
    auto featureVectorSize = 768;
    for (auto it = v.begin(); it != v.end(); ++it) {
        cv::Mat img = cv::imread(it->path().string());
        if (! img.data)
            continue;
        cv::normalize(img, img, 0, 255, cv::NORM_MINMAX);
        cv::imshow("Normalized image", img );
        vector<cv::Mat> bgr_planes;
        split( img, bgr_planes );
        int histSize = 256;
        /// Set the ranges ( for B,G,R) )
        float range[] = { 0, 256 } ;
        const float* histRange = { range };
        bool uniform = true; bool accumulate = false;
        cv::Mat b_hist, g_hist, r_hist;
        vector<cv::Mat> test(3);
        calcHist( &bgr_planes[0], 1, 0, cv::Mat(), test[0], 1, &histSize, &histRange, uniform, accumulate );
        calcHist( &bgr_planes[1], 1, 0, cv::Mat(), test[1], 1, &histSize, &histRange, uniform, accumulate );
        calcHist( &bgr_planes[2], 1, 0, cv::Mat(), test[2], 1, &histSize, &histRange, uniform, accumulate );

        cv::Mat dataPoint (cv::Size(featureVectorSize,1), CV_32F);
        //vector <cv::Mat> hists {b_hist, g_hist, r_hist};

        #if 1
        for (int i=0; i< test.size(); ++i) {
            int offset = (histSize) * i;
            for (int j =0; j < test[i].rows; ++j) {
                dataPoint.at<float>(0,offset + j) = test[i].at<float>(j,0);
            }
        }
        #endif
        count ++;
        tData.push_back(dataPoint);
    }
    return count;
}

cv::Mat getColorHistorgramDescriptorsMultipleChannels(std::vector<directory_entry> v) {
    bool uniform = true; bool accumulate = false;
    int histSize = 256;
    /// Set the ranges ( for B,G,R) )
    float range[] = { 0, 256 } ;
    const float* histRange = { range };
    cv::Mat colorHist(v.size(), 768, CV_32F);
    return colorHist;
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
    //int pCount = addTrainingData(pData, trainingData);
    //int nCount = addTrainingData(nData, trainingData);

    int pCount = addColorHistTrainingData(pData, trainingData);
    int nCount = addColorHistTrainingData(nData, trainingData);
    int dataCount = pCount + nCount;

    cv::Mat responses;

    cv::Mat positiveCode, negativeCode;
    negativeCode = cv::Mat::zeros(cv::Size(2,1), CV_32F);
    positiveCode = negativeCode.clone();

    positiveCode.at<float>(0,0) = 1; negativeCode.at<float>(0,1) = 1;

    cout << "positive code: "<< positiveCode <<endl;
    cout << "Negative code: "<< negativeCode <<endl;

    for(int i =0; i < pCount; ++i) {
        responses.push_back(positiveCode);
    }

    for (int i = 0; i < nCount; ++i) {
        responses.push_back(negativeCode);
    }

    cout << "Responses number of rows are "<<responses.rows<<endl;

    cout << "Size of training data is " << trainingData.size() << endl;


    shuffleTrainingData(trainingData, responses);

    auto dataSet = cv::ml::TrainData::create(trainingData, cv::ml::ROW_SAMPLE, responses,
            cv::noArray(), cv::noArray(), cv::noArray(), cv::noArray());


    //We will only use 80% of our data set for training.
    dataSet->setTrainTestSplitRatio(0.8);
    cv::Ptr<cv::ml::ANN_MLP> nn = cv::ml::ANN_MLP::create();
    nn->setActivationFunction(cv::ml::ANN_MLP::GAUSSIAN);

    /*
    //Neural network with 2 hidden layers
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
    */

    //Neural network with 3 hidden layers
    std::vector<int> colorHistogramLayerSizes {768, 1000, 1000, 2};
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

    auto weights = nn->getWeights(2);

    cout << "Weights of trained neural network for layer 2 are "<<weights<<endl;

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