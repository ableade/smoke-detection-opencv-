#include <boost/filesystem.hpp>
#include <iterator>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>

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
int addTrainingData(string directory, cv::Mat& m) {
    auto count =0;
    if(is_directory(directory))
    {
        std::vector<directory_entry> v;
        copy(directory_iterator(directory), directory_iterator(), back_inserter(v)); 
        for (auto it = v.begin(); it != v.end();  ++ it )
        {
            cv::Mat img = cv::imread(it->path().string());
            if(img.rows == 300 && img.cols == 300) {
                cv::Mat cnv;
                img.convertTo(cnv, CV_32FC1);
                cv::Mat m_flat = cnv.reshape(1,1); // unroll image into single channel vector
                m.push_back(m_flat);
                count++;
            }
        } 
    }
    return count;
}

int main(int argc, char *argv[])
{
    if (argc < 3) {
        cout << "Program usage : <training directory>   <test directory>"<<endl;
        exit(1);
    }

    path positivePath(argc>1? argv[1] : "."), negativePath(argc>1?argv[2]:".");
    cv::Mat trainingData;
    auto pCount = addTrainingData(positivePath.string(), trainingData);
    auto nCount = addTrainingData(negativePath.string(), trainingData);

    vector<float> responses((pCount + nCount), 1);
    for (int i = pCount -1; i < responses.size(); ++i) {
        responses[i] = 0;
    }
    
    cout <<"Size of training data is "<<responses.size()<<endl;

    auto trainDataSet = cv::ml::TrainData::create(trainingData, cv::ml::ROW_SAMPLE, cv::Mat(responses, true), 
        cv::noArray(), cv::noArray(), cv::noArray(), cv::noArray());

    //We will only use 80% of our data set for training.
    trainDataSet->setTrainTestSplitRatio(0.8);

	cv::Ptr<cv::ml::ANN_MLP> nn = cv::ml::ANN_MLP::create();

	//Neural network with three layers and 512 nodes
	std::vector<int> layerSizes = { 270000, 512, 1 };
	nn->setLayerSizes(layerSizes);
    cout << "Training neural network with 3 layers"<<endl;
	nn->train(trainDataSet);

    cout << "Calcuating error for neural network"<<endl;
    cv::Mat testResponses;
    auto error = nn->calcError(trainDataSet, true, testResponses);
    cout << "Percentage error was "<<error<<" percent"<<endl;
    cout << "Size of test set was "<<testResponses.size()<<endl;

    string answer;

    while (1) {
        cout << "Enter the path to an image to detect if it contains smoke or enter quit to exit"<<endl;
        cin >> answer;
        if (answer == "quit") {
            break;
        }
        path inputImage(answer);
        cv::Mat img = cv::imread(inputImage.string());
        if (img.empty())
        {
            cerr << "WARNING: Could not read image." << std::endl;
            continue;
        } else {
            cv::Mat cnv;
            img.convertTo(cnv, CV_32FC1);
            cv::Mat m_flat = cnv.reshape(1,1); // unroll image into single channel vector
            auto prediction = nn->predict(m_flat);
            cout << "prediciton was "<< prediction<<endl;
        }
    }
	return (0);
}