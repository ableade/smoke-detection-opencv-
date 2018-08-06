#include <boost/filesystem.hpp>
#include <iterator>
#include <iostream>
#include <vector>

using namespace boost::filesystem;


/*
Adds training data images from directory.
Returns number of images added
*/
int addTrainingData(string directory, cv::Mat m) {
    auto count =0;
    if(is_directory(p))
    {
        copy(directory_iterator(p), directory_iterator(), back_inserter(v));
        std::cout << p << " is a directory containing:\n";
        cv::Vec_2i positve {1,0};
        cv::Vec_2i negative {0,1}; 
        for ( std::vector<directory_entry>::const_iterator it = v.begin(); it != v.end();  ++ it )
        {
            cv::Mat img = cv::imread((*it).path);
            cv::Mat cnv;
            img.convertTo(cnv, CV_32FC1);
            cv::Mat m_flat = img.reshape(1,1); // unroll image into single channel vector
            m.push_back(m_flag);
            count++;
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
    path positivePath(argc>1? argv : "."), negativePath(argc>1?argv[2]:".");
    cv::Mat trainingData;
    auto pCount = addTrainingData(positivePath);
    auto nCount = addTrainingData(negativePath);
    vector<float> responses((pCount + nCount), 1);

    for (int i = pCount -1; i < responses.size(); ++i) {
        responses[i] = 0;
    }
}