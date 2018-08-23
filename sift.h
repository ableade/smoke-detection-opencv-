#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>

using std::vector;
using cv::KeyPoint;
using cv::Feature2D;
using cv::Ptr;
using cv::Mat;

using namespace cv::xfeatures2d;


struct SIFTDetector
{
    Ptr<Feature2D> sift;
    vector<vector< KeyPoint >> keypoints;

    SIFTDetector()
    {
        sift = SIFT::create();
    }

    template<class T>
    void operator()(const T& in, vector<vector<KeyPoint> >& pts, const T& mask)
    {
        sift->detect(in, pts, mask);
    }
    

    template<class T>
    void compute (const T& in, vector<std::vector<cv::KeyPoint> >& pts, const T& mask) {
        sift->compute(in, pts, mask);
    }
};

template<class KPMatcher>
struct SIFTMatcher
{
    KPMatcher matcher;
    template<class T>
    void match(const T& in1, const T& in2, std::vector<cv::DMatch>& matches)
    {
        matcher.match(in1, in2, matches);
    }
};