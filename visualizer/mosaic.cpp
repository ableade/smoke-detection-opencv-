#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <boost/filesystem.hpp>
#include <string>
#include <iostream>
#include <vector>

using cv::Mat;
using cv::Size;
using cv::xfeatures2d;

struct SIFTDetector
{
    Ptr<Feature2D> sift;
    SIFTDetector()
    {
        sift = SIFT::create();
    }
    template<class T>
    void operator()(const T& in, const T& mask, std::vector<cv::KeyPoint>& pts, T& descriptors, bool useProvided = false)
    {
        surf->detectAndCompute(in, mask, pts, descriptors, useProvided);
    }
};

template<class KPMatcher>
struct SURFMatcher
{
    KPMatcher matcher;
    template<class T>
    void match(const T& in1, const T& in2, std::vector<cv::DMatch>& matches)
    {
        matcher.match(in1, in2, matches);
    }
};

Mat loadImages(string directory) {
	assert (is_directory(directory));
	std::vector<directory_entry> v;
    copy(directory_iterator(directory), directory_iterator(), back_inserter(v));
    for (auto it = v.begin(); it != v.end();  ++ it ) {
    	img = imread(it->path().string());
    	if (!img.data()) 
    		continue

    	cv::resize(img, img, cv::Size(500, 300), 0, 0);
  		
    }
}

int main (int argc, char ** argv) {
	SIFTDetector sift;
    SIFTMatcher<BFMatcher> matcher;
	std::vector<KeyPoint> keypoints1, keypoints2;
    std::vector<DMatch> matches;
    if (argc < 2) {
        cout << "Program usage : <image directory>"<<endl;
        exit(1);
    }
    path imageDirectory(argc>1? argv[1] : ".");
}