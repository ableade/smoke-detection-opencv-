#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <boost/filesystem.hpp>
#include <string>
#include <iostream>
#include <vector>

using cv::Mat;
using cv::Size;
using cv::xfeatures2d;
using std::vector;

Mat loadImages(string directory) {
    SIFTDetector sift;
    vector<KeyPoint> keypoints;
	assert (is_directory(directory));
	std::vector<directory_entry> v;
    copy(directory_iterator(directory), directory_iterator(), back_inserter(v));
    for (auto it = v.begin(); it != v.end();  ++ it ) {
    	img = imread(it->path().string());
    	if (!img.data()) 
    		continue
        surf(img1.getMat(ACCESS_READ), Mat(), keypoints1, descriptors1);
    	cv::resize(img, img, cv::Size(500, 300), 0, 0);
  		
    }
}

int main (int argc, char ** argv) {
    SIFTMatcher<BFMatcher> matcher;
	std::vector<KeyPoint> keypoints1, keypoints2;
    std::vector<DMatch> matches;
    if (argc < 2) {
        cout << "Program usage : <image directory>"<<endl;
        exit(1);
    }
    path imageDirectory(argc>1? argv[1] : ".");
}