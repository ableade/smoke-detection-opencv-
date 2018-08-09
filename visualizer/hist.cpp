/**
 * Displays color histogram for images in a specified directory 
 * Code gotten from https://docs.opencv.org/2.4/doc/tutorials/imgproc/histograms/histogram_calculation/histogram_calculation.html
 * 
 */
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <boost/filesystem.hpp>
#include <string>
#include <iostream>
#include <vector>

using std::cout;
using cv::Mat;
using cv::waitKey;
using std::string;
using std::endl;
using cv::Point;
using cv::Scalar;
using cv::NORM_MINMAX;
using cv::namedWindow;
using std::vector;

using namespace boost::filesystem;

void loadImageAndPlotHistogram (string directory) {
    namedWindow("calcHist Demo", CV_WINDOW_AUTOSIZE );
    assert (is_directory(directory));
    Mat src, dst;
    std::vector<directory_entry> v;
    copy(directory_iterator(directory), directory_iterator(), back_inserter(v)); 
    for (auto it = v.begin(); it != v.end();  ++ it )
    {
        src = cv::imread(it->path().string());
        if( !src.data ) {
            continue;
        }
        /// Separate the image in 3 places ( B, G and R )
        vector<Mat> bgr_planes;
        split( src, bgr_planes );
        /// Establish the number of bins
        int histSize = 256;
        /// Set the ranges ( for B,G,R) )
        float range[] = { 0, 256 } ;
        const float* histRange = { range };
        bool uniform = true; bool accumulate = false;
        Mat b_hist, g_hist, r_hist;
        /// Compute the histograms:
        calcHist( &bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate );
        calcHist( &bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate );
        calcHist( &bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate );
         // Draw the histograms for B, G and R
        int hist_w = 512; int hist_h = 400;
        int bin_w = cvRound( (double) hist_w/histSize );
        Mat histImage( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );

        /// Normalize the result to [ 0, histImage.rows ]
        normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
        normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
        normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );

        /// Draw for each channel
        for( int i = 1; i < histSize; i++ )
        {
            line( histImage, Point( bin_w*(i-1), hist_h - cvRound(b_hist.at<float>(i-1)) ) ,
                       Point( bin_w*(i), hist_h - cvRound(b_hist.at<float>(i)) ),
                       Scalar( 255, 0, 0), 2, 8, 0  );
            line( histImage, Point( bin_w*(i-1), hist_h - cvRound(g_hist.at<float>(i-1)) ) ,
                       Point( bin_w*(i), hist_h - cvRound(g_hist.at<float>(i)) ),
                       Scalar( 0, 255, 0), 2, 8, 0  );
            line( histImage, Point( bin_w*(i-1), hist_h - cvRound(r_hist.at<float>(i-1)) ) ,
                       Point( bin_w*(i), hist_h - cvRound(r_hist.at<float>(i)) ),
                       Scalar( 0, 0, 255), 2, 8, 0  );
        }
        imshow("calcHist Demo", histImage );
        char c = waitKey(0);

        if (c == 'n') {
            continue;
        } else {
            break;
        }
    }
}

int main (int argc, char ** argv) {
    Mat src, dst;

      if (argc < 2) {
        cout << "Program usage : <image directory>"<<endl;
        exit(1);
    }
    path imageDirectory(argc>1? argv[1] : ".");
    
    loadImageAndPlotHistogram(imageDirectory.string());
}