// std
#include <iostream>
#include <iomanip>

// cctag
#include "cctag/ICCTag.hpp"
#include "cctag/CCTag.hpp"

// opencv
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/opencv.hpp>

// file
#include <fstream>

// clock
#include <time.h>
clock_t start_time,end_time;

using namespace std;
using namespace cctag;


const string pwd = "/home/gy/Desktop/HIT_study/computer-vision/project/pro1/";

inline double getTime()
{
    cout << "used time: " << double(end_time - start_time) / CLOCKS_PER_SEC<< "s" <<endl; 
}

void drawMarkers(const boost::ptr_list<ICCTag>& markers, cv::Mat& image, ofstream& out, bool showUnreliable = true)
{
    for(const ICCTag& marker : markers)
    {
        const cv::Point center = cv::Point(marker.x(), marker.y());
        const int radius = 10;
        const int fontSize = 3;
        if(marker.getStatus() == status::id_reliable)
        {   

            const cv::Scalar color = cv::Scalar(0, 255, 0, 255);
            const auto rescaledOuterEllipse = marker.rescaledOuterEllipse();

            cv::circle(image, center, radius, color, 3);
            cv::putText(image, std::to_string(marker.id()), center, cv::FONT_HERSHEY_SIMPLEX, fontSize, color, 3);
            cv::ellipse(image,
                        center,
                        cv::Size(rescaledOuterEllipse.a(), rescaledOuterEllipse.b()),
                        rescaledOuterEllipse.angle() * 180 / boost::math::constants::pi<double>(),
                        0,
                        360,
                        color,
                        3);
        }
        else if(showUnreliable)
        {
            const cv::Scalar color = cv::Scalar(0, 0, 255, 255);
            cv::circle(image, center, radius, color, 2);
            cv::putText(image, std::to_string(marker.id()), center, cv::FONT_HERSHEY_SIMPLEX, fontSize, color, 3);
        }
    }
}


void test_read_bmp()
{
    string src_path = "/home/gy/Desktop/HIT_study/computer-vision/project/images/big_circle/";
    string src_name = "1.bmp";

    cv::Mat src = cv::imread(src_path + src_name);

    cv::imshow("test", src);
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void test_detect_one_image()
{
    string src_path = "/home/liuxiao/Downloads/";
    string src_name = "temp.svg";

    cv::Mat src = cv::imread("/home/liuxiao/test/st_handeye_eval/cctag_test/005_image.jpg");
    std::cout<<"here";
    cv::Mat graySrc;
    cv::cvtColor(src, graySrc, CV_BGR2GRAY);

    size_t frameId = 0;
    int pipeId = 0;

    boost::ptr_list<ICCTag> markers{};

    const size_t nCrowns = 3;
    Parameters params(nCrowns);
    params.setUseCuda(false);

    start_time = clock();
    cctagDetection(markers, pipeId, frameId, graySrc, params);
    end_time = clock();
    getTime();

    cv::Mat showImg;
    cv::cvtColor(graySrc, showImg, cv::COLOR_GRAY2BGRA);

    ofstream out(pwd + "test_detect_one_image.txt");
    drawMarkers(markers, showImg, out);
    out.close();

    int height = showImg.rows;
    int width = showImg.cols;
    string winname = "test";

    cv::namedWindow(winname, cv::WINDOW_NORMAL);
    cv::resizeWindow(winname, width / 2, height / 2);
    cv::imshow("test", showImg);
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void detectAllImages()
{
    string src_path = "/home/gy/Desktop/HIT_study/computer-vision/project/images/big_circle/";
    int num = 31;
    string img_format = ".bmp";

    size_t frameId = 0;
    int pipeId = 0;
    const size_t nCrowns = 3;
    Parameters params(nCrowns);
    params.setUseCuda(false);

    // save the points for detection time is too long
    ofstream out(pwd + "big_circles.txt");

    for (int i = 0; i < num; i++)
    {
        string img_name = src_path + to_string(i+1) + img_format;
        cv::Mat src = cv::imread(img_name);
        cv::Mat graySrc, showImg;
        cv::cvtColor(src, graySrc, CV_BGR2GRAY);
        src.copyTo(showImg);

        boost::ptr_list<ICCTag> markers{};
        start_time = clock();
        cctagDetection(markers, pipeId, frameId, graySrc, params);
        end_time = clock();
        cout << i+1 << " image ";
        getTime();

        drawMarkers(markers, showImg, out);

        int height = showImg.rows;
        int width = showImg.cols;
        string winname = "test";
        cv::namedWindow(winname, cv::WINDOW_NORMAL);
        cv::resizeWindow(winname, width / 2, height / 2);
        cv::imshow("test", showImg);
        cv::waitKey(1000);
    }
    
    out.close();
}


int main(int, char**) {

    // test_read_bmp();
    test_detect_one_image();
    // detectAllImages();

    return 0;    

}
