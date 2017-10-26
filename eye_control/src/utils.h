//
// Created by Head on 9. 5. 2017.
//

#ifndef EYE_CONTROL_UTILS_H
#define EYE_CONTROL_UTILS_H

#include <iostream>
#include <vector>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
// using namespace cv;

#define CAP_W_L 1280
#define CAP_H_L 720
#define CAP_W_S 640
#define CAP_H_S 480
#define N_RAYS 40
#define PI 3.14
#define WIN_DBG_H 80

void show_normalized_img(string window_name, const cv::Mat& img, double resize_height = -1, bool normalize_ranges = true);

bool resize_rect(const cv::Rect& r, cv::Rect& result, double dx, double dy, const cv::Mat* constrain = nullptr);

cv::Point scale_pt(cv::Point2d p, const double resize_ratio);
cv::RotatedRect scale_rrect(const cv::RotatedRect r, const float resize_ratio);

cv::Mat mask_image(const cv::Mat& img, const cv::Mat& mask);

#endif //EYE_CONTROL_UTILS_H
