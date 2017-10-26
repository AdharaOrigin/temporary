//
// Created by Head on 10. 5. 2017.
//

#ifndef EYE_CONTROL_EYE_ROI_H
#define EYE_CONTROL_EYE_ROI_H

#include <iostream>
#include <vector>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv/cv.hpp>

#include "utils.h"

using namespace std;


class ROI {
public:
    ROI(const cv::Mat& frame, cv::Rect reg_of_interest, const cv::Mat& eye_mask) {
        cv::Mat tmp(frame, reg_of_interest);
        img_int = tmp.clone();
        mask = eye_mask.clone();
        iris_r_max = (int)(reg_of_interest.width * 0.625 * 4/16);
        iris_r_min = (int)(reg_of_interest.width * 0.625 * 1/8);
        preprocess_image();
    }

    void find_rough_eye_centre();
    void resize(double ratio, int to_height = -1);

    int         iris_r_max, iris_r_min;
    cv::Point   rough_centre;
    cv::Mat     img_int;
    cv::Mat     img_flp;
    cv::Mat     der_x, der_y;
    cv::Mat     inverse_flp;
    cv::Mat     mask;

private:
    void preprocess_image();
    void init_ring_masks(cv::Mat& ring_dx, cv::Mat& ring_dy);
};

#endif //EYE_CONTROL_EYE_ROI_H
