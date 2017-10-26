//
// Created by Head on 10. 5. 2017.
//

#include "eye_roi.h"


void ROI::preprocess_image()
{
    fastNlMeansDenoising(img_int, img_int, 3, 5, 11);

    cv::Mat img_eye_hist;
    equalizeHist(img_int, img_eye_hist);
    addWeighted(img_int, 0.35, img_eye_hist, 0.65, 0.0, img_int);

    img_int.convertTo(img_flp, CV_64FC1, 1.0 / 255);
    cvtColor(img_int, img_int, CV_GRAY2BGR);

    inverse_flp = cv::Mat::ones(img_flp.size(), CV_64FC1);
    addWeighted(inverse_flp, 1, img_flp, -1, 0, inverse_flp);

    // GaussianBlur(mask, mask, cv::Size(3, 3), 2);
}


void ROI::init_ring_masks(cv::Mat &ring_dx, cv::Mat &ring_dy)
{
    int kernel_size = 2*iris_r_max+1;
    ring_dx = cv::Mat::zeros(kernel_size, kernel_size, CV_64FC1);
    ring_dy = cv::Mat::zeros(kernel_size, kernel_size, CV_64FC1);

    for(int y = -iris_r_max; y <= iris_r_max; y++) {
        for (int x = -iris_r_max; x <= iris_r_max; x++) {
            if (x * x + y * y < iris_r_min * iris_r_min || x * x + y * y > iris_r_max * iris_r_max)
                continue;
            ring_dx.at<double>(y + iris_r_max, x + iris_r_max, 0) = x / (double)(abs(x) + abs(y));
            ring_dy.at<double>(y + iris_r_max, x + iris_r_max, 0) = y / (double)(abs(x) + abs(y));
        }
    }
}

void ROI::find_rough_eye_centre()
{
    // Convolution
    cv::Mat ring_dx, ring_dy;
    init_ring_masks(ring_dx, ring_dy);

    Scharr(img_flp, der_x, -1, 1, 0);
    Scharr(img_flp, der_y, -1, 0, 1);

    cv::Mat acc, accX, accY;
    filter2D(der_x, accX, -1, ring_dx);
    filter2D(der_y, accY, -1, ring_dy);

    // show_normalized_img("mask", mask, -1, false);

    acc = accX + accY;
    acc = acc.mul(inverse_flp.mul(mask));
    // acc = mask_image(acc, inverse_flp.mul(mask));

    minMaxLoc(acc, NULL, NULL, NULL, &rough_centre);
}


void ROI::resize(double ratio, int to_height)
{
    if(to_height > 0)
        ratio = (double)to_height / (double)img_flp.rows;

    iris_r_max *= ratio;
    iris_r_min *= ratio;
    rough_centre = cv::Point(rough_centre.x*ratio, rough_centre.y*ratio);
    cv::resize(img_int, img_int, cv::Size(0,0), ratio, ratio);
    cv::resize(img_flp, img_flp, cv::Size(0,0), ratio, ratio);
    cv::resize(der_x, der_x, cv::Size(0,0), ratio, ratio);
    cv::resize(der_y, der_y, cv::Size(0,0), ratio, ratio);
    cv::resize(inverse_flp, inverse_flp, cv::Size(0,0), ratio, ratio);
    cv::resize(mask, mask, cv::Size(0,0), ratio, ratio);
}