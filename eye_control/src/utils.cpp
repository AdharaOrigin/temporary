//
// Created by Head on 9. 5. 2017.
//

#include "utils.h"

void show_normalized_img(string window_name, const cv::Mat& img, double resize_height, bool normalize_ranges)
{
    double min = 0, max = 1;
    cv::Mat tmp = img.clone();
    if(tmp.rows < 1)
        return;

    if(resize_height != -1) {
        double resize_ratio = resize_height/tmp.rows;
        resize(tmp, tmp, cv::Size(0,0), resize_ratio, resize_ratio);
    }
    if(normalize_ranges) {
        minMaxLoc(tmp, &min, &max);
        if (min < 0) {
            tmp = tmp - min;
            max -= min;
            min = 0;
        }
    }
    if(tmp.type() == CV_64FC1)
        tmp.convertTo(tmp, CV_8UC1, 255/(max-min));
    imshow(window_name, tmp);
}

bool resize_rect(const cv::Rect& r, cv::Rect& result, double dx, double dy, const cv::Mat* constrain)
{
    int width = dx*r.width, height = dy*r.height;
    int x = r.x - ((width-r.width)/2);
    int y = r.y - ((height-r.height)/2);

    if(x < 0) {
        width += x;
        x = 0;
    }
    if(y < 0) {
        height += y;
        y = 0;
    }
    if(constrain != nullptr) {
        if(x >= constrain->cols-1 || y >= constrain->rows-1 || width < 1 || height < 1) {
            return false;
        }
        if(x+width >= constrain->cols)
            width -= x+width - constrain->cols + 1;
        if(y+height >= constrain->rows)
            height -= y+height - constrain->rows + 1;
    }

    result = cv::Rect(x, y, width, height);
    return true;
}

cv::Point scale_pt(cv::Point2d p, double resize_ratio)
{
    return cv::Point(p.x*resize_ratio, p.y*resize_ratio);
}

cv::RotatedRect scale_rrect(const cv::RotatedRect r, float resize_ratio)
{
    float x = r.center.x * resize_ratio;
    float y = r.center.y * resize_ratio;
    cv::Size scaled_size(r.size.width*resize_ratio, r.size.height*resize_ratio);
    return cv::RotatedRect(cv::Point2f(x,y), scaled_size, r.angle);
}

cv::Mat mask_image(const cv::Mat& img, const cv::Mat& mask) {
    double max_val;
    cv::Mat tmp = img.clone();
    tmp = tmp.mul(mask);

    minMaxLoc(tmp, NULL, &max_val);
    if(max_val > 1) {
        tmp = tmp * (1/max_val);
    }
    return tmp;
}
