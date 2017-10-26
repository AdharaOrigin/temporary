//
// Created by Head on 29. 4. 2017.
//
#include <iostream>
#include <vector>
#include <ctime>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

#ifndef EYE_CONTROL_PROFILER_H
#define EYE_CONTROL_PROFILER_H

class Profiler {
public:
    void mark(string msg);
    void stop();
    void print_results(Mat& img, double time_window = 0.75);

private:
    int     it = 0, counter = 0, n;
    bool    init_phase = true;
    bool    stopped = false;
    string  fps = "", parts_titles = "", parts = "";
    clock_t tmp = 0, tick = 0;
    vector<pair<double, string>> timestamps;
};

#endif //EYE_CONTROL_PROFILER_H
