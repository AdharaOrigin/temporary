//
// Created by Head on 29. 4. 2017.
//
#include "profiler.h"

void Profiler::mark(std::string msg) {
    if(tmp != 0)
        timestamps[it++].first += (clock()-tmp)/(double)CLOCKS_PER_SEC;
    if(init_phase)
        timestamps.push_back(make_pair(0,msg));

    tmp = clock();
    stopped = false;
}

void Profiler::stop() {
    if(init_phase) {
        n = (int)timestamps.size();
        for(int i = 0; i < n; i++)
            parts_titles += timestamps[i].second + "  ||  ";
    }
    if(!stopped)
        timestamps[it].first += (clock()-tmp)/(double)CLOCKS_PER_SEC;

    it = 0;
    tmp = 0;
    stopped = true;
    init_phase = false;
}

void Profiler::print_results(Mat& img, double time_window = 0.75) {
    stop();
    counter++;

    double total_time = 0;
    for(int i = 0; i < n; i++)
        total_time += timestamps[i].first;

    if((clock()-tick)/(double)CLOCKS_PER_SEC > time_window) {
        fps = "FPS: " + to_string(((double)counter/total_time));
        parts = "  ";
        for(int i = 0; i < n; i++) {
            parts += to_string(timestamps[i].first/total_time) + "   ||   ";
            timestamps[i].first = 0;
        }
        counter = 0;
        tick = clock();
    }

    img = Mat(cv::Size(600, 80), CV_64F, cvScalar(1.));
    putText(img, fps, cvPoint(10,15), FONT_HERSHEY_COMPLEX_SMALL, 0.6, cvScalar(0,0,0));
    putText(img, parts_titles, cvPoint(10,30), FONT_HERSHEY_COMPLEX_SMALL, 0.6, cvScalar(0,0,0));
    putText(img, parts, cvPoint(10,45), FONT_HERSHEY_COMPLEX_SMALL, 0.6, cvScalar(0,0,0));
}
