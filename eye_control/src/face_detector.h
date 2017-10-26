//
// Created by Head on 10. 5. 2017.
//

#ifndef EYE_CONTROL_FACE_DETECTOR_H
#define EYE_CONTROL_FACE_DETECTOR_H

#include <iostream>
#include <vector>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv/cv.hpp>

#include "landmark_detector.h"
#include "utils.h"

using namespace std;


class FaceDetector {
public:
    FaceDetector(string detector_source)
    {
        if( !face_cascade.load(detector_source) ){
            cerr << "Error loading face cascade." << endl;
        };
        lead_idx = -1;
    }

    bool        detect_leading_face(const cv::Mat& frame);
    int         choose_leading_face();
    cv::Rect    get_left_eye_region();
    cv::Rect    get_right_eye_region();
    cv::Mat     get_eye_mask(bool left_eye, cv::Rect eye_region);
    cv::Mat     get_left_eye_mask();
    cv::Mat     get_right_eye_mask();
    void        render_faces(cv::Mat& frame, bool render_regions = true,
                             bool render_landmarks = true, bool render_eye_regions = false);

    // Return to private*
    vector<cv::Point>       landmarks;
    cv::Rect                left_eye_region, right_eye_region;

private:
    cv::CascadeClassifier   face_cascade;
    LandmarkDetector        fl_detector;

    int                     lead_idx;
    vector<cv::Rect>        faces;
    // *here                here

    void                    determine_eye_regions();
};

#endif //EYE_CONTROL_FACE_DETECTOR_H
