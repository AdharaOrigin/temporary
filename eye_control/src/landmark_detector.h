//
// Created by Head on 10. 5. 2017.
//

#ifndef EYE_CONTROL_LANDMARKDETECTOR_H
#define EYE_CONTROL_LANDMARKDETECTOR_H

#include <opencv2/imgproc.hpp>

#include <dlib/opencv/cv_image.h>
#include <dlib/image_processing.h>
#include <opencv/cv.hpp>

using namespace std;

class LandmarkDetector {

public:
    LandmarkDetector() {
        dlib::deserialize("shape_predictor_68_face_landmarks.dat") >> pose_model;
        mean_face_initiated = false;
    }

    void    get_face_landmarks(const cv::Mat& frame, const cv::Rect& face, vector<cv::Point>& result);
    bool    get_mean_face_init();
    void    set_mean_face_init(bool initiated);

private:
    bool                        mean_face_initiated;
    dlib::shape_predictor       pose_model;
    dlib::full_object_detection mean_face, actual_face;

    void filter_face_landmarks();
};

#endif //EYE_CONTROL_LANDMARKDETECTOR_H
