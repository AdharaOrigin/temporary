//
// Created by Head on 10. 5. 2017.
//

#include "landmark_detector.h"

void LandmarkDetector::filter_face_landmarks()
{
    if(!mean_face_initiated) {
        mean_face = actual_face;
        mean_face_initiated = true;
    } else {
        int dx, dy;
        for(unsigned int i = 0; i < 68; i++) {
            dx = actual_face.part(i).x() - mean_face.part(i).x();
            dy = actual_face.part(i).y() - mean_face.part(i).y();
            if(abs(dx) > 5 || abs(dy) > 5)
                mean_face.part(i) = actual_face.part(i);
            else if(abs(dx) > 2)
                mean_face.part(i).x() += dx/2;
            else if(abs(dy) > 2)
                mean_face.part(i).y() += dy/2;
        }
    }
}

void LandmarkDetector::get_face_landmarks(const cv::Mat& frame, const cv::Rect& face, vector<cv::Point>& result)
{
    dlib::cv_image<unsigned char> img(frame);
    dlib::rectangle face_region = dlib::rectangle(face.x, face.y, face.x+face.width, face.y+face.height);

    actual_face = pose_model(img, face_region);
    filter_face_landmarks();

    result.clear();
    for(int j = 0; j < 68; j++) {
        result.push_back( cv::Point(mean_face.part(j).x(), mean_face.part(j).y()) );
    }
}

bool LandmarkDetector::get_mean_face_init()
{
    return mean_face_initiated;
}

void LandmarkDetector::set_mean_face_init(bool initiated)
{
    mean_face_initiated = initiated;
}
