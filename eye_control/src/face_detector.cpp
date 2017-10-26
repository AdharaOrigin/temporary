//
// Created by Head on 10. 5. 2017.
//

#include "face_detector.h"

bool FaceDetector::detect_leading_face(const cv::Mat& frame)
{
    // Detect all faces in frame
    face_cascade.detectMultiScale(frame, faces, 1.1, 2,
                                  0 | CV_HAAR_SCALE_IMAGE | CV_HAAR_FIND_BIGGEST_OBJECT, cv::Size(150, 150));

    lead_idx = choose_leading_face();
    if (lead_idx == -1) {
        fl_detector.set_mean_face_init(false);
        return false;
    }
    fl_detector.get_face_landmarks(frame, faces[lead_idx], landmarks);

    determine_eye_regions();
    resize_rect(left_eye_region, left_eye_region, 1.6, 2, &frame);
    resize_rect(right_eye_region, right_eye_region, 1.6, 2, &frame);
    return true;
}

int FaceDetector::choose_leading_face() {
    // TODO: Implement algorithm deciding which face has control.
    if(faces.size() > 0)
        return 0;
    else
        return -1;
}

void FaceDetector::determine_eye_regions() {
    vector<cv::Point> tmp;
    for(int i = 36; i <= 36+5; i++)
        tmp.push_back(landmarks[i]);
    left_eye_region = boundingRect(tmp);

    tmp.clear();
    for(int i = 42; i <= 42+5; i++)
        tmp.push_back(landmarks[i]);
    right_eye_region = boundingRect(tmp);
}


cv::Rect FaceDetector::get_left_eye_region() {
    return left_eye_region;
}
cv::Rect FaceDetector::get_right_eye_region() {
    return right_eye_region;
}


cv::Mat FaceDetector::get_eye_mask(bool left_eye, cv::Rect eye_region)
{
    int start_point;
    cv::Mat eye_mask = cv::Mat::zeros(eye_region.size(), CV_64FC1);
    if(left_eye) { start_point = 36; }
    else { start_point = 42; }

    std::vector<vector<cv::Point>> tmp;
    tmp.push_back(vector<cv::Point>());
    for(int i = start_point; i <= start_point+5; i++) {
        tmp[0].push_back(cv::Point(landmarks[i].x-eye_region.x, landmarks[i].y-eye_region.y));
    }

    drawContours(eye_mask, tmp, -1, cv::Scalar(1.0), CV_FILLED);
    return eye_mask;
}

cv::Mat FaceDetector::get_left_eye_mask() {
    return get_eye_mask(true, left_eye_region);
}

cv::Mat FaceDetector::get_right_eye_mask() {
    return get_eye_mask(false, right_eye_region);
}


void FaceDetector::render_faces(cv::Mat& frame, bool render_regions, bool render_landmarks, bool render_eye_regions) {
    if(render_regions) {
        for (unsigned int i = 0; i < faces.size(); i++)
            rectangle(frame, faces[i], cv::Scalar(0, 0, 0));
    }

    if(render_landmarks) {
        if(lead_idx != -1)
            for (unsigned int j = 0; j < 68; j++)
                circle(frame, landmarks[j], 2, cv::Scalar(0, 0, 0), -1);
    }

    if(render_eye_regions) {
        if(lead_idx != -1) {
            rectangle(frame, left_eye_region, cv::Scalar(50, 50, 50));
            rectangle(frame, right_eye_region,cv::Scalar(50, 50, 50));
        }
    }
}
