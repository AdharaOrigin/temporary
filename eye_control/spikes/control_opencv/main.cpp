//
// Created by Head on 20. 4. 2017.
//
#include <iostream>
#include <vector>
#include <ctime>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/imgproc.hpp>

#include <dlib/gui_widgets.h>
#include <dlib/opencv/cv_image.h>
#include <dlib/image_processing.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing/full_object_detection.h>
#include <opencv/cv.hpp>

#include "findEyeCenter.h"

using namespace std;
using namespace cv;

#define CAP_W_L 1280
#define CAP_H_L 720
#define CAP_W_S 640
#define CAP_H_S 480

#define EYES_WIN_WIDTH  350
#define EYES_WIN_HEIGHT 550


class Profiler {
public:
    void mark(string msg) {
        if(tmp != 0)
            timestamps[it++].first += (clock()-tmp)/(double)CLOCKS_PER_SEC;
        if(init_phase)
            timestamps.push_back(make_pair(0,msg));

        tmp = clock();
        stopped = false;
    }

    void stop() {
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

    void print_results(Mat& img, double time_window = 0.75) {
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

private:
    int     it = 0, counter = 0, n;
    bool    init_phase = true;
    bool    stopped = false;
    string  fps = "", parts_titles = "", parts = "";
    clock_t tmp = 0, tick = 0;
    vector<pair<double, string>> timestamps;
};

bool image_flip(Mat& src) {
    Mat tmp;
    flip(src, tmp, 1);
    tmp.copyTo(src);
    return true;
}

bool image_grayscale(Mat& src) {
    Mat tmp;
    cvtColor(src, tmp, COLOR_BGR2GRAY);
    tmp.copyTo(src);
    return true;
}

void render_face_landmarks(dlib::image_window& win, dlib::full_object_detection face) {
    // for (unsigned long i = 0; i < faces_fld.size(); ++i) {}
    //    dlib::full_object_detection& d = faces_fld[i];
    vector<dlib::image_window::overlay_circle> dots;
    for (unsigned long j = 0; j < face.num_parts(); ++j)
        dots.push_back(dlib::image_window::overlay_circle(face.part(j), 2, dlib::rgb_pixel(255,0,0)));

    if(dots.size() > 0)
        win.add_overlay(dots);
}

void filter_face_landmarks(const dlib::full_object_detection src,
                           dlib::full_object_detection& mean_face, bool& initiated) {
    if(!initiated) {
        mean_face = src;
        initiated = true;
    } else {
        int dx, dy;
        for(unsigned int i = 0; i < 68; i++) {
            dx = src.part(i).x() - mean_face.part(i).x();
            dy = src.part(i).y() - mean_face.part(i).y();
            if(abs(dx) > 5 || abs(dy) > 5)
                mean_face.part(i) = src.part(i);
            else if(abs(dx) > 2)
                mean_face.part(i).x() += dx/2;
            else if(abs(dy) > 2)
                mean_face.part(i).y() += dy/2;
        }
    }
}

Rect get_eye_region(const dlib::full_object_detection& face, bool left_eye) {

    int top = INT_MAX, left = INT_MAX, right = 0, bottom = 0, start_point;
    if(left_eye) start_point = 36;
    else start_point = 42;

    for(int i = start_point; i <= start_point+5; i++)
    {
        if(face.part(i).y() < top )
            top = face.part(i).y();
        if(face.part(i).x() < left )
            left = face.part(i).x();
        if(face.part(i).y() > bottom )
            bottom = face.part(i).y();
        if(face.part(i).x() > right )
            right = face.part(i).x();
    }
    int width = (right-left);
    int height = (bottom-top);
    return Rect(left, top, width, height);
    /*
    if(left_eye)
        return Rect((int)(left-width*0.5), top-height, (int)(width*1.75), 3*height);
    else
        return Rect((int)(left-width*0.25),top-height, (int)(width*1.75), 3*height);
    */
}

void resize_eye(Mat& eye) {
    double resize_factor = 70.0/eye.rows;
    resize(eye, eye, cv::Size((int)(eye.cols*resize_factor),(int)(eye.rows*resize_factor)));
}

void show_all_eyes(std::string window_name, const std::vector<Mat> mats) {

    for(unsigned int i = 0; i < mats.size(); i++)
        mats[0].type();

    int pt_y = 50;
    Mat eye_all(EYES_WIN_HEIGHT, EYES_WIN_WIDTH, CV_8UC1);
    eye_all.setTo(Scalar(0xff,0xff,0xff));

    for(unsigned int i = 0; i < mats.size(); i++) {

        int width = mats[i].cols, height = mats[i].rows;
        if(pt_y + height > EYES_WIN_HEIGHT) {
            cout << "Window too small!" << endl;
            break;
        }
        if(width >= EYES_WIN_WIDTH)
            width = EYES_WIN_WIDTH - 5;

        mats[i].copyTo(eye_all(Rect(0, pt_y, width, height)));
        pt_y += height;
    }
    imshow(window_name, eye_all);
}

Mat get_eye_mat(Mat& img, const dlib::full_object_detection& face) {

    Mat eye_mask = Mat::zeros(img.size(), img.type());
    Mat result   = Mat::zeros(img.size(), img.type());
    result.setTo(Scalar(255,255,255));

    std::vector<cv::Point> xtmp;
    std::vector<std::vector<cv::Point>> contours;
    for(int i = 36; i <= 36+5; i++) {
        Point p(face.part(i).x(), face.part(i).y());
        xtmp.push_back(p);
    }
    contours.push_back(xtmp);

    drawContours(eye_mask, contours, -1, Scalar(255, 0, 0), -1);
    img.copyTo(result, eye_mask);

    Rect eye_region = get_eye_region(face, true);
    Mat eye_tmp(result, eye_region);
    Mat eye = eye_tmp.clone();

    return eye;
}

// --- OPENCV VERSION ---
int main() {

    Profiler prof;
    Mat img, tmp, stats;
    dlib::image_window win;

    // Initialize camera
    VideoCapture cam(0);
    if (!cam.isOpened()) {
        cerr << "Unable to connect to camera" << endl;
        return 1;
    }
    cam.set(CAP_PROP_FRAME_WIDTH, CAP_W_L);
    cam.set(CAP_PROP_FRAME_HEIGHT,CAP_H_L);

    // Load FLD detector & dlib support
    dlib::shape_predictor pose_model;
    dlib::deserialize("shape_predictor_68_face_landmarks.dat") >> pose_model;
    dlib::full_object_detection mean_face;
    bool mean_face_initiated = false;


    // Load OpenCV CascadeClassifier
    cv::String face_cascade_name = "../include/haarcascade_frontalface_alt.xml";
    cv::CascadeClassifier face_cascade;
    vector<cv::Rect> faces;
    if( !face_cascade.load( face_cascade_name ) ){
        cout << "--(!)Error loading face cascade." << endl;
        return -1;
    };

    // Windows

    cvNamedWindow("Eyes", 0);
    resizeWindow("Eyes", EYES_WIN_WIDTH, EYES_WIN_HEIGHT);
    cvNamedWindow("Canny sliders", 0);
    resizeWindow("Canny sliders", 600, 400);

    // Trackers
    int hist_orig = 0, hist_modif = 150, darker = 30;
    createTrackbar("Hist_org", "Canny sliders", &hist_orig, 300);
    createTrackbar("Hist_mod", "Canny sliders", &hist_modif, 300);
    createTrackbar("Darker", "Canny sliders", &darker, 100);

    // GaussianBlur
    int strength = 0;
    createTrackbar("Gauss", "Canny sliders", &strength, 5);

    // Canny
    int threshold1 = 100, threshold2 = 300, apertureSize = 2;
    createTrackbar("Thresh1", "Canny sliders", &threshold1, 500);
    createTrackbar("Thresh2", "Canny sliders", &threshold2, 500);
    createTrackbar("ApertSize", "Canny sliders", &apertureSize, 3);

    int hough_min = 30, hough_max = 150;
    createTrackbar("HoughMin", "Canny sliders", &hough_min, 50);
    createTrackbar("HoughMax", "Canny sliders", &hough_max, 150);


    // Process webcam stream
    for(;;) {
        // prof.mark("GET FRAME");
        cam >> img;
        imshow("this", img);

        prof.mark("CONVERT ");
        image_flip(img);
        vector<Mat> rgbChannels(3);
        split(img, rgbChannels);
        Mat frame_gray = rgbChannels[2];


        prof.mark("DETECT  ");
        face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE|CV_HAAR_FIND_BIGGEST_OBJECT, cv::Size(150, 150) );
        dlib::cv_image<unsigned char> img_dlib(frame_gray);
        vector<dlib::full_object_detection> faces_fld;
        for(unsigned int i = 0; i < faces.size(); i++) {
            dlib::rectangle r = dlib::rectangle(faces[i].x, faces[i].y, faces[i].x+faces[i].width, faces[i].y+faces[i].height);
            faces_fld.push_back(pose_model(img_dlib, r));
        }

        imshow("this", frame_gray);
        win.clear_overlay();
        win.set_image(img_dlib);

        if(faces.size() == 0) {
            mean_face_initiated = false;
        }
        else {
            filter_face_landmarks(faces_fld[0], mean_face, mean_face_initiated);
            render_face_landmarks(win, mean_face);

            prof.mark("EYE OPT ");
            Mat eye = get_eye_mat(frame_gray, mean_face);
            Mat eye_hist, eye_hist_sum, eye_blured, eye_canny, result;
            resize_eye(eye);
            imshow("Bitch1", eye);

        }

        // Histogram
        equalizeHist(eye, eye_hist);
        addWeighted(eye, (double)hist_orig/100, eye_hist, (double)hist_modif/100, 0, eye_hist_sum);
        Mat white = Mat::zeros(eye_hist_sum.size(), eye_hist_sum.type());
        white.setTo(Scalar(255,255,255));
        addWeighted(white, 1, eye_hist_sum, -1, 0, white);
        addWeighted(eye_hist_sum, 1, white, (double)darker/-100, 0, eye_hist_sum);

        // Denoise
        GaussianBlur(eye_hist_sum, eye_blured, cv::Size(0,0), 2*strength+1);
        addWeighted(eye_hist_sum, 2, eye_blured, -1, 0, eye_blured);
        // fastNlMeansDenoising(eye_blured, eye_blured, 5, 7, 21);


/*
        if(apertureSize == 0) {
            apertureSize = 1;
            setTrackbarPos("ApertSize", "Canny sliders", 1);
        }
        Canny(eye_blured, eye_canny, threshold1, threshold2, 2*apertureSize+1);
        //cvtColor(left_eye2, canny, CV_GRAY2RGB);
*/

        //imshow("Bitch2", eye_hist_sum);
        //imshow("Bitch3", eye_blured);
        //imshow("Bitch4", eye_canny);
/*
        // Hough Circles
        vector<Vec3f> circles;
        cvtColor(eye_hist_sum, result, CV_GRAY2RGB);
        HoughCircles(eye_canny, circles, CV_HOUGH_GRADIENT, 2, 20, 100, 100, hough_min, hough_max);
        for( size_t i = 0; i < circles.size(); i++ )
        {
            Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
            int radius = cvRound(circles[i][2]);
            // draw the circle center
            circle( result, center, 3, Scalar(255,0,0), -1, 8, 0 );
            // draw the circle outline
            circle( result, center, radius, Scalar(0,0,255), 2, 8, 0 );
        }
        imshow("HoughCircles", result);
*/

        /*
            prof.mark("EYE CNTR");
            Point left_eye_pupil = findEyeCenter(left_eye);
            circle(left_eye3, left_eye_pupil, 3, cvScalar(0xff,0xff,0xff), 2);
            cout << "Pupil: " << left_eye_pupil.x << ":" << left_eye_pupil.y << endl;
        */
        /*
        prof.mark("DISPLAY ");
        // Display OpenCV screen
        for(unsigned int i = 0; i < faces.size(); i++)
            rectangle(frame_gray, faces[i], cvScalar(0.));
        imshow("Origin", frame_gray);
        */

        // prof.print_results(stats);
        // imshow("Statistics", stats);
        int c = waitKey(30);
        if(c != 255) break;
    }

    return 0;
}