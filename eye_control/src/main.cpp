//
// Created by Head on 29. 4. 2017.
//
#include <iostream>
#include <vector>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv/cv.hpp>
#include <unistd.h>

#include "face_detector.h"
#include "eye_roi.h"
#include "utils.h"

using namespace std;

#define CAP_W_L 1280
#define CAP_H_L 720
#define CAP_W_S 640
#define CAP_H_S 480
#define N_RAYS 40
#define PI 3.14
#define RANSAC_ITERS 40


bool fit_circle_least_squares(const vector<cv::Point>& points, cv::Point2d& centre, double& radius)
{
    int n = points.size();
    double x_avg = 0, y_avg = 0, kc, lc;
    vector<cv::Point2d> pts;

    for(int i = 0; i < n; i++) {
        x_avg += points[i].x;
        y_avg += points[i].y;
    }
    x_avg /= n;
    y_avg /= n;

    for(int i = 0; i < n; i++) {
        pts.push_back(cv::Point2d(points[i].x-x_avg, points[i].y-y_avg));
    }

    double Sk3 = 0, Sl3 = 0, Sk2l = 0, Skl2 = 0, Skl_sq = 0, Sk2 = 0, Sl2 = 0, Skl = 0;
    for(int i = 0; i < n; i++) {
        Sk3     += (pts[i].x * pts[i].x * pts[i].x);
        Sl3     += (pts[i].y * pts[i].y * pts[i].y);
        Sk2l    += (pts[i].x * pts[i].y * pts[i].y);
        Skl2    += (pts[i].x * pts[i].x * pts[i].y);
        Skl_sq  += (pts[i].x * pts[i].x * pts[i].y * pts[i].y);
        Sk2     += (pts[i].x * pts[i].x);
        Sl2     += (pts[i].y * pts[i].y);
        Skl     += (pts[i].x * pts[i].y);
    }

    if(Skl == 0)
        return false;
    double alfa = (Sk3 + Skl2) / (2*Skl);
    double beta = (Sk2*(Sl3+Sk2l)) / (2*(Skl*Skl));
    double gamm = 1 - (Sl2*Sk2) / (Skl*Skl);

    if(gamm == 0)
        return false;
    lc = (alfa-beta) / gamm;

    if(2*Skl-lc*(Sl2 / Skl) == 0)
        return false;
    kc = (Sl3+Sk2l) / (2*Skl) - lc*(Sl2 / Skl);

    centre = cv::Point2d(kc + x_avg, lc + y_avg);
    radius = sqrt( kc*kc + lc*lc + (Sk2+Sl2)/n );
    return true;
}


struct gaze {
    gaze() { iris_radius = -1; };
    gaze(cv::Point2d p, double r) : pupil_centre(p), iris_radius(r) {};
    cv::Point2d pupil_centre;
    double iris_radius;
};

double dist(const cv::Point c, const cv::Point a)
{
    return sqrt((c.x-a.x)*(c.x-a.x) + (c.y-a.y)*(c.y-a.y));
}

void get_circles(cv::Point& c1, cv::Point& c2, int r, const cv::Point a, const cv::Point b)
{
    double q = sqrt( (b.x-a.x)*(b.x-a.x) + (b.y-a.y)*(b.y-a.y) );
    double x3 = (a.x+b.x) / 2;
    double y3 = (a.y+b.y) / 2;

    c1.x = (int)(x3 + sqrt(r*r - (q/2)*(q/2)) * (a.y-b.y)/q);
    c1.y = (int)(y3 + sqrt(r*r - (q/2)*(q/2)) * (b.x-a.x)/q);

    c2.x = (int)(x3 - sqrt(r*r - (q/2)*(q/2)) * (a.y-b.y)/q);
    c2.y = (int)(y3 - sqrt(r*r - (q/2)*(q/2)) * (b.x-a.x)/q);
}


gaze ransac(const vector<cv::Point>& points, ROI& eye)
{
    cv::Mat debug = eye.img_int.clone();
    vector<cv::Point> inliers;
    int r_max = (int)(1.25 * eye.iris_r_max);
    int r_min = (int)(0.75 * eye.iris_r_max);
    int n_in, n_in_best, n_in_best_all_r = 0, eps = 3;
    double radius_best = eye.iris_r_max;
    cv::Point2d centre_best, centre_best_all_r = eye.rough_centre;

    for (int radius = r_max-eps; radius > r_min; radius -= 3)
    {
        n_in_best = 0;
        for (int iter = 0; iter < RANSAC_ITERS; iter++)
        {
            n_in = 0;
            unsigned int i1 = 0, i2 = 0;
            while (i1 == i2) {
                i1 = (unsigned int) (rand() % points.size());
                i2 = (unsigned int) (rand() % points.size());
            }

            cv::Point centre1, centre2;
            get_circles(centre1, centre2, radius, points[i1], points[i2]);

            if (dist(eye.rough_centre, centre1) > dist(eye.rough_centre, centre2))
                centre1 = centre2;

            for (unsigned int i = 0; i < points.size(); i++) {
                if (dist(centre1, points[i]) > radius - eps && dist(centre1, points[i]) < radius + eps)
                    n_in++;
            }
            if (n_in > n_in_best) {
                n_in_best = n_in;
                centre_best = centre1;
            }
        }

        if (n_in_best > n_in_best_all_r) {
            radius_best = radius;
            n_in_best_all_r = n_in_best;
            centre_best_all_r = centre_best;
        }
    }

    // DEBUG

    // circle(debug, centre_best_all_r, radius_best+eps, cv::Scalar(0,255,0), 1);
    // circle(debug, centre_best_all_r, radius_best-eps, cv::Scalar(0,255,0), 1);

    for(unsigned int i = 0; i < points.size(); i++) {
        if (dist(centre_best_all_r, points[i]) > radius_best-eps && dist(centre_best_all_r, points[i]) < radius_best+eps) {
            circle(debug, points[i], 2, cv::Scalar(0,0,255), -1);
            inliers.push_back(points[i]);
        }
        else
            circle(debug, points[i], 2, cv::Scalar(255,0,0), -1);
    }

    // fit_circle_least_squares(inliers, centre_best_all_r, radius_best);
    //std::vector<std::vector<cv::Point>> tmp;
    //tmp.push_back(inliers);
    // cv::RotatedRect box;
    // if(inliers.size() >= 5) {
    //     box = cv::fitEllipse({inliers});
    //     cv::ellipse(debug, box, cv::Scalar(0,0,255));
    // }
    // circle(debug, centre_best_all_r, radius_best, cv::Scalar(0,0,255), 1);

    imshow("RANSAC", debug);

    // return gaze(box.center, radius_best);
    return gaze(centre_best_all_r, radius_best);
}

gaze rays_run(ROI eye)
{
    cv::Mat img_dbg;
    double ratio = 200.0 / (double)eye.img_int.rows;
    eye.resize(0, 200);
    eye.img_flp.convertTo(img_dbg, CV_8UC3, 255);
    cvtColor(img_dbg, img_dbg, CV_GRAY2BGR);

    cv::Point2d p;
    vector<cv::Point> candidates;

    for(int i = 1; i < 5; i++) {
        eye.der_x.row(eye.der_x.rows-i).setTo(0.0);
        eye.der_y.row(eye.der_y.rows-i).setTo(0.0);
    }

    for(int r = 0; r < N_RAYS; r++)
    {
        if((r >= 7 && r <= 13) || (r >= 27 && r <= 33))
            continue;

        bool found = false;
        cv::Point2d ray_max;
        double derv_max = 0.8;
        double angle = r*(2*PI/N_RAYS);
        double x_step = 2*cos(angle), y_step = 2*sin(angle);

        p.x = eye.rough_centre.x + cos(angle)*(0.3*eye.iris_r_min);
        p.y = eye.rough_centre.y - sin(angle)*(0.3*eye.iris_r_min);


        int range = 2*24;
        for (int i = 0; i < range; i++)
        {
            double derv = 0, distance_coef = 1;
            p.x += x_step;
            p.y -= y_step;
            if(p.x < 0 || p.x >= eye.img_flp.cols || p.y < 0 || p.y >= eye.img_flp.rows-5 ) {
                break;
            }

            if((p.x-eye.rough_centre.x)*(p.x-eye.rough_centre.x) +
               (p.y-eye.rough_centre.y)*(p.y-eye.rough_centre.y) < (eye.iris_r_min)*(eye.iris_r_min)) {
                distance_coef = 0.5;
            }

            if(r/(N_RAYS/4) == 0)
                derv =  3*eye.der_x.at<double>(p) - eye.der_y.at<double>(p);
            else if(r/(N_RAYS/4) == 1)
                derv = -3*eye.der_x.at<double>(p) - eye.der_y.at<double>(p);
            else if(r/(N_RAYS/4) == 2)
                derv = -3*eye.der_x.at<double>(p) + eye.der_y.at<double>(p);
            else if(r/(N_RAYS/4) == 3)
                derv =  3*eye.der_x.at<double>(p) + eye.der_y.at<double>(p);

            derv = distance_coef * derv;
            if(derv > derv_max) {
                ray_max.x = p.x;
                ray_max.y = p.y;
                derv_max = derv;
                found = true;
            }
        }
        if(found) {
            candidates.push_back(ray_max);
            circle(img_dbg, ray_max, 1, cv::Scalar(0,0,255), -1);
        }
    }
    // imshow("Rays debug", img_dbg);

    if(candidates.size() >= 3) {
        gaze g = ransac(candidates, eye);
        return gaze(cv::Point2d(g.pupil_centre.x * (1/ratio),
                                g.pupil_centre.y * (1/ratio)), g.iris_radius * (1/ratio));
    }
    else
        return gaze();
}


int main() {

    cv::Mat frame, frame_clean;

    // Initialize camera
    cv::VideoCapture cam(0);
    if (!cam.isOpened()) {
        cerr << "Unable to connect to camera" << endl;
        return 1;
    }
    cam.set(cv::CAP_PROP_FRAME_WIDTH, CAP_W_L);
    cam.set(cv::CAP_PROP_FRAME_HEIGHT,CAP_H_L);

    bool recording = false;
    int point = 0, folder = 1;
    int state = 0, it = 0, imgit = 0;
    int person_t = 1;
    double lperc = 0.5, rperc = 0.5, tperc = 0.5, bperc = 0.5, pup_perc_x = 0.5, pup_perc_y = 0.5;
    std::vector<double> perc;
    perc.push_back(0);
    perc.push_back(0);
    perc.push_back(0);
    srand(time(0));

    FaceDetector face_detector("../include/haarcascade_frontalface_alt.xml");

    for(;;) {
        time_t start = clock();
        cam >> frame;
        flip(frame, frame, 1);

        /*
        string num;
        if(imgit < 10)
            num = "000" + to_string(imgit);
        else if(imgit < 100)
            num = "00" + to_string(imgit);
        else if(imgit < 1000)
            num = "0" + to_string(imgit);
        else
            to_string(imgit);
        */

        // frame = cv::imread("../tests/person" + person_t + "/img_" + point + "_" + imgit + ".jpg", CV_LOAD_IMAGE_COLOR);
        cvtColor(frame, frame, CV_BGR2GRAY);

        gaze lp, rp;
        if ( face_detector.detect_leading_face(frame) ) {

            cv::Rect eye_reg_l = face_detector.get_left_eye_region();
            ROI eye_l(frame, eye_reg_l, face_detector.get_left_eye_mask());
            eye_l.find_rough_eye_centre();

            cv::Rect eye_reg_r = face_detector.get_right_eye_region();
            ROI eye_r(frame, eye_reg_r, face_detector.get_right_eye_mask());
            eye_r.find_rough_eye_centre();

            // Rays run
            lp = rays_run(eye_l);
            lp.pupil_centre.x += eye_reg_l.x;
            lp.pupil_centre.y += eye_reg_l.y;

            rp = rays_run(eye_r);
            rp.pupil_centre.x += eye_reg_r.x;
            rp.pupil_centre.y += eye_reg_r.y;

        }
        time_t end = clock();
        // cout << (end-start) / (double)CLOCKS_PER_SEC << endl;

        // DEBUG ---------------------------------------------------
        cvtColor(frame, frame, CV_GRAY2BGR);
        face_detector.render_faces(frame, true, true);
        if(lp.iris_radius != -1 && lp.iris_radius > 0 && rp.iris_radius > 0) {
            cv::circle(frame, lp.pupil_centre, 1, cv::Scalar(0,0,255), -1);
            cv::circle(frame, lp.pupil_centre, lp.iris_radius, cv::Scalar(0,255,255), 1);

            cv::circle(frame, rp.pupil_centre, 1, cv::Scalar(0,0,255), -1);
            cv::circle(frame, rp.pupil_centre, rp.iris_radius, cv::Scalar(0,255,255), 1);
        }

        // DEBUG CONTROL --------------------------------------------------------------------

        /*
        if(lp.iris_radius != -1)
        {
            cv::Point icorner = face_detector.landmarks[39];
            cv::Point ocorner = face_detector.landmarks[36];
            cv::Point2d pupil = lp.pupil_centre;

            int fh = frame.rows, fw = frame.cols, corners_w = icorner.x - ocorner.x, corners_h = ocorner.y - icorner.y + 30;

            if(state > -1) {
                cv::circle(frame, cv::Point(5, 2 * fh / 5), 4, cv::Scalar(0, 255, 255), -1);
                cv::circle(frame, cv::Point(fw - 6, 2 * fh / 5), 4, cv::Scalar(0, 255, 255), -1);
                // cv::circle(frame, cv::Point(fw/2, 5), 4, cv::Scalar(0, 255, 255), -1);
                // cv::circle(frame, cv::Point(fw/2, fh - 6), 4, cv::Scalar(0, 255, 255), -1);
                icorner.y += 15;
                ocorner.y -= 15;
                cv::rectangle(frame, ocorner, icorner, cv::Scalar(0, 0, 0));
            }
            if(state == 0) {
                lperc = (pupil.x - ocorner.x) / (double)corners_w;
                cv::circle(frame, cv::Point(5, 2 * fh / 5), 4, cv::Scalar(0, 0, 255), -1);
            }
            if(state == 1) {
                rperc = (pupil.x - ocorner.x) / (double)corners_w;
                cv::circle(frame, cv::Point(fw - 6, 2 * fh / 5), 4, cv::Scalar(0, 0, 255), -1);
            }
            / *
            if(state == 3) {
                cout << "State 5" << endl;
                tperc = (pupil.y - ocorner.y) / (double)corners_h;
            }
            if(state == 4) {
                cout << "State 7" << endl;
                bperc = (pupil.y - ocorner.y) / (double)corners_h;
            }
            * /
            if(state == 2) {
                cout  << pupil.x << endl;
                perc[it++] = (pupil.x - ocorner.x) / (double)corners_w;
                it = it % 2;

                pup_perc_x = (perc[0] + perc[1] + perc[2]) / 2;
                double gpcx = (pup_perc_x - lperc) / (rperc - lperc);

                cv::circle(frame, cv::Point(fw*gpcx, 2 * fh / 5), 4, cv::Scalar(0, 0, 255), -1);
                // cout << pup_perc_x << " >> " << gpcx << endl;

                // double gpcy = (pup_perc_y - tperc) / (bperc - tperc);
                // pup_perc_y = (pupil.y - ocorner.y) / (double)corners_h;
                // cout << pup_perc_y << " >> " << gpcy << endl;
                //cv::circle(frame, cv::Point(fw/2, fh*gpcy), 4, cv::Scalar(255, 255, 0), -1);
            }
        }
        */

        imshow("Main", frame);

        int pressed_key = cv::waitKey(30);
        if (pressed_key == 'r') {
            recording = true;
        }
        else if (pressed_key == 'q') {
            break;
        }
        else if(pressed_key == 'c') {
            state++;
            state = state % 3;
            cout << "State: " << state << endl;
        }
        else if(pressed_key == 'n') {
            continue;
        }
    }

    return 0;
}


/*
            if(state > 3) {
                cv::line(frame, cv::Point(ocorner.x + (int)(rperc*corners_w), icorner.y),
                         cv::Point(ocorner.x + (int)(rperc*corners_w), ocorner.y), cv::Scalar(0,255,0));
                state++;
            }
 */