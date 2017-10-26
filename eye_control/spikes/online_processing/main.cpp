// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*

    This example program shows how to find frontal human faces in an image and
    estimate their pose.  The pose takes the form of 68 landmarks.  These are
    points on the face such as the corners of the mouth, along the eyebrows, on
    the eyes, and so forth.


    This example is essentially just a version of the face_landmark_detection_ex.cpp
    example modified to use OpenCV's VideoCapture object to read from a camera instead
    of files.


    Finally, note that the face detector is fastest when compiled with at least
    SSE2 instructions enabled.  So if you are using a PC with an Intel or AMD
    chip then you should enable at least SSE2 instructions.  If you are using
    cmake to compile this program you can enable them by using one of the
    following commands when you create the build project:
        cmake path_to_dlib_root/examples -DUSE_SSE2_INSTRUCTIONS=ON
        cmake path_to_dlib_root/examples -DUSE_SSE4_INSTRUCTIONS=ON
        cmake path_to_dlib_root/examples -DUSE_AVX_INSTRUCTIONS=ON
    This will set the appropriate compiler options for GCC, clang, Visual
    Studio, or the Intel compiler.  If you are using another compiler then you
    need to consult your compiler's manual to determine how to enable these
    instructions.  Note that AVX is the fastest but requires a CPU from at least
    2011.  SSE4 is the next fastest and is supported by most current machines.
*/

#include <ctime>
#include <queue>
#include <climits>

#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/image_transforms/draw_abstract.h>
#include <dlib/image_transforms/spatial_filtering_abstract.h>
#include <dlib/gui_widgets.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/photo.hpp>

#include "findEyeCenter.h"

using namespace dlib;
using namespace std;

#define FACE_DOWNSAMPLE_RATIO 4
#define SKIP_FRAMES 1
#define SCREEN_WIDTH 1024
#define SCREEN_HEIGHT 720

#define FLD_FILTER_SIZE 3

rectangle get_eye_region(const full_object_detection& face, bool left_eye) {
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

    int padding_x = (right-left)/2;
    int padding_y = (bottom-top)/1;
    return rectangle(left-padding_x, top-padding_y, right+padding_x, bottom+padding_y);

}

cv::Rect dlib_rect_to_cv(rectangle r) {
    int x = r.left();
    int y = r.top();
    int width = r.right() - r.left();
    int height = r.bottom() - r.top();

    if(x < 0) x = 0;
    if(x > SCREEN_WIDTH) x = SCREEN_WIDTH;
    if(y < 0) y = 0;
    if(y > SCREEN_HEIGHT) y = SCREEN_HEIGHT;

    if(x+width > SCREEN_WIDTH) width = width - ((x+width) % SCREEN_WIDTH);
    if(y+height > SCREEN_HEIGHT) height = height - ((y+height) % SCREEN_HEIGHT);

    return cv::Rect(x, y, width, height);
}

std::pair<point,point> linear_regression(std::vector<point> data) {
    int n = (int)data.size(), x1, y1, x2, y2;
    double avgX = 0, avgY = 0, numerator = 0.0, denominator = 0.0, slope;
    for(unsigned int i = 0; i<n; i++) {
        avgX += data[i].x();
        avgY += data[i].y();
    }
    avgX /= n;
    avgY /= n;

    for(int i=0; i<n; ++i){
        numerator += (data[i].x() - avgX) * (data[i].y() - avgY);
        denominator += (data[i].x() - avgX) * (data[i].x() - avgX);
    }
    slope = numerator / denominator;
    y1 = avgY - 150;
    y2 = avgY + 150;
    x1 = avgX - (150.0 / slope);
    x2 = avgX + (150.0 / slope);

    point p1(x1, y1);
    point p2(x2, y2);

    return make_pair(p1, p2);
}

int main()
{
    try
    {
        image_window win, win2;

        // Start camera
        cv::VideoCapture cap(0);
        if (!cap.isOpened())
        {
            cerr << "Unable to connect to camera" << endl;
            return 1;
        }
        cap.set(CV_CAP_PROP_FRAME_WIDTH, SCREEN_WIDTH);
        cap.set(CV_CAP_PROP_FRAME_HEIGHT, SCREEN_HEIGHT);

        // Load face detection and pose estimation models.
        frontal_face_detector detector = get_frontal_face_detector();
        shape_predictor pose_model;
        deserialize("shape_predictor_68_face_landmarks.dat") >> pose_model;

        // Grab and process frames until the main window is closed by the user.
        int fd_counter = -1;
        std::vector<rectangle> faces;
        std::vector<full_object_detection> shapes_vec;
        int it = 0;
        while(!win.is_closed())
        {
            std::clock_t begin = clock();

            // Grab a frame
            cv::Mat tmp, tmp2, tmp_orig, tmp_small;
            cap >> tmp;

            // std::clock_t c1 = clock();

            cv::flip(tmp, tmp2, 1);
            cv::cvtColor(tmp2, tmp, CV_BGR2GRAY);

            tmp.copyTo(tmp_orig);
            cv::resize(tmp, tmp_small, cv::Size(), 1.0/FACE_DOWNSAMPLE_RATIO, 1.0/FACE_DOWNSAMPLE_RATIO);
            cv::fastNlMeansDenoising(tmp_small, tmp2, 5, 5, 15);
            cv::resize(tmp2, tmp, cv::Size(), FACE_DOWNSAMPLE_RATIO, FACE_DOWNSAMPLE_RATIO);
            // tmp2.copyTo(tmp);
            // cv::GaussianBlur(tmp, tmp2, cv::Size(101,101), 1);

            cv_image<unsigned char> img(tmp);
            cv_image<unsigned char> img_orig(tmp_orig);
            cv_image<unsigned char> img_small(tmp_small);

            // std::clock_t c2 = clock();

            // Detect faces
            fd_counter = (fd_counter + 1) % SKIP_FRAMES;
            if(!fd_counter) {
                faces = detector(img_small);

                for (unsigned long i = 0; i < faces.size(); ++i) {
                    faces[i].set_right( faces[i].right() * FACE_DOWNSAMPLE_RATIO );
                    faces[i].set_bottom( faces[i].bottom() * FACE_DOWNSAMPLE_RATIO );
                    faces[i].set_top( faces[i].top() * FACE_DOWNSAMPLE_RATIO );
                    faces[i].set_left( faces[i].left() * FACE_DOWNSAMPLE_RATIO );
                }
            }

            // std::clock_t c3 = clock();
            /*
            std::vector<full_object_detection> shapes;
            if(faces.size() > 0)
                shapes.push_back(pose_model(img, faces[0]));
            */

            std::vector<image_window::overlay_circle> dots;
            std::vector<image_window::overlay_line> line;
            std::vector<full_object_detection> shapes;
            std::pair<point,point> reg_line;
            if(faces.size() == 0) {
                it = 0;
                shapes_vec.clear();
            } else {
                full_object_detection face = pose_model(img, faces[0]);
                shapes.push_back(face);
                if(shapes_vec.size() < FLD_FILTER_SIZE) {
                    shapes_vec.push_back(face);
                } else {
                    shapes_vec[it] = face;
                    it = (it+1) % FLD_FILTER_SIZE;
                }

                for(unsigned int j=0; j<68; j++) {
                    int px=0, py=0;
                    for(unsigned int i=0; i<shapes_vec.size(); i++) {
                        px += (int)shapes_vec[i].part(j).x();
                        py += (int)shapes_vec[i].part(j).y();
                    }
                    px = px / (int)shapes_vec.size();
                    py = py / (int)shapes_vec.size();
                    point p(px,py);
                    dots.push_back(image_window::overlay_circle(p, 2, rgb_pixel(255,0,0)));
                }

                // reg_line = linear_regression({face.part(27),face.part(28),face.part(29),face.part(30)});
                // line.push_back(image_window::overlay_line(reg_line.first, reg_line.second, rgb_pixel(0,255,0)));
            }



            // Find the pose of each face
            /*
            std::vector<full_object_detection> shapes;
            for (unsigned long i = 0; i < faces.size(); ++i) {
                full_object_detection x = pose_model(img, faces[i]);
                shapes.push_back(x);
            }
            */

            // Get eyes regions

            std::vector<rectangle> eyes;
            //std::vector<image_window::overlay_circle> dots;
            if(shapes.size() > 0) {
                rectangle left_eye_region  = get_eye_region(shapes[0], true);
                rectangle right_eye_region = get_eye_region(shapes[0], false);
                eyes.push_back(left_eye_region);
                eyes.push_back(right_eye_region);

                // Get pupils location
                cv::Rect cv_face_region = dlib_rect_to_cv(faces[0]);
                cv::Rect cv_left_eye_region = dlib_rect_to_cv(left_eye_region);
                cv_left_eye_region.x -= faces[0].left();
                cv_left_eye_region.y -= faces[0].top();

                cv::Mat cv_face = tmp(cv_face_region);
                cv::Point cv_left_pupil = findEyeCenter(cv_face, cv_left_eye_region);
                cv_left_pupil.x = cv_face_region.x + cv_left_eye_region.x + cv_left_pupil.x;
                cv_left_pupil.y = cv_face_region.y + cv_left_eye_region.y + cv_left_pupil.y;

                point left_pupil(cv_left_pupil.x, cv_left_pupil.y);
                dots.push_back(image_window::overlay_circle(left_pupil, 2, rgb_pixel(255,255,0)));
            }


                /*
            std::vector<image_window::overlay_circle> dots;
            for (unsigned long i = 0; i < shapes.size(); ++i) {
                full_object_detection& d = shapes[i];
                for (unsigned long j = 0; j < 68; ++j) {
                    dots.push_back(image_window::overlay_circle(d.part(j), 2, rgb_pixel(255,0,0)));
                }
            }
            */

            // Display it all on the screen
            win.clear_overlay();
            win.set_image(img_orig);
            win.add_overlay(faces);
            // win.add_overlay(render_face_detections(shapes));
            win.add_overlay(eyes);

            if(dots.size() != 0) {
                win.add_overlay(dots);
                win.add_overlay(line);
            }

            /*
            std::clock_t end = clock();
            cout << "GET FRAME // CONVERT // DETECT FACES // LANDMARKS // OVERLAY // DISPLAY" << endl;
                     0.6-0.7      0.05-0.1   0.15-0.25       0.05         0          0.05
            double total = (double)(end - begin) / CLOCKS_PER_SEC;
            double t1 = ((double)(c1 - begin) / CLOCKS_PER_SEC ) / total;
            double t2 = ((double)(c2 - c1) / CLOCKS_PER_SEC ) / total;
            double t3 = ((double)(c3 - c2) / CLOCKS_PER_SEC ) / total;
            double t4 = ((double)(c4 - c3) / CLOCKS_PER_SEC ) / total;
            double t5 = ((double)(c5 - c4) / CLOCKS_PER_SEC ) / total;
            double t6 = ((double)(end - c5) / CLOCKS_PER_SEC ) / total;
            cout << t1 << " // " << t2 << " // " << t3 << " // " << t4 << " // " << t5 << " // " << t6 << endl;

             */
        }
    }
    catch(exception& e)
    {
        cout << e.what() << endl;
    }
}