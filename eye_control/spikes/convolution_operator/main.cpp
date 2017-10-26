#include <iostream>
#include <vector>
#include <ctime>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

#define N_RAYS 20
#define PI 3.14
#define CAP_W_L 1280
#define CAP_H_L 720
#define CAP_W_S 640
#define CAP_H_S 480


void show_normalized_img(string window_name, const Mat& img) {
    double min, max;
    Mat tmp = img.clone();
    minMaxLoc(tmp, &min, &max);
    if(min < 0) {
        tmp = tmp - min;
        max -= min;
        min = 0;
    }
    tmp.convertTo(tmp, CV_8UC1, 255/(max-min));
    imshow(window_name, tmp);
}


// --- CONVOLUTION OPERATOR (CHT filter) ---
int main(int, char**)
{
    /*
    VideoCapture cam(0);
    if (!cam.isOpened()) {
        cerr << "Cannot open camera." << endl;
        return -1;
    }
    while (true) {
        Mat frame;
        cam >> frame;
        flip(frame, frame, 1);
        cvtColor(frame, frame, CV_BGR2GRAY);

        imshow("cam", frame);
        if (waitKey(30) != 255)
            break;
    }
    */

    clock_t start = clock();

    Mat img_origin = imread("C:\\Users\\Head\\sources\\eye_control\\references\\me3.jpg", 1);
    Mat img = Mat::zeros(img_origin.rows, img_origin.cols, CV_8UC1);
    cvtColor(img_origin, img, COLOR_BGR2GRAY);
    Mat imgf, result;
    img.convertTo(imgf, CV_64FC1, 1.0/255);

    Mat inverted = img.clone(), inverted_blured;
    inverted.setTo(255);
    addWeighted(inverted, 1, img, -1, 0, inverted);
    inverted.convertTo(inverted, CV_64FC1, 1.0/255);
    GaussianBlur(inverted, inverted_blured, cv::Size(3,3), 1);


    int offset = 1;
    int Rmax = 34, Rmin = 28;
    Rmax -= offset;
    Rmin -= offset;
    int kernel_size = 2*Rmax+1;

    Mat ring_dx = Mat::zeros(kernel_size, kernel_size, CV_64FC1);
    Mat ring_dy = Mat::zeros(kernel_size, kernel_size, CV_64FC1);

    if(Rmin <= 0) {
        cerr << "Specify minimal circle radius" << endl;
        return -1;
    }
    for(int y = -Rmax; y <= Rmax; y++) {
        for (int x = -Rmax; x <= Rmax; x++) {
            if(x*x+y*y < Rmin*Rmin || x*x+y*y > Rmax*Rmax)
                continue;
            ring_dx.at<double>(y+Rmax, x+Rmax, 0) = x / (double)(abs(x)+abs(y));
            ring_dy.at<double>(y+Rmax, x+Rmax, 0) = y / (double)(abs(x)+abs(y));
        }
    }


    double minVal, maxVal, alfa = 2;
    Mat accX, accY, acc;
    Mat img_der_x, img_der_y, img_der_x_inv, img_der_y_inv;

    Scharr(imgf, img_der_x, -1, 1, 0);
    Scharr(imgf, img_der_y, -1, 0, 1);

    img_der_x_inv = (-1) * img_der_x;
    img_der_y_inv = (-1) * img_der_y;

    filter2D(img_der_x, accX, -1, ring_dx);
    filter2D(img_der_y, accY, -1, ring_dy);

    acc = alfa * accX + accY;
    // acc = acc * inverted_blured;


    // SHOW OFF ---------------------------------------------------------------
    Point minLoc, maxLoc;
    minMaxLoc(acc, &minVal, &maxVal, &minLoc, &maxLoc );


    imgf.convertTo(result, CV_8UC1, 255);
    //cvtColor(result, result, COLOR_GRAY2BGR);
    //circle(result, maxLoc, 2, Scalar(0,255,0), -1);

    Mat ring_mask = result.clone();
    ring_mask.setTo(0.0);
    circle(ring_mask, maxLoc, Rmax, Scalar(0xff,0xff,0xff), -1);
    circle(ring_mask, maxLoc, Rmin/1.5, Scalar(0x0,0x0,0x0), -1);
    // ring_mask = imgf * ring_mask;


    // Ray marching
    Point2d p;
    vector<Point> candidates;
    double threshold = 0.8;
    for(int r = 0; r < N_RAYS; r++) {
        double angle = r*(2*PI/N_RAYS);
        double x_step = cos(angle), y_step = sin(angle);

        p.x = maxLoc.x;
        p.y = maxLoc.y;
        p.x = maxLoc.x + cos(angle)*(0.5*Rmin);
        p.y = maxLoc.y - sin(angle)*(0.5*Rmin);

        for (int i = 0; i < 25; i++) {
            p.x += 2*x_step;
            p.y -= 2*y_step;
            circle(result, p, 1, Scalar(100,0,0), -1);

            double derv;
            if(r/(N_RAYS/4) == 0)
                derv = img_der_x.at<double>(p)      + img_der_y_inv.at<double>(p);
            else if(r/(N_RAYS/4) == 1)
                derv = img_der_x_inv.at<double>(p)  + img_der_y_inv.at<double>(p);
            else if(r/(N_RAYS/4) == 2)
                derv = img_der_x_inv.at<double>(p)  + img_der_y.at<double>(p);
            else if(r/(N_RAYS/4) == 3)
                derv = img_der_x.at<double>(p)      + img_der_y.at<double>(p);

            if(derv > threshold) {
                if(i < 5)
                    continue;
                circle(result, p, 2, Scalar(0,255,255), -1);
                candidates.push_back(Point((int)p.x,(int)p.y));
                break;
            } else if(i == 24)
                circle(result, p, 2, Scalar(255,0,0), -1);
            // else
            //     circle(result, p, 1, Scalar(0,255,0), -1);
        }
    }

/*
    // Fit Ellipse
    RotatedRect box = fitEllipse(candidates);
    ellipse(result, box, Scalar(0,0,255), 1, LINE_AA);
    circle(result, box.center, 2, Scalar(0,0,255), -1);
*/
    clock_t end = clock();
    cout << "Operation time: " << (end - start)/(double)CLOCKS_PER_SEC << endl;


    for(;;) {
        imshow("Result", result);
        imshow("Ring mask", ring_mask);
        //imshow("Inverted Blured", inverted_blured);
        show_normalized_img("Ring X", ring_dx);
        show_normalized_img("Ring Y", ring_dy);
        //show_normalized_img("Derivative X", img_der_x);
        //show_normalized_img("Derivative Y", img_der_y);
        //show_normalized_img("Derivative X inv", img_der_x_inv);
        //show_normalized_img("Derivative Y inv", img_der_y_inv);
        // show_normalized_img("Accumulator", acc);
        //imshow("Result2", result_sc);

        int c = waitKey(30);
        if(c != 255) break;
    }

    return 0;
}