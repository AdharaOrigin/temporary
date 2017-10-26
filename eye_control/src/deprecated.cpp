//
// Created by Head on 11. 5. 2017.
//

// Circle Hough Transform; ransac() => Vote for centre from inliers. Unstable.
/*
    cv::Mat acc = cv::Mat::zeros(eye.img_flp.size(), eye.img_flp.type());
    cv::Mat tmp = cv::Mat::zeros(eye.img_flp.size(), eye.img_flp.type());
    for(unsigned int i = 0; i < inliers.size(); i++) {
    tmp.setTo(0);
    circle(tmp, cv::Point(inliers[i].x, inliers[i].y), eye.iris_r_max, cv::Scalar(0.25), 3);
    circle(tmp, cv::Point(inliers[i].x, inliers[i].y), eye.iris_r_max, cv::Scalar(0.50), 2);
    circle(tmp, cv::Point(inliers[i].x, inliers[i].y), eye.iris_r_max, cv::Scalar(1.00), 1);
    acc += tmp;
    }
    show_normalized_img("Acc", acc);

    cv::Point shit;
    minMaxLoc(acc, NULL, NULL, NULL, &shit);
*/


// Rays of sums of intensities.
/*
vector<cv::Point> rays_run(ROI eye, const cv::Point corner_left, const cv::Point corner_right) {
    cv::Mat img_dbg;
    eye.resize(0, 100);
    eye.img_flp.convertTo(img_dbg, CV_8UC3, 255);
    cvtColor(img_dbg, img_dbg, CV_GRAY2BGR);

    cv::Point2d p, pout, pin;
    vector <cv::Point> candidates;

    cv::GaussianBlur(eye.img_int, eye.img_int, cv::Size(5, 5), 2);
    imshow("Ble", eye.img_int);

    for (int r = 0; r < N_RAYS; r++) {
        if ((r > 5 && r < 15) || (r > 25 && r < 35))
            continue;

        double angle = r * (2 * PI / N_RAYS);
        double x_step = 2 * cos(angle), y_step = 2 * sin(angle);
        p.x = eye.rough_centre.x + cos(angle) * (0.3 * eye.iris_r_min);
        p.y = eye.rough_centre.y - sin(angle) * (0.3 * eye.iris_r_min);

        int range = 24;
        double k_max = 1.0;
        bool found = false;
        cv::Point2d ray_max;

        for (int i = 0; i < range; i++) {
            p.x += x_step;
            p.y -= y_step;
            if (out_of_image(eye.img_int, p))
                break;

            double k = 0, sum_out = 0, sum_in = 0.00001, distance_coef = 1;
            if ((p.x - eye.rough_centre.x) * (p.x - eye.rough_centre.x) +
                (p.y - eye.rough_centre.y) * (p.y - eye.rough_centre.y) < (eye.iris_r_min) * (eye.iris_r_min)) {
                distance_coef = 0.5;
            }

            pout.x = p.x;
            pin.x = p.x;
            pout.y = p.y;
            pin.y = p.y;
            for (int j = 0; j < 8; j++) {
                pout.x += x_step;
                pout.y -= y_step;
                pin.x -= x_step;
                pin.y += y_step;

                if (i == -12) {
                    circle(img_dbg, pout, 1, cv::Scalar(0, 120, 0), -1);
                    circle(img_dbg, pin, 1, cv::Scalar(120, 0, 0), -1);
                }

                if (out_of_image(eye.img_int, pout) || out_of_image(eye.img_int, pin))
                    break;

                sum_out += eye.img_int.at<int>(cv::Point((int) pout.x, (int) pout.y));
                sum_in += eye.img_int.at<int>(cv::Point((int) pin.x, (int) pin.y));
            }
            k = sum_out / sum_in;
            k *= distance_coef;

            if (k > k_max) {
                ray_max.x = p.x;
                ray_max.y = p.y;
                k_max = k;
                found = true;
            }
            if (i == range - 1)
                circle(img_dbg, p, 1, cv::Scalar(120, 0, 0), -1);
        }
        if (found) {
            candidates.push_back(ray_max);
            circle(img_dbg, ray_max, 1, cv::Scalar(0, 0, 255), -1);
        }
    }
}
*/

// Helper ^^^
/*
bool out_of_image(const cv::Mat& img, const cv::Point2d p) {
    return (p.x < 0 || p.x >= img.cols || p.y < 0 || p.y >= img.rows-2);
}
*/


// Non-linear Least Squares
/*
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
*/


