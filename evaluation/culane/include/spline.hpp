#ifndef SPLINE_HPP
#define SPLINE_HPP
#include <vector>
#include <cstdio>
#include <math.h>
#include <opencv2/core.hpp>

using namespace cv;
using namespace std;

struct Func {
    double a_x;
    double b_x;
    double c_x;
    double d_x;
    double a_y;
    double b_y;
    double c_y;
    double d_y;
    double h;
};
class Spline {
public:
	vector<Point2f> splineInterpTimes(const vector<Point2f> &tmp_line, int times);
    vector<Point2f> splineInterpStep(vector<Point2f> tmp_line, double step);
	vector<Func> cal_fun(const vector<Point2f> &point_v);
};
#endif
