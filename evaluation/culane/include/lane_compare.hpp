#ifndef LANE_COMPARE_HPP
#define LANE_COMPARE_HPP

#include "spline.hpp"
#include <vector>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

class LaneCompare{
	public:
		enum CompareMode{
			IOU,
			Caltech
		};

		LaneCompare(int _im_width, int _im_height, int _lane_width = 10, CompareMode _compare_mode = IOU){
			im_width = _im_width;
			im_height = _im_height;
			compare_mode = _compare_mode;
			lane_width = _lane_width;
		}

		double get_lane_similarity(const vector<Point2f> &lane1, const vector<Point2f> &lane2);
		void resize_lane(vector<Point2f> &curr_lane, int curr_width, int curr_height);
	private:
		CompareMode compare_mode;
		int im_width;
		int im_height;
		int lane_width;
		Spline splineSolver;
};

#endif
