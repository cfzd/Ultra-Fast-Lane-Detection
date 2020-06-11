/*************************************************************************
	> File Name: lane_compare.cpp
	> Author: Xingang Pan, Jun Li
	> Mail: px117@ie.cuhk.edu.hk
	> Created Time: Fri Jul 15 10:26:32 2016
 ************************************************************************/

#include "lane_compare.hpp"

double LaneCompare::get_lane_similarity(const vector<Point2f> &lane1, const vector<Point2f> &lane2)
{
	if(lane1.size()<2 || lane2.size()<2)
	{
		cerr<<"lane size must be greater or equal to 2"<<endl;
		return 0;
	}
	Mat im1 = Mat::zeros(im_height, im_width, CV_8UC1);
	Mat im2 = Mat::zeros(im_height, im_width, CV_8UC1);
	// draw lines on im1 and im2
	vector<Point2f> p_interp1;
	vector<Point2f> p_interp2;
	if(lane1.size() == 2)
	{
		p_interp1 = lane1;
	}
	else
	{
		p_interp1 = splineSolver.splineInterpTimes(lane1, 50);
	}

	if(lane2.size() == 2)
	{
		p_interp2 = lane2;
	}
	else
	{
		p_interp2 = splineSolver.splineInterpTimes(lane2, 50);
	}
	
	Scalar color_white = Scalar(1);
	for(int n=0; n<p_interp1.size()-1; n++)
	{
		line(im1, p_interp1[n], p_interp1[n+1], color_white, lane_width);
	}
	for(int n=0; n<p_interp2.size()-1; n++)
	{
		line(im2, p_interp2[n], p_interp2[n+1], color_white, lane_width);
	}

	double sum_1 = cv::sum(im1).val[0];
	double sum_2 = cv::sum(im2).val[0];
	double inter_sum = cv::sum(im1.mul(im2)).val[0];
	double union_sum = sum_1 + sum_2 - inter_sum; 
	double iou = inter_sum / union_sum;
	return iou;
}


// resize the lane from Size(curr_width, curr_height) to Size(im_width, im_height)
void LaneCompare::resize_lane(vector<Point2f> &curr_lane, int curr_width, int curr_height)
{
	if(curr_width == im_width && curr_height == im_height)
	{
		return;
	}
	double x_scale = im_width/(double)curr_width;
	double y_scale = im_height/(double)curr_height;
	for(int n=0; n<curr_lane.size(); n++)
	{
		curr_lane[n] = Point2f(curr_lane[n].x*x_scale, curr_lane[n].y*y_scale);
	}
}

