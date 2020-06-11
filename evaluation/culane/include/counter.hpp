#ifndef COUNTER_HPP
#define COUNTER_HPP

#include "lane_compare.hpp"
#include "hungarianGraph.hpp"
#include <iostream>
#include <algorithm>
#include <tuple>
#include <vector>
#include <opencv2/core.hpp>

using namespace std;
using namespace cv;

// before coming to use functions of this class, the lanes should resize to im_width and im_height using resize_lane() in lane_compare.hpp
class Counter
{
	public:
		Counter(int _im_width, int _im_height, double _iou_threshold=0.4, int _lane_width=10):tp(0),fp(0),fn(0){
			im_width = _im_width;
			im_height = _im_height;
			sim_threshold = _iou_threshold;
			lane_compare = new LaneCompare(_im_width, _im_height,  _lane_width, LaneCompare::IOU);
		};
		double get_precision(void);
		double get_recall(void);
		long getTP(void);
		long getFP(void);
		long getFN(void);
		void setTP(long);
		void setFP(long);
		void setFN(long);
		// direct add tp, fp, tn and fn
		// first match with hungarian
		tuple<vector<int>, long, long, long, long> count_im_pair(const vector<vector<Point2f> > &anno_lanes, const vector<vector<Point2f> > &detect_lanes);
		void makeMatch(const vector<vector<double> > &similarity, vector<int> &match1, vector<int> &match2);

	private:
		double sim_threshold;
		int im_width;
		int im_height;
		long tp;
		long fp;
		long fn;
		LaneCompare *lane_compare;
};
#endif
