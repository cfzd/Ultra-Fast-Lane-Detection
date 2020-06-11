/*************************************************************************
	> File Name: counter.cpp
	> Author: Xingang Pan, Jun Li
	> Mail: px117@ie.cuhk.edu.hk
	> Created Time: Thu Jul 14 20:23:08 2016
 ************************************************************************/

#include "counter.hpp"

double Counter::get_precision(void)
{
	cerr<<"tp: "<<tp<<" fp: "<<fp<<" fn: "<<fn<<endl;
	if(tp+fp == 0)
	{
		cerr<<"no positive detection"<<endl;
		return -1;
	}
	return tp/double(tp + fp);
}

double Counter::get_recall(void)
{
	if(tp+fn == 0)
	{
		cerr<<"no ground truth positive"<<endl;
		return -1;
	}
	return tp/double(tp + fn);
}

long Counter::getTP(void)
{
	return tp;
}

long Counter::getFP(void)
{
	return fp;
}

long Counter::getFN(void)
{
	return fn;
}

void Counter::setTP(long value) 
{
	tp = value;
}

void Counter::setFP(long value)
{
  fp = value;
}

void Counter::setFN(long value)
{
	fn = value;
}

tuple<vector<int>, long, long, long, long> Counter::count_im_pair(const vector<vector<Point2f> > &anno_lanes, const vector<vector<Point2f> > &detect_lanes)
{
	vector<int> anno_match(anno_lanes.size(), -1);
	vector<int> detect_match;
	if(anno_lanes.empty())
	{
		return make_tuple(anno_match, 0, detect_lanes.size(), 0, 0);
	}

	if(detect_lanes.empty())
	{
		return make_tuple(anno_match, 0, 0, 0, anno_lanes.size());
	}
	// hungarian match first
	
	// first calc similarity matrix
	vector<vector<double> > similarity(anno_lanes.size(), vector<double>(detect_lanes.size(), 0));
	for(int i=0; i<anno_lanes.size(); i++)
	{
		const vector<Point2f> &curr_anno_lane = anno_lanes[i];
		for(int j=0; j<detect_lanes.size(); j++)
		{
			const vector<Point2f> &curr_detect_lane = detect_lanes[j];
			similarity[i][j] = lane_compare->get_lane_similarity(curr_anno_lane, curr_detect_lane);
		}
	}



	makeMatch(similarity, anno_match, detect_match);

	
	int curr_tp = 0;
	// count and add
	for(int i=0; i<anno_lanes.size(); i++)
	{
		if(anno_match[i]>=0 && similarity[i][anno_match[i]] > sim_threshold)
		{
			curr_tp++;
		}
		else
		{
			anno_match[i] = -1;
		}
	}
	int curr_fn = anno_lanes.size() - curr_tp;
	int curr_fp = detect_lanes.size() - curr_tp;
	return make_tuple(anno_match, curr_tp, curr_fp, 0, curr_fn);
}


void Counter::makeMatch(const vector<vector<double> > &similarity, vector<int> &match1, vector<int> &match2) {
	int m = similarity.size();
	int n = similarity[0].size();
    pipartiteGraph gra;
    bool have_exchange = false;
    if (m > n) {
        have_exchange = true;
        swap(m, n);
    }
    gra.resize(m, n);
    for (int i = 0; i < gra.leftNum; i++) {
        for (int j = 0; j < gra.rightNum; j++) {
			if(have_exchange)
				gra.mat[i][j] = similarity[j][i];
			else
				gra.mat[i][j] = similarity[i][j];
        }
    }
    gra.match();
    match1 = gra.leftMatch;
    match2 = gra.rightMatch;
    if (have_exchange) swap(match1, match2);
}
