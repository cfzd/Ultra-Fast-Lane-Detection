/*************************************************************************
	> File Name: evaluate.cpp
	> Author: Xingang Pan, Jun Li
	> Mail: px117@ie.cuhk.edu.hk
	> Created Time: 2016年07月14日 星期四 18时28分45秒
 ************************************************************************/

#include "counter.hpp"
#include "spline.hpp"
#if __linux__
#include <unistd.h>
#elif _MSC_VER
#include "getopt.h"
#endif
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
using namespace std;
using namespace cv;

void help(void)
{
	cout<<"./evaluate [OPTIONS]"<<endl;
	cout<<"-h                  : print usage help"<<endl;
	cout<<"-a                  : directory for annotation files (default: /data/driving/eval_data/anno_label/)"<<endl;
	cout<<"-d                  : directory for detection files (default: /data/driving/eval_data/predict_label/)"<<endl;
	cout<<"-i                  : directory for image files (default: /data/driving/eval_data/img/)"<<endl;
	cout<<"-l                  : list of images used for evaluation (default: /data/driving/eval_data/img/all.txt)"<<endl;
	cout<<"-w                  : width of the lanes (default: 10)"<<endl;
	cout<<"-t                  : threshold of iou (default: 0.4)"<<endl;
	cout<<"-c                  : cols (max image width) (default: 1920)"<<endl;
	cout<<"-r                  : rows (max image height) (default: 1080)"<<endl;
	cout<<"-s                  : show visualization"<<endl;
	cout<<"-f                  : start frame in the test set (default: 1)"<<endl;
}


void read_lane_file(const string &file_name, vector<vector<Point2f> > &lanes);
void visualize(string &full_im_name, vector<vector<Point2f> > &anno_lanes, vector<vector<Point2f> > &detect_lanes, vector<int> anno_match, int width_lane);

int main(int argc, char **argv)
{
	// process params
	string anno_dir = "/data/driving/eval_data/anno_label/";
	string detect_dir = "/data/driving/eval_data/predict_label/";
	string im_dir = "/data/driving/eval_data/img/";
	string list_im_file = "/data/driving/eval_data/img/all.txt";
	string output_file = "./output.txt";
	int width_lane = 10;
	double iou_threshold = 0.4;
	int im_width = 1920;
	int im_height = 1080;
	int oc;
	bool show = false;
	int frame = 1;
	while((oc = getopt(argc, argv, "ha:d:i:l:w:t:c:r:sf:o:")) != -1)
	{
		switch(oc)
		{
			case 'h':
				help();
				return 0;
			case 'a':
				anno_dir = optarg;
				break;
			case 'd':
				detect_dir = optarg;
				break;
			case 'i':
				im_dir = optarg;
				break;
			case 'l':
				list_im_file = optarg;
				break;
			case 'w':
				width_lane = atoi(optarg);
				break;
			case 't':
				iou_threshold = atof(optarg);
				break;
			case 'c':
				im_width = atoi(optarg);
				break;
			case 'r':
				im_height = atoi(optarg);
				break;
			case 's':
				show = true;
				break;
			case 'f':
				frame = atoi(optarg);
				break;
			case 'o':
				output_file = optarg;
				break;
		}
	}


	cout<<"------------Configuration---------"<<endl;
	cout<<"anno_dir: "<<anno_dir<<endl;
	cout<<"detect_dir: "<<detect_dir<<endl;
	cout<<"im_dir: "<<im_dir<<endl;
	cout<<"list_im_file: "<<list_im_file<<endl;
	cout<<"width_lane: "<<width_lane<<endl;
	cout<<"iou_threshold: "<<iou_threshold<<endl;
	cout<<"im_width: "<<im_width<<endl;
	cout<<"im_height: "<<im_height<<endl;
	cout<<"-----------------------------------"<<endl;
	cout<<"Evaluating the results..."<<endl;
	// this is the max_width and max_height

	if(width_lane<1)
	{
		cerr<<"width_lane must be positive"<<endl;
		help();
		return 1;
	}


	ifstream ifs_im_list(list_im_file, ios::in);
	if(ifs_im_list.fail())
	{
		cerr<<"Error: file "<<list_im_file<<" not exist!"<<endl;
		return 1;
	}


	Counter counter(im_width, im_height, iou_threshold, width_lane);
	
	vector<int> anno_match;
	string sub_im_name;
  // pre-load filelist
  vector<string> filelists;
  while (getline(ifs_im_list, sub_im_name)) {
    filelists.push_back(sub_im_name);
  }
  ifs_im_list.close();

  vector<tuple<vector<int>, long, long, long, long>> tuple_lists;
  tuple_lists.resize(filelists.size());

#pragma omp parallel for
	for (int i = 0; i < filelists.size(); i++)
	{
		auto sub_im_name = filelists[i];
		string full_im_name = im_dir + sub_im_name;
		string sub_txt_name =  sub_im_name.substr(0, sub_im_name.find_last_of(".")) + ".lines.txt";
		string anno_file_name = anno_dir + sub_txt_name;
		string detect_file_name = detect_dir + sub_txt_name;
		vector<vector<Point2f> > anno_lanes;
		vector<vector<Point2f> > detect_lanes;
		read_lane_file(anno_file_name, anno_lanes);
		read_lane_file(detect_file_name, detect_lanes);
		//cerr<<count<<": "<<full_im_name<<endl;
		tuple_lists[i] = counter.count_im_pair(anno_lanes, detect_lanes);
		if (show)
		{
			auto anno_match = get<0>(tuple_lists[i]);
			visualize(full_im_name, anno_lanes, detect_lanes, anno_match, width_lane);
			waitKey(0);
		}
	}

	long tp = 0, fp = 0, tn = 0, fn = 0;
  for (auto result: tuple_lists) {
    tp += get<1>(result);
    fp += get<2>(result);
    // tn = get<3>(result);
    fn += get<4>(result);
  }
	counter.setTP(tp);
	counter.setFP(fp);
	counter.setFN(fn);
	
	double precision = counter.get_precision();
	double recall = counter.get_recall();
	double F = 2 * precision * recall / (precision + recall);	
	cerr<<"finished process file"<<endl;
	cout<<"precision: "<<precision<<endl;
	cout<<"recall: "<<recall<<endl;
	cout<<"Fmeasure: "<<F<<endl;
	cout<<"----------------------------------"<<endl;

	ofstream ofs_out_file;
	ofs_out_file.open(output_file, ios::out);
	ofs_out_file<<"file: "<<output_file<<endl;
	ofs_out_file<<"tp: "<<counter.getTP()<<" fp: "<<counter.getFP()<<" fn: "<<counter.getFN()<<endl;
	ofs_out_file<<"precision: "<<precision<<endl;
	ofs_out_file<<"recall: "<<recall<<endl;
	ofs_out_file<<"Fmeasure: "<<F<<endl<<endl;
	ofs_out_file.close();
	return 0;
}

void read_lane_file(const string &file_name, vector<vector<Point2f> > &lanes)
{
	lanes.clear();
	ifstream ifs_lane(file_name, ios::in);
	if(ifs_lane.fail())
	{
		return;
	}

	string str_line;
	while(getline(ifs_lane, str_line))
	{
		vector<Point2f> curr_lane;
		stringstream ss;
		ss<<str_line;
		double x,y;
		while(ss>>x>>y)
		{
			curr_lane.push_back(Point2f(x, y));
		}
		lanes.push_back(curr_lane);
	}

	ifs_lane.close();
}

void visualize(string &full_im_name, vector<vector<Point2f> > &anno_lanes, vector<vector<Point2f> > &detect_lanes, vector<int> anno_match, int width_lane)
{
	Mat img = imread(full_im_name, 1);
	Mat img2 = imread(full_im_name, 1);
	vector<Point2f> curr_lane;
	vector<Point2f> p_interp;
	Spline splineSolver;
	Scalar color_B = Scalar(255, 0, 0);
	Scalar color_G = Scalar(0, 255, 0);
	Scalar color_R = Scalar(0, 0, 255);
	Scalar color_P = Scalar(255, 0, 255);
	Scalar color;
	for (int i=0; i<anno_lanes.size(); i++)
	{
		curr_lane = anno_lanes[i];
		if(curr_lane.size() == 2)
		{
			p_interp = curr_lane;
		}
		else
		{
			p_interp = splineSolver.splineInterpTimes(curr_lane, 50);
		}
		if (anno_match[i] >= 0)
		{
			color = color_G;
		}
		else
		{
			color = color_G;
		}
		for (int n=0; n<p_interp.size()-1; n++)
		{
			line(img, p_interp[n], p_interp[n+1], color, width_lane);
			line(img2, p_interp[n], p_interp[n+1], color, 2);
		}
	}
	bool detected;
	for (int i=0; i<detect_lanes.size(); i++)
	{
		detected = false;
		curr_lane = detect_lanes[i];
		if(curr_lane.size() == 2)
		{
			p_interp = curr_lane;
		}
		else
		{
			p_interp = splineSolver.splineInterpTimes(curr_lane, 50);
		}
		for (int n=0; n<anno_lanes.size(); n++)
		{
			if (anno_match[n] == i)
			{
				detected = true;
				break;
			}
		}
		if (detected == true)
		{
			color = color_B;
		}
		else
		{
			color = color_R;
		}
		for (int n=0; n<p_interp.size()-1; n++)
		{
			line(img, p_interp[n], p_interp[n+1], color, width_lane);
			line(img2, p_interp[n], p_interp[n+1], color, 2);
		}
	}
	namedWindow("visualize", 1);
	imshow("visualize", img);
	namedWindow("visualize2", 1);
	imshow("visualize2", img2);
}
