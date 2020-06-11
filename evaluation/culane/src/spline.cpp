#include <vector>
#include <iostream>
#include "spline.hpp"
using namespace std;
using namespace cv;

vector<Point2f> Spline::splineInterpTimes(const vector<Point2f>& tmp_line, int times) {
    vector<Point2f> res;

    if(tmp_line.size() == 2) {
        double x1 = tmp_line[0].x;
        double y1 = tmp_line[0].y;
        double x2 = tmp_line[1].x;
        double y2 = tmp_line[1].y;

        for (int k = 0; k <= times; k++) {
            double xi =  x1 + double((x2 - x1) * k) / times;
            double yi =  y1 + double((y2 - y1) * k) / times;
            res.push_back(Point2f(xi, yi));
        }
    }

    else if(tmp_line.size() > 2)
    {
        vector<Func> tmp_func;
        tmp_func = this->cal_fun(tmp_line);
        if (tmp_func.empty()) {
            cout << "in splineInterpTimes: cal_fun failed" << endl;
            return res;
        }
        for(int j = 0; j < tmp_func.size(); j++)
        {
            double delta = tmp_func[j].h / times;
            for(int k = 0; k < times; k++)
            {
                double t1 = delta*k;
                double x1 = tmp_func[j].a_x + tmp_func[j].b_x*t1 + tmp_func[j].c_x*pow(t1,2) + tmp_func[j].d_x*pow(t1,3);
                double y1 = tmp_func[j].a_y + tmp_func[j].b_y*t1 + tmp_func[j].c_y*pow(t1,2) + tmp_func[j].d_y*pow(t1,3);
                res.push_back(Point2f(x1, y1));
            }
        }
        res.push_back(tmp_line[tmp_line.size() - 1]);
    }
	else {
		cerr << "in splineInterpTimes: not enough points" << endl;
	}
    return res;
}
vector<Point2f> Spline::splineInterpStep(vector<Point2f> tmp_line, double step) {
	vector<Point2f> res;
	/*
	if (tmp_line.size() == 2) {
		double x1 = tmp_line[0].x;
		double y1 = tmp_line[0].y;
		double x2 = tmp_line[1].x;
		double y2 = tmp_line[1].y;

		for (double yi = std::min(y1, y2); yi < std::max(y1, y2); yi += step) {
            double xi;
			if (yi == y1) xi = x1;
			else xi = (x2 - x1) / (y2 - y1) * (yi - y1) + x1;
			res.push_back(Point2f(xi, yi));
		}
	}*/
	if (tmp_line.size() == 2) {
		double x1 = tmp_line[0].x;
		double y1 = tmp_line[0].y;
		double x2 = tmp_line[1].x;
		double y2 = tmp_line[1].y;
		tmp_line[1].x = (x1 + x2) / 2;
		tmp_line[1].y = (y1 + y2) / 2;
		tmp_line.push_back(Point2f(x2, y2));
	}
	if (tmp_line.size() > 2) {
		vector<Func> tmp_func;
		tmp_func = this->cal_fun(tmp_line);
		double ystart = tmp_line[0].y;
		double yend = tmp_line[tmp_line.size() - 1].y;
		bool down;
		if (ystart < yend) down = 1;
		else down = 0;
		if (tmp_func.empty()) {
			cerr << "in splineInterpStep: cal_fun failed" << endl;
		}

		for(int j = 0; j < tmp_func.size(); j++)
        {
            for(double t1 = 0; t1 < tmp_func[j].h; t1 += step)
            {
                double x1 = tmp_func[j].a_x + tmp_func[j].b_x*t1 + tmp_func[j].c_x*pow(t1,2) + tmp_func[j].d_x*pow(t1,3);
                double y1 = tmp_func[j].a_y + tmp_func[j].b_y*t1 + tmp_func[j].c_y*pow(t1,2) + tmp_func[j].d_y*pow(t1,3);
                res.push_back(Point2f(x1, y1));
            }
        }
        res.push_back(tmp_line[tmp_line.size() - 1]);
	}
    else {
        cerr << "in splineInterpStep: not enough points" << endl;
    }
    return res;
}

vector<Func> Spline::cal_fun(const vector<Point2f> &point_v)
{
    vector<Func> func_v;
    int n = point_v.size();
    if(n<=2) {
        cout << "in cal_fun: point number less than 3" << endl;
        return func_v;
    }

    func_v.resize(point_v.size()-1);

    vector<double> Mx(n);
    vector<double> My(n);
    vector<double> A(n-2);
    vector<double> B(n-2);
    vector<double> C(n-2);
    vector<double> Dx(n-2);
    vector<double> Dy(n-2);
    vector<double> h(n-1);
    //vector<func> func_v(n-1);

    for(int i = 0; i < n-1; i++)
    {
        h[i] = sqrt(pow(point_v[i+1].x - point_v[i].x, 2) + pow(point_v[i+1].y - point_v[i].y, 2));
    }

    for(int i = 0; i < n-2; i++)
    {
        A[i] = h[i];
        B[i] = 2*(h[i]+h[i+1]);
        C[i] = h[i+1];

        Dx[i] =  6*( (point_v[i+2].x - point_v[i+1].x)/h[i+1] - (point_v[i+1].x - point_v[i].x)/h[i] );
        Dy[i] =  6*( (point_v[i+2].y - point_v[i+1].y)/h[i+1] - (point_v[i+1].y - point_v[i].y)/h[i] );
    }

    //TDMA
    C[0] = C[0] / B[0];
    Dx[0] = Dx[0] / B[0];
    Dy[0] = Dy[0] / B[0];
    for(int i = 1; i < n-2; i++)
    {
        double tmp = B[i] - A[i]*C[i-1];
        C[i] = C[i] / tmp;
        Dx[i] = (Dx[i] - A[i]*Dx[i-1]) / tmp;
        Dy[i] = (Dy[i] - A[i]*Dy[i-1]) / tmp;
    }
    Mx[n-2] = Dx[n-3];
    My[n-2] = Dy[n-3];
    for(int i = n-4; i >= 0; i--)
    {
        Mx[i+1] = Dx[i] - C[i]*Mx[i+2];
        My[i+1] = Dy[i] - C[i]*My[i+2];
    }

    Mx[0] = 0;
    Mx[n-1] = 0;
    My[0] = 0;
    My[n-1] = 0;

    for(int i = 0; i < n-1; i++)
    {
        func_v[i].a_x = point_v[i].x;
        func_v[i].b_x = (point_v[i+1].x - point_v[i].x)/h[i] - (2*h[i]*Mx[i] + h[i]*Mx[i+1]) / 6;
        func_v[i].c_x = Mx[i]/2;
        func_v[i].d_x = (Mx[i+1] - Mx[i]) / (6*h[i]);

        func_v[i].a_y = point_v[i].y;
        func_v[i].b_y = (point_v[i+1].y - point_v[i].y)/h[i] - (2*h[i]*My[i] + h[i]*My[i+1]) / 6;
        func_v[i].c_y = My[i]/2;
        func_v[i].d_y = (My[i+1] - My[i]) / (6*h[i]);

        func_v[i].h = h[i];
    }
    return func_v;
}
