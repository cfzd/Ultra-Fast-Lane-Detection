#include <iostream>
#include <memory>
#include <chrono>
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/videoio.hpp"
#include <torch/script.h>
#include <vector>

using namespace std;
using namespace cv;
using namespace torch::indexing;

torch::jit::script::Module module_;

std::vector<double> linspace(double start_in, double end_in, int num_in)
{
    std::vector<double> linspaced;

    double start = static_cast<double>(start_in);
    double end = static_cast<double>(end_in);
    double num = static_cast<double>(num_in);

    if (num == 0) { return linspaced; }
    if (num == 1) 
    {
        linspaced.push_back(start);
        return linspaced;
    }

    double delta = (end - start) / (num - 1);

    for(int i=0; i < num-1; ++i)
    {
      linspaced.push_back(start + delta * i);
    }
    linspaced.push_back(end); // I want to ensure that start and end
                            // are exactly the same as the input

    return linspaced;
}

std::vector<int> arrange(int num)
{
    std::vector<int> result;
    for (int i = 0; i < num; i++)
    {
        result.push_back(i);
    }
    return result;
}

int counter = 0;
Mat RunLaneDetection(Mat frame)
{

    int img_w = 1280;
    int img_h = 720; 
    
    Mat dest;
    // CV Resize
    cv::resize(frame, dest, cv::Size(800, 288));
    cv::cvtColor(dest, dest, cv::COLOR_BGR2RGB);  // BGR -> RGB
    dest.convertTo(dest, CV_32FC3, 1.0f / 255.0f);  // normalization 1/255
    int culane_row_anchor[] = {121, 131, 141, 150, 160, 170, 180, 189, 199, 209, 219, 228, 238, 248, 258, 267, 277, 287};


    auto tensor_img = torch::from_blob(dest.data, {1, dest.rows, dest.cols, dest.channels()}).to(torch::kCUDA);



    tensor_img = tensor_img.permute({0, 3, 1, 2}).contiguous();  // BHWC -> BCHW (Batch, Channel, Height, Width)

    tensor_img = tensor_img.to(torch::kHalf);
    std::vector<torch::jit::IValue> inputs;
    inputs.emplace_back(tensor_img);
    torch::jit::IValue output = module_.forward(inputs);
    torch::Tensor outputTensor = output.toTensor();
   
    // Logic
    int cuLaneGriding_num = 200;
    std::vector<double> linSpaceVector = linspace(0, 800 - 1, cuLaneGriding_num);
    double linSpace = linSpaceVector[1] - linSpaceVector[0];
    // Remove 1
    outputTensor = outputTensor.squeeze(0);
    // Flip
    outputTensor = outputTensor.flip(1);


    // Calculate SoftMax
    torch::Tensor prob = outputTensor.index({Slice(None, -1)}).softmax(0);

    // Calculate idx
    std::vector<int> idx = arrange(cuLaneGriding_num + 1);  
    auto arrange_idx = torch::from_blob(idx.data(), {cuLaneGriding_num, 1, 1}).to(torch::kCUDA);
    outputTensor = outputTensor.argmax(0);
		


    auto mult = prob * arrange_idx;

    auto loc = mult.sum(1);
    for (int i = 0; i < outputTensor.size(0); i++)
    {  
   			if (outputTensor[i][0].item<long>() == cuLaneGriding_num)
				{
					outputTensor[i][0] = 0;
				}	
   			if (outputTensor[i][1].item<long>() == cuLaneGriding_num)
				{
					outputTensor[i][1] = 0;
				}	
   			if (outputTensor[i][2].item<long>() == cuLaneGriding_num)
				{
					outputTensor[i][2] = 0;
				}	     
   			if (outputTensor[i][3].item<long>() == cuLaneGriding_num)
				{
					outputTensor[i][3] = 0;
				}	
    }
		torch::Tensor res = outputTensor;
   
    for (int i = 0; i < outputTensor.size(1); i++)
    {
        for (int k = 0; k < outputTensor.size(0); k++)
        {
            if (outputTensor[k][i].item<long>() > 0)
            {

    
                long widht = outputTensor[k][i].item<long>()  * linSpace * img_w /800;
                long height = img_h * (float(culane_row_anchor[18-1-k])/288);
								
								if (counter == 0)
								{
									cout << widht << ' ' << height;
								  cout << '\n';
								}

  
                circle( frame, Point( widht, height ), 5, Scalar( 0, 255, 0 ), -1);
            }
        }
    }
		counter = counter + 1;

    

    return frame;
}

void RunVideo()
{
    VideoCapture cap("/data/video/dout.mp4"); 
		cout << "Prepare to load";
    
    Mat frame;
    while (true)
    {
        cap.read(frame); // read a new frame from video 
        cv::imshow("", RunLaneDetection(frame));

        if (waitKey(10) >= 0)
            break;
    }
}

int main() {
    // Load JIT
    module_ = torch::jit::load("/data/Models/UltraFastLaneDetection/UFLD.torchscript.pt");
    module_.to(torch::kCUDA);
    module_.to(torch::kHalf);
    module_.eval();

    // check if gpu flag is set
    bool is_gpu = true;

    RunVideo();
    cv::destroyAllWindows();
    return 0;
}
