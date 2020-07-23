
# Install
1. Clone the project

    ```
    git clone https://github.com/cfzd/Ultra-Fast-Lane-Detection
    cd Ultra-Fast-Lane-Detection
    ```

2. Create a conda virtual environment and activate it

    ```
    conda create -n lane-det python=3.7 -y
    conda activate lane-det
    ```

3. Install dependencies

    ```
    # If you don't have pytorch
    conda install pytorch torchvision cudatoolkit=10.1 -c pytorch 

    pip install -r requirements.txt
    ```

4. Install CULane evaluation tools. 

    This tools requires OpenCV C++. Please follow [here](https://docs.opencv.org/master/d7/d9f/tutorial_linux_install.html) to install OpenCV C++. ***When you build OpenCV, remove the paths of anaconda from PATH or it will be failed.***
    ```
    # First you need to install OpenCV C++. 
    # After installation, make a soft link of OpenCV include path.

    ln -s /usr/local/include/opencv4/opencv2 /usr/local/include/opencv2
    ```
    We provide three kinds of complie pipelines to build the evaluation tool of CULane.

    Option 1:

    ```
    cd evaluation/culane
    make
    ```

    Option 2:
    ```
    cd evaluation/culane
    mkdir build && cd build
    cmake ..
    make
    mv culane_evaluator ../evaluate
    ```

    For Windows user:
    ```
    mkdir build-vs2017
    cd build-vs2017
    cmake .. -G "Visual Studio 15 2017 Win64"
    cmake --build . --config Release  
    # or, open the "xxx.sln" file by Visual Studio and click build button
    move culane_evaluator ../evaluate
    ```


5. Data preparation

    Download [CULane](https://xingangpan.github.io/projects/CULane.html) and [Tusimple](https://github.com/TuSimple/tusimple-benchmark/issues/3). Then extract them to `$CULANEROOT` and `$TUSIMPLEROOT`. The directory arrangement of Tusimple should look like:
    ```
    $TUSIMPLEROOT
    |──clips
    |──label_data_0313.json
    |──label_data_0531.json
    |──label_data_0601.json
    |──test_tasks_0627.json
    |──test_label.json
    |──readme.md
    ```
    The The directory arrangement of CULane should look like:
    ```
    $CULANEROOT
    |──driver_100_30frame
    |──driver_161_90frame
    |──driver_182_30frame
    |──driver_193_90frame
    |──driver_23_30frame
    |──driver_37_30frame
    |──laneseg_label_w16
    |──list
    ```
    
    For Tusimple, the segmentation annotation is not provided, hence we need to generate segmentation from the json annotation. 

    ```
    python scripts/convert_tusimple.py --root $TUSIMPLEROOT
    # this will generate segmentations and 2 list files train_gt.txt test.txt
    ```