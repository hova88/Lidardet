
# Inference on Nvidia Xavier 

- convert model to tensorrt verson and detect object with Ros topics
- Based on [wangguojun2018/PointPillars_Tensorrt](https://github.com/wangguojun2018/PointPillars_Tensorrt)



# Lidar-detection
 - ONLY support python 3.6+, pytorch 1.3.0+. Tested in Ubuntu 16.04.

 - Suppose you have installed the arm64 version of `Pytorch_1.3.0+, TensorRt, CUDA, CuDNN `on your Xavier device
 
## Install

### 1. Clone code

```bash
git clone https://github.com/hova88/Lidardet.git
```

### 2. Install dependence python packages

```bash
pip3 install numba pyntcloud pyyaml rospkg pyquaternion protobuf
```
  
 - Ros-numpy:
```bash
git clone https://github.com/eric-wieser/ros_numpy
cd ros_numpy && python setup.py install
```

 - if `LLVM_CONFIG`not found, as below:
```bash
sudo apt-get install llvm-8
export LLVM_CONFIG=/usr/bin/llvm-config-8
pip3 install numba
```
 - if `No lapack/blas resources found`, as below：   
```bash
apt-get install gfortran libopenblas-dev liblapack-dev
pip3 install scipy
```

 - if `can not find CUDNN`:
```bash
sudo ln -s libcudnn.so.7 libcudnn.so
```

 - Follow instructions in [torch2trt](https://github.com/NVIDIA-AI-IOT/torch2trt) to install torch2trt


### 3. Installing ROS dependencies：
```bash
apt install ros-melodic-rospy ros-melodic-ros-base ros-melodic-sensor-msgs ros-melodic-jsk-recognition-msgs ros-melodic-visualization-msgs
```
### 4. Setup cuda for numba

you need to add following environment variable for numba.cuda, you can add them to ~/.bashrc:

```bash
export NUMBAPRO_CUDA_DRIVER=/usr/lib/aarch64-linux-gnu/libcuda.so
export NUMBAPRO_NVVM=/usr/local/cuda/nvvm/lib64/libnvvm.so
export NUMBAPRO_LIBDEVICE=/usr/local/cuda/nvvm/libdevice
```

### 5. add Lidardet to PYTHONPATH

## Usage

1. Generating tensorrt model script
```bash
cd ./Lidardet

python3 libs/tools/convert2rt.py convert --config_path=./params/configs/pointpillars_kitti_car_xy16.yaml --weights_file=./params/weights/path/to/your.ckpt --trt_path=/home/hova/Lidardet/params/TensorRT/XXX
```

2. Inference   
```bash
python script.py --weights_file=/home/hova/Lidardet/params/weights/pointpillars/PointPillars.tckpt --config_path=/home/hova/Lidardet/params/configs/pointpillars_kitti_car_xy16.yaml --trt_dir=/home/hova/Lidardet/params/TensorRT/XXX

```
4. rviz view 
```
rviz 
```
## Concepts


* Kitti lidar box

A kitti lidar box is consist of 7 elements: [x, y, z, w, l, h, rz], see figure.

![Kitti Box Image](https://raw.githubusercontent.com/traveller59/second.pytorch/master/images/kittibox.png)

All training and inference code use kitti box format. So we need to convert other format to KITTI format before training.

* Kitti camera box

A kitti camera box is consist of 7 elements: [x, y, z, l, h, w, ry].
