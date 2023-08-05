# Training + Deployment on Intel dGPU

A deep learning pipeline consists of model training with Intel® Extension for PyTorch (IPEX), and model optimization and deployment with OpenVINO™ for YOLOv7 on Intel® discrete GPU (dGPU).

![Image_text](https://github.com/zhuo-yoyowz/classification/blob/24c62a825b84fcabe53671c718780178a48c48c5/DL_pipeline.jpg)

## Environment set up for model training

After successfully installed dGPU driver on your machine, we could start setting up the environment for model training with the following installations.

### Install required library for training on dGPU with IPEX

```bash
wget -qO - https://repositories.intel.com/graphics/intel-graphics.key | sudo gpg --dearmor --output /usr/share/keyrings/intel-graphics.gpg
echo "deb [arch=amd64,i386 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/graphics/ubuntu jammy arc" | sudo tee /etc/apt/sources.list.d/intel-gpu-jammy.list

sudo apt-get install -y \
  intel-opencl-icd intel-level-zero-gpu level-zero \
  intel-media-va-driver-non-free libmfx1 libmfxgen1 libvpl2 \
  libegl-mesa0 libegl1-mesa libegl1-mesa-dev libgbm1 libgl1-mesa-dev libgl1-mesa-dri \
  libglapi-mesa libgles2-mesa-dev libglx-mesa0 libigdgmm12 libxatracker2 mesa-va-drivers \
  mesa-vdpau-drivers mesa-vulkan-drivers va-driver-all vainfo hwinfo clinfo
```

After installing the GPU driver and the required library, we then install Intel® oneAPI Base Toolkit and IPEX, which will be used to perform training on Intel® dGPU. 

### Install Intel® oneAPI Base Toolkit 2023.1

```bash
sudo ls
wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor | sudo tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null
echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | sudo tee /etc/apt/sources.list.d/oneAPI.list
sudo apt update
sudo apt install intel-basekit
```

### Install Intel® Extension for PyTorch (IPEX)

```bash
# Intel® oneAPI Base Toolkit 2023.1 is installed to /opt/intel/oneapi/
export ONEAPI_ROOT=/opt/intel/oneapi

# A DPC++ compiler patch is required to use with oneAPI Basekit 2023.1.0. Use the command below to download the patch package.

wget https://registrationcenter-download.intel.com/akdlm/IRC_NAS/89283df8-c667-47b0-b7e1-c4573e37bd3e/2023.1-linux-hotfix.zip
unzip 2023.1-linux-hotfix.zip
cd 2023.1-linux-hotfix
source ${ONEAPI_ROOT}/setvars.sh
sudo -E bash installpatch.sh

sudo apt install python3-venv
cd
python3 -m venv ipex
source ipex/bin/activate
python -m pip install torch==1.13.0a0+git6c9b55e torchvision==0.14.1a0 intel_extension_for_pytorch==1.13.120+xpu -f https://developer.intel.com/ipex-whl-stable-xpu
```

After successful installation, you can run IPEX. Remember activation of oneAPI environment is required every time you open a new terminal, using the following command:
```bash
source /opt/intel/oneapi/setvars.sh
source ipex/bin/activate
```

### Install XPU manager for obtaining GPU running information
We could use XPU Manager to get GPU power, frequency, GPU memory used, compute engine %, copy engine % and throttle reason. Installation uses the following command:

```bash
wget -c https://github.com/intel/xpumanager/releases/download/V1.2.13/xpumanager_1.2.13_20230629.055631.aeeedfec.u22.04_amd64.deb
sudo apt install intel-gsc libmetee
sudo dpkg -i xpumanager_1.2.13_20230629.055631.aeeedfec.u22.04_amd64.deb
```
You could use the following command to observe GPU related info with
```bash
xpumcli dump -d 0 -m 1,2,18,22,26,35
```

Now we’ve set up the environment for model training on dGPU. Next steps show how to train a YOLOv7 model with custom data.
## Train YOLOv7 on custom data

### 1)	Download custom dataset
Download the pothole dataset.

```bash
wget https://learnopencv.s3.us-west-2.amazonaws.com/pothole_dataset.zip
unzip -q pothole_dataset.zip
```

### 2)	Clone YOLOv7 repository from GitHub

```bash
git clone https://github.com/WongKinYiu/yolov7.git
cd yolov7
pip install -r requirements.txt
```

### 3) Download yolov7-tiny model and add custom data and model configuration files for training

```bash
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt
```

Move the pothole.yaml file into "yolov7/data" folder
```bash
mv ../training-deployment-dGPU/pothole.yaml ./data
```

Move the pothole-tiny.yaml into "yolov7/cfg/tranining" folder 
```bash
mv ../training-deployment-dGPU/yolov7_pothole-tiny.yaml ./cfg/training
```

Make the xpu.patch file effective using the following
```bash
patch -p1 < ../training-deployment-dGPU/yolov7_xpu.patch
```

### 4) Perform model training

```bash
python train.py --epochs 50 --workers 4 --device xpu --batch-size 32 --data data/pothole.yaml --img 640 640 --cfg cfg/training/yolov7_pothole-tiny.yaml --weights 'yolov7-tiny.pt' --name yolov7_tiny_pothole_fixed_res --hyp data/hyp.scratch.tiny.yaml
```

After training is done, model weights with the best accuracy will be saved at "runs/train/yolov7_tiny_pothole_fixed_res/weights/best.pt".

## Deploy trained YOLOv7 model with OpenVINO
### 1)	Check model inference from the trained model

```bash
python -W ignore detect.py --weights ./runs/train/yolov7_tiny_pothole_fixed_res/weights/best.pt --conf 0.25 --img-size 640 --source test.jpg
```

The test result is saved at 'runs/detect/exp/test.jpg'.


### 2) Export model to ONNX
```bash
python -W ignore export.py --weights ./runs/train/yolov7_tiny_pothole_fixed_res/weights/best.pt --grid
```

### 3) Convert to OpenVINO IR format and run inference with OpenVINO runtime on dGPU
```bash
python ../training-deployment-dGPU/run_openvino_inference.py
```
The final inference result is like this

![Image_text](https://github.com/zhuo-yoyowz/classification/blob/master/G0026953.jpg)

