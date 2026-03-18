### Introduction

from InternVLA : https://github.com/InternRobotics/InternNav  


### Installation

```bash

git clone https://github.com/InternRobotics/InternNav.git --recursive
conda create -n internnav python=3.10
conda activate internnav

pip install -U pip setuptools==80.9.0 wheel packaging ninja
pip install -U huggingface_hub

pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

MAX_JOBS=4 pip install flash-attn==2.7.2.post1 --no-build-isolation
pip install numpy-quaternion

cd InternNav
pip install -e .[model] --no-build-isolation

# 로그인해야 빠름
huggingface-cli login

pip install setuptools==80.9.0

mkdir checkpoints
huggingface-cli download InternRobotics/InternVLA-N1-DualVLN --local-dir checkpoints/InternVLA-N1-DualVLN

# 긴 명령어 주의
curl -L https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Small/resolve/main/depth_anything_v2_metric_hypersim_vits.pth -o checkpoints/depth_anything_v2_metric_hypersim_vits.pth
```

### Check the Installation

[서버실행]  
/workspace/internVLA/InternNav$ 이 디렉토리에서  
export PYTHONWARNINGS="ignore::UserWarning"  
python3 scripts/eval/start_server.py --port 8087  

[클라이언트 실행]  
/workspace/internVLA/InternNav$ 이 디렉토리에서  

python3 test.py

### On Real World

1. install and execute realsense_ws ros  
   1-1) install the sdk first  
   ```bash
   sudo apt install git build-essential cmake libssl-dev libusb-1.0-0-dev pkg-config libgtk-3-dev libglfw3-dev libgl1-mesa-dev libglu1-mesa-dev
   git clone https://github.com/IntelRealSense/librealsense.git
   cd librealsense
   ./scripts/setup_udev_rules.sh
   mkdir build && cd build
   cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=true
   make -j$(nproc)
   sudo make install
   ```
   1-2) install realsense ros2 repo  
   follow this repo : https://github.com/realsenseai/realsense-ros
   1-3) exeute the ros2 realsense TODO

2. excute the server  
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  
CUDA_VISIBLE_DEVICES=1 python /home/tenstorrent/workspace/internVLA/InternNav/scripts/realworld/http_internvla_server.py --device cuda:0

3. execute the client  
TODO
