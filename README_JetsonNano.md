# Guía JetsonNano

## Uso ej
```
python track.py --source 0 --weights /yolov5/weights/yolov5s_7C.py --distance 40 --save-vid
```

## Instalar OpenCV y activar CUDA
Seguir pasos de QEngineering para OpenCV 4.5.1: https://qengineering.eu/install-opencv-4.5-on-jetson-nano.html
```bash
free -m
wget https://github.com/Qengineering/Install-OpenCV-Jetson-Nano/raw/main/OpenCV-4-5-1.sh
sudo chmod 755 ./OpenCV-4-5-1.sh
./OpenCV-4-5-1.sh
rm OpenCV-4-5-1.sh 
```

## Instalar PyTorch y TorchVision
Seguir pasos de QEngineering para PyTorch 1.8.1: https://qengineering.eu/install-pytorch-on-jetson-nano.html
```bash
sudo apt-get install python3-pip libjpeg-dev libopenblas-dev libopenmpi-dev libomp-dev
sudo -H pip3 install --upgrade pip
sudo -H pip3 install future
sudo pip3 install -U --user wheel mock pillow
sudo -H pip3 install --upgrade setuptools
sudo -H pip3 install Cython
sudo -H pip3 install gdown
gdown https://drive.google.com/uc?id=1XL6k3wfWTJVKXHvCbZSfIVdz6IDJUAkt
sudo -H pip3 install torch-1.8.1a0+56b43f4-cp36-cp36m-linux_aarch64.whl
rm torch-1.8.1a0+56b43f4-cp36-cp36m-linux_aarch64.whl
```

Seguir pasos de QEngineering para TorchVision 0.9.1
```bash
sudo apt-get install libjpeg-dev zlib1g-dev libpython3-dev
sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev
sudo pip3 install -U pillow
sudo -H pip3 install gdown
gdown https://drive.google.com/uc?id=1HYmjUrv9o2hZWVz7GpGplaKhqMPMtESL
sudo -H pip3 install torchvision-0.9.1a0+8fb5838-cp36-cp36m-linux_aarch64.whl
rm torchvision-0.9.1a0+8fb5838-cp36-cp36m-linux_aarch64.whl
sudo -H pip3 install -U protobuf
```

## Instalar TensorFlow
Seguir pasos de QEngineering para TenshorFlow 2.4.1: https://qengineering.eu/install-tensorflow-2.4.0-on-jetson-nano.html
```bash
sudo apt-get update
sudo apt-get upgrade
sudo pip uninstall tensorflow
sudo pip3 uninstall tensorflow
sudo apt-get install gfortran
sudo apt-get install libhdf5-dev libc-ares-dev libeigen3-dev
sudo apt-get install libatlas-base-dev libopenblas-dev libblas-dev
sudo apt-get install liblapack-dev
sudo -H pip3 install Cython==0.29.21
sudo -H pip3 install h5py==2.10.0
sudo -H pip3 install -U testresources numpy
sudo -H pip3 install --upgrade setuptools
sudo -H pip3 install pybind11 protobuf google-pasta
sudo -H pip3 install -U six mock wheel requests gast
sudo -H pip3 install keras_applications --no-deps
sudo -H pip3 install keras_preprocessing --no-deps
pip3 install gdown
gdown https://drive.google.com/uc?id=1DLk4Tjs8Mjg919NkDnYg02zEnbbCAzOz
sudo -H pip3 install tensorflow-2.4.1-cp36-cp36m-linux_aarch64.whl
```

## Instalar otras librerías

### Instalar dependencias necesarias
```bash
sudo cp /etc/apt/sources.list /etc/apt/sources.list~
sudo sed -Ei 's/^# deb-src /deb-src /' /etc/apt/sources.list
sudo apt-get update
sudo apt-get build-dep python3-matplotlib -y
sudo mv /etc/apt/sources.list~ /etc/apt/sources.list
sudo apt-get update
```

### Instalar Matplotlib
```bash
cd ~/.local/lib/python3.6/site-packages
git clone -b v3.3.4 --depth 1 https://github.com/matplotlib/matplotlib.git
cd ~/.local/lib/python3.6/site-packages/matplotlib
sudo -H pip3 install . -v
```

### Instalar Pandas
```bash
cd ~/.local/lib/python3.6/site-packages
git clone -b v1.1.5 --depth 1 https://github.com/pandas-dev/pandas.git
cd ~/.local/lib/python3.6/site-packages/matplotlib
sudo -H pip3 install . -v
```

## Instalar otras librerías
````bash
sudo -H pip3 install seaborn
```
