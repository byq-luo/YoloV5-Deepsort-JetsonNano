#!/bin/bash
echo perceptron | sudo -S python3 track.py --source 0 --weights yolov5/weights/yolov5s_7C.pt --distance 40 --show-vid
