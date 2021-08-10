#!/usr/bin/env bash
find images -name "*.jpg" >images.txt
./darknet detector test obj.data yolov4-obj.cfg yolov4-obj_best.weights -dont_show -out result.json <images.txt
python3 detect.py
