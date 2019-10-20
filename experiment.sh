#!/usr/bin/env bash

# python main.py -in ./videos/black_and_white.yuv -out ./videos/q3_r1.yuv -y_res 288 -x_res 352 -frames 10 -o both -q 3 -i 8 -n 3 -r 1 &

# python main.py -in ./videos/black_and_white.yuv -out ./videos/q3_r4.yuv -y_res 288 -x_res 352 -frames 10 -o both -q 3 -i 8 -n 3 -r 4 &

# python main.py -in ./videos/black_and_white.yuv -out ./videos/q3_r8.yuv -y_res 288 -x_res 352 -frames 10 -o both -q 3 -i 8 -n 3 -r 8 &


python main.py -in ./videos/black_and_white.yuv -out ./videos/q3_n1.yuv -y_res 288 -x_res 352 -frames 10 -o both -q 3 -i 8 -n 1 -r 4 &

python main.py -in ./videos/black_and_white.yuv -out ./videos/q3_n2.yuv -y_res 288 -x_res 352 -frames 10 -o both -q 3 -i 8 -n 2 -r 4 &

python main.py -in ./videos/black_and_white.yuv -out ./videos/q3_n3.yuv -y_res 288 -x_res 352 -frames 10 -o both -q 3 -i 8 -n 3 -r 4 &