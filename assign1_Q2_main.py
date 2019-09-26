import numpy as np
import sys

file = "../videos/black_and_white.yuv"
y_res = 288
x_res = 352
number_frames = 300
total_pixels_frame = x_res * y_res
bytes_frame = (int)(total_pixels_frame)

y_frame = np.empty((number_frames, y_res, x_res), dtype=int)

y_file = open(file,"rb")
for frame in range(number_frames) :
    raw = y_file.read(bytes_frame)
    y_frame = np.concatenate((y_frame, raw[: x_res * y_res]), axis=None)

    for y_it in range(y_res) :    
      for x_it in range(x_res):
        it = y_it * x_res + x_it
        y_frame[frame][y_it][x_it] = int.from_bytes((raw[it : it + 1]), byteorder=sys.byteorder)      
y_file.close()


    


