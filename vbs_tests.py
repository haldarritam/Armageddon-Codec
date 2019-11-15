import numpy as np
import math
import matplotlib.pyplot as plt
import sys
import os

def PSNR(y_frame_1, in_file_2, y_res, x_res, number_frames):
  total_pixels_frame = x_res * y_res
  y_frame_2 = np.empty((number_frames, y_res, x_res), dtype=int)

  yuv_file_2 = open(in_file_2,"rb")

  for frame in range(number_frames):
    raw_2 = yuv_file_2.read(total_pixels_frame)
    for y_it in range(y_res) :    
      for x_it in range(x_res):
        it = (y_it * x_res) + x_it
        y_frame_2[frame][y_it][x_it] = int.from_bytes((raw_2[it: it + 1]), byteorder=sys.byteorder)
        
  yuv_file_2.close()

  # PSNR
  mse = np.mean( (y_frame_1 - y_frame_2) ** 2 )
  if mse == 0:
    mse += 0.000000000001
  PIXEL_MAX = 255.0
  psnr = 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

  return psnr

if __name__ == "__main__":

  font = {'family' : 'normal',
        # 'weight' : 'bold',
        'size'   : 16}

  plt.rc('font', **font)

  in_file_1 = "./videos/black_and_white.yuv"
  y_res = 288
  x_res = 352
  number_frames = 10

  total_pixels_frame = x_res * y_res
  y_frame_1 = np.empty((number_frames, y_res, x_res), dtype=int)

  yuv_file_1 = open(in_file_1,"rb")

  for frame in range(number_frames):
    raw_1 = yuv_file_1.read(total_pixels_frame)
    for y_it in range(y_res) :    
      for x_it in range(x_res):
        it = (y_it * x_res) + x_it
        y_frame_1[frame][y_it][x_it] = int.from_bytes((raw_1[it: it + 1]), byteorder=sys.byteorder)
        
  yuv_file_1.close()

  psnr = []
  file_size = []

  psnr_temp = []
  file_size_temp = []

  in_file_2 = "./temp/assign2_base_QP0.yuv"
  psnr_temp += [PSNR(y_frame_1, in_file_2, y_res, x_res, number_frames)]
  psnr += [psnr_temp]
  encoded_in_file_2 = "./temp/assign2_base_QP0.far"
  file_size_temp += [os.path.getsize(encoded_in_file_2) * 8]
  file_size += [file_size_temp]

  psnr_temp = []
  file_size_temp = []

  in_file_2 = "./temp/assign2_vbs_QP0.yuv"
  psnr_temp += [PSNR(y_frame_1, in_file_2, y_res, x_res, number_frames)]
  psnr += [psnr_temp]
  encoded_in_file_2 = "./temp/assign2_vbs_QP0.far"
  file_size_temp += [os.path.getsize(encoded_in_file_2) * 8]
  file_size += [file_size_temp]
  
  psnr_temp = []
  file_size_temp = []

  in_file_2 = "./temp/assign2_base.yuv"
  psnr_temp += [PSNR(y_frame_1, in_file_2, y_res, x_res, number_frames)]
  psnr += [psnr_temp]
  encoded_in_file_2 = "./temp/assign2_base.far"
  file_size_temp += [os.path.getsize(encoded_in_file_2) * 8]
  file_size += [file_size_temp]

  psnr_temp = []
  file_size_temp = []

  in_file_2 = "./temp/assign2_vbs.yuv"
  psnr_temp += [PSNR(y_frame_1, in_file_2, y_res, x_res, number_frames)]
  psnr += [psnr_temp]
  encoded_in_file_2 = "./temp/assign2_vbs.far"
  file_size_temp += [os.path.getsize(encoded_in_file_2) * 8]
  file_size += [file_size_temp]


  fig, ax = plt.subplots()

  ax.plot(file_size[0], psnr[0], 'x', label='Base QP0')
  ax.plot(file_size[1], psnr[1], 'x', label='VBS QP0')
  ax.plot(file_size[2], psnr[2], 'x', label='Base QP3')
  ax.plot(file_size[3], psnr[3], 'x', label='VBS QP3')
  
  
  
  ax.set(xlabel='Size in bits', ylabel='PSNR (dB)')
  ax.grid()
  ax.legend()

  # fig.savefig("test.png")
  plt.show()