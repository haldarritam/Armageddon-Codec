import numpy as np
import math
import matplotlib.pyplot as plt
import sys
import os

def sad_per_frame(in_file_1, in_file_2, y_res, x_res, number_frames):
  total_pixels_frame = x_res * y_res
  y_frame_1 = np.empty((number_frames, y_res, x_res), dtype=int)
  y_frame_2 = np.empty((number_frames, y_res, x_res), dtype=int)

  yuv_file_1 = open(in_file_1,"rb")
  yuv_file_2 = open(in_file_2,"rb")

  for frame in range(number_frames):
    raw_1 = yuv_file_1.read(total_pixels_frame)
    raw_2 = yuv_file_2.read(total_pixels_frame)
    for y_it in range(y_res) :    
      for x_it in range(x_res):
        it = (y_it * x_res) + x_it
        y_frame_1[frame][y_it][x_it] = int.from_bytes((raw_1[it : it + 1]), byteorder=sys.byteorder) 
        y_frame_2[frame][y_it][x_it] = int.from_bytes((raw_2[it : it + 1]), byteorder=sys.byteorder)        
  
  yuv_file_1.close()
  yuv_file_2.close()

  SAD = []
  for frame in range(number_frames):
    SAD += [np.sum(np.abs(np.subtract(y_frame_1[frame], y_frame_2[frame], dtype=int)))]

  return SAD

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

  # ############################################################################
  # ######################## Experiment for varying i ##########################
  # ############################################################################
  # # i=2
  # in_file_2 = "./videos/q3_decoded_i2.yuv"
  # SAD_1 = sad_per_frame(in_file_1, in_file_2, y_res, x_res, number_frames)
  # SAD_1[:] = [x / 100000 for x in SAD_1]

  # # i=8
  # in_file_2 = "./videos/q3_decoded_i8.yuv"
  # SAD_2 = sad_per_frame(in_file_1, in_file_2, y_res, x_res, number_frames)
  # SAD_2[:] = [x / 100000 for x in SAD_2]

  # # i=64
  # in_file_2 = "./videos/q3_decoded_i64.yuv"
  # SAD_3 = sad_per_frame(in_file_1, in_file_2, y_res, x_res, number_frames)
  # SAD_3[:] = [x / 100000 for x in SAD_3]
  # ############################################################################
  # ############################################################################

  # ############################################################################
  # ######################## Experiment for varying r ##########################
  # ############################################################################
  # # r=1
  # in_file_2 = "./videos/q3_r1_decoded.yuv"
  # SAD_1 = sad_per_frame(in_file_1, in_file_2, y_res, x_res, number_frames)
  # SAD_1[:] = [x / 100000 for x in SAD_1]

  # # r=4
  # in_file_2 = "./videos/q3_r4_decoded.yuv"
  # SAD_2 = sad_per_frame(in_file_1, in_file_2, y_res, x_res, number_frames)
  # SAD_2[:] = [x / 100000 for x in SAD_2]

  # # r=8
  # in_file_2 = "./videos/q3_r8_decoded.yuv"
  # SAD_3 = sad_per_frame(in_file_1, in_file_2, y_res, x_res, number_frames)
  # SAD_3[:] = [x / 100000 for x in SAD_3]
  # ############################################################################
  # ############################################################################

  # ############################################################################
  # ######################## Experiment for varying n ##########################
  # ############################################################################
  # # n=1
  # in_file_2 = "./videos/q3_n1_decoded.yuv"
  # SAD_1 = sad_per_frame(in_file_1, in_file_2, y_res, x_res, number_frames)
  # SAD_1[:] = [x / 100000 for x in SAD_1]

  # # n=2
  # in_file_2 = "./videos/q3_n2_decoded.yuv"
  # SAD_2 = sad_per_frame(in_file_1, in_file_2, y_res, x_res, number_frames)
  # SAD_2[:] = [x / 100000 for x in SAD_2]

  # # n=3
  # in_file_2 = "./videos/q3_n3_decoded.yuv"
  # SAD_3 = sad_per_frame(in_file_1, in_file_2, y_res, x_res, number_frames)
  # SAD_3[:] = [x / 100000 for x in SAD_3]
  # ############################################################################
  # ############################################################################

  # ############################################################################
  # ######################## Experiment for varying n ##########################
  # ############################################################################

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
  for ip in [1, 4, 10]:
    psnr_temp = []
    file_size_temp = []
    for QP in range(0, 12):
      in_file_2 = "./videos/report/q4/i16/q4_ip" + str(ip) + "_qp" + str(QP) + "_decoded.yuv"
      psnr_temp += [PSNR(y_frame_1, in_file_2, y_res, x_res, number_frames)]

      encoded_in_file_2 = "./videos/report/q4/i16/q4_ip" + str(ip) + "_qp" + str(QP) + "_encoded.far"
      file_size_temp += [os.path.getsize(encoded_in_file_2) * 8]

    psnr += [psnr_temp]
    file_size += [file_size_temp]


  # ############################################################################
  # ############################################################################


  fig, ax = plt.subplots()

  ax.plot(file_size[0], psnr[0], label='I_Period=1')
  ax.plot(file_size[1], psnr[1], label='I_Period=4')
  ax.plot(file_size[2], psnr[2], label='I_Period=10')
  
  
  
  ax.set(xlabel='Size in bits', ylabel='PSNR (dB)')
  ax.grid()
  ax.legend()

  # fig.savefig("test.png")
  plt.show()