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

  # in_file_1 = "./videos/black_and_white.yuv"
  # y_res = 288
  # x_res = 352
  # number_frames = 10

  # in_file_1 = "./videos/hall_qcif_bw.yuv"
  # y_res = 144
  # x_res = 176
  # number_frames = 10

  in_file_1 = "./videos/synthetic_bw.yuv"
  y_res = 288
  x_res = 288
  number_frames = 30

  # ############################################################################
  # ######################## Experiment for varying i ##########################
  # ############################################################################
  # # i=2
  # in_file_2 = "./videos/q3_i2_decoded.yuv"
  # SAD_1 = sad_per_frame(in_file_1, in_file_2, y_res, x_res, number_frames)
  # SAD_1[:] = [x / 100000 for x in SAD_1]

  # # i=8
  # in_file_2 = "./videos/q3_i8_decoded.yuv"
  # SAD_2 = sad_per_frame(in_file_1, in_file_2, y_res, x_res, number_frames)
  # SAD_2[:] = [x / 100000 for x in SAD_2]

  # # i=64
  # in_file_2 = "./videos/q3_i64_decoded.yuv"
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
  # ######################## Experiment for A2 ##########################
  # ############################################################################
  # n=1
  in_file_2 = "./videos/a2_plot/synthetic_nref_1.yuv"
  SAD_1 = sad_per_frame(in_file_1, in_file_2, y_res, x_res, number_frames)
  SAD_1[:] = [x / 100000 for x in SAD_1]

  in_file_2 = "./videos/a2_plot/synthetic_nref_2.yuv"
  SAD_2 = sad_per_frame(in_file_1, in_file_2, y_res, x_res, number_frames)
  SAD_2[:] = [x / 100000 for x in SAD_2]

  in_file_2 = "./videos/a2_plot/synthetic_nref_3.yuv"
  SAD_3 = sad_per_frame(in_file_1, in_file_2, y_res, x_res, number_frames)
  SAD_3[:] = [x / 100000 for x in SAD_3]

  in_file_2 = "./videos/a2_plot/synthetic_nref_4.yuv"
  SAD_4 = sad_per_frame(in_file_1, in_file_2, y_res, x_res, number_frames)
  SAD_4[:] = [x / 100000 for x in SAD_4]

  frames = range(1, number_frames+1)
  fig, ax = plt.subplots()

  ax.plot(frames, SAD_1, 'o:', label='nRef_Frame = 1')
  ax.plot(frames, SAD_2, 'v-.', label='nRef_Frame = 2')
  ax.plot(frames, SAD_3, 'D--', label='nRef_Frame = 3')
  ax.plot(frames, SAD_4, 's:', label='nRef_Frame = 4')

  plt.xticks(frames)
  ax.set(xlabel='Frame', ylabel=r'SAD $(\times 10^6)$')
  ax.grid()
  ax.legend()
  plt.show()

  # x_axis = range(1, 4+1)
  # fig, ax = plt.subplots()

  # sizes = [754747, 664479, 616103, 551989]

  # ax.plot(x_axis, sizes, 'o-')

  # plt.xticks(x_axis)
  # ax.set(xlabel='nRefFrame', ylabel='Size (Bytes)')
  # ax.grid()
  # ax.legend()
  # plt.show()
  
  


  # ############################################################################
  # ############################################################################

  # frames = range(1,11)
  # fig, ax = plt.subplots()

  # ax.plot(frames, SAD_1, label='n=1')
  # ax.plot(frames, SAD_2, label='n=2')
  # ax.plot(frames, SAD_3, label='n=3')
  
  
  
  # ax.set(xlabel='Frame', ylabel=r'SAD $(\times 10^6)$')
  # ax.grid()
  # ax.legend()

  # # fig.savefig("test.png")
  # plt.show()


  # ############################################################################
  # ############################# Experiment for Q4 ############################
  # ############################################################################

  # total_pixels_frame = x_res * y_res
  # y_frame_1 = np.empty((number_frames, y_res, x_res), dtype=int)

  # yuv_file_1 = open(in_file_1,"rb")

  # for frame in range(number_frames):
  #   raw_1 = yuv_file_1.read(total_pixels_frame)
  #   for y_it in range(y_res) :    
  #     for x_it in range(x_res):
  #       it = (y_it * x_res) + x_it
  #       y_frame_1[frame][y_it][x_it] = int.from_bytes((raw_1[it: it + 1]), byteorder=sys.byteorder)
        
  # yuv_file_1.close()

  # psnr = []
  # file_size = []
  # for ip in [1, 4, 10]:
  #   psnr_temp = []
  #   file_size_temp = []
  #   for QP in range(0, 11):
  #     in_file_2 = "./videos/report/q4/i8/q4_ip" + str(ip) + "_qp" + str(QP) + "_decoded.yuv"
  #     psnr_temp += [PSNR(y_frame_1, in_file_2, y_res, x_res, number_frames)]

  #     encoded_in_file_2 = "./videos/report/q4/i8/q4_ip" + str(ip) + "_qp" + str(QP) + "_encoded.far"
  #     file_size_temp += [os.path.getsize(encoded_in_file_2) * 8]

  #   psnr += [psnr_temp]
  #   file_size += [file_size_temp]

  # for ip in range(3):
  #   print("I_Period:", ip)
  #   for QP in range(0, 11):
  #     print(psnr[ip][QP], file_size[ip][QP])




  # # ############################################################################
  # # ############################################################################


  # fig, ax = plt.subplots()

  # ax.plot(file_size[0], psnr[0], label='I_Period=1')
  # ax.plot(file_size[1], psnr[1], label='I_Period=4')
  # ax.plot(file_size[2], psnr[2], label='I_Period=10')
  
  
  
  # ax.set(xlabel='Size in bits', ylabel='PSNR (dB)')
  # ax.grid()
  # ax.legend()

  # # fig.savefig("test.png")
  # plt.show()