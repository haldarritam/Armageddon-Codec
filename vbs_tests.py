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

def PSNR_per_frame(y_frame_1, in_file_2, y_res, x_res, number_frames):
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

  psnr = []
  # PSNR
  for frame in range(number_frames):
    mse = np.mean( (y_frame_1[frame] - y_frame_2[frame]) ** 2 )
    if mse == 0:
      mse += 0.000000000001
    PIXEL_MAX = 255.0
    psnr += [20 * math.log10(PIXEL_MAX / math.sqrt(mse))]

  return psnr

if __name__ == "__main__":

  font = {'family' : 'normal',
        # 'weight' : 'bold',
        'size'   : 16}

  plt.rc('font', **font)

  in_file_1 = "./videos/CIF_bw.yuv"
  y_res = 288
  x_res = 352
  number_frames = 21

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

  # # ---------------------------------------------------------------------------
  # psnr_temp = []
  # file_size_temp = []

  # in_file_2 = "./videos/tests/QP1-1000_encoded.yuv"
  # psnr_temp += [PSNR(y_frame_1, in_file_2, y_res, x_res, number_frames)]
  # encoded_in_file_2 = "./videos/tests/QP1-1000_encoded.far"
  # file_size_temp += [os.path.getsize(encoded_in_file_2) * 8]

  # in_file_2 = "./videos/tests/QP4-1000_encoded.yuv"
  # psnr_temp += [PSNR(y_frame_1, in_file_2, y_res, x_res, number_frames)]
  # encoded_in_file_2 = "./videos/tests/QP4-1000_encoded.far"
  # file_size_temp += [os.path.getsize(encoded_in_file_2) * 8]

  # in_file_2 = "./videos/tests/QP7-1000_encoded.yuv"
  # psnr_temp += [PSNR(y_frame_1, in_file_2, y_res, x_res, number_frames)]
  # encoded_in_file_2 = "./videos/tests/QP7-1000_encoded.far"
  # file_size_temp += [os.path.getsize(encoded_in_file_2) * 8]

  # in_file_2 = "./videos/tests/QP10-1000_encoded.yuv"
  # psnr_temp += [PSNR(y_frame_1, in_file_2, y_res, x_res, number_frames)]
  # encoded_in_file_2 = "./videos/tests/QP10-1000_encoded.far"
  # file_size_temp += [os.path.getsize(encoded_in_file_2) * 8]

  # psnr += [psnr_temp]
  # file_size += [file_size_temp]

  # # ---------------------------------------------------------------------------
  # # ---------------------------------------------------------------------------
  # psnr_temp = []
  # file_size_temp = []

  # in_file_2 = "./videos/tests/QP1-1100_encoded.yuv"
  # psnr_temp += [PSNR(y_frame_1, in_file_2, y_res, x_res, number_frames)]
  # encoded_in_file_2 = "./videos/tests/QP1-1100_encoded.far"
  # file_size_temp += [os.path.getsize(encoded_in_file_2) * 8]

  # in_file_2 = "./videos/tests/QP4-1100_encoded.yuv"
  # psnr_temp += [PSNR(y_frame_1, in_file_2, y_res, x_res, number_frames)]
  # encoded_in_file_2 = "./videos/tests/QP4-1100_encoded.far"
  # file_size_temp += [os.path.getsize(encoded_in_file_2) * 8]

  # in_file_2 = "./videos/tests/QP7-1100_encoded.yuv"
  # psnr_temp += [PSNR(y_frame_1, in_file_2, y_res, x_res, number_frames)]
  # encoded_in_file_2 = "./videos/tests/QP7-1100_encoded.far"
  # file_size_temp += [os.path.getsize(encoded_in_file_2) * 8]

  # in_file_2 = "./videos/tests/QP10-1100_encoded.yuv"
  # psnr_temp += [PSNR(y_frame_1, in_file_2, y_res, x_res, number_frames)]
  # encoded_in_file_2 = "./videos/tests/QP10-1100_encoded.far"
  # file_size_temp += [os.path.getsize(encoded_in_file_2) * 8]

  # psnr += [psnr_temp]
  # file_size += [file_size_temp]

  # # ---------------------------------------------------------------------------
  # # ---------------------------------------------------------------------------
  # psnr_temp = []
  # file_size_temp = []

  # in_file_2 = "./videos/tests/QP1-1010_encoded.yuv"
  # psnr_temp += [PSNR(y_frame_1, in_file_2, y_res, x_res, number_frames)]
  # encoded_in_file_2 = "./videos/tests/QP1-1010_encoded.far"
  # file_size_temp += [os.path.getsize(encoded_in_file_2) * 8]

  # in_file_2 = "./videos/tests/QP4-1010_encoded.yuv"
  # psnr_temp += [PSNR(y_frame_1, in_file_2, y_res, x_res, number_frames)]
  # encoded_in_file_2 = "./videos/tests/QP4-1010_encoded.far"
  # file_size_temp += [os.path.getsize(encoded_in_file_2) * 8]

  # in_file_2 = "./videos/tests/QP7-1010_encoded.yuv"
  # psnr_temp += [PSNR(y_frame_1, in_file_2, y_res, x_res, number_frames)]
  # encoded_in_file_2 = "./videos/tests/QP7-1010_encoded.far"
  # file_size_temp += [os.path.getsize(encoded_in_file_2) * 8]

  # in_file_2 = "./videos/tests/QP10-1010_encoded.yuv"
  # psnr_temp += [PSNR(y_frame_1, in_file_2, y_res, x_res, number_frames)]
  # encoded_in_file_2 = "./videos/tests/QP10-1010_encoded.far"
  # file_size_temp += [os.path.getsize(encoded_in_file_2) * 8]

  # psnr += [psnr_temp]
  # file_size += [file_size_temp]

  # # ---------------------------------------------------------------------------
  # # ---------------------------------------------------------------------------
  # psnr_temp = []
  # file_size_temp = []

  # in_file_2 = "./videos/tests/QP1-1001_encoded.yuv"
  # psnr_temp += [PSNR(y_frame_1, in_file_2, y_res, x_res, number_frames)]
  # encoded_in_file_2 = "./videos/tests/QP1-1001_encoded.far"
  # file_size_temp += [os.path.getsize(encoded_in_file_2) * 8]

  # in_file_2 = "./videos/tests/QP4-1001_encoded.yuv"
  # psnr_temp += [PSNR(y_frame_1, in_file_2, y_res, x_res, number_frames)]
  # encoded_in_file_2 = "./videos/tests/QP4-1001_encoded.far"
  # file_size_temp += [os.path.getsize(encoded_in_file_2) * 8]

  # in_file_2 = "./videos/tests/QP7-1001_encoded.yuv"
  # psnr_temp += [PSNR(y_frame_1, in_file_2, y_res, x_res, number_frames)]
  # encoded_in_file_2 = "./videos/tests/QP7-1001_encoded.far"
  # file_size_temp += [os.path.getsize(encoded_in_file_2) * 8]

  # in_file_2 = "./videos/tests/QP10-1001_encoded.yuv"
  # psnr_temp += [PSNR(y_frame_1, in_file_2, y_res, x_res, number_frames)]
  # encoded_in_file_2 = "./videos/tests/QP10-1001_encoded.far"
  # file_size_temp += [os.path.getsize(encoded_in_file_2) * 8]

  # psnr += [psnr_temp]
  # file_size += [file_size_temp]

  # # ---------------------------------------------------------------------------
  # # ---------------------------------------------------------------------------
  # psnr_temp = []
  # file_size_temp = []

  # in_file_2 = "./videos/tests/QP1-2000_encoded.yuv"
  # psnr_temp += [PSNR(y_frame_1, in_file_2, y_res, x_res, number_frames)]
  # encoded_in_file_2 = "./videos/tests/QP1-2000_encoded.far"
  # file_size_temp += [os.path.getsize(encoded_in_file_2) * 8]

  # in_file_2 = "./videos/tests/QP4-2000_encoded.yuv"
  # psnr_temp += [PSNR(y_frame_1, in_file_2, y_res, x_res, number_frames)]
  # encoded_in_file_2 = "./videos/tests/QP4-2000_encoded.far"
  # file_size_temp += [os.path.getsize(encoded_in_file_2) * 8]

  # in_file_2 = "./videos/tests/QP7-2000_encoded.yuv"
  # psnr_temp += [PSNR(y_frame_1, in_file_2, y_res, x_res, number_frames)]
  # encoded_in_file_2 = "./videos/tests/QP7-2000_encoded.far"
  # file_size_temp += [os.path.getsize(encoded_in_file_2) * 8]

  # in_file_2 = "./videos/tests/QP10-2000_encoded.yuv"
  # psnr_temp += [PSNR(y_frame_1, in_file_2, y_res, x_res, number_frames)]
  # encoded_in_file_2 = "./videos/tests/QP10-2000_encoded.far"
  # file_size_temp += [os.path.getsize(encoded_in_file_2) * 8]

  # psnr += [psnr_temp]
  # file_size += [file_size_temp]

  # # ---------------------------------------------------------------------------
  # # ---------------------------------------------------------------------------
  # psnr_temp = []
  # file_size_temp = []

  # in_file_2 = "./videos/tests/QP1-2111_encoded.yuv"
  # psnr_temp += [PSNR(y_frame_1, in_file_2, y_res, x_res, number_frames)]
  # encoded_in_file_2 = "./videos/tests/QP1-2111_encoded.far"
  # file_size_temp += [os.path.getsize(encoded_in_file_2) * 8]

  # in_file_2 = "./videos/tests/QP4-2111_encoded.yuv"
  # psnr_temp += [PSNR(y_frame_1, in_file_2, y_res, x_res, number_frames)]
  # encoded_in_file_2 = "./videos/tests/QP4-2111_encoded.far"
  # file_size_temp += [os.path.getsize(encoded_in_file_2) * 8]

  # in_file_2 = "./videos/tests/QP7-2111_encoded.yuv"
  # psnr_temp += [PSNR(y_frame_1, in_file_2, y_res, x_res, number_frames)]
  # encoded_in_file_2 = "./videos/tests/QP7-2111_encoded.far"
  # file_size_temp += [os.path.getsize(encoded_in_file_2) * 8]

  # in_file_2 = "./videos/tests/QP10-2111_encoded.yuv"
  # psnr_temp += [PSNR(y_frame_1, in_file_2, y_res, x_res, number_frames)]
  # encoded_in_file_2 = "./videos/tests/QP10-2111_encoded.far"
  # file_size_temp += [os.path.getsize(encoded_in_file_2) * 8]

  # psnr += [psnr_temp]
  # file_size += [file_size_temp]

  # # ---------------------------------------------------------------------------
  # ---------------------------------------------------------------------------
  # psnr_temp = []
  # file_size_temp = []

  # in_file_2 = "./videos/tests/CIF_RC0_QP0_encoded.yuv"
  # psnr_temp += [PSNR(y_frame_1, in_file_2, y_res, x_res, number_frames)]
  # encoded_in_file_2 = "./videos/tests/CIF_RC0_QP0_encoded.far"
  # file_size_temp += [os.path.getsize(encoded_in_file_2) * 8]

  # in_file_2 = "./videos/tests/CIF_RC0_QP1_encoded.yuv"
  # psnr_temp += [PSNR(y_frame_1, in_file_2, y_res, x_res, number_frames)]
  # encoded_in_file_2 = "./videos/tests/CIF_RC0_QP1_encoded.far"
  # file_size_temp += [os.path.getsize(encoded_in_file_2) * 8]

  # in_file_2 = "./videos/tests/CIF_RC0_QP2_encoded.yuv"
  # psnr_temp += [PSNR(y_frame_1, in_file_2, y_res, x_res, number_frames)]
  # encoded_in_file_2 = "./videos/tests/CIF_RC0_QP2_encoded.far"
  # file_size_temp += [os.path.getsize(encoded_in_file_2) * 8]

  # in_file_2 = "./videos/tests/CIF_RC0_QP3_encoded.yuv"
  # psnr_temp += [PSNR(y_frame_1, in_file_2, y_res, x_res, number_frames)]
  # encoded_in_file_2 = "./videos/tests/CIF_RC0_QP3_encoded.far"
  # file_size_temp += [os.path.getsize(encoded_in_file_2) * 8]

  # in_file_2 = "./videos/tests/CIF_RC0_QP6_encoded.yuv"
  # psnr_temp += [PSNR(y_frame_1, in_file_2, y_res, x_res, number_frames)]
  # encoded_in_file_2 = "./videos/tests/CIF_RC0_QP6_encoded.far"
  # file_size_temp += [os.path.getsize(encoded_in_file_2) * 8]

  # in_file_2 = "./videos/tests/CIF_RC0_QP9_encoded.yuv"
  # psnr_temp += [PSNR(y_frame_1, in_file_2, y_res, x_res, number_frames)]
  # encoded_in_file_2 = "./videos/tests/CIF_RC0_QP9_encoded.far"
  # file_size_temp += [os.path.getsize(encoded_in_file_2) * 8]

  # in_file_2 = "./videos/tests/CIF_RC0_QP11_encoded.yuv"
  # psnr_temp += [PSNR(y_frame_1, in_file_2, y_res, x_res, number_frames)]
  # encoded_in_file_2 = "./videos/tests/CIF_RC0_QP11_encoded.far"
  # file_size_temp += [os.path.getsize(encoded_in_file_2) * 8]

  # psnr += [psnr_temp]
  # file_size += [file_size_temp]
  # # ---------------------------------------------------------------------------
  # # ---------------------------------------------------------------------------
  # psnr_temp = []
  # file_size_temp = []

  # in_file_2 = "./videos/tests/CIF_RC1_BR7m_encoded.yuv"
  # psnr_temp += [PSNR(y_frame_1, in_file_2, y_res, x_res, number_frames)]
  # encoded_in_file_2 = "./videos/tests/CIF_RC1_BR7m_encoded.far"
  # file_size_temp += [os.path.getsize(encoded_in_file_2) * 8]

  # in_file_2 = "./videos/tests/CIF_RC1_BR2_4m_encoded.yuv"
  # psnr_temp += [PSNR(y_frame_1, in_file_2, y_res, x_res, number_frames)]
  # encoded_in_file_2 = "./videos/tests/CIF_RC1_BR2_4m_encoded.far"
  # file_size_temp += [os.path.getsize(encoded_in_file_2) * 8]

  # in_file_2 = "./videos/tests/CIF_RC1_BR360k_encoded.yuv"
  # psnr_temp += [PSNR(y_frame_1, in_file_2, y_res, x_res, number_frames)]
  # encoded_in_file_2 = "./videos/tests/CIF_RC1_BR360k_encoded.far"
  # file_size_temp += [os.path.getsize(encoded_in_file_2) * 8]

  # psnr += [psnr_temp]
  # file_size += [file_size_temp]
  # # ---------------------------------------------------------------------------
  # # ---------------------------------------------------------------------------
  # psnr_temp = []
  # file_size_temp = []

  # in_file_2 = "./videos/tests/CIF_RC2_BR7m_encoded.yuv"
  # psnr_temp += [PSNR(y_frame_1, in_file_2, y_res, x_res, number_frames)]
  # encoded_in_file_2 = "./videos/tests/CIF_RC2_BR7m_encoded.far"
  # file_size_temp += [os.path.getsize(encoded_in_file_2) * 8]

  # in_file_2 = "./videos/tests/CIF_RC2_BR2_4m_encoded.yuv"
  # psnr_temp += [PSNR(y_frame_1, in_file_2, y_res, x_res, number_frames)]
  # encoded_in_file_2 = "./videos/tests/CIF_RC2_BR2_4m_encoded.far"
  # file_size_temp += [os.path.getsize(encoded_in_file_2) * 8]

  # in_file_2 = "./videos/tests/CIF_RC2_BR360k_encoded.yuv"
  # psnr_temp += [PSNR(y_frame_1, in_file_2, y_res, x_res, number_frames)]
  # encoded_in_file_2 = "./videos/tests/CIF_RC2_BR360k_encoded.far"
  # file_size_temp += [os.path.getsize(encoded_in_file_2) * 8]

  # psnr += [psnr_temp]
  # file_size += [file_size_temp]
  # # ---------------------------------------------------------------------------
  # # ---------------------------------------------------------------------------
  # psnr_temp = []
  # file_size_temp = []

  # in_file_2 = "./videos/tests/CIF_RC3_BR7m_encoded.yuv"
  # psnr_temp += [PSNR(y_frame_1, in_file_2, y_res, x_res, number_frames)]
  # encoded_in_file_2 = "./videos/tests/CIF_RC3_BR7m_encoded.far"
  # file_size_temp += [os.path.getsize(encoded_in_file_2) * 8]

  # in_file_2 = "./videos/tests/CIF_RC3_BR2_4m_encoded.yuv"
  # psnr_temp += [PSNR(y_frame_1, in_file_2, y_res, x_res, number_frames)]
  # encoded_in_file_2 = "./videos/tests/CIF_RC3_BR2_4m_encoded.far"
  # file_size_temp += [os.path.getsize(encoded_in_file_2) * 8]

  # in_file_2 = "./videos/tests/CIF_RC3_BR360k_encoded.yuv"
  # psnr_temp += [PSNR(y_frame_1, in_file_2, y_res, x_res, number_frames)]
  # encoded_in_file_2 = "./videos/tests/CIF_RC3_BR360k_encoded.far"
  # file_size_temp += [os.path.getsize(encoded_in_file_2) * 8]

  # psnr += [psnr_temp]
  # file_size += [file_size_temp]
  # # ---------------------------------------------------------------------------
  # ---------------------------------------------------------------------------
  # ---------------------------------------------------------------------------



  psnr_temp = []
  frame_numbers = list(range(1, 22))

  in_file_2 = "./videos/tests/CIF_RC1_BR2m_encoded.yuv"
  psnr += [PSNR_per_frame(y_frame_1, in_file_2, y_res, x_res, number_frames)]

  in_file_2 = "./videos/tests/CIF_RC2_BR2m_encoded.yuv"
  psnr += [PSNR_per_frame(y_frame_1, in_file_2, y_res, x_res, number_frames)]

  in_file_2 = "./videos/tests/CIF_RC3_BR2m_encoded.yuv"
  psnr += [PSNR_per_frame(y_frame_1, in_file_2, y_res, x_res, number_frames)]


  fig, ax = plt.subplots()

  ax.plot(frame_numbers, psnr[0], 'x:', label='RC=1')
  ax.plot(frame_numbers, psnr[1], 'x-.', label='RC=2')
  ax.plot(frame_numbers, psnr[2], 'x--', label='RC=3')
  
  
  
  ax.set(xlabel='Frame', ylabel='PSNR (dB)')
  ax.grid()
  ax.set_xticks(frame_numbers)
  ax.legend(loc='lower right')
  # ax.set_yscale('log')

  # fig.savefig("test.png")
  plt.show()

  # ---------------------------------------------------------------------------
  # ---------------------------------------------------------------------------




  # fig, ax = plt.subplots()

  # ax.plot(file_size[0], psnr[0], 'x:', label='RC=0')
  # ax.plot(file_size[1], psnr[1], 'x-.', label='RC=1')
  # ax.plot(file_size[2], psnr[2], 'x--', label='RC=2')
  # ax.plot(file_size[3], psnr[3], 'x:', label='RC=3')
  
  
  
  # ax.set(xlabel='Size in bits', ylabel='PSNR (dB)')
  # ax.grid()
  # ax.legend(loc='lower right')
  # # ax.set_yscale('log')

  # # fig.savefig("test.png")
  # plt.show()