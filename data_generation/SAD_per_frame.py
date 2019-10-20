import numpy as np
import matplotlib.pyplot as plt
import sys

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


  fig, ax = plt.subplots()

  frames = range(1, number_frames+1)
  ax.plot(frames, SAD_1, label='n=1')
  ax.plot(frames, SAD_2, label='n=2')
  ax.plot(frames, SAD_3, label='n=3')
  
  
  
  ax.set(xlabel='Frame', ylabel=r'SAD $(\times 10^6)$')
  ax.grid()
  ax.legend()

  # fig.savefig("test.png")
  plt.show()