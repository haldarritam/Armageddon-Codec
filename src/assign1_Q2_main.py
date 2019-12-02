import numpy as np
import sys

def progress(message, current, total):
  if (total != 1):
    progress = (int) (current / (total - 1) * 100)
    sys.stdout.write(message + " %d%%   \r" % (progress) )
    sys.stdout.flush()
  else:
    sys.stdout.write(message + " %d%%   \r" % (100) )
    sys.stdout.flush()

def block(file, y_res, x_res, number_frames, i):
  ext_y_res = y_res
  ext_x_res = x_res
  total_pixels_frame = x_res * y_res
  bytes_frame = (int)(total_pixels_frame)

  n_y_blocks =(int)(y_res / i)
  n_x_blocks = (int)(x_res / i)

  if (y_res % i != 0):
    n_y_blocks += 1
    ext_y_res += (i - (y_res % i)) 
  if (x_res % i != 0):
    n_x_blocks += 1
    ext_x_res += (i - (x_res % i)) 

  bl_y_frame = np.empty((number_frames, n_y_blocks, n_x_blocks, i, i), dtype=int)

  y_file = open(file,"rb")
  for frame in range(number_frames):
    progress("Blocking frames: ", frame, number_frames)
    raw = y_file.read(bytes_frame)

    for bl_y_it in range(n_y_blocks) :    
      for bl_x_it in range(n_x_blocks):
        for y_it in range(i) :    
          for x_it in range(i):

            if (((bl_y_it * i) + y_it) >= y_res or ((bl_x_it * i + x_it)) >= x_res):
              bl_y_frame[frame][bl_y_it][bl_x_it][y_it][x_it] = 128
            else:
              it = (((bl_y_it * i) + y_it) * x_res) + ((bl_x_it * i + x_it))
              bl_y_frame[frame][bl_y_it][bl_x_it][y_it][x_it] = int.from_bytes((raw[it : it + 1]), byteorder=sys.byteorder)
  y_file.close()

  return bl_y_frame, n_y_blocks, n_x_blocks, ext_y_res, ext_x_res

def average(bl_y_frame, number_frames, n_y_blocks, n_x_blocks, out_file):
  for frame in range(number_frames):
    progress("Averaging frames: ", frame, number_frames)
    for bl_y_it in range(n_y_blocks) :    
      for bl_x_it in range(n_x_blocks):
        mean = [np.round(np.mean(bl_y_frame[frame][bl_y_it][bl_x_it]))]
        bl_y_frame[frame][bl_y_it][bl_x_it] = mean

  converted = open(out_file, "wb")

  for frame in range(number_frames):
    counter = 0
    progress("Saving averaged: ", frame, number_frames)
    for bl_y_it in range(n_y_blocks):
      conc = np.concatenate((bl_y_frame[frame][bl_y_it][0], bl_y_frame[frame][bl_y_it][1]), axis=1)   
      for bl_x_it in range(2, n_x_blocks):
        conc = np.concatenate((conc, bl_y_frame[frame][bl_y_it][bl_x_it]), axis=1)
      
      for i_it in range(i):
        if (counter < y_res):
          counter += 1
          for x_it in range(x_res):
            converted.write(((int)(conc[i_it][x_it])).to_bytes(1, byteorder=sys.byteorder))

  converted.close()


if __name__=="__main__":
  in_file = "./videos/black_and_white.yuv"
  out_file = "./videos/averaged.yuv"
  number_frames = 300
  y_res = 288
  x_res = 352
  i = 16

  bl_y_frame, n_y_blocks, n_x_blocks,_,_ = block(in_file, y_res, x_res, number_frames, i)
  
  average(bl_y_frame, number_frames, n_y_blocks, n_x_blocks, out_file)
