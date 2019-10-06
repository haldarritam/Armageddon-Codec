import numpy as np
import sys
import assign1_Q2_main as pre
from scipy.fftpack import dct, idct

def dct2D(block):  # Transform Function
  res = dct(dct(block, axis=0, norm='ortho'), axis=1, norm='ortho')
  return res

def idct2D(block): # Inverse Transform Function
  res = idct(idct(block, axis=0, norm='ortho'), axis=1, norm='ortho')
  return res

def motion_vector_estimation(block, reconstructed, r, head_idy, head_idx, ext_y_res, ext_x_res, i):

  best_SAD = i * i * 255 + 1  # The sum can never exceed (i * i * 255 + 1) 
  mv = (0,0)
  for y_dir in range(-r, (r + 1)):
    for x_dir in range(-r, (r + 1)):

      if ((head_idy + y_dir) >= 0 and (head_idy + y_dir + i) < ext_y_res and (head_idx + x_dir) >= 0 and (head_idx + x_dir + i) < ext_x_res):
        extracted = reconstructed[head_idy + y_dir : head_idy + y_dir + i, head_idx + x_dir : head_idx + x_dir + i]

        SAD = np.sum(np.abs(np.subtract(extracted, block, dtype=int)))
        
        if (SAD < best_SAD):
          best_SAD = SAD
          mv = (y_dir, x_dir)
        elif (SAD == best_SAD):
          if ((abs(y_dir) + abs(x_dir)) < (abs(mv[0]) + abs(mv[1]))):
            best_SAD = SAD
            mv = (y_dir, x_dir)
          elif ((abs(y_dir) + abs(x_dir)) == (abs(mv[0]) + abs(mv[1]))):
            if (y_dir < mv[0]):
              best_SAD = SAD
              mv = (y_dir, x_dir)
            elif (y_dir == mv[0]):
              if (x_dir < mv[1]):
                best_SAD = SAD
                mv = (y_dir, x_dir)

  return mv

def calculate_residual_block(block, reconstructed, head_idy, head_idx, i, mv):
  extracted = np.empty((i, i))
  extracted = reconstructed[head_idy + mv[0] : head_idy + mv[0] + i, head_idx + mv[1] : head_idx + mv[1] + i]
  residual = np.subtract(block, extracted)

  return residual

def calculate_Q(i, QP):
  
  Q = np.empty((i, i), dtype=int)

  for y_it in range(i):
    for x_it in range(i):
      if ((x_it + y_it) < (i - 1)):
        Q[y_it][x_it] = 2 ** QP
      elif ((y_it + x_it) == (i - 1)):
        Q[y_it][x_it] = 2 ** (QP + 1)
      else:
        Q[y_it][x_it] = 2 ** (QP + 2)

  return Q

def quantize_dct(transformed_dct, Q):

  QTC = np.round(np.divide(transformed_dct, Q))
  return QTC

def rescale_IDCT(QTC, Q):
  approx_residual = idct2D(np.multiply(QTC, Q))
  return approx_residual



def decoder_core(residual_matrix, reconstructed, mv, block_option=False):
  y_it = residual_matrix.shape[0]
  x_it = residual_matrix.shape[1]
  i = residual_matrix.shape[2]

  new_reconstructed = np.empty((y_it * i, x_it * i), dtype=int)
  
  if (block_option):
    new_reconstructed.reshape(y_it, x_it, i, i)

  for bl_y_it in range(y_it) :    
      for bl_x_it in range(x_it):
        head_idy = bl_y_it * i
        head_idx = bl_x_it * i
        
        extracted = reconstructed[head_idy + mv[bl_y_it][bl_x_it][0] : head_idy + mv[bl_y_it][bl_x_it][0] + i, head_idx + mv[bl_y_it][bl_x_it][1] : head_idx + mv[bl_y_it][bl_x_it][1] + i]

        if (block_option):
          new_reconstructed[bl_y_it][bl_x_it] = np.add(residual_matrix[bl_y_it][bl_x_it], extracted)
        else:
          new_reconstructed[head_idy:head_idy + i, head_idx:head_idx + i] = np.add(residual_matrix[bl_y_it][bl_x_it], extracted)

  if (block_option):
    for bl_y_it in range(y_it) :    
      for bl_x_it in range(x_it):
        for y_it in range (i):
          for x_it in range (i):
            new_reconstructed[bl_y_it][bl_x_it][y_it][x_it] = max(0, min(new_reconstructed[bl_y_it][bl_x_it][y_it][x_it], 255))
  else:
    for y_it in range (new_reconstructed.shape[0]):
      for x_it in range (new_reconstructed.shape[1]):
        new_reconstructed[y_it][x_it] = max(0, min(new_reconstructed[y_it][x_it], 255))

  return new_reconstructed

def calculate_reconstructed_image(residual_matrix, reconstructed, ext_y_res, ext_x_res, n_y_blocks, n_x_blocks, mv):

  new_reconstructed = decoder_core(residual_matrix, reconstructed, mv)

  return new_reconstructed

##############################################################################
##############################################################################

def encoder(in_file, out_file, number_frames, y_res, x_res, i, r, QP, i_period):

  bl_y_frame, n_y_blocks, n_x_blocks, ext_y_res, ext_x_res = pre.block(in_file, y_res, x_res, number_frames, i)

  reconst = np.full((ext_y_res, ext_x_res), 128, dtype=int)
  mv = np.empty((number_frames, n_y_blocks, n_x_blocks, 2), dtype=int)

  residual_matrix = np.empty((number_frames, n_y_blocks, n_x_blocks, i, i), dtype=np.int16)

  QTC = np.empty((number_frames, n_y_blocks, n_x_blocks, i, i), dtype=int)

  approx_residual = np.empty((number_frames, n_y_blocks, n_x_blocks, i, i), dtype=int)

  Q = calculate_Q(i, QP)


  converted = open(out_file, "wb")

  for frame in range(number_frames):

    print("Loop of frame: ", frame)



# (frame_n +1) % i_period



  # Calculate Motion Vector
    for bl_y_it in range(n_y_blocks) :    
      for bl_x_it in range(n_x_blocks):

        mv[frame][bl_y_it][bl_x_it] = motion_vector_estimation(bl_y_frame[frame][bl_y_it][bl_x_it], reconst, r, bl_y_it*i, bl_x_it*i, ext_y_res, ext_x_res, i)


    # Calculate Residual Matrix
    
    for bl_y_it in range(n_y_blocks) :    
      for bl_x_it in range(n_x_blocks):
        residual_matrix[frame][bl_y_it][bl_x_it] = calculate_residual_block(bl_y_frame[frame][bl_y_it][bl_x_it], reconst, bl_y_it * i, bl_x_it * i, i, mv[frame][bl_y_it][bl_x_it])

    #  Trans/Quant/Rescaling/InvTrans
    for bl_y_it in range(n_y_blocks) :    
      for bl_x_it in range(n_x_blocks):        
        transformed_dct = dct2D(residual_matrix[frame][bl_y_it][bl_x_it])
        QTC[frame][bl_y_it][bl_x_it] = quantize_dct(transformed_dct, Q)
        approx_residual[frame][bl_y_it][bl_x_it] = rescale_IDCT(QTC[frame][bl_y_it][bl_x_it], Q)



    new_reconstructed = decoder_core(approx_residual[frame], reconst, mv[frame])

    for y_it in range(y_res):
      for x_it in range(x_res):
        # print((int)(new_reconstructed[y_it][x_it]))
        converted.write(((int)(new_reconstructed[y_it][x_it])).to_bytes(1, byteorder=sys.byteorder))

    reconst = new_reconstructed

  converted.close()
    
  # np.savez("./temp/motion_vectors.npz", mv=mv)
  # np.savez("./temp/residual_matrix.npz", residual_matrix=residual_matrix)

##############################################################################
##############################################################################

def decoder(y_res, x_res):

  mv = np.load("./temp/motion_vectors.npz")['mv']
  residual_matrix = np.load("./temp/residual_matrix.npz")['residual_matrix']

  number_of_frames = residual_matrix.shape[0]
  i = residual_matrix.shape[3]
  ext_y_res = residual_matrix.shape[1] * i
  ext_x_res = residual_matrix.shape[2] * i
  
  reconst = np.full((ext_y_res, ext_x_res), 128, dtype=int)

  converted = open("./videos/decoder_test.yuv", "wb")

  for frame in range(number_of_frames):

    print(frame)

    new_reconstructed = decoder_core(residual_matrix[frame], reconst, mv[frame])

    for y_it in range(y_res):
      for x_it in range(x_res):

        converted.write(((int)(new_reconstructed[y_it][x_it])).to_bytes(1, byteorder=sys.byteorder))

    reconst = new_reconstructed
  
  converted.close()

##############################################################################
##############################################################################

if __name__ == "__main__":
  
  in_file = "./videos/black_and_white.yuv"
  out_file = "./videos/encoder_test.yuv"
  number_frames = 1
  y_res = 288
  x_res = 352
  i = 32
  r = 1
  QP = 6  # from 0 to (log_2(i) + 7)
  i_period = 5

  encoder(in_file, out_file, number_frames, y_res, x_res, i, r, QP, i_period)
  # decoder(y_res, x_res)