import numpy as np
import sys
import assign1_Q2_main as pre


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

def calculate_no_motion_comp_residual_block(block, reconstructed, head_idy, head_idx, i):
  extracted = np.empty((i, i))
  extracted = reconstructed[head_idy : head_idy + i, head_idx : head_idx + i]
  residual = np.subtract(block, extracted)

  return residual

def calculate_approximate_residual_block(block, n):

  sign_block = np.sign(block)
  abs_block = np.absolute(block)

  factor = 2**n

  for y_it in range(abs_block.shape[0]):
    for x_it in range(abs_block.shape[1]):

      remainder = abs_block[y_it][x_it] % factor

      if (abs_block[y_it][x_it] < factor):
        abs_block[y_it][x_it] = 0

      elif (remainder < (factor / 2)):
        abs_block[y_it][x_it] = abs_block[y_it][x_it] - remainder

      else:
        abs_block[y_it][x_it] = abs_block[y_it][x_it] + factor - remainder
  
    block = np.multiply(abs_block, sign_block)

  return block


def decoder_core(residual_matrix, reconstructed, mv):
  y_it = residual_matrix.shape[0]
  x_it = residual_matrix.shape[1]
  i = residual_matrix.shape[2]

  new_reconstructed = np.empty((y_it*i, x_it*i), dtype=int)

  for bl_y_it in range(y_it) :    
      for bl_x_it in range(x_it):
        head_idy = bl_y_it * i
        head_idx = bl_x_it * i
        
        extracted = reconstructed[head_idy + mv[bl_y_it][bl_x_it][0] : head_idy + mv[bl_y_it][bl_x_it][0] + i, head_idx + mv[bl_y_it][bl_x_it][1] : head_idx + mv[bl_y_it][bl_x_it][1] + i]

        new_reconstructed[head_idy:head_idy + i, head_idx:head_idx + i] = np.add(residual_matrix[bl_y_it][bl_x_it], extracted)

  for y_it in range (new_reconstructed.shape[0]):
    for x_it in range (new_reconstructed.shape[1]):
      new_reconstructed[y_it][x_it] = max(0, min(new_reconstructed[y_it][x_it], 255))

  return new_reconstructed

def calculate_reconstructed_image(residual_matrix, reconstructed, ext_y_res, ext_x_res, n_y_blocks, n_x_blocks, mv):

  new_reconstructed = decoder_core(residual_matrix, reconstructed, mv)

  return new_reconstructed

def concatinate_and_write(ext_y_res, ext_x_res, y_res, x_res, n_y_blocks, n_x_blocks, input_frame, output_file):
  #  Concatenate
  counter = 0
  conc_reconstructed = np.empty((ext_y_res, ext_x_res), dtype = int)
  for bl_y_it in range(n_y_blocks):
    conc = np.concatenate((input_frame[bl_y_it][0], input_frame[bl_y_it][1]), axis=1)
    for bl_x_it in range(2, n_x_blocks):
      conc = np.concatenate((conc, input_frame[bl_y_it][bl_x_it]), axis=1)
    # Write frame (decoder output video)
    for i_it in range(i):
      if (counter < y_res):
        counter += 1
        for x_it in range(x_res):
          conc_reconstructed[bl_y_it*i + i_it][x_it] = conc[i_it][x_it]
          output_file.write(((int)(max(0, min(conc[i_it][x_it], 255)))).to_bytes(1, byteorder=sys.byteorder))

  return conc_reconstructed

def encoder(in_file, out_file, number_frames, y_res, x_res, i, r, n):

  print("----------------------------------------------")
  print("----------------------------------------------")
  print("Q3 Encoder Parameters-")
  print("----------------------------------------------")
  print("in_file: ", in_file)
  print("out_file: ", out_file)
  print("number_frames: ", number_frames)
  print("y_res: ", y_res)
  print("x_res: ", x_res)
  print("i: ", i)
  print("r: ", r)
  print("n: ", n)
  print("----------------------------------------------")

  bl_y_frame, n_y_blocks, n_x_blocks, ext_y_res, ext_x_res= pre.block(in_file, y_res, x_res, number_frames, i)

  reconst = np.full((ext_y_res, ext_x_res), 128, dtype=int)
  mv = np.empty((number_frames, n_y_blocks, n_x_blocks, 2), dtype=int)

  residual_matrix = np.empty((number_frames, n_y_blocks,n_x_blocks, i, i), dtype=np.int16)
  
  no_motion_comp_residual_matrix = np.empty((number_frames, n_y_blocks,n_x_blocks, i, i), dtype=np.int16)


  file_and_extension = out_file.split(".")
  converted_name = ".".join(file_and_extension[:-1]) + ".yuv"
  encoded_name = ".".join(file_and_extension[:-1]) + ".npz"
  # no_motion_comp_name = ".".join(file_and_extension[:-1]) + "_no_motion_comp.yuv"
  # motion_comp_name = ".".join(file_and_extension[:-1]) + "_motion_comp.yuv"

  converted = open(converted_name, "wb")
  # no_motion_comp = open(no_motion_comp_name, "wb")
  # motion_comp = open(motion_comp_name, "wb")

  for frame in range(number_frames):

    pre.progress("Encoding frames: ", frame, number_frames)

  # Calculate Motion Vector
    for bl_y_it in range(n_y_blocks) :    
      for bl_x_it in range(n_x_blocks):

        mv[frame][bl_y_it][bl_x_it] = motion_vector_estimation(bl_y_frame[frame][bl_y_it][bl_x_it], reconst, r, bl_y_it*i, bl_x_it*i, ext_y_res, ext_x_res, i)


    # Calculate Residual Matrix
    
        residual_matrix[frame][bl_y_it][bl_x_it] = calculate_residual_block(bl_y_frame[frame][bl_y_it][bl_x_it], reconst, bl_y_it * i, bl_x_it * i, i, mv[frame][bl_y_it][bl_x_it])
        
        # no_motion_comp_residual_matrix[frame][bl_y_it][bl_x_it] = calculate_no_motion_comp_residual_block(bl_y_frame[frame][bl_y_it][bl_x_it], reconst, bl_y_it * i, bl_x_it * i, i)
        
        residual_matrix[frame][bl_y_it][bl_x_it] = calculate_approximate_residual_block(residual_matrix[frame][bl_y_it][bl_x_it], n)
        
        # no_motion_comp_residual_matrix[frame][bl_y_it][bl_x_it] = calculate_approximate_residual_block(no_motion_comp_residual_matrix[frame][bl_y_it][bl_x_it], n)

    new_reconstructed = decoder_core(residual_matrix[frame], reconst, mv[frame])

    # concatinate_and_write(ext_y_res, ext_x_res, y_res, x_res, n_y_blocks, n_x_blocks, no_motion_comp_residual_matrix[frame], no_motion_comp)
    
    # concatinate_and_write(ext_y_res, ext_x_res, y_res, x_res, n_y_blocks, n_x_blocks, residual_matrix[frame], motion_comp)

    for y_it in range(y_res):
      for x_it in range(x_res):
        # print((int)(new_reconstructed[y_it][x_it]))
        converted.write(((int)(new_reconstructed[y_it][x_it])).to_bytes(1, byteorder=sys.byteorder))

    reconst = new_reconstructed

  converted.close()
  # no_motion_comp.close()
  # motion_comp.close()
    
  np.savez(encoded_name, mv=mv, residual_matrix=residual_matrix, y_res=y_res, x_res=x_res)
  print("Encoding Completed")


def decoder(in_file, out_file):

  mv = np.load(in_file)['mv']
  residual_matrix = np.load(in_file)['residual_matrix']
  y_res = np.load(in_file)['y_res']
  x_res = np.load(in_file)['x_res']

  number_of_frames = residual_matrix.shape[0]
  i = residual_matrix.shape[3]
  ext_y_res = residual_matrix.shape[1] * i
  ext_x_res = residual_matrix.shape[2] * i

  print("----------------------------------------------")
  print("----------------------------------------------")
  print("Q3 Decoder Parameters-")
  print("----------------------------------------------")
  print("in_file: ", in_file)
  print("out_file: ", out_file)
  print("y_res: ", y_res)
  print("x_res: ", x_res)
  print("i: ", i)
  print("----------------------------------------------")
  
  reconst = np.full((ext_y_res, ext_x_res), 128, dtype=int)

  converted = open(out_file, "wb")

  for frame in range(number_of_frames):

    pre.progress("Decoding frames: ", frame, number_of_frames)

    new_reconstructed = decoder_core(residual_matrix[frame], reconst, mv[frame])

    for y_it in range(y_res):
      for x_it in range(x_res):

        converted.write(((int)(new_reconstructed[y_it][x_it])).to_bytes(1, byteorder=sys.byteorder))

    reconst = new_reconstructed
  
  converted.close()
  print("Decoding Completed")



if __name__ == "__main__":
  
  in_file = "./videos/black_and_white.yuv"
  out_file = "./temp/q3_encoded.npz"
  decoder_in = "./temp/q3_encoded.npz"
  decoder_out = "./videos/q3_decoded_i64.yuv"
  number_frames = 10
  y_res = 288
  x_res = 352
  i = 64
  r = 4
  n = 2

  encoder(in_file, out_file, number_frames, y_res, x_res, i, r, n)
  decoder(decoder_in, decoder_out)