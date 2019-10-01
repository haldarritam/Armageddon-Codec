import numpy as np
import sys
import assign1_Q2_main as pre

def motion_vector_estimation(block, reconstructed, r, head_idy, head_idx, ext_y_res, ext_x_res, i):

  extracted = np.empty((i, i))
  best_SAD = 300  # since biggest difference can be 255 (8-Bit image)
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

def calculate_approximate_residual_block(block):

  sign_block = np.sign(block)
  abs_block = np.absolute(block)

  abs_block[abs_block == 0] = 2

  # block_c = np.power(2, np.ceil((np.log(abs_block) / np.log(2))))
  # block_f = np.power(2, np.floor((np.log(abs_block) / np.log(2))))

  # for y_it in range(abs_block.shape[0]):
  #   for x_it in range(abs_block.shape[1]):
      
  #     floor = abs_block[y_it][x_it] - block_f[y_it][x_it]
  #     ceil = block_c[y_it][x_it] - abs_block[y_it][x_it]
      
  #     if floor < ceil:
  #       block[y_it][x_it] = block_f[y_it][x_it]
  #     else:
  #       block[y_it][x_it] = block_c[y_it][x_it]
      
  #     block[y_it][x_it] = block[y_it][x_it] * sign_block[y_it][x_it]

  block_f = np.power(2, np.floor((np.log(abs_block) / np.log(2))))
  block = np.multiply(block_f, sign_block)

  return block

def calculate_reconstructed_image(residual_matrix, reconstructed, ext_y_res, ext_x_res, n_y_blocks, n_x_blocks):
  
  temp = np.empty((ext_y_res, ext_x_res), dtype=int)
  for bl_y_it in range(n_y_blocks):
    conc = np.concatenate((residual_matrix[bl_y_it][0], residual_matrix[bl_y_it][1]), axis=1)
    for bl_x_it in range(2, n_x_blocks):
      conc = np.concatenate((conc, residual_matrix[bl_y_it][bl_x_it]), axis=1)
    
    temp[bl_y_it * i : bl_y_it * i + i, :] = conc

  new_reconstructed = np.add(temp, reconstructed)


  # print(temp[0][3])
  # print("\n\n\n")
  # print(reconstructed[0][3])
  # print("\n\n\n")
  # print(new_reconstructed[0][3])

  return new_reconstructed


if __name__ == "__main__":
  
  in_file = "./videos/black_and_white.yuv"
  out_file = "./videos/averaged.yuv"
  number_frames = 300
  y_res = 288
  x_res = 352
  i = 8
  r = 8

  bl_y_frame, n_y_blocks, n_x_blocks, ext_y_res, ext_x_res= pre.block(in_file, y_res, x_res, number_frames, i)

  reconst = np.full((y_res, x_res), 128, dtype=int)
  mv = np.empty((number_frames, n_y_blocks, n_x_blocks, 2), dtype=int)


  converted = open("./videos/testing.yuv", "wb")

  for frame in range(number_frames):

    print("Loop of frame: ", frame)

  # Calculate Motion Vector
    for bl_y_it in range(n_y_blocks) :    
      for bl_x_it in range(n_x_blocks):

        mv[frame][bl_y_it][bl_x_it] = motion_vector_estimation(bl_y_frame[frame][bl_y_it][bl_x_it], reconst, r, bl_y_it*i, bl_x_it*i, ext_y_res, ext_x_res, i)


    # Calculate Residual Matrix
    residual_matrix = np.empty((n_y_blocks,n_x_blocks, i, i))
    for bl_y_it in range(n_y_blocks) :    
      for bl_x_it in range(n_x_blocks):
        residual_matrix[bl_y_it][bl_x_it] = calculate_residual_block(bl_y_frame[frame][bl_y_it][bl_x_it], reconst, bl_y_it * i, bl_x_it * i, i, mv[frame][bl_y_it][bl_x_it])
        
        residual_matrix[bl_y_it][bl_x_it] = calculate_approximate_residual_block(residual_matrix[bl_y_it][bl_x_it])

    new_reconstructed = calculate_reconstructed_image(residual_matrix, reconst, ext_y_res, ext_x_res, n_y_blocks, n_x_blocks)

    for y_it in range(y_res):
      for x_it in range(x_res):
        # print((int)(new_reconstructed[y_it][x_it]))
        converted.write(((int)(new_reconstructed[y_it][x_it])).to_bytes(1, byteorder=sys.byteorder))

  converted.close()
    
  np.save("./temp/motion_vectors.npy", mv)