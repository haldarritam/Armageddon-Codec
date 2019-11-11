import numpy as np
import sys
import matplotlib.pyplot as plt
import assign1_Q2_main as pre
from scipy.fftpack import dct, idct

def dct2D(block):  # Transform Function
  res = dct(dct(block, axis=0, norm='ortho'), axis=1, norm='ortho')
  return res

def idct2D(block): # Inverse Transform Function
  res = idct(idct(block, axis=0, norm='ortho'), axis=1, norm='ortho')
  return res

def FME_extraction(FMEEnabled, i, head_idy, head_idx, y_dir, x_dir, ext_y_res, ext_x_res, reconstructed):
  extracted = 0
  if(FMEEnabled):
    dy_dir = int(y_dir/2)
    dx_dir = int(x_dir/2)
    move_is_inside_frame = (head_idy + dy_dir) >= 0 and (head_idy + dy_dir + i) < ext_y_res and (head_idx + dx_dir) >= 0 and (head_idx + dx_dir + i) < ext_x_res
  else:
    move_is_inside_frame = (head_idy + y_dir) >= 0 and (head_idy + y_dir + i) < ext_y_res and (head_idx + x_dir) >= 0 and (head_idx + x_dir + i) < ext_x_res
    
  if (move_is_inside_frame):
    if (not FMEEnabled):
      extracted = reconstructed[head_idy + y_dir : head_idy + y_dir + i, head_idx + x_dir : head_idx + x_dir + i]

    elif ((y_dir % 2 == 0) and (x_dir % 2 == 0)): # none fractional
      #print("NONE FRAC", y_dir, x_dir)
      dy_dir = int(y_dir/2)
      dx_dir = int(x_dir/2)
      
      extracted = reconstructed[head_idy + dy_dir : head_idy + dy_dir + i, head_idx + dx_dir : head_idx + dx_dir + i]

    elif (x_dir % 2 and y_dir % 2): # both fractional
      #print("BOTH FRAC", y_dir, x_dir)
      dy_dir = int(y_dir/2)
      dx_dir = int(x_dir/2)
      
      extracted = (reconstructed[head_idy + dy_dir : head_idy + dy_dir + i, head_idx + dx_dir : head_idx + dx_dir + i] +
        reconstructed[head_idy + dy_dir + 1 : head_idy + dy_dir + i + 1, head_idx + dx_dir : head_idx + dx_dir + i] +
        reconstructed[head_idy + dy_dir : head_idy + dy_dir + i, head_idx + dx_dir + 1 : head_idx + dx_dir + i + 1] +
        reconstructed[head_idy + dy_dir + 1: head_idy + dy_dir + i + 1, head_idx + dx_dir + 1 : head_idx + dx_dir + i + 1]) // 4

    elif (x_dir % 2): # x fractional
      #print("X FRAC", y_dir, x_dir)
      dy_dir = int(y_dir/2)
      dx_dir = int(x_dir/2)
      
      extracted = (reconstructed[head_idy + dy_dir : head_idy + dy_dir + i, head_idx + dx_dir : head_idx + dx_dir + i] +
        reconstructed[head_idy + dy_dir : head_idy + dy_dir + i, head_idx + dx_dir + 1 : head_idx + dx_dir + i + 1]) // 2

    else: # y fractional
      #print("Y FRAC", y_dir, x_dir)
      dy_dir = int(y_dir/2)
      dx_dir = int(x_dir/2)
      
      extracted = (reconstructed[head_idy + dy_dir : head_idy + dy_dir + i, head_idx + dx_dir : head_idx + dx_dir + i] +
        reconstructed[head_idy + dy_dir + 1 : head_idy + dy_dir + i + 1, head_idx + dx_dir : head_idx + dx_dir + i]) // 2
        
    # print(head_idy, head_idx)
    # print(extracted)
    # print("-------")
    # print(reconstructed)

  return extracted

def motion_vector_estimation(block, rec_buffer, r, head_idy, head_idx, ext_y_res, ext_x_res, i, FMEEnabled):

  best_SAD = i * i * 255 + 1  # The sum can never exceed (i * i * 255 + 1)
  mv = (0, 0, 0)

  up_r = r
  if (FMEEnabled):
    up_r = 2 * r

  for buff_idx, reconstructed in enumerate(rec_buffer):
    for y_dir in range(-up_r, (up_r + 1)):
      for x_dir in range(-up_r, (up_r + 1)):

        extracted = FME_extraction(FMEEnabled, i, head_idy, head_idx, y_dir, x_dir, ext_y_res, ext_x_res, reconstructed)

        SAD = np.sum(np.abs(np.subtract(extracted, block, dtype=int)))

        if (SAD < best_SAD):
          best_SAD = SAD
          mv = (y_dir, x_dir, buff_idx)
        elif (SAD == best_SAD):
          if ((abs(y_dir) + abs(x_dir)) < (abs(mv[0]) + abs(mv[1]))):
            best_SAD = SAD
            mv = (y_dir, x_dir, buff_idx)
          elif ((abs(y_dir) + abs(x_dir)) == (abs(mv[0]) + abs(mv[1]))):
            if (y_dir < mv[0]):
              best_SAD = SAD
              mv = (y_dir, x_dir, buff_idx)
            elif (y_dir == mv[0]):
              if (x_dir < mv[1]):
                best_SAD = SAD
                mv = (y_dir, x_dir, buff_idx)

  return [mv]

def RDO_sel(extracted, block, Q, lambda_const, y_dir, x_dir, buff_idx, mv, best_RDO):

  RDO = calc_RDO(extracted, block, Q, lambda_const)

  if (RDO < best_RDO):
    best_RDO = RDO
    mv = (y_dir, x_dir, buff_idx)
  elif (RDO == best_RDO):
    if ((abs(y_dir) + abs(x_dir)) < (abs(mv[0]) + abs(mv[1]))):
      best_RDO = RDO
      mv = (y_dir, x_dir, buff_idx)
    elif ((abs(y_dir) + abs(x_dir)) == (abs(mv[0]) + abs(mv[1]))):
      if (y_dir < mv[0]):
        best_RDO = RDO
        mv = (y_dir, x_dir, buff_idx)
      elif (y_dir == mv[0]):
        if (x_dir < mv[1]):
          best_RDO = RDO
          mv = (y_dir, x_dir, buff_idx)

  return best_RDO, mv


def motion_vector_estimation_vbs(block, rec_buffer, r, head_idy, head_idx, ext_y_res, ext_x_res, i, Q, sub_Q, lambda_const, FMEEnabled):

  if (i != 4 and i != 8 and i != 16):
    print("Block size should be 4, 8 or 16 when VBS enabled!")
    print("Exiting...")
    exit()

  sub_i = (int)(i / 2)

  up_r = r
  if (FMEEnabled):
    up_r = 2 * r

  # Extract the sub-blocks
  sub_block = []

  sub_block += [block[0: sub_i, 0: sub_i]]
  sub_block += [block[0: sub_i, sub_i: sub_i + sub_i]]
  sub_block += [block[sub_i: sub_i + sub_i, 0: sub_i]]
  sub_block += [block[sub_i: sub_i + sub_i, sub_i: sub_i + sub_i]]

  sub_head_idy = [head_idy, head_idy, head_idy + sub_i, head_idy + sub_i]
  sub_head_idx = [head_idx, head_idx + sub_i, head_idx, head_idx + sub_i]


  # Setting up the initial best block value using mv=(0, 0, 0)
  extracted_block = FME_extraction(FMEEnabled, i, head_idy, head_idx, 0, 0, ext_y_res, ext_x_res, rec_buffer[0])

  best_RDO_block = calc_RDO(extracted_block, block, Q, lambda_const) + 1

  # Setting up the initial best sub-block value using mv=(0, 0, 0)
  extracted_sub = []
  
  extracted_sub += [FME_extraction(FMEEnabled, sub_i, sub_head_idy[0], sub_head_idx[0], 0, 0, ext_y_res, ext_x_res, rec_buffer[0])]
  
  extracted_sub += [FME_extraction(FMEEnabled, sub_i, sub_head_idy[1], sub_head_idx[1], 0, 0, ext_y_res, ext_x_res, rec_buffer[0])]
    
  extracted_sub += [FME_extraction(FMEEnabled, sub_i, sub_head_idy[2], sub_head_idx[2], 0, 0, ext_y_res, ext_x_res, rec_buffer[0])]
    
  extracted_sub += [FME_extraction(FMEEnabled, sub_i, sub_head_idy[3], sub_head_idx[3], 0, 0, ext_y_res, ext_x_res, rec_buffer[0])]
  
  best_RDO_sub = []

  best_RDO_sub += [calc_RDO(extracted_sub[0], sub_block[0], sub_Q, lambda_const)+1]
  best_RDO_sub += [calc_RDO(extracted_sub[1], sub_block[1], sub_Q, lambda_const)+1]
  best_RDO_sub += [calc_RDO(extracted_sub[2], sub_block[2], sub_Q, lambda_const)+1]
  best_RDO_sub += [calc_RDO(extracted_sub[3], sub_block[3], sub_Q, lambda_const)+1]

  mv = (0, 0, 0)
  sub_mv = [(0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)]

  for buff_idx, reconstructed in enumerate(rec_buffer):
    # Block prediction
    for y_dir in range(-up_r, (up_r + 1)):
      for x_dir in range(-up_r, (up_r + 1)):

        if ((head_idy + y_dir) >= 0 and (head_idy + y_dir + i) < ext_y_res and (head_idx + x_dir) >= 0 and (head_idx + x_dir + i) < ext_x_res):

          extracted = FME_extraction(FMEEnabled, i, head_idy, head_idx, y_dir, x_dir, ext_y_res, ext_x_res, reconstructed)

          best_RDO_block, mv = RDO_sel(extracted, block, Q, lambda_const, y_dir, x_dir, buff_idx, mv, best_RDO_block)

    # Sub-block prediction
    for sub_idx in range(4):
      for y_dir in range(-up_r, (up_r + 1)):
        for x_dir in range(-up_r, (up_r + 1)):

          if ((sub_head_idy[sub_idx] + y_dir) >= 0 and (sub_head_idy[sub_idx] + y_dir + sub_i) < ext_y_res and (sub_head_idx[sub_idx] + x_dir) >= 0 and (sub_head_idx[sub_idx] + x_dir + sub_i) < ext_x_res):

            extracted_sub = FME_extraction(FMEEnabled, sub_i, sub_head_idy[sub_idx], sub_head_idx[sub_idx], y_dir, x_dir, ext_y_res, ext_x_res, reconstructed)

            best_RDO_sub[sub_idx], sub_mv[sub_idx] = RDO_sel(extracted_sub, sub_block[sub_idx], sub_Q, lambda_const, y_dir, x_dir, buff_idx, sub_mv[sub_idx], best_RDO_sub[sub_idx])

  RDO_sub = best_RDO_sub[0] + best_RDO_sub[1] + best_RDO_sub[2] + best_RDO_sub[3]

  if (best_RDO_block < RDO_sub):
    return [0, mv]
  else:
    return [1, sub_mv[0], sub_mv[1], sub_mv[2], sub_mv[3]]

def intra_prediction(frame, y_idx, x_idx):

  i = frame.shape[2]

  grey = 128

  top_edge_block = np.empty((i,i), dtype=int)
  left_edge_block = np.empty((i,i), dtype=int)

  mode = 0 #Horizontal

  top_edge = np.full((1, i), grey)
  left_edge = np.full((i, 1), grey)

  if ((y_idx - 1) >= 0):
    top_edge = frame[y_idx - 1][x_idx][-1, :]

  if ((x_idx - 1) >= 0):
    left_edge = frame[y_idx][x_idx - 1][:, -1].reshape((i, 1))

  top_edge_block[:,:] = top_edge
  left_edge_block[:, :] = left_edge

  SAD_top = np.sum(np.abs(np.subtract(frame[y_idx][x_idx], top_edge_block, dtype=int)))
  SAD_left = np.sum(np.abs(np.subtract(frame[y_idx][x_idx], left_edge_block, dtype=int)))

  if (SAD_top < SAD_left):
    mode = 1
    return [mode], top_edge_block

  return [mode], left_edge_block

def extract_predicted_block(frame_buff, head_idy, head_idx, mv, i, FMEEnable):
  if (FMEEnable):
    if ((mv[0] % 2 == 0) and (mv[1] % 2 == 0)): # none fractional
      extracted = frame_buff[mv[2]][head_idy + mv[0] : head_idy + mv[0] + i, head_idx + mv[1] : head_idx + mv[1] + i]
    elif (mv[0] % 2 and mv[1] % 2): # both fractional
      dy_dir = int(mv[0]/2)
      dx_dir = int(mv[1]/2)
            
      extracted = (frame_buff[mv[2]][head_idy + dy_dir : head_idy + dy_dir + i, head_idx + dx_dir : head_idx + dx_dir + i] +
        frame_buff[mv[2]][head_idy + dy_dir + 1 : head_idy + dy_dir + i + 1, head_idx + dx_dir : head_idx + dx_dir + i] +
        frame_buff[mv[2]][head_idy + dy_dir : head_idy + dy_dir + i, head_idx + dx_dir + 1 : head_idx + dx_dir + i + 1] +
        frame_buff[mv[2]][head_idy + dy_dir + 1: head_idy + dy_dir + i + 1, head_idx + dx_dir + 1 : head_idx + dx_dir + i + 1]) // 4
    elif (mv[1] % 2): # x fractional
      dy_dir = int(mv[0]/2)
      dx_dir = int(mv[1]/2)
            
      extracted = (frame_buff[mv[2]][head_idy + dy_dir : head_idy + dy_dir + i, head_idx + dx_dir : head_idx + dx_dir + i] +
        frame_buff[mv[2]][head_idy + dy_dir : head_idy + dy_dir + i, head_idx + dx_dir + 1 : head_idx + dx_dir + i + 1]) // 2
    else: # y fractional
      dy_dir = int(mv[0]/2)
      dx_dir = int(mv[1]/2)
            
      extracted = (frame_buff[mv[2]][head_idy + dy_dir : head_idy + dy_dir + i, head_idx + dx_dir : head_idx + dx_dir + i] +
        frame_buff[mv[2]][head_idy + dy_dir + 1 : head_idy + dy_dir + i + 1, head_idx + dx_dir : head_idx + dx_dir + i]) // 2

  else:
    # print("\n")
    # print(mv)
    # print(head_idy, head_idx, i)
    # print(frame_buff.shape)
    # quit()
    extracted = frame_buff[mv[2]][head_idy + mv[0] : head_idy + mv[0] + i, head_idx + mv[1] : head_idx + mv[1] + i]
    
    # print(extracted.shape)

  return extracted

def extract_block(frame_buff, head_idy, head_idx, modes_mv, i, VBSEnable, FMEEnable):

  if(VBSEnable):
    extracted = []
    # print(modes_mv)
    for idx, mv in reversed(list(enumerate(modes_mv))):
      if (mv == 0):
        extracted = extract_predicted_block(frame_buff, head_idy, head_idx, modes_mv[idx + 1], i, FMEEnable)
        
        return 0, extracted

      elif (mv == 1):
        sub_i = (int)(i / 2)
        sub_head_idy = [head_idy, head_idy, head_idy + sub_i, head_idy + sub_i]
        sub_head_idx = [head_idx, head_idx + sub_i, head_idx, head_idx + sub_i]

        for sub_idx in range(4):
          extracted += [extract_predicted_block(frame_buff, sub_head_idy[sub_idx], sub_head_idx[sub_idx], modes_mv[idx + sub_idx + 1], sub_i, FMEEnable)]

        conc_0 = np.concatenate((extracted[0], extracted[1]), axis=1)
        conc_1 = np.concatenate((extracted[2], extracted[3]), axis=1)

        extracted = np.concatenate((conc_0, conc_1), axis=0)
        
        return 1, extracted
  else:
    extracted = extract_predicted_block(frame_buff, head_idy, head_idx, modes_mv[-1], i, FMEEnable)

    return 0, extracted

def calculate_residual_block(block, predicted):

  residual = np.subtract(block, predicted)

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



def decoder_core(QTC, Q, sub_Q, predicted_block, split):
  approx_residual = []
  recontructed_block = 0
  if (split == 0):
    approx_residual = idct2D(np.multiply(QTC, Q))
    recontructed_block = np.add(approx_residual, predicted_block)
  else:
    sub_i = (int)(i / 2)

    # Extract the sub-blocks
    sub_block = []
    residual = []

    sub_block += [QTC[0: sub_i, 0: sub_i]]
    sub_block += [QTC[0: sub_i, sub_i: sub_i + sub_i]]
    sub_block += [QTC[sub_i: sub_i + sub_i, 0: sub_i]]
    sub_block += [QTC[sub_i: sub_i + sub_i, sub_i: sub_i + sub_i]]

    for idx in range(4):
      residual += [idct2D(np.multiply(sub_block[idx], sub_Q))]

    conc_0 = np.concatenate((residual[0], residual[1]), axis=1)
    conc_1 = np.concatenate((residual[2], residual[3]), axis=1)
    approx_residual = np.concatenate((conc_0, conc_1), axis=0)

    recontructed_block = np.add(approx_residual, predicted_block)

  for y_it in range(recontructed_block.shape[0]):
    for x_it in range(recontructed_block.shape[1]):
      recontructed_block[y_it][x_it] = max(0, min(recontructed_block[y_it][x_it], 255))

  return recontructed_block

def calculate_reconstructed_image(residual_matrix, reconstructed, ext_y_res, ext_x_res, n_y_blocks, n_x_blocks, mv):

  new_reconstructed = decoder_core(residual_matrix, reconstructed, mv)

  return new_reconstructed

def predict_block(rec_buffer, new_reconstructed, modes_mv, bl_y_it, bl_x_it, i, is_p_block, VBSEnable, FMEEnable):

  grey = 128
  predicted_block = np.empty((i, i), dtype=int)
  split = 0
  if(is_p_block):
          # predicted_block = extract_block(rec_buffer, bl_y_it * i, bl_x_it * i, modes_mv, i, FMEEnable)
          
          split, predicted_block = extract_block(rec_buffer, bl_y_it * i, bl_x_it * i, modes_mv, i, VBSEnable, FMEEnable)

  else:
    modes_mv = modes_mv[-1]
    top_edge = np.full((1, i), grey)
    left_edge = np.full((i, 1), grey)

    predicted_block[:,:] = top_edge

    if ((modes_mv == 1) and ((bl_y_it - 1) >= 0)):
      top_edge = new_reconstructed[bl_y_it - 1][bl_x_it][-1, :]
      predicted_block[:,:] = top_edge

    if ((modes_mv == 0) and ((bl_x_it - 1) >= 0)):
      left_edge = new_reconstructed[bl_y_it][bl_x_it - 1][:, -1].reshape((i, 1))
      predicted_block[:, :] = left_edge

  return split, predicted_block

def differential_encoder_decoder(modes_mv_block, is_p_block, not_first_bl):
  return_data = 0
  if (is_p_block):
    if (not_first_bl):
      return_data = list(np.array(modes_mv_block[-2]) - np.array(modes_mv_block[-1]))
    else:
      return_data = list(np.array([0, 0, 0]) - np.array(modes_mv_block[-1]))

  else:
    if (not_first_bl):
      return_data = modes_mv_block[-2] - modes_mv_block[-1]
    else:
      return_data = 0 - modes_mv_block[-1]

  return [return_data]

def differential_encoder_decoder_vbs(modes_mv_block, is_p_block, not_first_bl):
  return_data = []
  if (is_p_block):
    if (not_first_bl):
      for idx, mv in reversed(list(enumerate(modes_mv_block))):
        
        if (mv == 0):
          for prev_idx, prev_mv in reversed(list(enumerate(modes_mv_block[:idx]))):
            if ((prev_mv == 0) or (prev_mv == 1)):
              return_data += [modes_mv_block[prev_idx] - modes_mv_block[idx]]
              break
          return_data += [list(np.array(modes_mv_block[idx - 1]) - np.array(modes_mv_block[idx + 1]))]
          break
        
        elif (mv == 1):

          for prev_idx, prev_mv in reversed(list(enumerate(modes_mv_block[:idx]))):
            if ((prev_mv == 0) or (prev_mv == 1)):
              return_data += [modes_mv_block[prev_idx] - modes_mv_block[idx]]
              break
          return_data += [list(np.array(modes_mv_block[idx - 1]) - np.array(modes_mv_block[idx + 1]))]
          return_data += [list(np.array(modes_mv_block[idx + 1]) - np.array(modes_mv_block[idx + 2]))]
          return_data += [list(np.array(modes_mv_block[idx + 2]) - np.array(modes_mv_block[idx + 3]))]
          return_data += [list(np.array(modes_mv_block[idx + 3]) - np.array(modes_mv_block[idx + 4]))]
          break      
    else:
      for idx, mv in reversed(list(enumerate(modes_mv_block))):          
        if (mv == 0):
          return_data += [0 - modes_mv_block[idx]]
          return_data += [list(np.array([0, 0, 0]) - np.array(modes_mv_block[idx + 1]))]
          break          
        elif (mv == 1):
          return_data += [0 - modes_mv_block[idx]]
          return_data += [list(np.array([0, 0, 0]) - np.array(modes_mv_block[idx + 1]))]
          return_data += [list(np.array(modes_mv_block[idx + 1]) - np.array(modes_mv_block[idx + 2]))]
          return_data += [list(np.array(modes_mv_block[idx + 2]) - np.array(modes_mv_block[idx + 3]))]
          return_data += [list(np.array(modes_mv_block[idx + 3]) - np.array(modes_mv_block[idx + 4]))]
          break
  else:
    if (not_first_bl):
      return_data += modes_mv_block[-2] - modes_mv_block[-1]
    else:
      return_data += 0 - modes_mv_block[-1]

  return return_data

  

def exp_golomb_coding(number):

  if (number <= 0):
    number =  -2 * number
  else:
    number = (2 * number) - 1

  bitstream = "{0:b}".format(number + 1)
  bitstream = ('0' * (len(bitstream) - 1)) + bitstream

  return bitstream
  

def scanning(block):
  i = block.shape[0]
  scanned = []
  for k in range(i * 2):
    for y in range(k+1):
      x = k - y
      if ( y < i and x < i ):
        scanned += [block[y][x]]
  return scanned

def RLE(data):
  i = 0
  skip = False
  rled = []
  while (i < len(data)):
    k = i
    if (data[k] != 0):
      while (data[k] != 0):
        if ((k+1) < len(data)):
          k += 1
        else:
          k += 1
          rled += [-(k - i)]
          rled += data[i: (i + (k - i))]
          skip = True
          break
      if (skip):
        break
      rled += [-(k - i)]
      rled += data[i: (i + (k - i))]
      i = k

    else:
      while (data[k] == 0):
        if ((k+1) < len(data)):
          k += 1
        else:
          k += 1
          rled += [0]
          skip = True
          break

      if (skip):
        break
      rled += [(k - i)]
      i = k
  return rled

def _to_Bytes(data):
  b = bytearray()
  for i in range(0, len(data), 8):
    b.append(int(data[i:i+8], 2))
  return bytes(b)

def write_encoded(bitstream, file):
  file.write(_to_Bytes(bitstream))

def I_golomb(bitstream, idx):
  if (int(bitstream[idx]) == 1):
    idx += 1
    return 0, idx

  else:
    zero_counter = 0
    while (int(bitstream[idx]) == 0):
      zero_counter += 1
      idx += 1

    number = int(bitstream[idx: idx + zero_counter + 1], 2) - 1
    idx += zero_counter + 1

  if (number % 2):
    return (int)((1 + number)/2), idx
  else:
    return (int)(-number/2), idx

def I_RLE(data, i):
    qtc_line = []
    elements_per_block = i*i
    iterat = 0
    while (iterat < len(data)):
        if(data[iterat]<0): #non-zero
            add_index = abs(data[iterat])
            qtc_line += data[iterat+1:iterat + add_index + 1]
            iterat += add_index + 1
        elif(data[iterat]>0): #zero
            qtc_line += [0]*data[iterat]
            iterat +=  1
        else: #end of block
            missing_elements = elements_per_block - (len(qtc_line) % elements_per_block)
            qtc_line += [0]*missing_elements
            iterat += 1
    return qtc_line

def I_scanning(qtc, i, lin_it):
  bl_y_frame = np.empty((i, i), dtype=int)
  for k in range(i * 2):
    for y in range(k+1):
      x = k - y
      if (y < i and x < i):
        bl_y_frame[y][x] = qtc[lin_it]
        lin_it += 1
  return bl_y_frame, lin_it   

def calc_RDO(pred_block, cur_block, Q, lambda_coeff):
  residual = np.subtract(cur_block, pred_block, dtype=int)
  transformed = dct2D(residual)
  quantized = quantize_dct(transformed, Q)
  scanned = scanning(quantized)
  rled_block = RLE(scanned)

  qtc_bitstream = ''
  for rled in rled_block:
    qtc_bitstream += exp_golomb_coding(int(rled))

  R = len(qtc_bitstream) # Number of bits needed

  D = np.sum(np.abs(residual))  # SAD
  
  J = D + (lambda_coeff * R)  # RDO
  
  return J

def transform_quantize(residual_matrix, frame, bl_y_it, bl_x_it, Q, sub_Q, split, i):
  
  QTC = 0

  if (split == 0):
    transformed_dct = dct2D(residual_matrix[frame][bl_y_it][bl_x_it])
    QTC = quantize_dct(transformed_dct, Q)

  else:

    block = residual_matrix[frame][bl_y_it][bl_x_it]
    sub_i = (int)(i / 2)

    # Extract the sub-blocks
    sub_block = []
    quant = []

    sub_block += [block[0: sub_i, 0: sub_i]]
    sub_block += [block[0: sub_i, sub_i: sub_i + sub_i]]
    sub_block += [block[sub_i: sub_i + sub_i, 0: sub_i]]
    sub_block += [block[sub_i: sub_i + sub_i, sub_i: sub_i + sub_i]]

    for idx in range(4):
      transformed_dct = dct2D(sub_block[idx])
      quant += [quantize_dct(transformed_dct, sub_Q)]

    conc_0 = np.concatenate((quant[0], quant[1]), axis=1)
    conc_1 = np.concatenate((quant[2], quant[3]), axis=1)
    QTC = np.concatenate((conc_0, conc_1), axis=0)


  return QTC


##############################################################################
##############################################################################

def encoder(in_file, out_file, number_frames, y_res, x_res, i, r, QP, i_period, nRefFrames, VBSEnable, FMEEnable):
  if (nRefFrames > (i_period - 1)):
    print("nRefFrames is incompatible with i_period.")
    return

  print("----------------------------------------------")
  print("----------------------------------------------")
  print("Q4 Encoder Parameters-")
  print("----------------------------------------------")
  print("in_file: ", in_file)
  print("out_file: ", out_file)
  print("number_frames: ", number_frames)
  print("y_res: ", y_res)
  print("x_res: ", x_res)
  print("i: ", i)
  print("r: ", r)
  print("QP: ", QP)
  print("i_period: ", i_period)
  print("nRefFrames: ", nRefFrames)
  print("----------------------------------------------")

  bl_y_frame, n_y_blocks, n_x_blocks, ext_y_res, ext_x_res = pre.block(in_file, y_res, x_res, number_frames, i)
  # reconst = np.empty((ext_y_res, ext_x_res), dtype=int)

  rec_buffer = np.empty((nRefFrames, ext_y_res, ext_x_res), dtype=int)

  residual_matrix = np.empty((number_frames, n_y_blocks, n_x_blocks, i, i), dtype=np.int16)

  QTC = np.empty((number_frames, n_y_blocks, n_x_blocks, i, i), dtype=int)

  approx_residual = np.empty((number_frames, n_y_blocks, n_x_blocks, i, i), dtype=int)

  Q = calculate_Q(i, QP)

  sub_QP = 0
  if (QP == 0):
    sub_QP = 0
  else:
    sub_QP = QP - 1

  sub_Q = calculate_Q((int)(i/2), sub_QP)

  Constant = 1
  lambda_const = Constant * 2 ** ((QP - 12) / 3)
  split = 0

  new_reconstructed = np.empty((n_y_blocks, n_x_blocks, i, i), dtype = int)

  file_and_extension = out_file.split(".")
  converted_name = ".".join(file_and_extension[:-1]) + ".yuv"
  encoded_name = ".".join(file_and_extension[:-1]) + ".far"

  converted = open(converted_name, "wb")
  encoded_file = open(encoded_name, "wb")

  differentiated_modes_mv_bitstream = ''
  qtc_bitstream = ''

  len_of_frame = []

  for frame in range(number_frames):

    differentiated_modes_mv_frame = ''

    bits_in_frame = ''

    modes_mv_block = []

    is_p_block = frame % i_period

    for bl_y_it in range(n_y_blocks) :
      for bl_x_it in range(n_x_blocks):

        # print("--------------- : ", frame)

        predicted_block = np.empty((i, i), dtype=int)
        if (is_p_block):
          # Calculate Motion Vector (inter)
          if(VBSEnable):
            modes_mv_block += motion_vector_estimation_vbs(bl_y_frame[frame][bl_y_it][bl_x_it], rec_buffer, r, bl_y_it * i, bl_x_it * i, ext_y_res, ext_x_res, i, Q, sub_Q, lambda_const, FMEEnable)
          else:
            modes_mv_block += motion_vector_estimation(bl_y_frame[frame][bl_y_it][bl_x_it], rec_buffer, r, bl_y_it * i, bl_x_it * i, ext_y_res, ext_x_res, i, FMEEnable)

          split, predicted_block = extract_block(rec_buffer, bl_y_it * i, bl_x_it * i, modes_mv_block, i, VBSEnable, FMEEnable)

          # print(frame, bl_y_it, bl_x_it, predicted_block[int(i/2)])

        else:
          # Calculate mode (intra)
          temp_mode, predicted_block = intra_prediction(new_reconstructed, bl_y_it, bl_x_it)
          modes_mv_block += temp_mode
          
      #  Calculate Residual Matrix
        residual_matrix[frame][bl_y_it][bl_x_it] = calculate_residual_block(bl_y_frame[frame][bl_y_it][bl_x_it], predicted_block)

      #  Trans/Quant/Rescaling/InvTrans
        QTC[frame][bl_y_it][bl_x_it] = transform_quantize(residual_matrix, frame, bl_y_it, bl_x_it, Q, sub_Q, split, i)

      #  Decode
        new_reconstructed[bl_y_it][bl_x_it] = decoder_core(QTC[frame][bl_y_it][bl_x_it], Q, sub_Q, predicted_block, split)      

      # Differential Encoding
        if (is_p_block):
          if (VBSEnable):
            if (split == 0):
              y_range = 2
            else:
              y_range = 5
            differentiated_modes_mv_frame += exp_golomb_coding(differential_encoder_decoder_vbs(modes_mv_block, is_p_block, bl_x_it)[0])

            for num_mv in range(1, y_range):
              for index in range(3):
                differentiated_modes_mv_frame += exp_golomb_coding(differential_encoder_decoder_vbs(modes_mv_block, is_p_block, bl_x_it)[num_mv][index])
          else:
            differentiated_modes_mv_frame += exp_golomb_coding(differential_encoder_decoder(modes_mv_block, is_p_block, bl_x_it)[0][0])
            differentiated_modes_mv_frame += exp_golomb_coding(differential_encoder_decoder(modes_mv_block, is_p_block, bl_x_it)[0][1])
            differentiated_modes_mv_frame += exp_golomb_coding(differential_encoder_decoder(modes_mv_block, is_p_block, bl_x_it)[0][2])
        else:
          differentiated_modes_mv_frame += exp_golomb_coding(differential_encoder_decoder(modes_mv_block, is_p_block, bl_x_it)[0])

        # Scanning/RLE/writing (QTC)        
        scanned_block = scanning(QTC[frame][bl_y_it][bl_x_it])
        rled_block = RLE(scanned_block)

        
        for rled in rled_block:
          qtc_bitstream += exp_golomb_coding(rled)
          bits_in_frame += exp_golomb_coding(rled)

    len_of_frame += [len(bits_in_frame)]


    # insert i_period data/writing (modes_mv)
    if (is_p_block):
      differentiated_modes_mv_frame = exp_golomb_coding(0) + differentiated_modes_mv_frame
    else:
      differentiated_modes_mv_frame = exp_golomb_coding(1) + differentiated_modes_mv_frame

    differentiated_modes_mv_bitstream += differentiated_modes_mv_frame

    len_of_frame[-1] = len_of_frame[-1] + len(differentiated_modes_mv_frame)

    # Concatenate
    counter = 0
    conc_reconstructed = np.empty((ext_y_res, ext_x_res), dtype = int)
    for bl_y_it in range(n_y_blocks):
      conc = np.concatenate((new_reconstructed[bl_y_it][0], new_reconstructed[bl_y_it][1]), axis=1)
      for bl_x_it in range(2, n_x_blocks):
        conc = np.concatenate((conc, new_reconstructed[bl_y_it][bl_x_it]), axis=1)
      # Write frame (encoder output video)
      for i_it in range(i):
        if (counter < y_res):
          counter += 1
          for x_it in range(x_res):
            conc_reconstructed[bl_y_it*i + i_it][x_it] = conc[i_it][x_it]
            converted.write(((int)(conc[i_it][x_it])).to_bytes(1, byteorder=sys.byteorder))

    # reconst = conc_reconstructed
    if (not is_p_block):
      rec_buffer = np.delete(rec_buffer, np.s_[0:nRefFrames], 0)

    rec_buffer = np.insert(rec_buffer, 0, conc_reconstructed, axis=0)
        
    if(rec_buffer.shape[0] > nRefFrames):
      rec_buffer = np.delete(rec_buffer, (nRefFrames - 1), 0)

    # print(rec_buffer.shape[0])

    pre.progress("Encoding frames: ", frame, number_frames)
    


  # Padding with 1s to make the number of bits divisible by 8
  bits_in_a_byte = 8
  bits_in_mdiff = len(differentiated_modes_mv_bitstream)

  differentiated_modes_mv_bitstream = exp_golomb_coding(y_res) + exp_golomb_coding(x_res) + exp_golomb_coding(i) + exp_golomb_coding(QP) + exp_golomb_coding(nRefFrames) + exp_golomb_coding(FMEEnable) + exp_golomb_coding(VBSEnable) + exp_golomb_coding(bits_in_mdiff) + differentiated_modes_mv_bitstream

  final_bitstream = differentiated_modes_mv_bitstream + qtc_bitstream

  # Padding with 1s to make the number of bits divisible by 8
  final_bitstream = ('1' * (bits_in_a_byte - (len(final_bitstream) % bits_in_a_byte))) + final_bitstream

  write_encoded(final_bitstream, encoded_file)

  converted.close()
  encoded_file.close()

  # return len_of_frame
##############################################################################
##############################################################################

def decoder(in_file, out_file):

  qtc = []
  mdiff = []
  lin_it = 0
  size = 1  # bytes per pixel
  
  qtc_idx = 0
  mdiff_idx = 0
  encoded_idx = 0
  encoded_bitstream = ''


  with open(in_file, 'rb') as file:
    print("Reading " + in_file)
    while True:
        data = file.read(size)
        if not data:
            # eof
            break
        # Get RLEd QTC
        byte = "{0:b}".format(int.from_bytes(data, byteorder=sys.byteorder, signed=False))

        byte = ('0' * (8 - len(byte))) + byte
        
        encoded_bitstream += byte

  # Reading metadata
  while (int(encoded_bitstream[encoded_idx]) == 1):
    encoded_idx += 1
  
  y_res, encoded_idx = I_golomb(encoded_bitstream, encoded_idx)
  x_res, encoded_idx = I_golomb(encoded_bitstream, encoded_idx)
  i, encoded_idx = I_golomb(encoded_bitstream, encoded_idx)
  QP, encoded_idx = I_golomb(encoded_bitstream, encoded_idx)
  nRefFrames, encoded_idx = I_golomb(encoded_bitstream, encoded_idx)
  FMEEnable, encoded_idx = I_golomb(encoded_bitstream, encoded_idx)
  VBSEnable, encoded_idx = I_golomb(encoded_bitstream, encoded_idx)
  

  print("----------------------------------------------")
  print("----------------------------------------------")
  print("Q4 Decoder Parameters-")
  print("----------------------------------------------")
  print("in_file: ", in_file)
  print("out_file: ", out_file)
  print("y_res: ", y_res)
  print("x_res: ", x_res)
  print("i: ", i)
  print("QP: ", QP)
  print("nRefFrames: ", nRefFrames)
  print("FMEEnable: ", FMEEnable)
  print("VBSEnable: ", VBSEnable)
  print("----------------------------------------------")

  bits_in_mdiff, encoded_idx = I_golomb(encoded_bitstream, encoded_idx)
  mdiff_encoded_bitstream = encoded_bitstream[encoded_idx: encoded_idx + bits_in_mdiff]
  qtc_encoded_bitstream = encoded_bitstream[encoded_idx + bits_in_mdiff: ]


  # Reading qtc data
  while (qtc_idx < len(qtc_encoded_bitstream)):
    
    temp, qtc_idx = I_golomb(qtc_encoded_bitstream, qtc_idx)
    qtc += [temp]

  # Skipping padded 1s
  while (int(mdiff_encoded_bitstream[mdiff_idx]) == 1):
    mdiff_idx += 1

  # Reading mdiff data
  while (mdiff_idx < len(mdiff_encoded_bitstream)):
    temp, mdiff_idx = I_golomb(mdiff_encoded_bitstream, mdiff_idx)
    mdiff += [temp]

  ########################################
  # Can be converted to a function
  ########################################
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
  ########################################
  ########################################

  Q = calculate_Q(i, QP)
  sub_QP = 0
  if (QP == 0):
    sub_QP = 0
  else:
    sub_QP = QP - 1
  sub_Q = calculate_Q((int)(i / 2), sub_QP)
  
  # reconst = np.empty((ext_y_res, ext_x_res), dtype=int)
  new_reconstructed = np.empty((n_y_blocks, n_x_blocks, i, i), dtype=int)
  rec_buffer = np.empty((nRefFrames, ext_y_res, ext_x_res), dtype=int)
  
  decoded = open(out_file, "wb")

  # Recover QTC Block:
  # Performing inverse RLE to get scanned QTC  
  qtc = I_RLE(qtc, i)

  number_of_frames = (int)((len(qtc) / (i*i)) / (n_x_blocks * n_y_blocks))

  modes_mv = []
  lin_idx = 0
  
  for frame in range(number_of_frames):
    
    pre.progress("Decoding frames: ", frame, number_of_frames)

    is_i_frame = mdiff[lin_idx]
    lin_idx += 1


    for bl_y_it in range(n_y_blocks):
      prev_mode = 0
      prev_mv = [0, 0, 0]
      for bl_x_it in range(n_x_blocks):

        # Inverse scanning
        QTC_recovered, lin_it = I_scanning(qtc, i, lin_it)
             
        if (is_i_frame):
          prev_mode = prev_mode - mdiff[lin_idx]
          modes_mv += [prev_mode]
          lin_idx += 1

        else:
          if (VBSEnable):
            prev_mode = prev_mode - mdiff[lin_idx]
            vbs_mode = prev_mode
            modes_mv += [vbs_mode]
            lin_idx += 1

            if (vbs_mode == 0):
              y_range = 2
            else:
              y_range = 5

            for num_mv in range(1, y_range):
                prev_mv[0] = prev_mv[0] - mdiff[lin_idx+0]
                prev_mv[1] = prev_mv[1] - mdiff[lin_idx+1]
                prev_mv[2] = prev_mv[2] - mdiff[lin_idx+2]
                modes_mv += [[prev_mv[0], prev_mv[1], prev_mv[2]]]
                lin_idx += 3
            
          else:
            prev_mv[0] = prev_mv[0] - mdiff[lin_idx+0]
            prev_mv[1] = prev_mv[1] - mdiff[lin_idx+1]
            prev_mv[2] = prev_mv[2] - mdiff[lin_idx+2]
            modes_mv += [[prev_mv[0], prev_mv[1], prev_mv[2]]]
            lin_idx += 3
          
        #  Decode
        # print(modes_mv)
        # print(modes_mv[-1])

        split, predicted_block = predict_block(rec_buffer, new_reconstructed, modes_mv, bl_y_it, bl_x_it, i, (not is_i_frame), VBSEnable, FMEEnable)
        
        # print(frame, bl_y_it, bl_x_it, split, predicted_block.shape)

        # print(predicted_block.shape)
        # new_reconstructed[bl_y_it][bl_x_it] = decoder_core(QTC_recovered, Q, predicted_block)

        new_reconstructed[bl_y_it][bl_x_it] = decoder_core(QTC_recovered, Q, sub_Q, predicted_block, split)
        
        # print(frame, bl_y_it, bl_x_it, new_reconstructed[bl_y_it][bl_x_it][int(i/2)])

    # if (frame == 1):
    #   in_block = (n_y_blocks*n_x_blocks)
    #   print(modes_mv[in_block:in_block+30])
    #   quit()
    
    #  Concatenate
    counter = 0
    conc_reconstructed = np.empty((ext_y_res, ext_x_res), dtype = int)
    for bl_y_it in range(n_y_blocks):
      conc = np.concatenate((new_reconstructed[bl_y_it][0], new_reconstructed[bl_y_it][1]), axis=1)
      for bl_x_it in range(2, n_x_blocks):
        conc = np.concatenate((conc, new_reconstructed[bl_y_it][bl_x_it]), axis=1)
      # Write frame (decoder output video)
      for i_it in range(i):
        if (counter < y_res):
          counter += 1
          for x_it in range(x_res):
            conc_reconstructed[bl_y_it*i + i_it][x_it] = conc[i_it][x_it]
            decoded.write(((int)(conc[i_it][x_it])).to_bytes(1, byteorder=sys.byteorder))

    # reconst = conc_reconstructed

    if (is_i_frame):
      rec_buffer = np.delete(rec_buffer, np.s_[0:nRefFrames], 0)

    rec_buffer = np.insert(rec_buffer, 0, conc_reconstructed, axis=0)
        
    if(rec_buffer.shape[0] > nRefFrames):
      rec_buffer = np.delete(rec_buffer, (nRefFrames - 1), 0)

  decoded.close()
  print("Decoding Completed")

##############################################################################
##############################################################################

if __name__ == "__main__":

  in_file = "./videos/black_and_white.yuv"
  out_file = "./temp/assign2_vbs_QP0.far"

  number_frames = 10
  y_res = 288
  x_res = 352
  i = 16
  r = 2
  QP = 0  # from 0 to (log_2(i) + 7)
  i_period = 50
  nRefFrames = 2
  FMEEnable = False
  VBSEnable = True

  # bits_in_each_frame = []

  decoder_infile = out_file
  decoder_outfile = "./videos/q4_decoded.yuv"

  encoder(in_file, out_file, number_frames, y_res, x_res, i, r, QP, i_period, nRefFrames, VBSEnable, FMEEnable)
  decoder(decoder_infile, decoder_outfile)
  
  # y_res = 3
  # x_res = 3
  # reconstructed = np.zeros((y_res, x_res), dtype=int)
  # reconstructed = np.array([[25, 28, 29],[50, 57, 53],[44, 52, 56]])
  # i = 2
  # upscale_frame(reconstructed, y_res, x_res, i)

# ##############################################################################
# ######################### Code for producing plots ###########################
# ##############################################################################
  
#   number_frames = 10
#   y_res = 288
#   x_res = 352
#   i = 8
#   r = 4
#   QP = 3# from 0 to (log_2(i) + 7)
  
#   i_period = 1
#   bits_in_each_frame += [encoder(in_file, out_file, number_frames, y_res, x_res, i, r, QP, i_period)]
  
#   i_period = 4
#   bits_in_each_frame += [encoder(in_file, out_file, number_frames, y_res, x_res, i, r, QP, i_period)]
  
#   i_period = 10
#   bits_in_each_frame += [encoder(in_file, out_file, number_frames, y_res, x_res, i, r, QP, i_period)]

# ##############################################################################

#   fig, ax = plt.subplots()

#   frames = range(1, number_frames+1)
#   ax.plot(frames, bits_in_each_frame[0], '-', label='i_period = 1') 
#   ax.plot(frames, bits_in_each_frame[1], '--', label='i_period = 4') 
#   ax.plot(frames, bits_in_each_frame[2], '-.', label='i_period = 10') 
  
#   ax.set(xlabel='Frame', ylabel='Number of bits')
#   ax.grid()
#   ax.legend()
#   plt.show()

# ##############################################################################
# ##############################################################################