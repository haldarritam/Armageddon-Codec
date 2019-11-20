import numpy as np
import sys
import matplotlib.pyplot as plt
import assign1_Q2_main as pre
from scipy.fftpack import dct, idct

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


def idct2D(block): # Inverse Transform Function
  res = idct(idct(block, axis=0, norm='ortho'), axis=1, norm='ortho')
  return res

def decoder_core(QTC, Q, sub_Q, predicted_block, split, i):
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


def extract_intra_block(new_reconstructed, modes_mv, bl_y_it, bl_x_it, i, VBSEnable, QTC, sub_Q):

  # print("Here")
  grey = 128
  predicted_block = np.empty((i, i), dtype=int)
  split = 0

  if (VBSEnable):
    # print(len(modes_mv))

    for idx, mv in reversed(list(enumerate(modes_mv))):

      if (modes_mv[idx] == [0]):
        modes_mv = modes_mv[idx+1]
        top_edge = np.full((1, i), grey)
        left_edge = np.full((i, 1), grey)

        predicted_block[:,:] = top_edge

        if ((modes_mv == 1) and ((bl_y_it - 1) >= 0)):
          top_edge = new_reconstructed[bl_y_it - 1][bl_x_it][-1, :]
          predicted_block[:,:] = top_edge

        if ((modes_mv == 0) and ((bl_x_it - 1) >= 0)):
          left_edge = new_reconstructed[bl_y_it][bl_x_it - 1][:, -1].reshape((i, 1))
          predicted_block[:, :] = left_edge

          # print(predicted_block)

        break

      elif (modes_mv[idx] == [1]):

        split = 1
        i = int(i / 2)
        predicted_sub = np.zeros((4, i, i), dtype=int)
        sub_block = np.empty((i, i), dtype=int)

        sub_QTC = []
        sub_QTC += [QTC[0: i, 0: i]]
        sub_QTC += [QTC[0: i, i: i + i]]
        sub_QTC += [QTC[i: i + i, 0: i]]
        sub_QTC += [QTC[i: i + i, i: i + i]]

        sub_residuals = []
        sub_residuals += [idct2D(np.multiply(sub_QTC[0], sub_Q))]
        sub_residuals += [idct2D(np.multiply(sub_QTC[1], sub_Q))]
        sub_residuals += [idct2D(np.multiply(sub_QTC[2], sub_Q))]
        sub_residuals += [idct2D(np.multiply(sub_QTC[3], sub_Q))]

        # print(i, sub_block.shape)

        top_edge = [] 
        left_edge = []

        top_edge_1 = np.full((1, i), grey)
        top_edge_2 = np.full((1, i), grey)

        left_edge_1 = np.full((i, 1), grey)
        left_edge_2 = np.full((i, 1), grey)

        if ((bl_y_it - 1) >= 0):
          top_edge_1 = new_reconstructed[bl_y_it - 1][bl_x_it][-1, 0:i]
          top_edge_2 = new_reconstructed[bl_y_it - 1][bl_x_it][-1, i:i+i]
        if ((bl_x_it - 1) >= 0):
          left_edge_1 = new_reconstructed[bl_y_it][bl_x_it - 1][0:i, -1].reshape((i, 1))
          left_edge_2 = new_reconstructed[bl_y_it][bl_x_it - 1][i: i + i, -1].reshape((i, 1))
          
        top_edge += [top_edge_1, top_edge_2]
        left_edge += [left_edge_1, left_edge_2]

        lin_it = 0

        for y_idx in range(2):
          for x_idx in range(2):

            current_mode = modes_mv[idx + lin_it + 1]

            if (current_mode == 0):
              sub_block[:, :] = left_edge[y_idx]
              recontructed_block = np.add(sub_residuals[lin_it], sub_block)        
              top_edge[x_idx] = recontructed_block[-1, :]
              left_edge[y_idx] = recontructed_block[:, -1]
              
            elif (current_mode == 1):
              sub_block[:, :] = top_edge[x_idx]
              recontructed_block = np.add(sub_residuals[lin_it], sub_block)              
              top_edge[x_idx] = recontructed_block[-1, :]
              left_edge[y_idx] = recontructed_block[:, -1]

            predicted_sub[lin_it] = sub_block
            lin_it += 1
        
        conc_0 = np.concatenate((predicted_sub[0], predicted_sub[1]), axis=1)
        conc_1 = np.concatenate((predicted_sub[2], predicted_sub[3]), axis=1)
        predicted_block = np.concatenate((conc_0, conc_1), axis=0)

        break        
    return [split], predicted_block

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


def predict_block(rec_buffer, new_reconstructed, modes_mv, bl_y_it, bl_x_it, i, is_p_block, VBSEnable, FMEEnable, QTC, sub_Q):

  grey = 128
  predicted_block = np.empty((i, i), dtype=int)
  split = 0

  # print(modes_mv)

  if(is_p_block):    
    split, predicted_block = extract_block(rec_buffer, bl_y_it * i, bl_x_it * i, modes_mv, i, VBSEnable, FMEEnable)
  else:
    split, predicted_block = extract_intra_block(new_reconstructed, modes_mv, bl_y_it, bl_x_it, i, VBSEnable, QTC, sub_Q)

  return split, predicted_block


def I_scanning(qtc, i, lin_it):
  bl_y_frame = np.empty((i, i), dtype=int)
  for k in range(i * 2):
    for y in range(k+1):
      x = k - y
      if (y < i and x < i):
        bl_y_frame[y][x] = qtc[lin_it]
        lin_it += 1
  return bl_y_frame, lin_it   


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

def nRef_tool(n_past_frames, view_new_reconstructed):
  if(n_past_frames == 1):
    view_new_reconstructed[1,-2] = replace_value_for(view_new_reconstructed[1,-2])
  elif(n_past_frames == 2):
    view_new_reconstructed[1,1] = replace_value_for(view_new_reconstructed[1,1])
  elif(n_past_frames == 3):
    view_new_reconstructed[-2,1] = replace_value_for(view_new_reconstructed[-2,1])
  elif(n_past_frames == 4):
    view_new_reconstructed[-2,-2] = replace_value_for(view_new_reconstructed[-2,-2])
  elif(n_past_frames == 5):
    view_new_reconstructed[1,-2] = replace_value_for(view_new_reconstructed[1,-2])
    view_new_reconstructed[1,1] = replace_value_for(view_new_reconstructed[1,1])
  elif(n_past_frames == 6):
    view_new_reconstructed[1,1] = replace_value_for(view_new_reconstructed[1,1])
    view_new_reconstructed[-2,1] = replace_value_for(view_new_reconstructed[-2,1])
  elif(n_past_frames == 7):
    view_new_reconstructed[-2,1] = replace_value_for(view_new_reconstructed[-2,1])
    view_new_reconstructed[-2,-2] = replace_value_for(view_new_reconstructed[-2,-2])
  elif(n_past_frames == 8):
    view_new_reconstructed[-2,-2] = replace_value_for(view_new_reconstructed[-2,-2])
    view_new_reconstructed[1,-2] = replace_value_for(view_new_reconstructed[1,-2])
  elif(n_past_frames == 9):
    view_new_reconstructed[1,-2] = replace_value_for(view_new_reconstructed[1,-2])
    view_new_reconstructed[1,1] = replace_value_for(view_new_reconstructed[1,1])
    view_new_reconstructed[-2,1] = replace_value_for(view_new_reconstructed[-2,1])  
  elif(n_past_frames == 10):
    view_new_reconstructed[1,1] = replace_value_for(view_new_reconstructed[1,1])
    view_new_reconstructed[-2,1] = replace_value_for(view_new_reconstructed[-2,1])
    view_new_reconstructed[-2,-2] = replace_value_for(view_new_reconstructed[-2,-2])      
  elif(n_past_frames == 11):
    view_new_reconstructed[-2,1] = replace_value_for(view_new_reconstructed[-2,1])
    view_new_reconstructed[-2,-2] = replace_value_for(view_new_reconstructed[-2,-2])
    view_new_reconstructed[1,-2] = replace_value_for(view_new_reconstructed[1,-2])    
  elif(n_past_frames == 12):
    view_new_reconstructed[-2,-2] = replace_value_for(view_new_reconstructed[-2,-2])
    view_new_reconstructed[1,-2] = replace_value_for(view_new_reconstructed[1,-2])
    view_new_reconstructed[1,1] = replace_value_for(view_new_reconstructed[1,1])       
  elif(n_past_frames >= 13):
    view_new_reconstructed[1,1] = replace_value_for(view_new_reconstructed[1,1])
    view_new_reconstructed[1,-2] = replace_value_for(view_new_reconstructed[1,-2])
    view_new_reconstructed[-2,1] = replace_value_for(view_new_reconstructed[-2,1])
    view_new_reconstructed[-2, -2] = replace_value_for(view_new_reconstructed[-2, -2])

  return view_new_reconstructed


def draw_line(curr_pos, dest_pos, frame):

  y0 = curr_pos[0]
  x0 = curr_pos[1]

  y1 = dest_pos[0]
  x1 = dest_pos[1]

  deltax = 0
  deltay = 0
  error = 0
  x = 0
  y = 0
  y_step = 0
  is_steep = (abs(y1 - y0) > abs(x1 - x0))

  if (is_steep):
    x0, y0 = y0, x0
    x1, y1 = y1, x1
  if (x0 > x1):
    x0, x1 = x1, x0
    y0, y1 = y1, y0

  deltax = x1 - x0
  deltay = abs(y1 - y0)
  error = -(deltax / 2)
  y = y0

  if (y0 < y1):
    y_step = 1
  else:
    y_step = -1
	
  for x in range(x0, x1 + 1):
    
    # print(y, x)
    if (is_steep):
      frame[x][y] = 0
    else:
      frame[y][x] = 0

    error = error + deltay

    if (error > 0):
      y = y + y_step
      error = error - deltax
    
    # print(y1, x1, frame.shape)
    frame[curr_pos[0]][curr_pos[1]] = 255

  return frame


                
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

  lin_iterator = 0
  
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
          if (VBSEnable):
            prev_mode = prev_mode - mdiff[lin_idx]
            vbs_mode = prev_mode
            modes_mv += [[vbs_mode]]
            lin_idx += 1

            if (vbs_mode == 0):
              prev_mode = prev_mode - mdiff[lin_idx]
              modes_mv += [prev_mode]
              lin_idx += 1
            else:
              for _ in range(4):
                prev_mode = prev_mode - mdiff[lin_idx]
                modes_mv += [prev_mode]
                lin_idx += 1

          else:
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
        split, predicted_block = predict_block(rec_buffer, new_reconstructed, modes_mv, bl_y_it, bl_x_it, i, (not is_i_frame), VBSEnable, FMEEnable, QTC_recovered, sub_Q)

        new_reconstructed[bl_y_it][bl_x_it] = decoder_core(QTC_recovered, Q, sub_Q, predicted_block, split, i)
    
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
            # decoded.write(((int)(conc[i_it][x_it])).to_bytes(1, byteorder=sys.byteorder))

    if (is_i_frame):
      rec_buffer = np.delete(rec_buffer, np.s_[0:nRefFrames], 0)

    rec_buffer = np.insert(rec_buffer, 0, conc_reconstructed, axis=0)

    if(rec_buffer.shape[0] > nRefFrames):
      rec_buffer = np.delete(rec_buffer, nRefFrames, 0)        
    
    for y_mv in range(int(i/2), ext_y_res, i):
      for x_mv in range(int(i / 2), ext_x_res, i):
        if (VBSEnable):
          if (is_i_frame):
            if (modes_mv[lin_iterator] == [0]):
              lin_iterator += 1
              if ((y_mv < y_res) and (x_mv < x_res)):
                if (modes_mv[lin_iterator] == 0):
                  conc_reconstructed[y_mv][x_mv] = 0
                else:
                  conc_reconstructed[y_mv][x_mv] = 255
              lin_iterator += 1
            else:
              lin_iterator += 1

              sub_i_mv = int(i/4)
              sub_y_mv = [(y_mv+sub_i_mv), (y_mv+sub_i_mv), (y_mv-sub_i_mv), (y_mv-sub_i_mv)]
              sub_x_mv = [(x_mv - sub_i_mv), (x_mv + sub_i_mv), (x_mv - sub_i_mv), (x_mv + sub_i_mv)]
              
              for index in range(4):
                if ((sub_y_mv[index] < y_res) and (sub_x_mv[index] < x_res)):

                  if (modes_mv[lin_iterator] == 0):
                    conc_reconstructed[sub_y_mv[index]][sub_x_mv[index]] = 0
                  else:
                    conc_reconstructed[sub_y_mv[index]][sub_x_mv[index]] = 255

                lin_iterator += 1
          else:
            if (modes_mv[lin_iterator] == 0):
              lin_iterator += 1
              if ((y_mv < y_res) and (x_mv < x_res)):
                source = [y_mv, x_mv]
                destination = [(y_mv + modes_mv[lin_iterator][0]), (x_mv + modes_mv[lin_iterator][1])]

                conc_reconstructed = draw_line(source, destination, conc_reconstructed)
              lin_iterator += 1
            else:
              lin_iterator += 1
              
              sub_i_mv = int(i/4)
              sub_y_mv = [(y_mv+sub_i_mv), (y_mv+sub_i_mv), (y_mv-sub_i_mv), (y_mv-sub_i_mv)]
              sub_x_mv = [(x_mv - sub_i_mv), (x_mv + sub_i_mv), (x_mv - sub_i_mv), (x_mv + sub_i_mv)]
              
              for index in range(4):
                if ((sub_y_mv[index] < y_res) and (sub_x_mv[index] < x_res)):

                  source = [sub_y_mv[index], sub_x_mv[index]]
                  destination = [(sub_y_mv[index] + modes_mv[lin_iterator][0]), (sub_x_mv[index] + modes_mv[lin_iterator][1])]

                  conc_reconstructed = draw_line(source, destination, conc_reconstructed)

                lin_iterator += 1

        else:
          if (is_i_frame):
            if ((y_mv < y_res) and (x_mv < x_res)):
              if (modes_mv[lin_iterator] == 0):
                conc_reconstructed[y_mv][x_mv] = 0
              else:
                conc_reconstructed[y_mv][x_mv] = 255

            lin_iterator += 1
          else:
            if ((y_mv < y_res) and (x_mv < x_res)):
              # print(y_mv, x_mv, modes_mv[lin_iterator])
              source = [y_mv, x_mv]
              destination = [(y_mv + modes_mv[lin_iterator][0]), (x_mv + modes_mv[lin_iterator][1])]

              conc_reconstructed = draw_line(source, destination, conc_reconstructed)

            lin_iterator += 1

    for y_write in range(y_res):
      for x_write in range(x_res):
        decoded.write(((int)(conc_reconstructed[y_write][x_write])).to_bytes(1, byteorder=sys.byteorder))

  decoded.close()
  print("Decoding Completed")

def replace_value_for(current_value):
    if(current_value > 128):
        return 0
    else:
        return 255

##############################################################################
##############################################################################

if __name__ == "__main__":

  in_file = "./temp/mv_tool_test.far"
  decoder_outfile = "./videos/a2_mv_tool.yuv"

  decoder(in_file, decoder_outfile)