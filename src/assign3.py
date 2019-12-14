import numpy as np
import sys
import concurrent.futures
import matplotlib.pyplot as plt
import assign1_Q2_main as pre
from scipy.fftpack import dct, idct
import copy

def dct2D(block):  # Transform Function
  res = dct(dct(block, axis=0, norm='ortho'), axis=1, norm='ortho')
  return res

def idct2D(block): # Inverse Transform Function
  res = idct(idct(block, axis=0, norm='ortho'), axis=1, norm='ortho')
  return res

def find_mv(block, rec_buffer, r, head_idy, head_idx, ext_y_res, ext_x_res, i, origin, FMEEnabled, rec_buffer_pos = -1):
    mv = (0, 0, 0)
    best_SAD = i * i * 255 + 1  # The sum can never exceed (i * i * 255 + 1)
    up_r = r
    
    if (FMEEnabled):
      up_r = 2 * r

    negative = -up_r
    positive = (up_r + 1)
    check = 1
    
    if(origin!=1):
        # print('HERE')
        negative = -1
        positive = 2

    for buff_idx, reconstructed in enumerate(rec_buffer):
        if (rec_buffer_pos != -1 and buff_idx == rec_buffer_pos) or (rec_buffer_pos == -1):
          for y_dir in range(negative, positive):
              for x_dir in range(negative, positive):
                  if((origin != 1 and (check%2)==0) or origin == 1):
                      if ((head_idy + y_dir) >= 0 and (head_idy + y_dir + i) < ext_y_res and (head_idx + x_dir) >= 0 and (head_idx + x_dir + i) < ext_x_res):
                          #print(reconstructed.shape)
                          # extracted = reconstructed[head_idy + y_dir : head_idy + y_dir + i, head_idx + x_dir : head_idx + x_dir + i]
                          extracted = FME_extraction(FMEEnabled, i, head_idy, head_idx, y_dir, x_dir, ext_y_res, ext_x_res, reconstructed)

                          if(type(extracted)!= int):
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

                  if(origin!=1):
                      check = check + 1
    # print("Mv and best SAD are: ", mv,best_SAD)
    # print("+++++++")

    return mv, best_SAD

def find_mv_vbs(block, rec_buffer, r, head_idy, head_idx, ext_y_res, ext_x_res, i, origin, FMEEnabled, best_RDO_block, lambda_const, Q, rec_buffer_pos = -1):
    mv = (0, 0, 0)
    up_r = r    
    
    if (FMEEnabled):
      up_r = 2 * r

    negative = -up_r
    positive = (up_r + 1)
    check = 1
    
    if(origin!=1):
        # print('HERE')
        negative = -1
        positive = 2

    for buff_idx, reconstructed in enumerate(rec_buffer):
        if (rec_buffer_pos != -1 and buff_idx == rec_buffer_pos) or (rec_buffer_pos == -1):
          for y_dir in range(negative, positive):
              for x_dir in range(negative, positive):
                  if ((origin != 1 and (check % 2) == 0) or origin == 1):
                    
                      if ((head_idy + y_dir) >= 0 and (head_idy + y_dir + i) < ext_y_res and (head_idx + x_dir) >= 0 and (head_idx + x_dir + i) < ext_x_res):

                          extracted = FME_extraction(FMEEnabled, i, head_idy, head_idx, y_dir, x_dir, ext_y_res, ext_x_res, reconstructed)

                          if(type(extracted)!= int):
                            best_RDO_block, mv = RDO_sel(extracted, block, Q, lambda_const, y_dir, x_dir, buff_idx, mv, best_RDO_block)

                  if(origin!=1):
                      check = check + 1
    # print("Mv and best SAD are: ", mv,best_SAD)
    # print("+++++++")
    return mv, best_RDO_block
    
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

    elif ((x_dir % 2) and (y_dir % 2)): # both fractional
      #print("BOTH FRAC", y_dir, x_dir)
      dy_dir = int(y_dir/2)
      dx_dir = int(x_dir/2)
      
      #print(head_idy, dy_dir, head_idx, dx_dir, y_dir, x_dir)
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

    # print(extracted.shape)
  return extracted

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

def motion_vector_estimation(block, rec_buffer, r, head_idy, head_idx, ext_y_res, ext_x_res, i, FMEEnabled, FastME, nRefFrames):

    if(FastME):
      # print('Entered')
      origin = 1
      iterate = 1
      mv_accum = []
      mv_new = ()
      mv_test = ()

      mv_new,sad_new = find_mv(block, rec_buffer, r, head_idy, head_idx, ext_y_res, ext_x_res, i, origin, FMEEnabled)
      # print('Got mv')
      # print(type(mv_new))
      origin = 2

      if mv_new[0] == 0 and mv_new[1] == 0:
        #print("MV points to origin")
        return [mv_new]

      #print(head_idy, head_idx)
      head_idy += mv_new[0]
      head_idx += mv_new[1]
      #print(mv_new)

      if nRefFrames > 1:
        ref_frame = mv_new[2]
      else:
        ref_frame = -1

      #print("Ref Frame is ", ref_frame)

      while(iterate):
          #print("Head is ", head_idy, head_idx)
          mv_test,sad_test = find_mv(block, rec_buffer, r, head_idy, head_idx, ext_y_res, ext_x_res, i, origin, FMEEnabled, ref_frame)
          # print('Got sad')
          #print ("SADs: ", sad_test, sad_new)
          if(sad_test < sad_new):
              sad_new = sad_test
              head_idy += mv_test[0]
              head_idx += mv_test[1]
              #print(mv_new)
              #print(mv_new, mv_test)
              if abs(mv_new[0] + mv_test[0]) > 16:
                break

              if abs(mv_new[1] + mv_test[1]) > 16:
                break

              mv_new = (mv_new[0]+mv_test[0], mv_new[1]+mv_test[1], ref_frame)
              #print(mv_new)
              #print(mv_new)
              #print(mv_new)
              #print ("Better MV found!")
          else:
              iterate = 0

      return [mv_new]
    else:
      # print("new else")
      origin = 1
      mv_non_fme, sad_non_fme = find_mv(block, rec_buffer, r, head_idy, head_idx, ext_y_res, ext_x_res, i, origin, FMEEnabled)
      return [mv_non_fme]

def fast_mv_vbs(block, rec_buffer, r, head_idy, head_idx, ext_y_res, ext_x_res, i, FMEEnabled, FastME, lambda_const, Q, best_RDO_block, nRefFrames):

    if(FastME):
      origin = 1
      iterate = 1
      mv_accum = []
      mv_new = ()
      mv_test = ()

      mv_new, sad_new = find_mv_vbs(block, rec_buffer, r, head_idy, head_idx, ext_y_res, ext_x_res, i, origin, FMEEnabled, best_RDO_block, lambda_const, Q)
      
      origin = 2

      if mv_new[0] == 0 and mv_new[1] == 0:
        return mv_new, sad_new

      head_idy += mv_new[0]
      head_idx += mv_new[1]

      if nRefFrames > 1:
        ref_frame = mv_new[2]
      else:
        ref_frame = -1

      while(iterate):          
          mv_test, sad_test = find_mv_vbs(block, rec_buffer, r, head_idy, head_idx, ext_y_res, ext_x_res, i, origin, FMEEnabled, best_RDO_block, lambda_const, Q)

          if(sad_test < sad_new):
              sad_new = sad_test
              head_idy += mv_test[0]
              head_idx += mv_test[1]

              if abs(mv_new[0] + mv_test[0]) > 16:
                break

              if abs(mv_new[1] + mv_test[1]) > 16:
                break

              mv_new = (mv_new[0]+mv_test[0], mv_new[1]+mv_test[1], ref_frame)
          else:
              iterate = 0

      return mv_new, sad_new
    else:
      origin = 1
      mv_non_fastme, sad_non_fastme = find_mv_vbs(block, rec_buffer, r, head_idy, head_idx, ext_y_res, ext_x_res, i, origin, FMEEnabled, best_RDO_block, lambda_const, Q)

      # print(mv_non_fastme)
      return mv_non_fastme, sad_non_fastme


def motion_vector_estimation_vbs(block, rec_buffer, r, head_idy, head_idx, ext_y_res, ext_x_res, i, Q, sub_Q, lambda_const, FMEEnabled, FastME, nRefFrames, RCflag, split, mv_input, mv_iterator):

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
  sub_mv_ret = sub_mv
  mv_factor = 1

  if RCflag == 3 and FMEEnabled:
    mv_factor = 2

  if ((RCflag == 0) or (RCflag == 1) or (RCflag == 2)):

    #print("RC: 0, 1 or 2" )

    mv, best_RDO_block = fast_mv_vbs(block, rec_buffer, r, head_idy, head_idx, ext_y_res, ext_x_res, i, FMEEnabled, FastME, lambda_const, Q, best_RDO_block, nRefFrames)
      # Sub-block prediction
    for sub_idx in range(4):
      sub_mv[sub_idx], best_RDO_sub[sub_idx] = fast_mv_vbs(sub_block[sub_idx], rec_buffer, r, sub_head_idy[sub_idx], sub_head_idx[sub_idx], ext_y_res, ext_x_res, sub_i, FMEEnabled, FastME, lambda_const, sub_Q, best_RDO_sub[sub_idx], nRefFrames)
    RDO_sub = best_RDO_sub[0] + best_RDO_sub[1] + best_RDO_sub[2] + best_RDO_sub[3]
    if (best_RDO_block < RDO_sub):
      return [0, mv], 0
    else:
      return [1, sub_mv[0], sub_mv[1], sub_mv[2], sub_mv[3]], 0

  elif ((RCflag == 3) and split == 0):

    #print("RC: 3, split: 0" )

    mv, best_RDO_block = fast_mv_vbs(block, rec_buffer, r, head_idy + mv_input[mv_iterator][0], head_idx + mv_input[mv_iterator][1], ext_y_res, ext_x_res, i, FMEEnabled, False, lambda_const, Q, best_RDO_block, nRefFrames)

    mv_ret = [mv[0] + mv_input[mv_iterator][0]*mv_factor, mv[1] + mv_input[mv_iterator][1]*mv_factor, mv_input[mv_iterator][2]]
    mv_iterator += 1
    return [0, mv_ret], mv_iterator

  elif ((RCflag == 3) and split == 1):

    #print("RC: 3, split: 1" )

    for sub_idx in range(4):
      sub_mv[sub_idx], best_RDO_sub[sub_idx] = fast_mv_vbs(sub_block[sub_idx], rec_buffer, r, sub_head_idy[sub_idx] + mv_input[mv_iterator][0], sub_head_idx[sub_idx] + mv_input[mv_iterator][1], ext_y_res, ext_x_res, sub_i, FMEEnabled, False, lambda_const, sub_Q, best_RDO_sub[sub_idx], nRefFrames)
      sub_mv_ret[sub_idx] = [sub_mv[sub_idx][0] + mv_input[mv_iterator][0]*mv_factor, sub_mv[sub_idx][1] + mv_input[mv_iterator][1]*mv_factor, sub_mv[sub_idx][2]]
      mv_iterator += 1

    RDO_sub = best_RDO_sub[0] + best_RDO_sub[1] + best_RDO_sub[2] + best_RDO_sub[3]
    return [1, sub_mv_ret[0], sub_mv_ret[1], sub_mv_ret[2], sub_mv_ret[3]], mv_iterator

def intra_prediction(frame, bl_y_frame, y_idx, x_idx):

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

  SAD_top = np.sum(np.abs(np.subtract(bl_y_frame, top_edge_block, dtype=int)))
  SAD_left = np.sum(np.abs(np.subtract(bl_y_frame, left_edge_block, dtype=int)))

  if (SAD_top < SAD_left):
    mode = 1
    return [mode], top_edge_block

  return [mode], left_edge_block

def intra_prediction_vbs(frame, block, y_idx, x_idx, Q, sub_Q, lambda_const):

  # print(frame[y_idx][x_idx])
  # print("---------------------------------")
  # print("---------------------------------")

  mode_block, intra_block, RDO_block = intra_prediction_block(frame, block, y_idx, x_idx, Q, lambda_const)

  sub_mode, sub_predicted_block, RDO_sub_block = intra_prediction_sub_block(frame, block, y_idx, x_idx, sub_Q, lambda_const)

  if (RDO_block < RDO_sub_block):
    return [0], [mode_block], intra_block
  else:
    return [1], sub_mode, sub_predicted_block

def intra_prediction_block(frame, block, y_idx, x_idx, Q, lambda_const):

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

  RDO_top  = calc_RDO(top_edge_block, block, Q, lambda_const)
  RDO_left = calc_RDO(left_edge_block, block, Q, lambda_const)

  if (RDO_top < RDO_left):
    mode = 1
    return mode, top_edge_block, RDO_top

  return mode, left_edge_block, RDO_left

def intra_prediction_sub_block(frame, block, bl_y_it, bl_x_it, sub_Q, lambda_const):

  i = frame.shape[2]

  if (i != 4 and i != 8 and i != 16):
    print("Block size should be 4, 8 or 16 when VBS enabled!")
    print("Exiting...")
    exit()

  i = int(i / 2)

  sub_block = []
  sub_block += [block[0: i, 0: i]]
  sub_block += [block[0: i, i: i + i]]
  sub_block += [block[i: i + i, 0: i]]
  sub_block += [block[i: i + i, i: i + i]]

  grey = 128

  lefted_sub_block = np.empty((i, i), dtype=int)
  topped_sub_block = np.empty((i, i), dtype=int)

  top_edge = [] 
  left_edge = []

  top_edge_1 = np.full((1, i), grey)
  top_edge_2 = np.full((1, i), grey)

  left_edge_1 = np.full((i, 1), grey)
  left_edge_2 = np.full((i, 1), grey)

  if ((bl_y_it - 1) >= 0):
    top_edge_1 = frame[bl_y_it - 1][bl_x_it][-1, 0:i]
    top_edge_2 = frame[bl_y_it - 1][bl_x_it][-1, i:i+i]
  if ((bl_x_it - 1) >= 0):
    left_edge_1 = frame[bl_y_it][bl_x_it - 1][0:i, -1].reshape((i, 1))
    left_edge_2 = frame[bl_y_it][bl_x_it - 1][i: i + i, -1].reshape((i, 1))
    
  top_edge += [top_edge_1, top_edge_2]
  left_edge += [left_edge_1, left_edge_2]

  lin_it = 0

  predicted_mode = []
  total_RDO = 0

  np_predicted_sub = np.zeros((4, i, i), dtype=int)

  for y_idx in range(2):
    for x_idx in range(2):
      temp = 0

      lefted_sub_block[:, :] = left_edge[y_idx]              
      topped_sub_block[:, :] = top_edge[x_idx]


      lefted_RDO, lefted_reconstructed = calc_RDO_intra_sub(lefted_sub_block, sub_block[lin_it], sub_Q, lambda_const)

      topped_RDO, topped_reconstructed = calc_RDO_intra_sub(topped_sub_block, sub_block[lin_it], sub_Q, lambda_const)


      if (lefted_RDO < topped_RDO):
        top_edge[x_idx] = lefted_reconstructed[-1, :]
        left_edge[y_idx] = lefted_reconstructed[:, -1]

        total_RDO += lefted_RDO
        predicted_mode += [0]
        temp = lefted_sub_block

      else:
        top_edge[x_idx] = topped_reconstructed[-1, :]
        left_edge[y_idx] = topped_reconstructed[:, -1]
        
        total_RDO += topped_RDO
        predicted_mode += [1]
        temp = topped_sub_block

      np_predicted_sub[lin_it] = temp
  
      lin_it += 1

  conc_0 = np.concatenate((np_predicted_sub[0], np_predicted_sub[1]), axis=1)
  conc_1 = np.concatenate((np_predicted_sub[2], np_predicted_sub[3]), axis=1)
  predicted_block = np.concatenate((conc_0, conc_1), axis=0)

  return predicted_mode, predicted_block, total_RDO

def extract_predicted_block(frame_buff, head_idy, head_idx, mv, i, FMEEnable):
  if (FMEEnable):
    if ((mv[0] % 2 == 0) and (mv[1] % 2 == 0)): # none fractional
      #print("None fractional")
      #print(mv[2], head_idy, mv[0], i, head_idx, mv[1])
      #print(mv[2], head_idy + mv[0], head_idy + mv[0] + i, head_idx + mv[1], head_idx + mv[1] + i)
      extracted = frame_buff[mv[2]][head_idy + mv[0] : head_idy + mv[0] + i, head_idx + mv[1] : head_idx + mv[1] + i]
    elif ((mv[0] % 2) and (mv[1] % 2)): # both fractional
      #print("Both fractional")
      dy_dir = int(mv[0]/2)
      dx_dir = int(mv[1]/2)
      
      extracted = (frame_buff[mv[2]][head_idy + dy_dir : head_idy + dy_dir + i, head_idx + dx_dir : head_idx + dx_dir + i] +
        frame_buff[mv[2]][head_idy + dy_dir + 1 : head_idy + dy_dir + i + 1, head_idx + dx_dir : head_idx + dx_dir + i] +
        frame_buff[mv[2]][head_idy + dy_dir : head_idy + dy_dir + i, head_idx + dx_dir + 1 : head_idx + dx_dir + i + 1] +
        frame_buff[mv[2]][head_idy + dy_dir + 1: head_idy + dy_dir + i + 1, head_idx + dx_dir + 1 : head_idx + dx_dir + i + 1]) // 4
    elif (mv[1] % 2): # x fractional
      #print("X fractional")
      dy_dir = int(mv[0]/2)
      dx_dir = int(mv[1]/2)
      
      extracted = (frame_buff[mv[2]][head_idy + dy_dir : head_idy + dy_dir + i, head_idx + dx_dir : head_idx + dx_dir + i] +
        frame_buff[mv[2]][head_idy + dy_dir : head_idy + dy_dir + i, head_idx + dx_dir + 1 : head_idx + dx_dir + i + 1]) // 2
    else: # y fractional
      #print("Y fractional")
      dy_dir = int(mv[0]/2)
      dx_dir = int(mv[1]/2)
            
      extracted = (frame_buff[mv[2]][head_idy + dy_dir : head_idy + dy_dir + i, head_idx + dx_dir : head_idx + dx_dir + i] +
        frame_buff[mv[2]][head_idy + dy_dir + 1 : head_idy + dy_dir + i + 1, head_idx + dx_dir : head_idx + dx_dir + i]) // 2

  else:
    #print("Exception")
    # print("\n")
    # print(mv)
    # print(head_idy, head_idx, i)
    # print(frame_buff.shape)
    # quit()
    extracted = frame_buff[mv[2]][head_idy + mv[0] : head_idy + mv[0] + i, head_idx + mv[1] : head_idx + mv[1] + i]
    
    # print(extracted.shape)

  #print(extracted)
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

        # print(extracted[0].shape, extracted[1].shape, extracted[2].shape, extracted[3].shape)
        
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

def calculate_reconstructed_image(residual_matrix, reconstructed, ext_y_res, ext_x_res, n_y_blocks, n_x_blocks, mv):

  new_reconstructed = decoder_core(residual_matrix, reconstructed, mv)

  return new_reconstructed
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
      for idx, mv in reversed(list(enumerate(modes_mv_block))):
        
        if (mv == [0]):
          return_data += [modes_mv_block[idx-1] - modes_mv_block[idx][0]]
          return_data += [modes_mv_block[idx][0] - modes_mv_block[idx+1]]
          break
        
        elif (mv == [1]):
          return_data += [modes_mv_block[idx-1] - modes_mv_block[idx][0]]
          return_data += [modes_mv_block[idx][0] - modes_mv_block[idx+1]]
          return_data += [modes_mv_block[idx+1] - modes_mv_block[idx+2]]
          return_data += [modes_mv_block[idx+2] - modes_mv_block[idx+3]]
          return_data += [modes_mv_block[idx+3] - modes_mv_block[idx+4]]
          break
        
    else:
      for idx, mv in reversed(list(enumerate(modes_mv_block))):          
        if (mv == [0]):
          return_data += [0 - modes_mv_block[idx][0]]
          return_data += [modes_mv_block[idx][0] - modes_mv_block[idx+1]]
          break 
        elif (mv == [1]):
          return_data += [0 - modes_mv_block[idx][0]]
          return_data += [modes_mv_block[idx][0] - modes_mv_block[idx+1]]
          return_data += [modes_mv_block[idx+1] - modes_mv_block[idx+2]]
          return_data += [modes_mv_block[idx+2] - modes_mv_block[idx+3]]
          return_data += [modes_mv_block[idx+3] - modes_mv_block[idx+4]]
          break

  return return_data

  

def exp_golomb_coding(number):

  # print(number)
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

def calc_RDO_intra_sub(pred_block, cur_block, Q, lambda_coeff):
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

  approx_residual = idct2D(np.multiply(quantized, Q))
  recontructed_block = np.add(approx_residual, pred_block)
  
  return J, recontructed_block

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

def entropy(is_p_block, VBSEnable, split, differentiated_modes_mv_frame, modes_mv_block, bl_x_it, bl_y_it, QTC, frame, ParallelMode=False):
  y_range = 0
  if (ParallelMode == 1):
    if (VBSEnable):
      if (split == 0):
        differentiated_modes_mv_frame += exp_golomb_coding(0)
        differentiated_modes_mv_frame += exp_golomb_coding(modes_mv_block[-1][0])
        differentiated_modes_mv_frame += exp_golomb_coding(modes_mv_block[-1][1])
        differentiated_modes_mv_frame += exp_golomb_coding(modes_mv_block[-1][2])
        

      else:
        differentiated_modes_mv_frame += exp_golomb_coding(1)
        for num_mv in range(-4, 0):
          for index in range(3):
            differentiated_modes_mv_frame += exp_golomb_coding(modes_mv_block[num_mv][index])
    else:
      differentiated_modes_mv_frame += exp_golomb_coding(modes_mv_block[-1][0])
      differentiated_modes_mv_frame += exp_golomb_coding(modes_mv_block[-1][1])
      differentiated_modes_mv_frame += exp_golomb_coding(modes_mv_block[-1][2])

  elif (is_p_block):
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
    if (VBSEnable):
      differentiated_modes_mv_frame += exp_golomb_coding(differential_encoder_decoder_vbs(modes_mv_block, is_p_block, bl_x_it)[0])
      if (split == [0]):
        differentiated_modes_mv_frame += exp_golomb_coding(differential_encoder_decoder_vbs(modes_mv_block, is_p_block, bl_x_it)[1])
      else:
        differentiated_modes_mv_frame += exp_golomb_coding(differential_encoder_decoder_vbs(modes_mv_block, is_p_block, bl_x_it)[1])
        differentiated_modes_mv_frame += exp_golomb_coding(differential_encoder_decoder_vbs(modes_mv_block, is_p_block, bl_x_it)[2])
        differentiated_modes_mv_frame += exp_golomb_coding(differential_encoder_decoder_vbs(modes_mv_block, is_p_block, bl_x_it)[3])
        differentiated_modes_mv_frame += exp_golomb_coding(differential_encoder_decoder_vbs(modes_mv_block, is_p_block, bl_x_it)[4])
    else:
      differentiated_modes_mv_frame += exp_golomb_coding(differential_encoder_decoder(modes_mv_block, is_p_block, bl_x_it)[0])

  # Scanning/RLE/writing (QTC)        
  scanned_block = scanning(QTC[frame][bl_y_it][bl_x_it])
  rled_block = RLE(scanned_block)

  return rled_block, differentiated_modes_mv_frame

def encode_one_block(bl_x_it, is_p_block, modes_mv_block, bl_y_frame, frame, bl_y_it, rec_buffer, ext_y_res, ext_x_res, Q, sub_Q, lambda_const, new_reconstructed, residual_matrix, QTC, differentiated_modes_mv_frame, qtc_bitstream, bits_in_frame, r, RC_pass, mv_mode_in, mv_modes_iterator, i, VBSEnable, RCflag, FMEEnable, FastME, nRefFrames, ParallelMode):
  
  splt = 0
  predicted_block = np.empty((i, i), dtype=int)

  if (is_p_block):
    # Calculate Motion Vector (inter)
    if (VBSEnable):
      if (RCflag == 3):
        splt = mv_mode_in[mv_modes_iterator]
        mv_modes_iterator += 1

      temp_mv, mv_modes_iterator = motion_vector_estimation_vbs(bl_y_frame[frame][bl_y_it][bl_x_it], rec_buffer, r, bl_y_it * i, bl_x_it * i, ext_y_res, ext_x_res, i, Q, sub_Q, lambda_const, FMEEnable, FastME, nRefFrames, RCflag, splt, mv_mode_in, mv_modes_iterator)
      modes_mv_block += temp_mv
    else:
      temp_mv = motion_vector_estimation(bl_y_frame[frame][bl_y_it][bl_x_it], rec_buffer, r, (bl_y_it * i) + mv_mode_in[mv_modes_iterator][0], (bl_x_it * i) + mv_mode_in[mv_modes_iterator][1], ext_y_res, ext_x_res, i, FMEEnable, FastME, nRefFrames)
      modes_mv_block += temp_mv
      mv_modes_iterator += 1
      
    split, predicted_block = extract_block(rec_buffer, bl_y_it * i, bl_x_it * i, modes_mv_block, i, VBSEnable, FMEEnable)

  else:
    # Calculate mode (intra)
    if (VBSEnable):
      split, temp_mode, predicted_block = intra_prediction_vbs(new_reconstructed, bl_y_frame[frame][bl_y_it][bl_x_it], bl_y_it, bl_x_it, Q, sub_Q, lambda_const)
      modes_mv_block += [split]
      modes_mv_block += temp_mode
    else:
      temp_mode, predicted_block = intra_prediction(new_reconstructed, bl_y_frame[frame][bl_y_it][bl_x_it], bl_y_it, bl_x_it)
      modes_mv_block += temp_mode
    
  #  Calculate Residual Matrix
  residual_matrix[frame][bl_y_it][bl_x_it] = calculate_residual_block(bl_y_frame[frame][bl_y_it][bl_x_it], predicted_block)

  #  Trans/Quant/Rescaling/InvTrans
  QTC[frame][bl_y_it][bl_x_it] = transform_quantize(residual_matrix, frame, bl_y_it, bl_x_it, Q, sub_Q, split, i)

  #  Decode
  new_reconstructed[bl_y_it][bl_x_it] = decoder_core(QTC[frame][bl_y_it][bl_x_it], Q, sub_Q, predicted_block, split, i)      

  # Differential Encoding

  rled_block, differentiated_modes_mv_frame = entropy(is_p_block, VBSEnable, split, differentiated_modes_mv_frame, modes_mv_block, bl_x_it, bl_y_it, QTC, frame, ParallelMode)

  # if (ParallelMode == 0):

  for rled in rled_block:
    qtc_bitstream += exp_golomb_coding(rled)
    bits_in_frame += exp_golomb_coding(rled)

  return qtc_bitstream, bits_in_frame, differentiated_modes_mv_frame, new_reconstructed, mv_modes_iterator
  
  # elif (ParallelMode == 1):
  #   for rled in rled_block:
  #     qtc_bitstream_temp += exp_golomb_coding(rled)
  #     bits_in_frame_temp += exp_golomb_coding(rled)

  #   return qtc_bitstream_temp, bits_in_frame_temp, differentiated_modes_mv_frame_temp, new_reconstructed, mv_modes_iterator

def block_encoding_sp(n_x_blocks, is_p_block, modes_mv_block, bl_y_frame, frame, bl_y_it, rec_buffer, ext_y_res, ext_x_res, Q, sub_Q, lambda_const, new_reconstructed, residual_matrix, QTC, differentiated_modes_mv_frame, qtc_bitstream, bits_in_frame, r, RC_pass, mv_mode_in, mv_modes_iterator, i, VBSEnable, RCflag, FMEEnable, FastME, nRefFrames, ParallelMode):

  if (RC_pass == 3):
    r = 1

  results = []

  if(ParallelMode == 1):      
    with concurrent.futures.ThreadPoolExecutor() as executor:
      for bl_x_it in range(n_x_blocks):
        modes_mv_copy = copy.deepcopy(modes_mv_block)
        qtc_bitstream_temp = ''
        bits_in_frame_temp = ''
        differentiated_modes_mv_frame_temp = ''

        results += [executor.submit(encode_one_block, bl_x_it, is_p_block, modes_mv_copy, bl_y_frame, frame, bl_y_it, rec_buffer, ext_y_res, ext_x_res, Q, sub_Q, lambda_const, new_reconstructed, residual_matrix, QTC, differentiated_modes_mv_frame_temp, qtc_bitstream_temp, bits_in_frame_temp, r, RC_pass, mv_mode_in, mv_modes_iterator, i, VBSEnable, RCflag, FMEEnable, FastME, nRefFrames, ParallelMode)]

      for f in concurrent.futures.as_completed(results):
        qtc_bitstream += f.result()[0]
        bits_in_frame += f.result()[1]
        differentiated_modes_mv_frame += f.result()[2]
        new_reconstructed = f.result()[3]

  else:
    for bl_x_it in range(n_x_blocks):
      qtc_bitstream, bits_in_frame, differentiated_modes_mv_frame, new_reconstructed, mv_modes_iterator = encode_one_block(bl_x_it, is_p_block, modes_mv_block, bl_y_frame, frame, bl_y_it, rec_buffer, ext_y_res, ext_x_res, Q, sub_Q, lambda_const, new_reconstructed, residual_matrix, QTC, differentiated_modes_mv_frame, qtc_bitstream, bits_in_frame, r, RC_pass, mv_mode_in, mv_modes_iterator, i, VBSEnable, RCflag, FMEEnable, FastME, nRefFrames, ParallelMode)

  # if(ParallelMode == 1):
  #   for proc in jobs:
  #     proc.join()

  #   for i in range(len(return_dict)):
  #     qtc_bitstream += return_dict[i][0]
  #     bits_in_frame += return_dict[i][1]
  #     differentiated_modes_mv_frame += return_dict[i][2]
  #     mv_modes_iterator += return_dict[i][3]
  
  return qtc_bitstream, bits_in_frame, differentiated_modes_mv_frame, new_reconstructed, mv_modes_iterator

def block_encoding_fp(n_x_blocks, is_p_block, modes_mv_block, bl_y_frame, frame, bl_y_it, rec_buffer, ext_y_res, ext_x_res, Q, sub_Q, lambda_const, new_reconstructed, residual_matrix, QTC, differentiated_modes_mv_frame, qtc_bitstream, bits_in_frame, r):

  mv_mode_out = []

  for bl_x_it in range(n_x_blocks):

    # print("--------------- : ", frame)

    predicted_block = np.empty((i, i), dtype=int)
    if (is_p_block):
      # Calculate Motion Vector (inter)
      if(VBSEnable):
        temp_mv, _ = motion_vector_estimation_vbs(bl_y_frame[frame][bl_y_it][bl_x_it], rec_buffer, r, bl_y_it * i, bl_x_it * i, ext_y_res, ext_x_res, i, Q, sub_Q, lambda_const, FMEEnable, FastME, nRefFrames, 0, None, None, None)

        mv_mode_out += temp_mv
        modes_mv_block += temp_mv
      else:
        temp_mv = motion_vector_estimation(bl_y_frame[frame][bl_y_it][bl_x_it], rec_buffer, r, bl_y_it * i, bl_x_it * i, ext_y_res, ext_x_res, i, FMEEnable, FastME, nRefFrames)

        mv_mode_out += temp_mv
        modes_mv_block += temp_mv

      # print(modes_mv_block)
      # print(bl_x_it * i)
      split, predicted_block = extract_block(rec_buffer, bl_y_it * i, bl_x_it * i, modes_mv_block, i, VBSEnable, FMEEnable)

      # print(frame, bl_y_it, bl_x_it, predicted_block[int(i/2)])

    else:
      # Calculate mode (intra)
      if (VBSEnable):
        split, temp_mode, predicted_block = intra_prediction_vbs(new_reconstructed, bl_y_frame[frame][bl_y_it][bl_x_it], bl_y_it, bl_x_it, Q, sub_Q, lambda_const)
        modes_mv_block += [split]
        mv_mode_out += [split]
        modes_mv_block += temp_mode
        mv_mode_out += temp_mode
      else:
        temp_mode, predicted_block = intra_prediction(new_reconstructed, bl_y_frame[frame][bl_y_it][bl_x_it], bl_y_it, bl_x_it)
        modes_mv_block += temp_mode
        mv_mode_out += temp_mode
      
  #  Calculate Residual Matrix
    residual_matrix[frame][bl_y_it][bl_x_it] = calculate_residual_block(bl_y_frame[frame][bl_y_it][bl_x_it], predicted_block)

  #  Trans/Quant/Rescaling/InvTrans
    QTC[frame][bl_y_it][bl_x_it] = transform_quantize(residual_matrix, frame, bl_y_it, bl_x_it, Q, sub_Q, split, i)

  #  Decode
    new_reconstructed[bl_y_it][bl_x_it] = decoder_core(QTC[frame][bl_y_it][bl_x_it], Q, sub_Q, predicted_block, split, i)      

  # Differential Encoding
    rled_block, differentiated_modes_mv_frame = entropy(is_p_block, VBSEnable, split, differentiated_modes_mv_frame, modes_mv_block, bl_x_it, bl_y_it, QTC, frame)

    for rled in rled_block:
      qtc_bitstream += exp_golomb_coding(rled)
      bits_in_frame += exp_golomb_coding(rled)

  return qtc_bitstream, bits_in_frame, differentiated_modes_mv_frame, mv_mode_out

def QP_select(bit_stats_list, remaining_bits):
  QP = min(range(len(bit_stats_list)), key=lambda i: abs(bit_stats_list[i] - remaining_bits))

  return QP

def calc_QP_dependents(QP, Constant):
  Q = calculate_Q(i, QP)
  sub_QP = 0
  if (QP == 0):
    sub_QP = 0
  else:
    sub_QP = QP - 1

  sub_Q = calculate_Q((int)(i/2), sub_QP)
  lambda_const = Constant * 2 ** ((QP - 12) / 3)

  return Q, sub_QP, sub_Q, lambda_const


def QP_selector(remaining_bits, is_p_block, Constant, cif_approx_p, cif_approx_i, qcif_approx_p, qcif_approx_i, approx_per_row_bits):
  if (type(approx_per_row_bits) != int):
    QP = QP_select(approx_per_row_bits, remaining_bits)
  elif (y_res == 288):
    if(is_p_block):
      QP = QP_select(cif_approx_p, remaining_bits)
    else:
      QP = QP_select(cif_approx_i, remaining_bits)
  elif (y_res == 144):
    if(is_p_block):
      QP = QP_select(qcif_approx_p, remaining_bits)
    else:
      QP = QP_select(qcif_approx_i, remaining_bits)
  else:
    print("Resolution not supported!")
    quit()

  Q, sub_QP, sub_Q, lambda_const = calc_QP_dependents(QP, Constant)

  return QP, Q, sub_QP, sub_Q, lambda_const

def detect_scene_change(curr_size, prev_size, curr_QP, prev_QP):
  # first_dim -> current QP
  # second_dim -> prev_QP

  scene_matrix = [[40000,	185640,	288279,	368579,	456575,	516351,	528368, 535336,	540061,	542462,	543681,	544546],
                  [62322,	70000,	176644,	256944,	344940,	404716,	416733,	423701,	428426,	430827,	432046,	432911],
                  [162591,	26264,	70000,	156675,	244671,	304447,	316464,	323432,	328157,	330558,	331777,	332642],
                  [243655,	107328,	4689,	70000,	163607,	223383,	235400,	242368,	247093,	249494,	250713,	251578],
                  [312503,	176176,	73537,	6763,	84311,	154535,	166552,	173520,	178245,	180646,	181865,	182730],
                  [380428,	244101,	141462,	61162,	26834,	80000,	98627,	105595,	110320,	112721,	113940,	114805],
                  [423876,	287549,	184910,	104610,	16614,	43162,	65789,	62147,	66872,	69273,	70492,	71357],
                  [454447,	318120,	215481,	135181,	47185,	12591,	24608,	30000,	36301,	38702,	39921,	40786],
                  [468569,	332242,	229603,	149303,	61307,	1531,	10486,	17454, 20000,	14562,	25799,	26664],
                  [478497,	342170,	239531,	159231,	71235,	80879,	558,	7526,	1213,	10000,	15871,	16736],
                  [483369,	347042,	244403,	164103,	76107,	16331,	4314,	2654,	7379,	1543,	10000,	11864],
                  [486345,	350018,	247379,	167079,	79083,	19307,	7290,	322,	4403,	6804,	8023,	8000]]

  difference = abs(curr_size - prev_size)

  # print("diff: ", difference, "curr_QP: ",curr_QP , "prev_QP: ", prev_QP, "scene_mat: ", scene_matrix[curr_QP][prev_QP])

  if (curr_QP <= prev_QP):
    if (difference >= scene_matrix[curr_QP][prev_QP]):
      # print("HERE 1")
      return 1
    else:
      return 0
  else:
    if (difference <= scene_matrix[curr_QP][prev_QP]):
      # print("HERE 2")
      return 1
    else:
      return 0
    

##############################################################################
##############################################################################

def encoder(in_file, out_file, number_frames, y_res, x_res, i, r, QP, i_period, nRefFrames, VBSEnable, FMEEnable, FastME, RCflag, targetBR):  

  cif_approx_i = [27248,27280,21128,15544,10680,6992,4360,2624,1416,664,264,176]
  cif_approx_p = [23192,16696,11904,7984,4888,2720,1608,1152,1024,904,736,664]
  qcif_approx_i = [7424,7424,5856,4464,3328,2368,1600,1000,552,256,96,64]
  qcif_approx_p = [5656,4216,3064,2192,1480,912,496,312,280,248,208,184]

  if (nRefFrames > (i_period - 1)):
    print("\n\n\n*****  nRefFrames is incompatible with i_period.  *****\n\n\n")
    return
  print("----------------------------------------------")
  print("----------------------------------------------")
  print("A3 Encoder Parameters-")
  print("----------------------------------------------")
  print("in_file: ", in_file)
  print("out_file: ", out_file)
  print("number_frames: ", number_frames)
  print("y_res: ", y_res)
  print("x_res: ", x_res)
  print("i: ", i)
  print("r: ", r)
  if(RCflag):
    print("QP: Variable (RC enabled)")
  else:
    print("QP: ", QP)

  print("i_period: ", i_period)
  print("nRefFrames: ", nRefFrames)
  print("VBSEnable: ", VBSEnable)
  print("FMEEnable: ", FMEEnable)
  print("FastME: ", FastME)
  print("RCflag: ", RCflag)
  if(RCflag):
    print("targetBR: ", targetBR)
  print("ParallelMode: ", ParallelMode)
  print("----------------------------------------------")


  targetBR_bps = 0
  bit_in_frame = 0
  remaining_bits = 0


  if (RCflag):
    targetBR_bps = targetBR * 1024
    bit_in_frame = (targetBR_bps / 30) # for 30fps

  bl_y_frame, n_y_blocks, n_x_blocks, ext_y_res, ext_x_res = pre.block(in_file, y_res, x_res, number_frames, i)
  # reconst = np.empty((ext_y_res, ext_x_res), dtype=int)

  rec_buffer = np.empty((nRefFrames, ext_y_res, ext_x_res), dtype=int)

  residual_matrix = np.empty((number_frames, n_y_blocks, n_x_blocks, i, i), dtype=np.int16)

  QTC = np.empty((number_frames, n_y_blocks, n_x_blocks, i, i), dtype=int)

  approx_residual = np.empty((number_frames, n_y_blocks, n_x_blocks, i, i), dtype=int)

  Constant = 35
  Q, sub_QP, sub_Q, lambda_const = calc_QP_dependents(QP, Constant)

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

  prev_frame_size = 0
  prev_avg_QP = 0
  user_FMEEnable = FMEEnable

  prev_QP_scene_detect = 0

  is_p_block = 0

  for frame in range(number_frames):


    # print("frame: ", frame)

    differentiated_modes_mv_frame = ''

    bits_in_frame = ''

    modes_mv_block = []

    is_p_block += 1

    if (ParallelMode == 1):
        is_p_block = 1
        if (frame == 0):
          rec_buffer = np.delete(rec_buffer, np.s_[0:nRefFrames], 0)
          rec_buffer = np.insert(rec_buffer, 0, np.full((ext_y_res, ext_x_res), 128, dtype=int), axis=0)
    elif ((frame % i_period) == 0):
      is_p_block = 0

    bits_used = 0
    prev_QP = 0
    QP_list = []
    true_QP = []

    bits_per_block_row = []
    prev_size = 0
    bit_proportion = 0

    scene_change = 0
    approx_per_row_bits = 0

    mv_mode_out = []
    mv_modes_iterator = 0

    if (RCflag > 1):

      row_bits = bit_in_frame // n_y_blocks

      if (frame==0):
        QP, Q, sub_QP, sub_Q, lambda_const = QP_selector(row_bits, is_p_block, Constant, cif_approx_p, cif_approx_i, qcif_approx_p, qcif_approx_i, approx_per_row_bits)

      else:
        QP = int(round(prev_avg_QP))
        Q, sub_QP, sub_Q, lambda_const = calc_QP_dependents(QP, Constant)

      for bl_y_it in range(n_y_blocks):

        # First Pass
        if RCflag == 3 :
          FMEEnable = False

        _, bits_in_frame, differentiated_modes_mv_frame, temp_mv_mode_out = block_encoding_fp(n_x_blocks, is_p_block, modes_mv_block, bl_y_frame, frame, bl_y_it, rec_buffer, ext_y_res, ext_x_res, Q, sub_Q, lambda_const, new_reconstructed, residual_matrix, QTC, differentiated_modes_mv_frame, qtc_bitstream, bits_in_frame, r)

        mv_mode_out += temp_mv_mode_out
        
        bits_per_block_row += [len(bits_in_frame) + len(differentiated_modes_mv_frame) - prev_size]
        prev_size = len(bits_in_frame) + len(differentiated_modes_mv_frame)

      total_bit_in_frame = len(bits_in_frame) + len(differentiated_modes_mv_frame)

      bit_proportion = np.array(bits_per_block_row) / total_bit_in_frame

      if (is_p_block and (ParallelMode != 1)):
        if (detect_scene_change(total_bit_in_frame, prev_frame_size, QP, prev_QP_scene_detect)):
          is_p_block = 0
          scene_change = 1

      print((frame+1), "-->", is_p_block, "-->", QP, "\n")

      prev_frame_size = total_bit_in_frame
      prev_QP_scene_detect = QP

      if ((y_res == 288) and is_p_block):
        approx_per_row_bits = np.array(cif_approx_p) * ((total_bit_in_frame/n_y_blocks)/cif_approx_p[QP])
      elif ((y_res == 288) and not is_p_block):
        approx_per_row_bits = np.array(cif_approx_i) * ((total_bit_in_frame/n_y_blocks)/cif_approx_i[QP])
      elif ((y_res == 144) and is_p_block):
        approx_per_row_bits = np.array(qcif_approx_p) * ((total_bit_in_frame/n_y_blocks)/qcif_approx_p[QP])
      elif ((y_res == 144) and not is_p_block):
        approx_per_row_bits = np.array(qcif_approx_i) * ((total_bit_in_frame/n_y_blocks)/qcif_approx_i[QP])

    bits_in_frame = ''
    differentiated_modes_mv_frame = ''

    for bl_y_it in range(n_y_blocks):

      # print(bits_used)

      # QP, Q, sub_QP, lambda_const, QP_list, prev_QP, remaining_bits = rate_controller(bit_in_frame, bits_used, i, y_res, n_y_blocks, bl_y_it, is_p_block, cif_approx_p, cif_approx_i, qcif_approx_p, qcif_approx_i, Constant, QP_list, prev_QP, remaining_bits, RCflag)

      if (RCflag):
        if((RCflag == 1) or scene_change):
          remaining_bits = (bit_in_frame - bits_used) // (n_y_blocks - bl_y_it)
        elif (RCflag > 1):
          remaining_bits = (bit_in_frame - bits_used)
          proportion_adj_factor =  1 / np.sum(bit_proportion[bl_y_it:])
          remaining_bits *= (bit_proportion[bl_y_it] * proportion_adj_factor)
      
        QP, Q, sub_QP, sub_Q, lambda_const = QP_selector(remaining_bits, is_p_block, Constant, cif_approx_p, cif_approx_i, qcif_approx_p, qcif_approx_i, approx_per_row_bits) 

        true_QP += [QP]
        QP_list += [prev_QP - QP]
        prev_QP = QP

      # Second Pass
      if RCflag == 3 :
        FMEEnable = user_FMEEnable
      
      qtc_bitstream, bits_in_frame, differentiated_modes_mv_frame, new_reconstructed, mv_modes_iterator = block_encoding_sp(n_x_blocks, is_p_block, modes_mv_block, bl_y_frame, frame, bl_y_it, rec_buffer, ext_y_res, ext_x_res, Q, sub_Q, lambda_const, new_reconstructed, residual_matrix, QTC, differentiated_modes_mv_frame, qtc_bitstream, bits_in_frame, r, RCflag, mv_mode_out, mv_modes_iterator, i, VBSEnable, RCflag, FMEEnable, FastME, nRefFrames, ParallelMode)

      bits_used = len(bits_in_frame) + len(differentiated_modes_mv_frame)

    if (RCflag):
      prev_avg_QP = np.mean(true_QP)
      for diff_QP_val in list(reversed(QP_list)):
        differentiated_modes_mv_frame = exp_golomb_coding(diff_QP_val) + differentiated_modes_mv_frame

    len_of_frame += [len(bits_in_frame)]

    # print(frame, frame_accumulated_bits_used)

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
      rec_buffer = np.delete(rec_buffer, nRefFrames, 0)
    
        

    # print(rec_buffer.shape[0])

    pre.progress("Encoding frames: ", frame, number_frames)
    


  # Padding with 1s to make the number of bits divisible by 8
  bits_in_a_byte = 8
  bits_in_mdiff = len(differentiated_modes_mv_bitstream)

  differentiated_modes_mv_bitstream = exp_golomb_coding(y_res) + exp_golomb_coding(x_res) + exp_golomb_coding(i) + exp_golomb_coding(QP) + exp_golomb_coding(nRefFrames) + exp_golomb_coding(FMEEnable) + exp_golomb_coding(VBSEnable) + exp_golomb_coding(RCflag) + exp_golomb_coding(ParallelMode) + exp_golomb_coding(bits_in_mdiff) + differentiated_modes_mv_bitstream

  final_bitstream = differentiated_modes_mv_bitstream + qtc_bitstream

  # Padding with 1s to make the number of bits divisible by 8
  final_bitstream = ('1' * (bits_in_a_byte - (len(final_bitstream) % bits_in_a_byte))) + final_bitstream

  write_encoded(final_bitstream, encoded_file)

  converted.close()
  encoded_file.close()

  return len_of_frame
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
  RCflag, encoded_idx = I_golomb(encoded_bitstream, encoded_idx)
  ParallelMode, encoded_idx = I_golomb(encoded_bitstream, encoded_idx)
  

  print("----------------------------------------------")
  print("----------------------------------------------")
  print("A3 Decoder Parameters-")
  print("----------------------------------------------")
  print("in_file: ", in_file)
  print("out_file: ", out_file)
  print("y_res: ", y_res)
  print("x_res: ", x_res)
  print("i: ", i)
  if(RCflag):
    print("QP: Variable (RC enabled)")
  else:
    print("QP: ", QP)
  print("nRefFrames: ", nRefFrames)
  print("FMEEnable: ", FMEEnable)
  print("VBSEnable: ", VBSEnable)
  print("RCflag: ", RCflag)
  print("ParallelMode: ", ParallelMode)
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

  if (ParallelMode == 1):
    rec_buffer = np.delete(rec_buffer, np.s_[0:nRefFrames], 0)
    rec_buffer = np.insert(rec_buffer, 0, np.full((ext_y_res, ext_x_res), 128, dtype=int), axis=0)
  
  decoded = open(out_file, "wb")

  # Recover QTC Block:
  # Performing inverse RLE to get scanned QTC  
  qtc = I_RLE(qtc, i)

  number_of_frames = (int)((len(qtc) / (i*i)) / (n_x_blocks * n_y_blocks))

  modes_mv = []
  lin_idx = 0
  
  for frame in range(number_of_frames):
    
    pre.progress("Decoding frames: ", frame, number_of_frames)

    #is_i_frame = mdiff[lin_idx]
    is_i_frame = False #ASS
    lin_idx += 1
    
    QP_list = []
    if (RCflag):
      QP_list += [0 - mdiff[lin_idx]]
      lin_idx += 1

      for _ in range(n_y_blocks - 1):
        QP_list += [QP_list[-1] - mdiff[lin_idx]]
        lin_idx += 1

    for bl_y_it in range(n_y_blocks):
      prev_mode = 0
      prev_mv = [0, 0, 0]

      if (RCflag):
        QP = QP_list[bl_y_it]
        Q = calculate_Q(i, QP)
        sub_QP = 0
        if (QP == 0):
          sub_QP = 0
        else:
          sub_QP = QP - 1

        sub_Q = calculate_Q((int)(i/2), sub_QP)

        if (QP_list == []):
          QP_list += [0 - QP]
        else:
          QP_list += [QP_list[-1] - QP]

      for bl_x_it in range(n_x_blocks):

        # Inverse scanning
        QTC_recovered, lin_it = I_scanning(qtc, i, lin_it)

        if (ParallelMode == 1):
          is_i_frame = 0
             
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
            prev_mode = mdiff[lin_idx]#prev_mode - mdiff[lin_idx]
            vbs_mode = prev_mode
            modes_mv += [vbs_mode]
            lin_idx += 1

            if (vbs_mode == 0):
              y_range = 2
            else:
              y_range = 5

            for num_mv in range(1, y_range):
                prev_mv[0] = mdiff[lin_idx+0]#prev_mv[0] - mdiff[lin_idx+0]
                prev_mv[1] = mdiff[lin_idx+1]#prev_mv[1] - mdiff[lin_idx+1]
                prev_mv[2] = mdiff[lin_idx+2]#prev_mv[2] - mdiff[lin_idx+2]
                modes_mv += [[prev_mv[0], prev_mv[1], prev_mv[2]]]
                lin_idx += 3
            
          else:
            prev_mv[0] = mdiff[lin_idx+0]#prev_mv[0] - mdiff[lin_idx+0]
            prev_mv[1] = mdiff[lin_idx+1]#prev_mv[1] - mdiff[lin_idx+1]
            prev_mv[2] = mdiff[lin_idx+2]#prev_mv[2] - mdiff[lin_idx+2]
            modes_mv += [[prev_mv[0], prev_mv[1], prev_mv[2]]]
            lin_idx += 3
          
        #  Decode
        #print(rec_buffer, new_reconstructed, modes_mv, bl_y_it, bl_x_it, i, (not is_i_frame), VBSEnable, FMEEnable, QTC_recovered, sub_Q)
        print(modes_mv)
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
            decoded.write(((int)(conc[i_it][x_it])).to_bytes(1, byteorder=sys.byteorder))

    # reconst = conc_reconstructed

    if (is_i_frame):
      rec_buffer = np.delete(rec_buffer, np.s_[0:nRefFrames], 0)

    rec_buffer = np.insert(rec_buffer, 0, conc_reconstructed, axis=0)

    if(rec_buffer.shape[0] > nRefFrames):
      rec_buffer = np.delete(rec_buffer, nRefFrames, 0)        

  decoded.close()
  print("Decoding Completed")

##############################################################################
##############################################################################

if __name__ == "__main__":

  in_file = "./videos/CIF_bw.yuv"
  out_file = "./temp/a3_CIF.far"
  # in_file = "./temp/white.yuv"

  # in_file = "./videos/synthetic_bw.yuv"
  # out_file = "./temp/synthetic_test.far"

  number_frames = 4
  y_res = 288
  x_res = 352
  i = 16
  r = 1
  QP = 2  # from 0 to (log_2(i) + 7)
  i_period = 4
  nRefFrames = 1
  VBSEnable = True
  FMEEnable = True
  FastME = True
  RCflag = 0
  targetBR = 2458 # kbps
  ParallelMode = 1

  # bits_in_each_frame = []

  if (ParallelMode != 0 and RCflag > 0):
    print("\n\n ##### Parallel mode is only available when rate control is disabled. Please disable rate control and try again. Aborting... #####\n")
    quit()

  if (ParallelMode == 1):
    i_period = number_frames + 1

  decoder_infile = out_file
  decoder_outfile = "./videos/a3_CIF_decoded.yuv"

  encoder(in_file, out_file, number_frames, y_res, x_res, i, r, QP, i_period, nRefFrames, VBSEnable, FMEEnable, FastME, RCflag, targetBR)
  decoder(decoder_infile, decoder_outfile)
  
# ##############################################################################
# ##############################################################################

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


# # ##############################################################################
#   nRefFrames = 1
#   out_file = "./videos/a2_plot/synthetic_nref_1.far"
#   decoder_infile = out_file
#   decoder_outfile = "./videos/a2_plot/synthetic_nref_1.yuv"

#   size_1 = encoder(in_file, out_file, number_frames, y_res, x_res, i, r, QP, i_period, nRefFrames, VBSEnable, FMEEnable, FastME)
#   decoder(decoder_infile, decoder_outfile)

# # ##############################################################################

#   nRefFrames = 2
#   out_file = "./videos/a2_plot/synthetic_nref_2.far"
#   decoder_infile = out_file
#   decoder_outfile = "./videos/a2_plot/synthetic_nref_2.yuv"

#   size_2 = encoder(in_file, out_file, number_frames, y_res, x_res, i, r, QP, i_period, nRefFrames, VBSEnable, FMEEnable, FastME)
#   decoder(decoder_infile, decoder_outfile)

# # ##############################################################################

#   nRefFrames = 3
#   out_file = "./videos/a2_plot/synthetic_nref_3.far"
#   decoder_infile = out_file
#   decoder_outfile = "./videos/a2_plot/synthetic_nref_3.yuv"

#   size_3 = encoder(in_file, out_file, number_frames, y_res, x_res, i, r, QP, i_period, nRefFrames, VBSEnable, FMEEnable, FastME)
#   decoder(decoder_infile, decoder_outfile)

# # ##############################################################################

#   nRefFrames = 4
#   out_file = "./videos/a2_plot/synthetic_nref_4.far"
#   decoder_infile = out_file
#   decoder_outfile = "./videos/a2_plot/synthetic_nref_4.yuv"

#   size_4 = encoder(in_file, out_file, number_frames, y_res, x_res, i, r, QP, i_period, nRefFrames, VBSEnable, FMEEnable, FastME)
#   decoder(decoder_infile, decoder_outfile)

# # ##############################################################################

#   x_axis = range(1, 31)
#   fig, ax = plt.subplots()

#   ax.plot(x_axis, size_1, 'o:', label='nRefFrame = 1')
#   ax.plot(x_axis, size_2, 'v-.', label='nRefFrame = 2')
#   ax.plot(x_axis, size_3, 'D--', label='nRefFrame = 3')
#   ax.plot(x_axis, size_4, 's:', label='nRefFrame = 4')


#   plt.xticks(x_axis)
#   ax.set(xlabel='Frame', ylabel='Size (Bits)')
#   ax.grid()
#   ax.legend()

#   # fig.savefig("test.png")
#   plt.show()



