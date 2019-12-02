import numpy as np
import sys
import matplotlib.pyplot as plt
import assign1_Q2_main as pre
from assign2 import extract_predicted_block, extract_block, idct2D, decoder_core, extract_intra_block, predict_block, I_scanning, I_RLE, calculate_Q, I_golomb
from scipy.fftpack import dct, idct

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
                
def decoder(in_file, out_file, view_blocks, view_ref_frame):

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
  view_new_reconstructed = np.empty((n_y_blocks, n_x_blocks, i, i), dtype=int)
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

        # print(modes_mv)

        split, predicted_block = predict_block(rec_buffer, new_reconstructed, modes_mv, bl_y_it, bl_x_it, i, (not is_i_frame), VBSEnable, FMEEnable, QTC_recovered, sub_Q)

        new_reconstructed[bl_y_it][bl_x_it] = decoder_core(QTC_recovered, Q, sub_Q, predicted_block, split, i)

        view_new_reconstructed[bl_y_it][bl_x_it] = new_reconstructed[bl_y_it][bl_x_it]
        if (view_blocks or view_ref_frame):
          view_new_reconstructed[bl_y_it][bl_x_it][0,:] = 0
          view_new_reconstructed[bl_y_it][bl_x_it][-1,:] = 0
          view_new_reconstructed[bl_y_it][bl_x_it][:,0] = 0
          view_new_reconstructed[bl_y_it][bl_x_it][:,-1] = 0

          if((split == 1) or (split == [1])):
            s = int(i/2)
            view_new_reconstructed[bl_y_it][bl_x_it][s,:] = 0
            view_new_reconstructed[bl_y_it][bl_x_it][:, s] = 0

            if (view_ref_frame and not is_i_frame):
              block = view_new_reconstructed[bl_y_it][bl_x_it]

              sub_block = []
              sub_block += [block[0: s, 0: s]]
              sub_block += [block[0: s, s: s + s]]
              sub_block += [block[s: s + s, 0: s]]
              sub_block += [block[s: s + s, s: s + s]]

              for idx, mv in reversed(list(enumerate(modes_mv))):

                if (mv == 1):
                  n_past_frames = modes_mv[idx+1][2] + 1
                  sub_block[0] = nRef_tool(n_past_frames, sub_block[0])

                  n_past_frames = modes_mv[idx+2][2] + 1
                  sub_block[1] = nRef_tool(n_past_frames, sub_block[1])

                  n_past_frames = modes_mv[idx+3][2] + 1
                  sub_block[2] = nRef_tool(n_past_frames, sub_block[2])

                  n_past_frames = modes_mv[idx+4][2] + 1
                  sub_block[3] = nRef_tool(n_past_frames, sub_block[3])

                  conc_0 = np.concatenate((sub_block[0], sub_block[1]), axis=1)
                  conc_1 = np.concatenate((sub_block[2], sub_block[3]), axis=1)
                  view_new_reconstructed[bl_y_it][bl_x_it] = np.concatenate((conc_0, conc_1), axis=0)

                  break

          else:
            if (view_ref_frame and not is_i_frame):
              n_past_frames = modes_mv[-1][2] + 1
              view_new_reconstructed[bl_y_it][bl_x_it] = nRef_tool(n_past_frames, view_new_reconstructed[bl_y_it][bl_x_it])
            
    
    #  Concatenate
    counter = 0
    conc_reconstructed = np.empty((ext_y_res, ext_x_res), dtype = int)
    
    for bl_y_it in range(n_y_blocks):
        conc = np.concatenate((new_reconstructed[bl_y_it][0], new_reconstructed[bl_y_it][1]), axis=1)
        if(view_blocks or view_ref_frame):
            conc_view = np.concatenate((view_new_reconstructed[bl_y_it][0], view_new_reconstructed[bl_y_it][1]), axis=1)
        for bl_x_it in range(2, n_x_blocks):
            conc = np.concatenate((conc, new_reconstructed[bl_y_it][bl_x_it]), axis=1)
            if(view_blocks or view_ref_frame):
                conc_view = np.concatenate((conc_view, view_new_reconstructed[bl_y_it][bl_x_it]), axis=1)
      # Write frame (decoder output video)
        for i_it in range(i):
            if (counter < y_res):
                counter += 1
                for x_it in range(x_res):
                    conc_reconstructed[bl_y_it*i + i_it][x_it] = conc[i_it][x_it]
                    
                    if(view_blocks or view_ref_frame):
                        decoded.write(((int)(conc_view[i_it][x_it])).to_bytes(1, byteorder=sys.byteorder))
                    else:
                        decoded.write(((int)(conc[i_it][x_it])).to_bytes(1, byteorder=sys.byteorder))

    # reconst = conc_reconstructed

    if (is_i_frame):
      rec_buffer = np.delete(rec_buffer, np.s_[0:nRefFrames], 0)

    if(rec_buffer.shape[0] > nRefFrames):
      rec_buffer = np.delete(rec_buffer, nRefFrames, 0)
    
    rec_buffer = np.insert(rec_buffer, 0, conc_reconstructed, axis=0)
        

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
  
  decoder_outfile = "./videos/a2_block_print.yuv"

  view_blocks = True
  view_ref_frame = False

  decoder(in_file, decoder_outfile, view_blocks, view_ref_frame)