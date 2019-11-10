import numpy as np
import sys
import operator
# import matplotlib.pyplot as plt
import assign1_Q2_main as pre
from scipy.fftpack import dct, idct

def dct2D(block):  # Transform Function
  res = dct(dct(block, axis=0, norm='ortho'), axis=1, norm='ortho')
  return res

def idct2D(block): # Inverse Transform Function
  res = idct(idct(block, axis=0, norm='ortho'), axis=1, norm='ortho')
  return res

def find_mv(mv, block, rec_buffer, r, head_idy, head_idx, ext_y_res, ext_x_res, i, origin):
    best_SAD = i * i * 255 + 1  # The sum can never exceed (i * i * 255 + 1)
    negative = -r
    positive = (r + 1)
    check = 1
    if(origin!=1):
        # print('HERE')
        negative = -1
        positive = 2

    for buff_idx, reconstructed in enumerate(rec_buffer):
        for y_dir in range(negative, positive):
            for x_dir in range(negative, positive):
                if((check%2)==0):
                    if ((head_idy + y_dir) >= 0 and (head_idy + y_dir + i) < ext_y_res and (head_idx + x_dir) >= 0 and (head_idx + x_dir + i) < ext_x_res):
                        extracted = reconstructed[head_idy + y_dir : head_idy + y_dir + i, head_idx + x_dir : head_idx + x_dir + i]

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
    # print(mv,best_SAD)
    # print("+++++++")
    return  mv,best_SAD

def motion_vector_estimation(block, rec_buffer, r, head_idy, head_idx, ext_y_res, ext_x_res, i, FastME):

    if(FastME):

        mv = (0, 0, 0)
        origin = 1
        iterate = 1
        mv_accum = []
        mv_new = ()
        mv_test = ()
        while(iterate):
            mv_new,sad_new = find_mv(mv, block, rec_buffer, r, head_idy, head_idx, ext_y_res, ext_x_res, i, origin)

            origin = 2
            mv_test,sad_test = find_mv(mv_new, block, rec_buffer, r, head_idy, head_idx, ext_y_res, ext_x_res, i, origin)

            if(sad_new > sad_test):
                origin = 1
                nframe = []
                nframe.append(mv_new[0])
                mv_new = mv_new[1:]
                mv_test = mv_test[1:]
                new = list(map(operator.add, mv_new, mv_test))
                if(len(mv_accum)==0):
                    mv_accum = tuple(nframe + new)
                else:
                    new = list(map(operator.add, new, list(mv_accum[1:])))
                    mv_accum = tuple(nframe + new)
                mv = mv_new
            else:
                iterate = 0
                if(len(mv_accum)==0):
                    mv_accum = mv_new

        return [mv_accum]
    else:
        pass

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

def extract_block(frame_buff, head_idy, head_idx, mv, i):
  # print('mv===',mv)
  extracted = frame_buff[mv[2]][head_idy + mv[0] : head_idy + mv[0] + i, head_idx + mv[1] : head_idx + mv[1] + i]

  return extracted

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



def decoder_core(QTC, Q, predicted_block):

  approx_residual = idct2D(np.multiply(QTC, Q))
  recontructed_block = np.add(approx_residual, predicted_block)

  for y_it in range(recontructed_block.shape[0]):
    for x_it in range(recontructed_block.shape[1]):
      recontructed_block[y_it][x_it] = max(0, min(recontructed_block[y_it][x_it], 255))
  return recontructed_block

def calculate_reconstructed_image(residual_matrix, reconstructed, ext_y_res, ext_x_res, n_y_blocks, n_x_blocks, mv):

  new_reconstructed = decoder_core(residual_matrix, reconstructed, mv)

  return new_reconstructed

def predict_block(rec_buffer, new_reconstructed, modes_mv, bl_y_it, bl_x_it, i, is_p_block):

  grey = 128
  predicted_block = np.empty((i, i), dtype=int)
  if(is_p_block):
          predicted_block = extract_block(rec_buffer, bl_y_it * i, bl_x_it * i, modes_mv, i)
  else:
    top_edge = np.full((1, i), grey)
    left_edge = np.full((i, 1), grey)

    predicted_block[:,:] = top_edge

    if ((modes_mv == 1) and ((bl_y_it - 1) >= 0)):
      top_edge = new_reconstructed[bl_y_it - 1][bl_x_it][-1, :]
      predicted_block[:,:] = top_edge

    if ((modes_mv == 0) and ((bl_x_it - 1) >= 0)):
      left_edge = new_reconstructed[bl_y_it][bl_x_it - 1][:, -1].reshape((i, 1))
      predicted_block[:, :] = left_edge

  return predicted_block

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



##############################################################################
##############################################################################

def encoder(in_file, out_file, number_frames, y_res, x_res, i, r, QP, i_period, nRefFrames):

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

    pre.progress("Encoding frames: ", frame, number_frames)

    modes_mv_block = []

    is_p_block = frame % i_period

    for bl_y_it in range(n_y_blocks) :
      for bl_x_it in range(n_x_blocks):

        predicted_block = np.empty((i, i), dtype=int)
        if (is_p_block):
          # Calculate Motion Vector (inter)
          # print('f=',frame,'by=',bl_y_it,'bx=',bl_x_it)
          # print(bl_y_frame[frame][bl_y_it][bl_x_it])
          # print( rec_buffer, r, bl_y_it * i, bl_x_it * i, ext_y_res, ext_x_res, i)
          modes_mv_block += motion_vector_estimation(bl_y_frame[frame][bl_y_it][bl_x_it], rec_buffer, r, bl_y_it * i, bl_x_it * i, ext_y_res, ext_x_res, i,1)

          predicted_block = extract_block(rec_buffer, bl_y_it * i, bl_x_it * i, modes_mv_block[-1], i)

        else:
          # Calculate mode (intra)
          temp_mode, predicted_block = intra_prediction(new_reconstructed, bl_y_it, bl_x_it)
          modes_mv_block += temp_mode


      #  Calculate Residual Matrix
        residual_matrix[frame][bl_y_it][bl_x_it] = calculate_residual_block(bl_y_frame[frame][bl_y_it][bl_x_it], predicted_block)

      #  Trans/Quant/Rescaling/InvTrans
        transformed_dct = dct2D(residual_matrix[frame][bl_y_it][bl_x_it])
        QTC[frame][bl_y_it][bl_x_it] = quantize_dct(transformed_dct, Q)

      #  Decode
        new_reconstructed[bl_y_it][bl_x_it] = decoder_core(QTC[frame][bl_y_it][bl_x_it], Q, predicted_block)

      # Differential Encoding
        if (is_p_block):
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



  # Padding with 1s to make the number of bits divisible by 8
  bits_in_a_byte = 8
  bits_in_mdiff = len(differentiated_modes_mv_bitstream)

  differentiated_modes_mv_bitstream = exp_golomb_coding(y_res) + exp_golomb_coding(x_res) + exp_golomb_coding(i) + exp_golomb_coding(QP) + exp_golomb_coding(nRefFrames) + exp_golomb_coding(bits_in_mdiff) + differentiated_modes_mv_bitstream

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
          prev_mv[0] = prev_mv[0] - mdiff[lin_idx+0]
          prev_mv[1] = prev_mv[1] - mdiff[lin_idx+1]
          prev_mv[2] = prev_mv[2] - mdiff[lin_idx+2]
          modes_mv += [[prev_mv[0], prev_mv[1], prev_mv[2]]]

          # print(modes_mv[-1])
          lin_idx += 3

        #  Decode
        predicted_block = predict_block(rec_buffer, new_reconstructed, modes_mv[-1], bl_y_it, bl_x_it, i, (not is_i_frame))

        new_reconstructed[bl_y_it][bl_x_it] = decoder_core(QTC_recovered, Q, predicted_block)

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

  in_file = "../videos/black_and_white.yuv"
  out_file = "../temp/q4_encoded.far"

  number_frames = 300
  y_res = 288
  x_res = 352
  i = 8
  r = 1
  QP = 6  # from 0 to (log_2(i) + 7)
  i_period = 5
  nRefFrames = 4

  # bits_in_each_frame = []

  decoder_infile = out_file
  decoder_outfile = "../videos/q4_decoded.yuv"

  encoder(in_file, out_file, number_frames, y_res, x_res, i, r, QP, i_period, nRefFrames)
  decoder(decoder_infile, decoder_outfile)

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
