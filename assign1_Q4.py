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
  mv = (0, 0)

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

  return [mv]

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

def extract_block(block, head_idy, head_idx, mv):
  extracted = block[head_idy + mv[0] : head_idy + mv[0] + i, head_idx + mv[1] : head_idx + mv[1] + i]

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

def differential_encoder_decoder(modes_mv_block, is_p_block, not_first_bl):
  return_data = 0
  if (is_p_block):
    if (not_first_bl):
      return_data = list(np.array(modes_mv_block[-2]) - np.array(modes_mv_block[-1]))
    else:
      return_data = list(np.array([0,0]) - np.array(modes_mv_block[-1]))

  else:
    if (not_first_bl):
      return_data = modes_mv_block[-2] - modes_mv_block[-1]
    else:
      return_data = 0 - modes_mv_block[-1]

  return [return_data]

def exp_golomb_coding(number):
  if (number <= 0):
    return [-2 * number]
  else:
    return [(2 * number) - 1]

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

def I_RLE(data, i):
    print("Inside")
    qtc_line = []
    elements_per_block = i*i
    counter = 0
    iterat = 0
    while (iterat< len(data)):
        #print(iterat)
        if(data[iterat]<0): #non-zero
            counter -= data[iterat]
            print('counter=, interator = ',counter,iterat)
            qtc_line += data[iterat+1:iterat + counter+1]
            counter += 1
            print(qtc_line)
            iterat += counter
            print('non-zero')
        elif(data[iterat]>0): #zero
            counter += data[iterat]
            qtc_line += [0]*data[iterat]
            print(qtc_line)
            print('zero')
            iterat +=  1
        else: #end of block
            missing_elements = elements_per_block - counter
            qtc_line += [0]*missing_elements
            print(qtc_line)
            counter = elements_per_block
            iterat +=  counter + 1
            print('EoB')

        if(counter==elements_per_block):
            counter = 0

    return qtc_line



##############################################################################
##############################################################################

def encoder(in_file, out_file, number_frames, y_res, x_res, i, r, QP, i_period):

  grey = 128
  bl_y_frame, n_y_blocks, n_x_blocks, ext_y_res, ext_x_res = pre.block(in_file, y_res, x_res, number_frames, i)
  reconst = np.full((ext_y_res, ext_x_res), grey, dtype=int)

  residual_matrix = np.empty((number_frames, n_y_blocks, n_x_blocks, i, i), dtype=np.int16)

  QTC = np.empty((number_frames, n_y_blocks, n_x_blocks, i, i), dtype=int)

  approx_residual = np.empty((number_frames, n_y_blocks, n_x_blocks, i, i), dtype=int)

  Q = calculate_Q(i, QP)

  new_reconstructed = np.empty((n_y_blocks, n_x_blocks, i, i), dtype = int)


  converted = open(out_file, "wb")
  mdiff_file = open("./temp/mdiff.far", "wb")
  qtc_file = open("./temp/qtc.far", "wb")

  for frame in range(number_frames):

    print("Loop of frame: ", frame)

    modes_mv_block = []
    differentiated_modes_mv_block = []
    intra_predicted_frame = np.empty((n_y_blocks, n_x_blocks, i, i), dtype=int)

    is_p_block = frame % i_period

    for bl_y_it in range(n_y_blocks) :
      for bl_x_it in range(n_x_blocks):

        predicted_block = np.empty((i, i), dtype=int)
        if (is_p_block):
          # Calculate Motion Vector (inter)
          modes_mv_block += motion_vector_estimation(bl_y_frame[frame][bl_y_it][bl_x_it], reconst, r, bl_y_it * i, bl_x_it * i, ext_y_res, ext_x_res, i)

          predicted_block = extract_block(reconst, bl_y_it * i, bl_x_it * i, modes_mv_block[-1])

        else:
          # Calculate mode (intra)
          temp_mode, predicted_block = intra_prediction(intra_predicted_frame, bl_y_it, bl_x_it)
          modes_mv_block += temp_mode


    #  Calculate Residual Matrix
        residual_matrix[frame][bl_y_it][bl_x_it] = calculate_residual_block(bl_y_frame[frame][bl_y_it][bl_x_it], predicted_block)

    #  Trans/Quant/Rescaling/InvTrans
        transformed_dct = dct2D(residual_matrix[frame][bl_y_it][bl_x_it])
        QTC[frame][bl_y_it][bl_x_it] = quantize_dct(transformed_dct, Q)

    #  Decoding Preparations
        if(is_p_block):
          predicted_block = extract_block(reconst, bl_y_it * i, bl_x_it * i, modes_mv_block[-1])
        else:
          top_edge = np.full((1, i), grey)
          left_edge = np.full((i, 1), grey)

          predicted_block[:,:] = top_edge

          if ((modes_mv_block[-1] == 1) and ((bl_y_it - 1) >= 0)):
            top_edge = new_reconstructed[bl_y_it - 1][bl_x_it][-1, :]
            predicted_block[:,:] = top_edge

          if ((modes_mv_block[-1] == 0) and ((bl_x_it - 1) >= 0)):
            left_edge = new_reconstructed[bl_y_it][bl_x_it - 1][:, -1].reshape((i, 1))
            predicted_block[:, :] = left_edge

        new_reconstructed[bl_y_it][bl_x_it] = decoder_core(QTC[frame][bl_y_it][bl_x_it], Q, predicted_block)

        intra_predicted_frame[bl_y_it][bl_x_it] = new_reconstructed[bl_y_it][bl_x_it]

        if (is_p_block):
          differentiated_modes_mv_block += exp_golomb_coding(differential_encoder_decoder(modes_mv_block, is_p_block, bl_x_it)[0][0])
          differentiated_modes_mv_block += exp_golomb_coding(differential_encoder_decoder(modes_mv_block, is_p_block, bl_x_it)[0][1])
        else:
          differentiated_modes_mv_block += exp_golomb_coding(differential_encoder_decoder(modes_mv_block, is_p_block, bl_x_it)[0])

        scanned_block = scanning(QTC[frame][bl_y_it][bl_x_it])
        rled_block = RLE(scanned_block)
        # qtc_file.write(bytearray(rled_block))
        for rled in rled_block:
          qtc_file.write((int)(exp_golomb_coding(rled)[0]).to_bytes(1, byteorder=sys.byteorder, signed=False))


    if (is_p_block):
      differentiated_modes_mv_block.insert(0, 0)
    else:
      differentiated_modes_mv_block.insert(0, 1)
    # mdiff_file.write(bytearray(differentiated_modes_mv_block))
    for mdiff in differentiated_modes_mv_block:
      mdiff_file.write((int)(mdiff).to_bytes(1, byteorder=sys.byteorder, signed=False))

    #  Concatenate
    counter = 0
    conc_reconstructed = np.empty((ext_y_res, ext_x_res), dtype = int)
    for bl_y_it in range(n_y_blocks):
      conc = np.concatenate((new_reconstructed[bl_y_it][0], new_reconstructed[bl_y_it][1]), axis=1)
      for bl_x_it in range(2, n_x_blocks):
        conc = np.concatenate((conc, new_reconstructed[bl_y_it][bl_x_it]), axis=1)

      for i_it in range(i):
        if (counter < y_res):
          counter += 1
          for x_it in range(x_res):
            conc_reconstructed[bl_y_it*i + i_it][x_it] = conc[i_it][x_it]
            converted.write(((int)(conc[i_it][x_it])).to_bytes(1, byteorder=sys.byteorder))

    reconst = conc_reconstructed

  # print(modes_mv)

  converted.close()
  mdiff_file.close()
  qtc_file.close()

##############################################################################
##############################################################################

def decoder(y_res, x_res):

  rled_qtc = []
  mdiff = []
  size = 1
  with open("./temp/qtc.far", 'rb') as file:
    while True:
        data = file.read(size)
        if not data:
            # eof
            break
        rled_qtc += [int.from_bytes(data, byteorder=sys.byteorder, signed=False)]

  with open("./temp/mdiff.far", 'rb') as file:
    while True:
        data = file.read(size)
        if not data:
            # eof
            break
        mdiff += [int.from_bytes(data, byteorder=sys.byteorder, signed=False)]

  print(rled_qtc)
  print(mdiff)

  # mv = np.load("./temp/motion_vectors.npz")['mv']
  # residual_matrix = np.load("./temp/residual_matrix.npz")['residual_matrix']

  # number_of_frames = residual_matrix.shape[0]
  # i = residual_matrix.shape[3]
  # ext_y_res = residual_matrix.shape[1] * i
  # ext_x_res = residual_matrix.shape[2] * i

  # reconst = np.full((ext_y_res, ext_x_res), 128, dtype=int)

  # converted = open("./videos/decoder_test.yuv", "wb")

  # for frame in range(number_of_frames):

  #   print(frame)

  #   new_reconstructed = decoder_core(residual_matrix[frame], reconst, mv[frame])

  #   for y_it in range(y_res):
  #     for x_it in range(x_res):

  #       converted.write(((int)(new_reconstructed[y_it][x_it])).to_bytes(1, byteorder=sys.byteorder))

  #   reconst = new_reconstructed

  # converted.close()

##############################################################################
##############################################################################

if __name__ == "__main__":

  in_file = "./videos/black_and_white.yuv"
  out_file = "./videos/encoder_test.yuv"
  number_frames = 5
  y_res = 288
  x_res = 352
  i = 16
  r = 3
  QP = 6  # from 0 to (log_2(i) + 7)
  i_period = 3

  # encoder(in_file, out_file, number_frames, y_res, x_res, i, r, QP, i_period)
  #decoder(y_res, x_res)
  result = I_RLE([-3,7,5,2,4,-2,1,2,-4,4,8,12,16,0],4)
  print(result)
  print(len(result))
