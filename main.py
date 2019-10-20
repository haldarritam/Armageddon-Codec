import sys
import argparse
sys.path.insert(1, './src')

from assign1_Q3 import encoder as q3_encoder, decoder as q3_decoder
from assign1_Q4 import encoder as q4_encoder, decoder as q4_decoder


if __name__ == "__main__":

  # Command line argument parser
  parser = argparse.ArgumentParser(description='Armageddon video codec.')
  
  parser.add_argument('-frames', help='Number of frames to encode.', default=300,type=int, metavar='default=300')

  parser.add_argument('-y_res', help='Y-Resolution of the video.', default=288, type=int, metavar='default=288')

  parser.add_argument('-x_res', help='X-Resolution of the video.', default=352, type=int, metavar='default=352')

  parser.add_argument('-i', help='Block size for the encoder.', default=16, type=int, metavar='default=16')

  parser.add_argument('-r', help='Inter prediction search range.', default=2, type=int, metavar='default=2')

  parser.add_argument('-n', help='Approximation parameter (Q3).', default=3, type=int, metavar='default=3')
  
  parser.add_argument('-QP', help='Quantization parameter (Q4).', default=3, type=int, metavar='default=3')  

  parser.add_argument('-ip', help='I-Period  (Q4).', default=6, type=int, metavar='default=6')

  parser.add_argument('-in', help='Input file location.', type=str, metavar='...', required=True)

  parser.add_argument('-out', help='Output file location.', type=str, metavar='...', required=True)  

  parser.add_argument('-o', help='Operation: (encode) or (decode)', default='encode', type=str, choices=['encode', 'decode'])
  
  parser.add_argument('-q', help='Question number.', default=4, type=int, choices=[3, 4], metavar='default=4')
  
  
  # Reading command line arguments
  args = vars(parser.parse_args())  
  number_frames = args['frames']
  y_res = args['y_res']
  x_res = args['x_res']
  i = args['i']
  r = args['r']
  n = args['n']
  QP = args['QP']  # from 0 to (log_2(i) + 7)
  i_period = args['ip']
  in_file = args['in']
  out_file = args['out']

  operation = args['o']
  q = args['q']
  

  if (q == 4):
    if (operation == 'encode'):
      q4_encoder(in_file, out_file, number_frames, y_res, x_res, i, r, QP, i_period)
    else:
      q4_decoder(in_file, out_file)

  elif (q == 3):
    if (operation == 'encode'):
      q3_encoder(in_file, out_file, number_frames, y_res, x_res, i, r, n)
    else:
      q3_decoder(in_file, out_file)