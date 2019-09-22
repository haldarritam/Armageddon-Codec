import numpy as np
import sys
import struct

number_frames = 300
x_res = 352
y_res = 288
total_pixels_frame = x_res * y_res
bytes_frame = (int) (3 * (x_res * y_res) / 2)

y_frame = np.empty((0,0))
u_frame = np.empty((number_frames, y_res, x_res), dtype=int)
#u_frame = [[[0 for col in range(y_res)] for col in range(x_res)] for row in range(number_frames)]
v_frame = np.empty((number_frames, y_res, x_res), dtype=int)
#v_frame = [[[0 for col in range(y_res)] for col in range(x_res)] for row in range(number_frames)]

yuv_file = open("foreman_cif.yuv","rb")


for frame in range(number_frames) :
    raw = yuv_file.read(bytes_frame)
    y_frame = np.concatenate((y_frame, raw[ : x_res * y_res]), axis=None)

    for y_it in range(y_res) :
        u_it = 0
        for x_it in range(x_res) :
            u_offset = total_pixels_frame + u_it
            v_offset = (int) (u_offset + total_pixels_frame / 4)
            
            if y_it % 2 == 0 :
                if x_it % 2 == 0 :
                    u_frame[frame][y_it][x_it] = int.from_bytes((raw[u_offset : u_offset + 1]), byteorder=sys.byteorder)
                    v_frame[frame][y_it][x_it] = int.from_bytes((raw[v_offset : v_offset + 1]), byteorder=sys.byteorder)
                    u_it = u_it + 1
                    #print(u_frame[y_it][x_it][frame], int.from_bytes((raw[u_offset : u_offset + 1]), byteorder=sys.byteorder), raw[u_offset : u_offset + 1])
                else :
                    u_frame[frame][y_it][x_it] = u_frame[frame][y_it][x_it - 1]
                    v_frame[frame][y_it][x_it] = v_frame[frame][y_it][x_it - 1]
            else :
                u_frame[frame][y_it][x_it] = u_frame[frame][y_it - 1][x_it]
                v_frame[frame][y_it][x_it] = v_frame[frame][y_it - 1][x_it]
          
    progress = (int) (frame / (number_frames - 1) * 100)
    if progress % 10 == 0 :
        sys.stdout.write("Converting: %d%%   \r" % (progress) )
        sys.stdout.flush()
    
yuv_file.close()


converted = open("converted.yuv", "wb")

for frame in range(number_frames) :
    converted.write(y_frame[frame])
    
    for y_it in range(y_res) :    
        for x_it in range(x_res) :
            converted.write((int)(u_frame[frame][y_it][x_it]).to_bytes(1, byteorder=sys.byteorder))
            
    for y_it in range(y_res) :    
        for x_it in range(x_res) :
            converted.write((int)(v_frame[frame][y_it][x_it]).to_bytes(1, byteorder=sys.byteorder))
    
    progress = (int) (frame / (number_frames - 1) * 100)
    if progress % 10 == 0 :
        sys.stdout.write("Saving file: %d%%   \r" % (progress) )
        sys.stdout.flush()
        
converted.close()


    


