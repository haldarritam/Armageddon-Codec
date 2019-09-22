import numpy as np
import sys
import struct

number_frames = 300
x_res = 352
y_res = 288
total_pixels_frame = x_res * y_res
bytes_frame = (int) (3 * (x_res * y_res) / 2)

y_frame = np.empty((0,0))
u_frame = np.empty((x_res, y_res, number_frames))
v_frame = np.empty((x_res, y_res, number_frames))

yuv_file = open("foreman_cif.yuv","rb")


for frame in range(number_frames) :
    raw = yuv_file.read(bytes_frame)
    y_frame = np.concatenate((y_frame, raw[ : x_res * y_res]), axis=None)

    for y_it in range(y_res) :    
        for x_it in range(x_res) :
            it_offset = y_it * x_res + x_it
            u_offset = total_pixels_frame + it_offset
            v_offset = (int) (u_offset + total_pixels_frame / 4)
            
            if y_it % 2 == 0 :
                if x_it % 2 == 0 :#if it_offset % 2 == 0 :
                    u_frame[x_it][y_it][frame] = int.from_bytes(raw[u_offset : u_offset + 1], byteorder=sys.byteorder)
                    v_frame[x_it][y_it][frame] = int.from_bytes(raw[v_offset : v_offset + 1], byteorder=sys.byteorder)
                else :
                    u_frame[x_it][y_it][frame] = u_frame[x_it - 1][y_it][frame]
                    v_frame[x_it][y_it][frame] = v_frame[x_it - 1][y_it][frame]
            else :
                u_frame[x_it][y_it][frame] = u_frame[x_it][y_it - 1][frame]
                v_frame[x_it][y_it][frame] = v_frame[x_it][y_it - 1][frame]
          
    progress = (int) (frame / (number_frames - 1) * 100)
    if progress % 10 == 0 :
        sys.stdout.write("Progress: %d%%   \r" % (progress) )
        sys.stdout.flush()
    
yuv_file.close()


converted = open("converted.yuv", "wb")

for frame in range(number_frames) :
    converted.write(y_frame[frame])
    
    for y_it in range(y_res) :    
        for x_it in range(x_res) :
            converted.write((int)(u_frame[x_it][y_it][frame]).to_bytes(1, byteorder=sys.byteorder))
            
    for y_it in range(y_res) :    
        for x_it in range(x_res) :
            converted.write((int)(v_frame[x_it][y_it][frame]).to_bytes(1, byteorder=sys.byteorder))
    
converted.close()


    


