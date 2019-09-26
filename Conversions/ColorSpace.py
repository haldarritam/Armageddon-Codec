import numpy as np
import sys

class CSC:
    def __init__(self):
        '''Initialize CSC'''
    def ftz_to_bw(self, file, y_res, x_res, number_frames):
        total_pixels_frame = x_res * y_res
        bytes_frame = (int) (3 * (x_res * y_res) / 2)

        y_frame = np.empty((0,0))

        yuv_file = open(file,"rb")

        for frame in range(number_frames) :
            raw = yuv_file.read(bytes_frame)
            y_frame = np.concatenate((y_frame, raw[: x_res * y_res]), axis=None)
                                    
            self._progress("Converting:", frame, number_frames)  
              
        yuv_file.close()

        converted = open("../videos/black_and_white.yuv", "wb")

        for frame in range(number_frames) :
            converted.write(y_frame[frame])    
            self._progress("Saving file:", frame, number_frames)
            
        converted.close()

    def ftz_to_fff(self, file, y_res, x_res, number_frames):
        total_pixels_frame = x_res * y_res
        bytes_frame = (int) (3 * (x_res * y_res) / 2)

        y_frame = np.empty((0,0))
        u_frame = np.empty((number_frames, y_res, x_res), dtype=int)
        v_frame = np.empty((number_frames, y_res, x_res), dtype=int)

        yuv_file = open(file,"rb")

        for frame in range(number_frames) :
            raw = yuv_file.read(bytes_frame)
            y_frame = np.concatenate((y_frame, raw[ : x_res * y_res]), axis=None)
            u_it = 0
    
            for y_it in range(y_res) :    
                for x_it in range(x_res) :
                    u_offset = total_pixels_frame + u_it
                    v_offset = (int) (u_offset + total_pixels_frame / 4)
            
                    if y_it % 2 == 0 :
                        if x_it % 2 == 0 :
                            u_frame[frame][y_it][x_it] = int.from_bytes((raw[u_offset : u_offset + 1]), byteorder=sys.byteorder)
                            v_frame[frame][y_it][x_it] = int.from_bytes((raw[v_offset : v_offset + 1]), byteorder=sys.byteorder)
                            u_it = u_it + 1
                        else :
                            u_frame[frame][y_it][x_it] = u_frame[frame][y_it][x_it - 1]
                            v_frame[frame][y_it][x_it] = v_frame[frame][y_it][x_it - 1]
                    else :
                        u_frame[frame][y_it][x_it] = u_frame[frame][y_it - 1][x_it]
                        v_frame[frame][y_it][x_it] = v_frame[frame][y_it - 1][x_it]
                        
            self._progress("Converting:", frame, number_frames)  
              
        yuv_file.close()

        converted = open(self._output_file_name(file), "wb")

        for frame in range(number_frames) :
            converted.write(y_frame[frame])
    
            for y_it in range(y_res) :    
                for x_it in range(x_res) :
                    converted.write((int)(u_frame[frame][y_it][x_it]).to_bytes(1, byteorder=sys.byteorder))
            
            for y_it in range(y_res) :    
                for x_it in range(x_res) :
                    converted.write((int)(v_frame[frame][y_it][x_it]).to_bytes(1, byteorder=sys.byteorder))
    
            self._progress("Saving file:", frame, number_frames)
            
        converted.close()
        
    def _progress(self, message, current, total):
        progress = (int) (current / (total - 1) * 100)
        if progress % 10 == 0 :
            sys.stdout.write(message + " %d%%   \r" % (progress) )
            sys.stdout.flush()
            
    def _output_file_name(self, orig_file_name) :
        file_and_extension = orig_file_name.split(".")
        return ".".join(file_and_extension[:-1]) + "_444.yuv"
        
    
if __name__ == "__main__":
    converter = CSC()
    number_frames = 300
    x_res = 352
    y_res = 288
    # converter.ftz_to_fff("../videos/foreman_cif.yuv", y_res, x_res, number_frames)
    converter.ftz_to_bw("../videos/foreman_cif.yuv", y_res, x_res, number_frames)


    


