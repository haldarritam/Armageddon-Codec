import numpy as np
import png
import sys

number_frames = 1
x_res = 352
y_res = 288
total_pixels_frame = x_res * y_res
bytes_frame = (int) (3 * (x_res * y_res) / 2)

yuv_to_rgb = np.array([1.164, 0, 1.596, 1.164, 0.392, 0.813, 1.164, 2.017, 0]).reshape(3,3)
yuv_array = np.empty((3,1))

rgb_pixels = np.empty((3*total_pixels_frame))

converted = open("./videos/foreman_cif_444.yuv", "rb")

for frame in range(number_frames) :
        raw = converted.read(bytes_frame)
        for it in range(total_pixels_frame) :
                u_it = it + total_pixels_frame
                v_it = u_it + total_pixels_frame

                yuv_array[0] = int.from_bytes((raw[it:it+1]), byteorder=sys.byteorder) - 16
                yuv_array[1] = int.from_bytes((raw[u_it:u_it+1]), byteorder=sys.byteorder) - 128
                yuv_array[2] = int.from_bytes((raw[v_it:v_it+1]), byteorder=sys.byteorder) - 128

                rgb = np.dot(yuv_to_rgb, yuv_array)

                rgb_pixels[it] = rgb[0]
                rgb_pixels[u_it] = rgb[1]
                rgb_pixels[v_it] = rgb[2]

        png.from_array(rgb_pixels, mode='RGB', info={'height': 288, 'width': 352, 'bitdepth':3}).save('foo.png')

converted.close()
