import numpy as np
import sys

number_frames = 300
x_res = 352
y_res = 288
total_pixels_frame = x_res * y_res
bytes_frame = (int) (3 * (x_res * y_res) / 2)

yuv_to_rgb = np.array([1.164, 0, 1.596, 1.164, 0.392, 0.813, 1.164, 2.017, 0]).reshape(3,3)
yuv_array = np.empty((3,1))

rgb_pixel = np.empty((3*total_pixels_frame), dtype=int)

converted = open("converted.yuv", "wb")

for frame in range(number_frames) :
    raw = converted.read(bytes_frame)
    for it in range(total_pixels_frame) :
        u_it = it + total_pixels_frame
        v_it = u_it + total_pixels_frame

        yuv_array[0] = int.from_bytes((raw[it]), byteorder=sys.byteorder) - 16
        yuv_array[1] = int.from_bytes((raw[u_it]), byteorder=sys.byteorder) - 128
        yuv_array[2] = int.from_bytes((raw[v_it]), byteorder=sys.byteorder) - 128

        rgb = np.dot(yuv_to_rgb, yuv_array)

        rgb_pixel[it] = rgb[0]
        rgb_pixel[u_it] = rgb[1]
        rgb_pixel[v_it] = rgb[2]
