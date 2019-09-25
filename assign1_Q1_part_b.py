import numpy as np
import png
import sys

number_frames = 1
x_res = 352
y_res = 288
total_pixels_frame = x_res * y_res
bytes_frame = (int) (3 * (total_pixels_frame))

yuv_to_rgb = np.array([1.164, 0, 1.596, 1.164, 0.392, 0.813, 1.164, 2.017, 0]).reshape(3, 3)
yuv_array = np.empty((3,1))

rgb_pixels = np.empty((y_res, 3*x_res), dtype=np.uint8)

converted = open("./videos/foreman_cif_444.yuv", "rb")

for frame in range(number_frames) :
    raw = converted.read(bytes_frame)
    for y_it in range(y_res):
        x_rgb_it = 0
        for x_it in range(x_res):
            it = (y_it * x_res) + x_it
            u_it = it + total_pixels_frame
            v_it = u_it + total_pixels_frame

            yuv_array[0] = int.from_bytes((raw[it:it+1]), byteorder=sys.byteorder) - 16
            yuv_array[1] = int.from_bytes((raw[u_it:u_it+1]), byteorder=sys.byteorder) - 128
            yuv_array[2] = int.from_bytes((raw[v_it:v_it+1]), byteorder=sys.byteorder) - 128

            rgb = np.dot(yuv_to_rgb, yuv_array)

            print(rgb)
            print(yuv_array)
            quit()

            rgb[0] = np.round(max(0, min(rgb[0], 255)))
            rgb[1] = np.round(max(0, min(rgb[1], 255)))
            rgb[2] = np.round(max(0, min(rgb[2], 255)))

            rgb_pixels[y_it][x_rgb_it]     = np.round(rgb[0])
            rgb_pixels[y_it][x_rgb_it + 1] = np.round(rgb[1])
            rgb_pixels[y_it][x_rgb_it + 2] = np.round(rgb[2])

            # print("---------------------")
            # print(rgb_pixels[y_it][x_rgb_it], "--->", rgb[0])
            # print(rgb_pixels[y_it][x_rgb_it + 1], "--->", rgb[1])
            # print(rgb_pixels[y_it][x_rgb_it + 2], "--->", rgb[2])
            # print("---------------------")

            # quit()

            x_rgb_it = x_rgb_it + 3
    file_name = "./png/frame_" + str(frame + 1) + ".png"

    png.from_array(rgb_pixels, mode='RGB').save(file_name)

converted.close()
