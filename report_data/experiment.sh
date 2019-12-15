#!/usr/bin/env bash

# ######################################################################
# Generating for various r values (Q3) - Foreman
# ######################################################################
# python main.py -in ./videos/black_and_white.yuv -out ./videos/q3_r1.yuv -y_res 288 -x_res 352 -frames 10 -o both -q 3 -i 8 -n 3 -r 1 &

# python main.py -in ./videos/black_and_white.yuv -out ./videos/q3_r4.yuv -y_res 288 -x_res 352 -frames 10 -o both -q 3 -i 8 -n 3 -r 4 &

# python main.py -in ./videos/black_and_white.yuv -out ./videos/q3_r8.yuv -y_res 288 -x_res 352 -frames 10 -o both -q 3 -i 8 -n 3 -r 8 &


# ######################################################################
# Generating for various n values (Q3) - Foreman
# ######################################################################
# python main.py -in ./videos/black_and_white.yuv -out ./videos/q3_n1.yuv -y_res 288 -x_res 352 -frames 10 -o both -q 3 -i 8 -n 1 -r 4 &

# python main.py -in ./videos/black_and_white.yuv -out ./videos/q3_n2.yuv -y_res 288 -x_res 352 -frames 10 -o both -q 3 -i 8 -n 2 -r 4 &

# python main.py -in ./videos/black_and_white.yuv -out ./videos/q3_n3.yuv -y_res 288 -x_res 352 -frames 10 -o both -q 3 -i 8 -n 3 -r 4 &

# ######################################################################
# Generating for various QP values (Q4) (RD Curve) - Foreman
# ######################################################################

# # i = 8
# # GOP is IIII

# python main.py -in ./videos/black_and_white.yuv -out ./videos/report/q4_ip1_qp0.yuv -y_res 288 -x_res 352 -frames 10 -o both -q 4 -i 8 -r 2 -ip 1 -QP 0
# python main.py -in ./videos/black_and_white.yuv -out ./videos/report/q4_ip1_qp1.yuv -y_res 288 -x_res 352 -frames 10 -o both -q 4 -i 8 -r 2 -ip 1 -QP 1
# python main.py -in ./videos/black_and_white.yuv -out ./videos/report/q4_ip1_qp2.yuv -y_res 288 -x_res 352 -frames 10 -o both -q 4 -i 8 -r 2 -ip 1 -QP 2
# python main.py -in ./videos/black_and_white.yuv -out ./videos/report/q4_ip1_qp3.yuv -y_res 288 -x_res 352 -frames 10 -o both -q 4 -i 8 -r 2 -ip 1 -QP 3
# python main.py -in ./videos/black_and_white.yuv -out ./videos/report/q4_ip1_qp4.yuv -y_res 288 -x_res 352 -frames 10 -o both -q 4 -i 8 -r 2 -ip 1 -QP 4
# python main.py -in ./videos/black_and_white.yuv -out ./videos/report/q4_ip1_qp5.yuv -y_res 288 -x_res 352 -frames 10 -o both -q 4 -i 8 -r 2 -ip 1 -QP 5
# python main.py -in ./videos/black_and_white.yuv -out ./videos/report/q4_ip1_qp6.yuv -y_res 288 -x_res 352 -frames 10 -o both -q 4 -i 8 -r 2 -ip 1 -QP 6
# python main.py -in ./videos/black_and_white.yuv -out ./videos/report/q4_ip1_qp7.yuv -y_res 288 -x_res 352 -frames 10 -o both -q 4 -i 8 -r 2 -ip 1 -QP 7
# python main.py -in ./videos/black_and_white.yuv -out ./videos/report/q4_ip1_qp8.yuv -y_res 288 -x_res 352 -frames 10 -o both -q 4 -i 8 -r 2 -ip 1 -QP 8
# python main.py -in ./videos/black_and_white.yuv -out ./videos/report/q4_ip1_qp9.yuv -y_res 288 -x_res 352 -frames 10 -o both -q 4 -i 8 -r 2 -ip 1 -QP 9
# python main.py -in ./videos/black_and_white.yuv -out ./videos/report/q4_ip1_qp10.yuv -y_res 288 -x_res 352 -frames 10 -o both -q 4 -i 8 -r 2 -ip 1 -QP 10

# # GOP is IPPPIPPPI...

# python main.py -in ./videos/black_and_white.yuv -out ./videos/report/q4_ip4_qp0.yuv -y_res 288 -x_res 352 -frames 10 -o both -q 4 -i 8 -r 2 -ip 4 -QP 0
# python main.py -in ./videos/black_and_white.yuv -out ./videos/report/q4_ip4_qp1.yuv -y_res 288 -x_res 352 -frames 10 -o both -q 4 -i 8 -r 2 -ip 4 -QP 1
# python main.py -in ./videos/black_and_white.yuv -out ./videos/report/q4_ip4_qp2.yuv -y_res 288 -x_res 352 -frames 10 -o both -q 4 -i 8 -r 2 -ip 4 -QP 2
# python main.py -in ./videos/black_and_white.yuv -out ./videos/report/q4_ip4_qp3.yuv -y_res 288 -x_res 352 -frames 10 -o both -q 4 -i 8 -r 2 -ip 4 -QP 3
# python main.py -in ./videos/black_and_white.yuv -out ./videos/report/q4_ip4_qp4.yuv -y_res 288 -x_res 352 -frames 10 -o both -q 4 -i 8 -r 2 -ip 4 -QP 4
# python main.py -in ./videos/black_and_white.yuv -out ./videos/report/q4_ip4_qp5.yuv -y_res 288 -x_res 352 -frames 10 -o both -q 4 -i 8 -r 2 -ip 4 -QP 5
# python main.py -in ./videos/black_and_white.yuv -out ./videos/report/q4_ip4_qp6.yuv -y_res 288 -x_res 352 -frames 10 -o both -q 4 -i 8 -r 2 -ip 4 -QP 6
# python main.py -in ./videos/black_and_white.yuv -out ./videos/report/q4_ip4_qp7.yuv -y_res 288 -x_res 352 -frames 10 -o both -q 4 -i 8 -r 2 -ip 4 -QP 7
# python main.py -in ./videos/black_and_white.yuv -out ./videos/report/q4_ip4_qp8.yuv -y_res 288 -x_res 352 -frames 10 -o both -q 4 -i 8 -r 2 -ip 4 -QP 8
# python main.py -in ./videos/black_and_white.yuv -out ./videos/report/q4_ip4_qp9.yuv -y_res 288 -x_res 352 -frames 10 -o both -q 4 -i 8 -r 2 -ip 4 -QP 9
# python main.py -in ./videos/black_and_white.yuv -out ./videos/report/q4_ip4_qp10.yuv -y_res 288 -x_res 352 -frames 10 -o both -q 4 -i 8 -r 2 -ip 4 -QP 10

# # GOP is IPPPPPPPPPIPPPPPPPPPI...

# python main.py -in ./videos/black_and_white.yuv -out ./videos/report/q4_ip10_qp0.yuv -y_res 288 -x_res 352 -frames 10 -o both -q 4 -i 8 -r 2 -ip 10 -QP 0
# python main.py -in ./videos/black_and_white.yuv -out ./videos/report/q4_ip10_qp1.yuv -y_res 288 -x_res 352 -frames 10 -o both -q 4 -i 8 -r 2 -ip 10 -QP 1
# python main.py -in ./videos/black_and_white.yuv -out ./videos/report/q4_ip10_qp2.yuv -y_res 288 -x_res 352 -frames 10 -o both -q 4 -i 8 -r 2 -ip 10 -QP 2
# python main.py -in ./videos/black_and_white.yuv -out ./videos/report/q4_ip10_qp3.yuv -y_res 288 -x_res 352 -frames 10 -o both -q 4 -i 8 -r 2 -ip 10 -QP 3
# python main.py -in ./videos/black_and_white.yuv -out ./videos/report/q4_ip10_qp4.yuv -y_res 288 -x_res 352 -frames 10 -o both -q 4 -i 8 -r 2 -ip 10 -QP 4
# python main.py -in ./videos/black_and_white.yuv -out ./videos/report/q4_ip10_qp5.yuv -y_res 288 -x_res 352 -frames 10 -o both -q 4 -i 8 -r 2 -ip 10 -QP 5
# python main.py -in ./videos/black_and_white.yuv -out ./videos/report/q4_ip10_qp6.yuv -y_res 288 -x_res 352 -frames 10 -o both -q 4 -i 8 -r 2 -ip 10 -QP 6
# python main.py -in ./videos/black_and_white.yuv -out ./videos/report/q4_ip10_qp7.yuv -y_res 288 -x_res 352 -frames 10 -o both -q 4 -i 8 -r 2 -ip 10 -QP 7
# python main.py -in ./videos/black_and_white.yuv -out ./videos/report/q4_ip10_qp8.yuv -y_res 288 -x_res 352 -frames 10 -o both -q 4 -i 8 -r 2 -ip 10 -QP 8
# python main.py -in ./videos/black_and_white.yuv -out ./videos/report/q4_ip10_qp9.yuv -y_res 288 -x_res 352 -frames 10 -o both -q 4 -i 8 -r 2 -ip 10 -QP 9
# python main.py -in ./videos/black_and_white.yuv -out ./videos/report/q4_ip10_qp10.yuv -y_res 288 -x_res 352 -frames 10 -o both -q 4 -i 8 -r 2 -ip 10 -QP 10

# # # i = 16
# # # GOP is IIII
# python main.py -in ./videos/black_and_white.yuv -out ./videos/report/q4_ip1_qp0.yuv -y_res 288 -x_res 352 -frames 10 -o both -q 4 -i 16 -r 2 -ip 1 -QP 0
# python main.py -in ./videos/black_and_white.yuv -out ./videos/report/q4_ip1_qp1.yuv -y_res 288 -x_res 352 -frames 10 -o both -q 4 -i 16 -r 2 -ip 1 -QP 1
# python main.py -in ./videos/black_and_white.yuv -out ./videos/report/q4_ip1_qp2.yuv -y_res 288 -x_res 352 -frames 10 -o both -q 4 -i 16 -r 2 -ip 1 -QP 2
# python main.py -in ./videos/black_and_white.yuv -out ./videos/report/q4_ip1_qp3.yuv -y_res 288 -x_res 352 -frames 10 -o both -q 4 -i 16 -r 2 -ip 1 -QP 3
# python main.py -in ./videos/black_and_white.yuv -out ./videos/report/q4_ip1_qp4.yuv -y_res 288 -x_res 352 -frames 10 -o both -q 4 -i 16 -r 2 -ip 1 -QP 4
# python main.py -in ./videos/black_and_white.yuv -out ./videos/report/q4_ip1_qp5.yuv -y_res 288 -x_res 352 -frames 10 -o both -q 4 -i 16 -r 2 -ip 1 -QP 5
# python main.py -in ./videos/black_and_white.yuv -out ./videos/report/q4_ip1_qp6.yuv -y_res 288 -x_res 352 -frames 10 -o both -q 4 -i 16 -r 2 -ip 1 -QP 6
# python main.py -in ./videos/black_and_white.yuv -out ./videos/report/q4_ip1_qp7.yuv -y_res 288 -x_res 352 -frames 10 -o both -q 4 -i 16 -r 2 -ip 1 -QP 7
# python main.py -in ./videos/black_and_white.yuv -out ./videos/report/q4_ip1_qp8.yuv -y_res 288 -x_res 352 -frames 10 -o both -q 4 -i 16 -r 2 -ip 1 -QP 8
# python main.py -in ./videos/black_and_white.yuv -out ./videos/report/q4_ip1_qp9.yuv -y_res 288 -x_res 352 -frames 10 -o both -q 4 -i 16 -r 2 -ip 1 -QP 9
# python main.py -in ./videos/black_and_white.yuv -out ./videos/report/q4_ip1_qp10.yuv -y_res 288 -x_res 352 -frames 10 -o both -q 4 -i 16 -r 2 -ip 1 -QP 10
# python main.py -in ./videos/black_and_white.yuv -out ./videos/report/q4_ip1_qp11.yuv -y_res 288 -x_res 352 -frames 10 -o both -q 4 -i 16 -r 2 -ip 1 -QP 11

# # GOP is IPPPIPPPI...

# python main.py -in ./videos/black_and_white.yuv -out ./videos/report/q4_ip4_qp0.yuv -y_res 288 -x_res 352 -frames 10 -o both -q 4 -i 16 -r 2 -ip 4 -QP 0
# python main.py -in ./videos/black_and_white.yuv -out ./videos/report/q4_ip4_qp1.yuv -y_res 288 -x_res 352 -frames 10 -o both -q 4 -i 16 -r 2 -ip 4 -QP 1
# python main.py -in ./videos/black_and_white.yuv -out ./videos/report/q4_ip4_qp2.yuv -y_res 288 -x_res 352 -frames 10 -o both -q 4 -i 16 -r 2 -ip 4 -QP 2
# python main.py -in ./videos/black_and_white.yuv -out ./videos/report/q4_ip4_qp3.yuv -y_res 288 -x_res 352 -frames 10 -o both -q 4 -i 16 -r 2 -ip 4 -QP 3
# python main.py -in ./videos/black_and_white.yuv -out ./videos/report/q4_ip4_qp4.yuv -y_res 288 -x_res 352 -frames 10 -o both -q 4 -i 16 -r 2 -ip 4 -QP 4
# python main.py -in ./videos/black_and_white.yuv -out ./videos/report/q4_ip4_qp5.yuv -y_res 288 -x_res 352 -frames 10 -o both -q 4 -i 16 -r 2 -ip 4 -QP 5
# python main.py -in ./videos/black_and_white.yuv -out ./videos/report/q4_ip4_qp6.yuv -y_res 288 -x_res 352 -frames 10 -o both -q 4 -i 16 -r 2 -ip 4 -QP 6
# python main.py -in ./videos/black_and_white.yuv -out ./videos/report/q4_ip4_qp7.yuv -y_res 288 -x_res 352 -frames 10 -o both -q 4 -i 16 -r 2 -ip 4 -QP 7
# python main.py -in ./videos/black_and_white.yuv -out ./videos/report/q4_ip4_qp8.yuv -y_res 288 -x_res 352 -frames 10 -o both -q 4 -i 16 -r 2 -ip 4 -QP 8
# python main.py -in ./videos/black_and_white.yuv -out ./videos/report/q4_ip4_qp9.yuv -y_res 288 -x_res 352 -frames 10 -o both -q 4 -i 16 -r 2 -ip 4 -QP 9
# python main.py -in ./videos/black_and_white.yuv -out ./videos/report/q4_ip4_qp10.yuv -y_res 288 -x_res 352 -frames 10 -o both -q 4 -i 16 -r 2 -ip 4 -QP 10
# python main.py -in ./videos/black_and_white.yuv -out ./videos/report/q4_ip4_qp11.yuv -y_res 288 -x_res 352 -frames 10 -o both -q 4 -i 16 -r 2 -ip 4 -QP 11

# # GOP is IPPPPPPPPPIPPPPPPPPPI...

# python main.py -in ./videos/black_and_white.yuv -out ./videos/report/q4_ip10_qp0.yuv -y_res 288 -x_res 352 -frames 10 -o both -q 4 -i 16 -r 2 -ip 10 -QP 0
# python main.py -in ./videos/black_and_white.yuv -out ./videos/report/q4_ip10_qp1.yuv -y_res 288 -x_res 352 -frames 10 -o both -q 4 -i 16 -r 2 -ip 10 -QP 1
# python main.py -in ./videos/black_and_white.yuv -out ./videos/report/q4_ip10_qp2.yuv -y_res 288 -x_res 352 -frames 10 -o both -q 4 -i 16 -r 2 -ip 10 -QP 2
# python main.py -in ./videos/black_and_white.yuv -out ./videos/report/q4_ip10_qp3.yuv -y_res 288 -x_res 352 -frames 10 -o both -q 4 -i 16 -r 2 -ip 10 -QP 3
# python main.py -in ./videos/black_and_white.yuv -out ./videos/report/q4_ip10_qp4.yuv -y_res 288 -x_res 352 -frames 10 -o both -q 4 -i 16 -r 2 -ip 10 -QP 4
# python main.py -in ./videos/black_and_white.yuv -out ./videos/report/q4_ip10_qp5.yuv -y_res 288 -x_res 352 -frames 10 -o both -q 4 -i 16 -r 2 -ip 10 -QP 5
# python main.py -in ./videos/black_and_white.yuv -out ./videos/report/q4_ip10_qp6.yuv -y_res 288 -x_res 352 -frames 10 -o both -q 4 -i 16 -r 2 -ip 10 -QP 6
# python main.py -in ./videos/black_and_white.yuv -out ./videos/report/q4_ip10_qp7.yuv -y_res 288 -x_res 352 -frames 10 -o both -q 4 -i 16 -r 2 -ip 10 -QP 7
# python main.py -in ./videos/black_and_white.yuv -out ./videos/report/q4_ip10_qp8.yuv -y_res 288 -x_res 352 -frames 10 -o both -q 4 -i 16 -r 2 -ip 10 -QP 8
# python main.py -in ./videos/black_and_white.yuv -out ./videos/report/q4_ip10_qp9.yuv -y_res 288 -x_res 352 -frames 10 -o both -q 4 -i 16 -r 2 -ip 10 -QP 9
# python main.py -in ./videos/black_and_white.yuv -out ./videos/report/q4_ip10_qp10.yuv -y_res 288 -x_res 352 -frames 10 -o both -q 4 -i 16 -r 2 -ip 10 -QP 10
# python main.py -in ./videos/black_and_white.yuv -out ./videos/report/q4_ip10_qp11.yuv -y_res 288 -x_res 352 -frames 10 -o both -q 4 -i 16 -r 2 -ip 10 -QP 11

# ######################################################################
# Generating for various i values (Q3) - Hall
# ######################################################################
# python main.py -in ./videos/hall_qcif_bw.yuv -out ./videos/q3_i2.yuv -y_res 144 -x_res 176 -frames 10 -o both -q 3 -i 2 -n 3 -r 4 &

# python main.py -in ./videos/hall_qcif_bw.yuv -out ./videos/q3_i8.yuv -y_res 144 -x_res 176 -frames 10 -o both -q 3 -i 8 -n 3 -r 4 &

# python main.py -in ./videos/hall_qcif_bw.yuv -out ./videos/q3_i64.yuv -y_res 144 -x_res 176 -frames 10 -o both -q 3 -i 64 -n 3 -r 4 &

# ######################################################################
# Generating for various r values (Q3) - Hall
# ######################################################################
# python main.py -in ./videos/hall_qcif_bw.yuv -out ./videos/q3_r1.yuv -y_res 144 -x_res 176 -frames 10 -o both -q 3 -i 8 -n 3 -r 1 &

# python main.py -in ./videos/hall_qcif_bw.yuv -out ./videos/q3_r4.yuv -y_res 144 -x_res 176 -frames 10 -o both -q 3 -i 8 -n 3 -r 4 &

# python main.py -in ./videos/hall_qcif_bw.yuv -out ./videos/q3_r8.yuv -y_res 144 -x_res 176 -frames 10 -o both -q 3 -i 8 -n 3 -r 8 &


# ######################################################################
# Generating for various n values (Q3) - Hall
# ######################################################################
# python main.py -in ./videos/hall_qcif_bw.yuv -out ./videos/q3_n1.yuv -y_res 144 -x_res 176 -frames 10 -o both -q 3 -i 8 -n 1 -r 4 &

# python main.py -in ./videos/hall_qcif_bw.yuv -out ./videos/q3_n2.yuv -y_res 144 -x_res 176 -frames 10 -o both -q 3 -i 8 -n 2 -r 4 &

# python main.py -in ./videos/hall_qcif_bw.yuv -out ./videos/q3_n3.yuv -y_res 144 -x_res 176 -frames 10 -o both -q 3 -i 8 -n 3 -r 4 &

# ######################################################################
# Generating for various QP values (Q3) (effect of the i and r)
# ######################################################################


# python main.py -in ./videos/black_and_white.yuv -out ./videos/report/q3_table/i2_r2.yuv -y_res 288 -x_res 352 -frames 10 -o both -q 3 -n 3 -i 2 -r 2

# python main.py -in ./videos/black_and_white.yuv -out ./videos/report/q3_table/i8_r2.yuv -y_res 288 -x_res 352 -frames 10 -o both -q 3 -n 3 -i 8 -r 2

# python main.py -in ./videos/black_and_white.yuv -out ./videos/report/q3_table/i64_r2.yuv -y_res 288 -x_res 352 -frames 10 -o both -q 3 -n 3 -i 64 -r 2

# python main.py -in ./videos/black_and_white.yuv -out ./videos/report/q3_table/i2_r4.yuv -y_res 288 -x_res 352 -frames 10 -o both -q 3 -n 3 -i 2 -r 4

# python main.py -in ./videos/black_and_white.yuv -out ./videos/report/q3_table/i8_r4.yuv -y_res 288 -x_res 352 -frames 10 -o both -q 3 -n 3 -i 8 -r 4

# python main.py -in ./videos/black_and_white.yuv -out ./videos/report/q3_table/i64_r4.yuv -y_res 288 -x_res 352 -frames 10 -o both -q 3 -n 3 -i 64 -r 4

# python main.py -in ./videos/black_and_white.yuv -out ./videos/report/q3_table/i2_r8.yuv -y_res 288 -x_res 352 -frames 10 -o both -q 3 -n 3 -i 2 -r 8

# python main.py -in ./videos/black_and_white.yuv -out ./videos/report/q3_table/i8_r8.yuv -y_res 288 -x_res 352 -frames 10 -o both -q 3 -n 3 -i 8 -r 8

# python main.py -in ./videos/black_and_white.yuv -out ./videos/report/q3_table/i64_r8.yuv -y_res 288 -x_res 352 -frames 10 -o both -q 3 -n 3 -i 64 -r 8

# ######################################################################
# Generating for various QP values (A2)
# ######################################################################

# python main.py -in ./videos/black_and_white.yuv -out ./videos/tests/QP1-1000.yuv -y_res 288 -x_res 352 -frames 10 -o both  -i 16 -r 4 -ip 8 -QP 1 -nRef 1 -vbs 0 -fme 0 -fastME 0

# python main.py -in ./videos/black_and_white.yuv -out ./videos/tests/QP4-1000.yuv -y_res 288 -x_res 352 -frames 10 -o both  -i 16 -r 4 -ip 8 -QP 4 -nRef 1 -vbs 0 -fme 0 -fastME 0

# python main.py -in ./videos/black_and_white.yuv -out ./videos/tests/QP4-1000.yuv -y_res 288 -x_res 352 -frames 10 -o both  -i 16 -r 4 -ip 8 -QP 4 -nRef 1 -vbs 0 -fme 0 -fastME 0

# python main.py -in ./videos/black_and_white.yuv -out ./videos/tests/QP7-1000.yuv -y_res 288 -x_res 352 -frames 10 -o both  -i 16 -r 4 -ip 8 -QP 7 -nRef 1 -vbs 0 -fme 0 -fastME 0

# python main.py -in ./videos/black_and_white.yuv -out ./videos/tests/QP10-1000.yuv -y_res 288 -x_res 352 -frames 10 -o both  -i 16 -r 4 -ip 8 -QP 10 -nRef 1 -vbs 0 -fme 0 -fastME 0




# python main.py -in ./videos/black_and_white.yuv -out ./videos/tests/QP1-2000.yuv -y_res 288 -x_res 352 -frames 10 -o both  -i 16 -r 4 -ip 8 -QP 1 -nRef 2 -vbs 0 -fme 0 -fastME 0

# python main.py -in ./videos/black_and_white.yuv -out ./videos/tests/QP4-2000.yuv -y_res 288 -x_res 352 -frames 10 -o both  -i 16 -r 4 -ip 8 -QP 4 -nRef 2 -vbs 0 -fme 0 -fastME 0

# python main.py -in ./videos/black_and_white.yuv -out ./videos/tests/QP4-2000.yuv -y_res 288 -x_res 352 -frames 10 -o both  -i 16 -r 4 -ip 8 -QP 4 -nRef 2 -vbs 0 -fme 0 -fastME 0

# python main.py -in ./videos/black_and_white.yuv -out ./videos/tests/QP7-2000.yuv -y_res 288 -x_res 352 -frames 10 -o both  -i 16 -r 4 -ip 8 -QP 7 -nRef 2 -vbs 0 -fme 0 -fastME 0

# python main.py -in ./videos/black_and_white.yuv -out ./videos/tests/QP10-2000.yuv -y_res 288 -x_res 352 -frames 10 -o both  -i 16 -r 4 -ip 8 -QP 10 -nRef 2 -vbs 0 -fme 0 -fastME 0




# python main.py -in ./videos/black_and_white.yuv -out ./videos/tests/QP1-1100.yuv -y_res 288 -x_res 352 -frames 10 -o both  -i 16 -r 4 -ip 8 -QP 1 -nRef 1 -vbs 1 -fme 0 -fastME 0

# python main.py -in ./videos/black_and_white.yuv -out ./videos/tests/QP4-1100.yuv -y_res 288 -x_res 352 -frames 10 -o both  -i 16 -r 4 -ip 8 -QP 4 -nRef 1 -vbs 1 -fme 0 -fastME 0

# python main.py -in ./videos/black_and_white.yuv -out ./videos/tests/QP4-1100.yuv -y_res 288 -x_res 352 -frames 10 -o both  -i 16 -r 4 -ip 8 -QP 4 -nRef 1 -vbs 1 -fme 0 -fastME 0

# python main.py -in ./videos/black_and_white.yuv -out ./videos/tests/QP7-1100.yuv -y_res 288 -x_res 352 -frames 10 -o both  -i 16 -r 4 -ip 8 -QP 7 -nRef 1 -vbs 1 -fme 0 -fastME 0

# python main.py -in ./videos/black_and_white.yuv -out ./videos/tests/QP10-1100.yuv -y_res 288 -x_res 352 -frames 10 -o both  -i 16 -r 4 -ip 8 -QP 10 -nRef 1 -vbs 1 -fme 0 -fastME 0




# python main.py -in ./videos/black_and_white.yuv -out ./videos/tests/QP1-1010.yuv -y_res 288 -x_res 352 -frames 10 -o both  -i 16 -r 4 -ip 8 -QP 1 -nRef 1 -vbs 0 -fme 1 -fastME 0

# python main.py -in ./videos/black_and_white.yuv -out ./videos/tests/QP4-1010.yuv -y_res 288 -x_res 352 -frames 10 -o both  -i 16 -r 4 -ip 8 -QP 4 -nRef 1 -vbs 0 -fme 1 -fastME 0

# python main.py -in ./videos/black_and_white.yuv -out ./videos/tests/QP4-1010.yuv -y_res 288 -x_res 352 -frames 10 -o both  -i 16 -r 4 -ip 8 -QP 4 -nRef 1 -vbs 0 -fme 1 -fastME 0

# python main.py -in ./videos/black_and_white.yuv -out ./videos/tests/QP7-1010.yuv -y_res 288 -x_res 352 -frames 10 -o both  -i 16 -r 4 -ip 8 -QP 7 -nRef 1 -vbs 0 -fme 1 -fastME 0

# python main.py -in ./videos/black_and_white.yuv -out ./videos/tests/QP10-1010.yuv -y_res 288 -x_res 352 -frames 10 -o both  -i 16 -r 4 -ip 8 -QP 10 -nRef 1 -vbs 0 -fme 1 -fastME 0




# python main.py -in ./videos/black_and_white.yuv -out ./videos/tests/QP1-1001.yuv -y_res 288 -x_res 352 -frames 10 -o both  -i 16 -r 4 -ip 8 -QP 1 -nRef 1 -vbs 0 -fme 0 -fastME 1

# python main.py -in ./videos/black_and_white.yuv -out ./videos/tests/QP4-1001.yuv -y_res 288 -x_res 352 -frames 10 -o both  -i 16 -r 4 -ip 8 -QP 4 -nRef 1 -vbs 0 -fme 0 -fastME 1

# python main.py -in ./videos/black_and_white.yuv -out ./videos/tests/QP4-1001.yuv -y_res 288 -x_res 352 -frames 10 -o both  -i 16 -r 4 -ip 8 -QP 4 -nRef 1 -vbs 0 -fme 0 -fastME 1

# python main.py -in ./videos/black_and_white.yuv -out ./videos/tests/QP7-1001.yuv -y_res 288 -x_res 352 -frames 10 -o both  -i 16 -r 4 -ip 8 -QP 7 -nRef 1 -vbs 0 -fme 0 -fastME 1

# python main.py -in ./videos/black_and_white.yuv -out ./videos/tests/QP10-1001.yuv -y_res 288 -x_res 352 -frames 10 -o both  -i 16 -r 4 -ip 8 -QP 10 -nRef 1 -vbs 0 -fme 0 -fastME 1




# python main.py -in ./videos/black_and_white.yuv -out ./videos/tests/QP1-2111.yuv -y_res 288 -x_res 352 -frames 10 -o both  -i 16 -r 4 -ip 8 -QP 1 -nRef 2 -vbs 1 -fme 1 -fastME 1

# python main.py -in ./videos/black_and_white.yuv -out ./videos/tests/QP4-2111.yuv -y_res 288 -x_res 352 -frames 10 -o both  -i 16 -r 4 -ip 8 -QP 4 -nRef 2 -vbs 1 -fme 1 -fastME 1

# python main.py -in ./videos/black_and_white.yuv -out ./videos/tests/QP4-2111.yuv -y_res 288 -x_res 352 -frames 10 -o both  -i 16 -r 4 -ip 8 -QP 4 -nRef 2 -vbs 1 -fme 1 -fastME 1

# python main.py -in ./videos/black_and_white.yuv -out ./videos/tests/QP7-2111.yuv -y_res 288 -x_res 352 -frames 10 -o both  -i 16 -r 4 -ip 8 -QP 7 -nRef 2 -vbs 1 -fme 1 -fastME 1

# python main.py -in ./videos/black_and_white.yuv -out ./videos/tests/QP10-2111.yuv -y_res 288 -x_res 352 -frames 10 -o both  -i 16 -r 4 -ip 8 -QP 10 -nRef 2 -vbs 1 -fme 1 -fastME 1

# ----------------------------------------------------------------#
# --------------------- Assignment 3 -----------------------------#
# ----------------------------------------------------------------#

# # --------------------- RC=0 -----------------------------#
# python main.py -in ./videos/CIF_bw.yuv -out ./videos/tests/CIF_RC0_QP3.yuv -y_res 288 -x_res 352 -frames 21 -o both  -i 16 -r 1 -ip 21 -QP 3 -nRef 1 -vbs 1 -fme 1 -fastME 1 -RCflag 0 -targetBR 2458 -ParallelMode 0

# python main.py -in ./videos/CIF_bw.yuv -out ./videos/tests/CIF_RC0_QP6.yuv -y_res 288 -x_res 352 -frames 21 -o both  -i 16 -r 1 -ip 21 -QP 6 -nRef 1 -vbs 1 -fme 1 -fastME 1 -RCflag 0 -targetBR 2458 -ParallelMode 0

# python main.py -in ./videos/CIF_bw.yuv -out ./videos/tests/CIF_RC0_QP9.yuv -y_res 288 -x_res 352 -frames 21 -o both  -i 16 -r 1 -ip 21 -QP 9 -nRef 1 -vbs 1 -fme 1 -fastME 1 -RCflag 0 -targetBR 2458 -ParallelMode 0

# python main.py -in ./videos/CIF_bw.yuv -out ./videos/tests/CIF_RC0_QP0.yuv -y_res 288 -x_res 352 -frames 21 -o both  -i 16 -r 1 -ip 21 -QP 0 -nRef 1 -vbs 1 -fme 1 -fastME 1 -RCflag 0 -targetBR 2458 -ParallelMode 0

# python main.py -in ./videos/CIF_bw.yuv -out ./videos/tests/CIF_RC0_QP1.yuv -y_res 288 -x_res 352 -frames 21 -o both  -i 16 -r 1 -ip 21 -QP 1 -nRef 1 -vbs 1 -fme 1 -fastME 1 -RCflag 0 -targetBR 2458 -ParallelMode 0

# python main.py -in ./videos/CIF_bw.yuv -out ./videos/tests/CIF_RC0_QP2.yuv -y_res 288 -x_res 352 -frames 21 -o both  -i 16 -r 1 -ip 21 -QP 2 -nRef 1 -vbs 1 -fme 1 -fastME 1 -RCflag 0 -targetBR 2458 -ParallelMode 0

# python main.py -in ./videos/CIF_bw.yuv -out ./videos/tests/CIF_RC0_QP11.yuv -y_res 288 -x_res 352 -frames 21 -o both  -i 16 -r 1 -ip 21 -QP 11 -nRef 1 -vbs 1 -fme 1 -fastME 1 -RCflag 0 -targetBR 2458 -ParallelMode 0

# # --------------------- RC=1 -----------------------------#
# python main.py -in ./videos/CIF_bw.yuv -out ./videos/tests/CIF_RC1_BR7m.yuv -y_res 288 -x_res 352 -frames 21 -o both  -i 16 -r 1 -ip 21 -QP 3 -nRef 1 -vbs 1 -fme 1 -fastME 1 -RCflag 1 -targetBR 7168 -ParallelMode 0

# python main.py -in ./videos/CIF_bw.yuv -out ./videos/tests/CIF_RC1_BR2_4m.yuv -y_res 288 -x_res 352 -frames 21 -o both  -i 16 -r 1 -ip 21 -QP 3 -nRef 1 -vbs 1 -fme 1 -fastME 1 -RCflag 1 -targetBR 2458 -ParallelMode 0

# python main.py -in ./videos/CIF_bw.yuv -out ./videos/tests/CIF_RC1_BR360k.yuv -y_res 288 -x_res 352 -frames 21 -o both  -i 16 -r 1 -ip 21 -QP 3 -nRef 1 -vbs 1 -fme 1 -fastME 1 -RCflag 1 -targetBR 360 -ParallelMode 0

# # --------------------- RC=2 -----------------------------#
# python main.py -in ./videos/CIF_bw.yuv -out ./videos/tests/CIF_RC2_BR7m.yuv -y_res 288 -x_res 352 -frames 21 -o both  -i 16 -r 1 -ip 21 -QP 3 -nRef 1 -vbs 1 -fme 1 -fastME 1 -RCflag 2 -targetBR 7168 -ParallelMode 0

# python main.py -in ./videos/CIF_bw.yuv -out ./videos/tests/CIF_RC2_BR2_4m.yuv -y_res 288 -x_res 352 -frames 21 -o both  -i 16 -r 1 -ip 21 -QP 3 -nRef 1 -vbs 1 -fme 1 -fastME 1 -RCflag 2 -targetBR 2458 -ParallelMode 0

# python main.py -in ./videos/CIF_bw.yuv -out ./videos/tests/CIF_RC2_BR360k.yuv -y_res 288 -x_res 352 -frames 21 -o both  -i 16 -r 1 -ip 21 -QP 3 -nRef 1 -vbs 1 -fme 1 -fastME 1 -RCflag 2 -targetBR 360 -ParallelMode 0

# # --------------------- RC=3 -----------------------------#
# python main.py -in ./videos/CIF_bw.yuv -out ./videos/tests/CIF_RC3_BR7m.yuv -y_res 288 -x_res 352 -frames 21 -o both  -i 16 -r 1 -ip 21 -QP 3 -nRef 1 -vbs 1 -fme 1 -fastME 1 -RCflag 3 -targetBR 7168 -ParallelMode 0

# python main.py -in ./videos/CIF_bw.yuv -out ./videos/tests/CIF_RC3_BR2_4m.yuv -y_res 288 -x_res 352 -frames 21 -o both  -i 16 -r 1 -ip 21 -QP 3 -nRef 1 -vbs 1 -fme 1 -fastME 1 -RCflag 3 -targetBR 2458 -ParallelMode 0

# python main.py -in ./videos/CIF_bw.yuv -out ./videos/tests/CIF_RC3_BR360k.yuv -y_res 288 -x_res 352 -frames 21 -o both  -i 16 -r 1 -ip 21 -QP 3 -nRef 1 -vbs 1 -fme 1 -fastME 1 -RCflag 3 -targetBR 360 -ParallelMode 0

# # --------------------- PSNR per frame -----------------------------#

python main.py -in ./videos/CIF_bw.yuv -out ./videos/tests/CIF_RC1_BR2m.yuv -y_res 288 -x_res 352 -frames 21 -o both  -i 16 -r 1 -ip 21 -QP 3 -nRef 1 -vbs 1 -fme 1 -fastME 1 -RCflag 1 -targetBR 2048 -ParallelMode 0

python main.py -in ./videos/CIF_bw.yuv -out ./videos/tests/CIF_RC2_BR2m.yuv -y_res 288 -x_res 352 -frames 21 -o both  -i 16 -r 1 -ip 21 -QP 3 -nRef 1 -vbs 1 -fme 1 -fastME 1 -RCflag 2 -targetBR 2048 -ParallelMode 0

python main.py -in ./videos/CIF_bw.yuv -out ./videos/tests/CIF_RC3_BR2m.yuv -y_res 288 -x_res 352 -frames 21 -o both  -i 16 -r 1 -ip 21 -QP 3 -nRef 1 -vbs 1 -fme 1 -fastME 1 -RCflag 3 -targetBR 2048 -ParallelMode 0