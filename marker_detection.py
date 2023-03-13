### Aruco marker generator : https://chev.me/arucogen/
import cv2
from cv2 import aruco
import numpy as np
import csv
import time
import pyrealsense2 as rs
import math
import pandas as pd
import os

from datetime import datetime
from RealSense_Utilities.realsense_api.realsense_api import RealSenseCamera
from RealSense_Utilities.realsense_api.realsense_api import find_realsense
from RealSense_Utilities.realsense_api.realsense_api import frame_to_np_array
from RealSense_Utilities.realsense_api.realsense_api import mediapipe_detection

# ARUCO_DICT = {
# 	"DICT_4X4_50": cv2.aruco.DICT_4X4_50,
# 	"DICT_4X4_100": cv2.aruco.DICT_4X4_100,
# 	"DICT_4X4_250": cv2.aruco.DICT_4X4_250,
# 	"DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
# 	"DICT_5X5_50": cv2.aruco.DICT_5X5_50,
# 	"DICT_5X5_100": cv2.aruco.DICT_5X5_100,
# 	"DICT_5X5_250": cv2.aruco.DICT_5X5_250,
# 	"DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
# 	"DICT_6X6_50": cv2.aruco.DICT_6X6_50,
# 	"DICT_6X6_100": cv2.aruco.DICT_6X6_100,
# 	"DICT_6X6_250": cv2.aruco.DICT_6X6_250,
# 	"DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
# 	"DICT_7X7_50": cv2.aruco.DICT_7X7_50,
# 	"DICT_7X7_100": cv2.aruco.DICT_7X7_100,
# 	"DICT_7X7_250": cv2.aruco.DICT_7X7_250,
# 	"DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
# 	"DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
# 	"DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
# 	"DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
# 	"DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
# 	"DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
# }

#realsense 카메라 사용하기 위해 존재함. :: aruco marker를 인식하기 위해 realsense 카메라를 사용한다.
def convert_depth_to_phys_coord(xp, yp, depth, intr):
    result = rs.rs2_deproject_pixel_to_point(intr, [int(xp), int(yp)], depth)

    return result[0], result[1], result[2]

#aruco marker 사용하기 위해 존재함
dict_aruco = aruco.Dictionary_get(aruco.DICT_4X4_50)    #: "정의된 객체": 딕셔너리
parameters = aruco.DetectorParameters_create()          #: "detection parametes"
marker_list=[]

#realsense 카메라 사용하기 위해 존재함.
cameras = {}
realsense_device = find_realsense()
rs_main = None

for serial, devices in realsense_device:
    if serial == '123622270472':
        cameras[serial] = RealSenseCamera(depth_stream_width=640, depth_stream_height=480,
                                          color_stream_width=640, color_stream_height=480,
                                          color_stream_fps=30, depth_stream_fps=90,
                                          device=devices, adv_mode_flag=True, device_type="d415")
    else:
        cameras[serial] = RealSenseCamera(depth_stream_width=640, depth_stream_height=480,
                                          color_stream_width=640, color_stream_height=480,
                                          color_stream_fps=30, depth_stream_fps=90,
                                          device=devices, device_type="d455", adv_mode_flag=True)


#realsense 카메라 사용하기 위해 존재함.
for ser, dev in cameras.items():
    rs_main = dev

    s_time=int(round(time.time()*1000))

    try:
        while True:
            rs_main.get_data()

            current_timestamp = rs_main.current_timestamp
            c_time=int(round(time.time()*1000))-s_time
            #print(c_time)

            frameset = rs_main.frameset
            rs_main.get_aligned_frames(frameset, aligned_to_color=True)

            frameset = rs_main.depth_to_disparity.process(rs_main.frameset)
            frameset = rs_main.spatial_filter.process(frameset)
            frameset = rs_main.temporal_filter.process(frameset)
            frameset = rs_main.disparity_to_depth.process(frameset)
            frameset = rs_main.hole_filling_filter.process(frameset).as_frameset()

            rs_main.frameset = frameset

            rs_main.color_frame = frameset.get_color_frame()
            rs_main.depth_frame = frameset.get_depth_frame()

            rs_main.color_image = frame_to_np_array(rs_main.color_frame)

            img_rs0 = np.copy(rs_main.color_image)
            # img_rs0 = zoom(img_rs0.copy(), scale=zoom_scale)
            img_raw = np.copy(img_rs0)

            #openCV: 색 변환 함수 : rgb에서 gray로 색을 변환하기.:: 그레이 스케일 이미지로 변환
            gray = cv2.cvtColor(img_rs0, cv2.COLOR_RGB2GRAY)    #: "마커를 찾을 이미지"
            #그레이로 변환 후 // 이진화를 하고 contour를 추출하는 과정이 없음.

            img_h, img_w, img_c = img_rs0.shape

#############aruco marker##########
            #detect marker: 실제 마커 검출하는 함수.
            corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, dict_aruco, parameters=parameters)

            #이 부분은 마커의 코너와 깊이에 따라 검출하는 로직
            if len(corners) > 0:
                """
                for i in range(0, len(ids)):
                    # Estimate pose of each marker and return the values rvec and tvec---(different from those of camera coefficients)
                    rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.02, cameraMatrix=cam_matrix,
                                                                             distCoeffs=dist_matrix)
                """
                #ids = ids.flatten()
                for (markerCorner, markerID) in zip(corners, ids):
                    corners_list = markerCorner.reshape((4, 2))
                    (topLeft, topRight, bottomRight, bottomLeft) = corners_list
                    topRight = (topRight[0], topRight[1])
                    bottomRight = (bottomRight[0], bottomRight[1])
                    bottomLeft = (bottomLeft[0], bottomLeft[1])
                    topLeft = (topLeft[0], topLeft[1])

                    #depth calculate
                    depth_tr = rs_main.depth_frame.get_distance(topRight[0], topRight[1])
                    depth_tl = rs_main.depth_frame.get_distance(topLeft[0], topLeft[1])
                    depth_br = rs_main.depth_frame.get_distance(bottomRight[0], bottomRight[1])
                    depth_bl = rs_main.depth_frame.get_distance(bottomLeft[0], bottomLeft[1])

                    topRight = (convert_depth_to_phys_coord(topRight[0], topRight[1],depth_tr,rs_main.color_intrinsics))
                    topLeft = (
                        convert_depth_to_phys_coord(topLeft[0], topLeft[1], depth_tl, rs_main.color_intrinsics))
                    bottomRight = (
                        convert_depth_to_phys_coord(bottomRight[0], bottomRight[1], depth_br, rs_main.color_intrinsics))
                    bottomLeft = (
                        convert_depth_to_phys_coord(bottomLeft[0], bottomLeft[1], depth_bl, rs_main.color_intrinsics))

                    marker_list.append([c_time, markerID, topRight, topLeft, bottomRight, bottomLeft])

            #적어도 하나의 마커가 검출 되었을 때.
            #draw : 검출된 마커들을 영상에 그려준다. :: 카메라를 통해 입력받은 영상에서 마커 검출 -> 결과물은 이미지
            frame_markers = aruco.drawDetectedMarkers(img_raw, corners, ids)
            marker = np.array(marker_list, dtype=object)

            ##realsense 카메라 -> cv2에서 이미지 출력하기
            cv2.imshow('frame', frame_markers)
            cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
            key = cv2.waitKey(1)

            if key & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break

            #save csv
            if key & 0xFF == ord('s'):
                suffix = datetime.now().strftime('%y%m%d_%H%M%S')
                fileName = suffix + '.csv'
                with open("marker"+fileName, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerows(marker)
                    marker_list.clear()

    finally:
        rs_main.stop()



