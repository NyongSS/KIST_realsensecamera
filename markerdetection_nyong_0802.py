import cv2
import cv2.aruco as aruco
import pyrealsense2 as rs
import numpy as np
import os
import pickle
from RealSense_Utilities.realsense_api.realsense_api import RealSenseCamera
from RealSense_Utilities.realsense_api.realsense_api import find_realsense
from RealSense_Utilities.realsense_api.realsense_api import frame_to_np_array
import csv
import time
import math
from datetime import datetime


# realsense 카메라 사용하기 위한 함수
def findRealsenseCamera():
    realsense_ctx = rs.context()
    connected_devices = []
    for i in range(len(realsense_ctx.devices)):
        detected_camera = realsense_ctx.devices[i].get_info(rs.camera_info.serial_number)
        connected_devices.append(detected_camera)

    return connected_devices


# def inversePerspective(paramrvec, paramtvec):
#     R, _ = cv2.Rodrigues(paramrvec)
#     R = np.matrix(R).T
#     invTvec = np.dot(R, np.matrix(-paramtvec))
#     invRvec, _ = cv2.Rodrigues(R)
#     return invRvec, invTvec
# def drawAxis(img, p, q, colour, scale=1):
#     angle = math.atan2(p.y - q.y, p.x - q.x)  # angle in radians
#     hypotenuse = math.sqrt((p.y - q.y) ** 2 + (p.x - q.x) ** 2)
#     # Here we lengthen the arrow by a factor of scale
#     q = (int(p.x - scale * hypotenuse * math.cos(angle)), int(p.y - scale * hypotenuse * math.sin(angle)))
#     cv2.line(img, p, q, colour, 1, cv2.LINE_AA)
#     # create the arrow hooks
#     p = (int(q[0] + 9 * math.cos(angle + math.pi / 4)), int(q[1] + 9 * math.sin(angle + math.pi / 4)))
#     cv2.line(img, p, q, colour, 1, cv2.LINE_AA)
#     p = (int(q[0] + 9 * math.cos(angle - math.pi / 4)), int(q[1] + 9 * math.sin(angle - math.pi / 4)))
#     cv2.line(img, p, q, colour, 1, cv2.LINE_AA)

# 칼만필터: 잡음이 포함되어있는 측정치를 바탕으로 선형 역학계의 상태를 추정하는 재귀필터. 과거에 수행한 측정값을 바탕으로 현재의 상태 변수의 결합분포를 추정.
# 근데 실제로 사용되지 않음 ㅎ\def TrackKalman(xm, ym):
    """
    input: xm, ym (측정값 - 가로, 세로 위치 정보)
    output: xh, yh (추정값 - 가로, 세로 위치 정보)
    """

    global x, P, A, H, Q, R

    # 상태변수: 동적 시스템의 동적 상태를 결정해 주는 최소개의 변수들.
    # 상태변수 추정값과 오차 공분산 예측
    xp = A.dot(x).reshape(4, 1)  # 4x4 * 4x1 = 4x1
    Pp = A.dot(P).dot(A.T) + Q  # 오차공분산에 시스템 오차 추가 4x4 * 4*4 * 4*4 + 4*4 = 4x4

    # 칼만 이득 계산
    K = Pp.dot(H.T).dot(np.linalg.inv(H.dot(P).dot(H.T) + R))  # 4x4 * 4*2 * (2x4 * 4x4 * 4*2 + 2x2) = 4x2

    z = np.array([[xm], [ym]])  # 2x1

    # 추정값 계산
    x = xp + (K.dot(z - (H.dot(xp)).reshape(2, 1))).reshape(4, 1)  # 4x1 + 4x2 * (2x1 - 2x4 * 4x1) = 4x1

    # 오차 공분산 계산
    P = Pp - K.dot(H).dot(Pp)  # 4x4 - 4x2 * 2x4 * 4x4 = 4x4

    return x[0][0], x[2][0]  # 위치


# color intrinsic과 depth intrinsic을 얻음.
# intrinsic: 카메라 파라미터 행렬에서 카메라 렌즈와 센서 위치에 의해 결정되는 항목의 파라미터.
def get_intrinsics(profileArg):
    Color_intrinsics = profileArg.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    Depth_intrinsics = profileArg.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
    cameramatrix = np.array(
        [[Color_intrinsics.fx, 0, Color_intrinsics.ppx], [0, Color_intrinsics.fy, Color_intrinsics.ppy], [0, 0, 1]],
        dtype=np.
        float64)
    distcoeffs = np.array([Color_intrinsics.coeffs], dtype=np.float64)

    model = Color_intrinsics.model

    return cameramatrix, distcoeffs, model, Color_intrinsics, Depth_intrinsics


# depth image를 physical로 projection하는 함수
def convert_depth_to_phys_coord(xp, yp, depth, intr):
    result = rs.rs2_deproject_pixel_to_point(intr, [int(xp), int(yp)], depth)
    return result[0], result[1], result[2]


# 카메라에 의한 왜곡 보정: 왜곡되지 않은 점 반환하기.
def return_undistortion_point(distPoint, intr):
    point = rs.rs2_deproject_pixel_to_point(intr, distPoint, 1)

    Ux = int(point[0] * intr.fx + intr.ppx)
    Uy = int(point[1] * intr.fy + intr.ppy)
    return Ux, Uy


# if not os.path.exists('./CameraCalibration.pckl'):
#     print("You need to calibrate the camera you'll be using. See calibration project directory for details.")
#     exit()
# else:
#     f = open('./CameraCalibration.pckl', 'rb')
#     cameraMatrix, distCoeffs, _, _ = pickle.load(f)
#     f.close()
#
#     print(cameraMatrix)
#     print(distCoeffs)
#     if cameraMatrix is None or distCoeffs is None:
#         print(
#             "Calibration issue. Remove ./CameraCalibration.pckl and recalibrate your camera with calibration_ChAruco.py.")
#         exit()

ARUCO_PARAMETERS = aruco.DetectorParameters()
ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

dt = 1
A = np.array([[1, dt, 0, 0],
              [0, 1, 0, 0],
              [0, 0, 1, dt],
              [0, 0, 0, 1]])
H = np.array([[1, 0, 0, 0],
              [0, 0, 1, 0]])
Q = 1.0 * np.eye(4)
R = np.array([[50, 0],
              [0, 50]])
P = 100 * np.eye(4)
x = np.array([0, 0, 0, 0])

# Create grid board object we're using in our stream
# board = aruco.GridBoard(
#     markersX=1,
#     markersY=1,
#     markerLength=0.027,
#     markerSeparation=0.01,
#     dictionary=ARUCO_DICT)

# translation vector 값의 볒ㄴ화 확인.    0.027->0.07 : 변화 없음.
board = aruco.GridBoard(
    (1, 1),
    0.07,
    0.01,
    ARUCO_DICT)

# Create vectors we'll be using for rotations and translations for postures
rotation_vectors, translation_vectors = None, None
axis = np.float32([[-.5, -.5, 0], [-.5, .5, 0], [.5, .5, 0], [.5, -.5, 0],
                   [-.5, -.5, 1], [-.5, .5, 1], [.5, .5, 1], [.5, -.5, 1]])

color_image_set = []
rs_device = findRealsenseCamera()
marker_list = []
rot_1 = []
rot_2 = []
trs_1 = []
trs_2 = []
# Configure depth and color streams...
# ...from Camera 1
pipeline_1 = rs.pipeline()
config_1 = rs.config()
config_1.enable_device(rs_device[0])
config_1.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config_1.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming from both cameras
profile = pipeline_1.start(config_1)

cameraMatrix, distCoeffs, cameraModel, C_Intr, D_Intr = get_intrinsics(profile)

depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: ", depth_scale)

clipping_distance_in_meters = 1.5  # 1 meter
clipping_distance = clipping_distance_in_meters / depth_scale

align_to = rs.stream.color
align = rs.align(align_to)

radian = 180 / math.pi
print(radian)
# millisecond로 현재 시간 받기
s_time = int(round(time.time() * 1000))
try:
    while True:
        # Camera 1
        # Wait for a coherent pair of frames: depth and color
        frames_1 = pipeline_1.wait_for_frames()
        aligned_frames = align.process(frames_1)
        depth_frame_1 = aligned_frames.get_depth_frame()
        color_frame_1 = aligned_frames.get_color_frame()
        if not depth_frame_1 or not color_frame_1:
            continue
        # Convert images to numpy arrays: 이미지 프레임을 depth, color 넘파이로 변환
        depth_image_1 = np.asanyarray(depth_frame_1.get_data())
        color_image_1 = np.asanyarray(color_frame_1.get_data())
        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        # depth_colormap_1 = cv2.applyColorMap(cv2.convertScaleAbs(depth_image_1, alpha=0.5), cv2.COLORMAP_JET)

        gray = cv2.cvtColor(color_image_1, cv2.COLOR_BGR2GRAY)

        # Detect Aruco markers
        #detect marker: 실제 마커 검출하는 함수. :: detector.detectMarkers(inputImage, markerCorners, markerIds, rejectCandidates);
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMETERS)
        # current_timestamp = rs_main.current_timestamp
        c_time = int(round(time.time() * 1000)) - s_time  # 앞에서 받은 현재의 밀리셐 시간에서 루프를 돌면서 받은 밀리셐 시간 차 계산
        timeNow = time.strftime('%H:%M:%S')
        #print(c_time)
        #print(timeNow)
        # Refine detected markers2
        # Eliminates markers not part of our board, adds missing markers to the board
        corners, ids, rejectedImgPoints, recoveredIds = aruco.refineDetectedMarkers(
            image=gray,
            board=board,
            detectedCorners=corners,
            detectedIds=ids,
            rejectedCorners=rejectedImgPoints,
            cameraMatrix=cameraMatrix,
            distCoeffs=distCoeffs)

        # detected id에 따라 아래의 것들을 수행. detected id가 여러개이면 여러번 수행할 것.
        # 마커 번호 달아서 네모 쳐서 보여주기
        # Outline all of the markers detected in our image
        # Uncomment below to show ids as well
        # ProjectImage = aruco.drawDetectedMarkers(ProjectImage, corners, ids, borderColor=(0, 0, 255))
        color_image_1 = aruco.drawDetectedMarkers(color_image_1, corners, borderColor=(0, 0, 255))

        # 이 부분은 마커의 코너와 깊이에 따라 검출하는 로직
        if len(corners) > 0:
            """
            for i in range(0, len(ids)):
                # Estimate pose of each marker and return the values rvec and tvec---(different from those of camera coefficients)
                rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.02, cameraMatrix=cam_matrix,
                                                                         distCoeffs=dist_matrix)
            """

            # print(rotation_vectors)

            Cx1 = 0
            Cy1 = 0
            Cz1 = 0
            Cx2 = 0
            Cy2 = 0
            Cz2 = 0
            for i in range(len(corners)):
                x_sum = corners[int(i)][0][0][0] + corners[int(i)][0][1][0] + corners[int(i)][0][2][0] + corners[int(i)][0][3][0]
                y_sum = corners[int(i)][0][0][1] + corners[int(i)][0][1][1] + corners[int(i)][0][2][1] + corners[int(i)][0][3][1]
                x_cen_0 = x_sum * .25
                y_cen_0 = y_sum * .25
                dist = depth_frame_1.get_distance(int(x_cen_0), int(y_cen_0))
                print(
                    "x_{} : {}, y_{} : {}, dist : {}".format(int(ids[i]), int(x_cen_0), int(ids[i]), int(y_cen_0),
                                                               dist / depth_scale))  # 인식된 실제 값

                if not dist == 0:
                    if ids[i] == 1:
                        # x, y, z를 모두 depth에서 physical coordinate로 convert한 후 depth scale로 나눈 값들.
                        Cx1, Cy1, Cz1 = convert_depth_to_phys_coord(x_cen_0, y_cen_0, dist, C_Intr)
                        print("Cx_{} : {}, Cy_{} : {}, Cz_{} : {}".format(int(ids[i]), Cx1 / depth_scale, int(ids[i]),
                                                                          Cy1 / depth_scale, int(ids[i]),
                                                                          Cz1 / depth_scale))  # depth에서 physical로 convert한 값

                    elif ids[i] == 2:
                        # x, y, z를 모두 depth에서 physical coordinate로 convert한 후 depth scale로 나눈 값들.
                        Cx2, Cy2, Cz2 = convert_depth_to_phys_coord(x_cen_0, y_cen_0, dist, C_Intr)
                        print("Cx_{} : {}, Cy_{} : {}, Cz_{} : {}".format(int(ids[i]), Cx2 / depth_scale, int(ids[i]),
                                                                          Cy2 / depth_scale, int(ids[i]),
                                                                          Cz2 / depth_scale))  # depth에서 physical로 convert한 값


                markerID = ids[i]
                rot_1_x = 0
                rot_1_y = 0
                rot_1_z = 0
                trs_1_x = 0
                trs_1_y = 0
                trs_1_z = 0

                rot_2_x = 0
                rot_2_y = 0
                rot_2_z = 0
                trs_2_x = 0
                trs_2_y = 0
                trs_2_z = 0

                trs_1_xx = 0
                trs_1_yy = 0
                trs_1_zz = 0
                trs_2_xx = 0
                trs_2_yy = 0
                trs_2_zz = 0

                # ids = ids.flatten()
                rotation_vectors, translation_vectors, _objPoints = aruco.estimatePoseSingleMarkers(corners, 7,
                                                                                                    cameraMatrix,
                                                                                                    distCoeffs)
                rvecss = rotation_vectors[0][0]
                rot_mat, _jacob = cv2.Rodrigues(rvecss)
                print(rot_mat)

                # id의 개수에 따라 인식되는 rvecs의 개수가 다름. 따라서 어떤 마커가 인식되냐에 따라 벡터의 종류가 달라질수도,
                # 개수가 달라질 수도 있음.
                # print(translation_vectors)
                if len(ids) == 2:
                    x_sum_ = (corners[0][0][0][0] + corners[0][0][1][0] + corners[0][0][2][0] + corners[0][0][3][
                        0]) * 0.25 + (corners[1][0][0][0] + corners[1][0][1][0] + corners[1][0][2][0] +
                                      corners[1][0][3][
                                          0]) * 0.25
                    y_sum_ = (corners[0][0][0][1] + corners[0][0][1][1] + corners[0][0][2][1] + corners[0][0][3][
                        1]) * 0.25 + (corners[1][0][0][1] + corners[1][0][1][1] + corners[1][0][2][1] +
                                      corners[1][0][3][
                                          1]) * 0.25
                    x_sum_ = x_sum_ / 2
                    y_sum_ = y_sum_ / 2

                    rot_1 = [rotation_vectors[0][0][0], rotation_vectors[0][0][1], rotation_vectors[0][0][2]]
                    rot_1_x = rotation_vectors[0][0][0] * radian
                    rot_1_y = rotation_vectors[0][0][1]
                    rot_1_z = rotation_vectors[0][0][2]

                    rot_2 = [rotation_vectors[1][0][0], rotation_vectors[1][0][1], rotation_vectors[1][0][2]]
                    rot_2_x = rotation_vectors[1][0][0]
                    rot_2_y = rotation_vectors[1][0][1]
                    rot_2_z = rotation_vectors[1][0][2]

                    trs_1 = [translation_vectors[0][0][0], translation_vectors[0][0][1], translation_vectors[0][0][2]]
                    trs_1_x = translation_vectors[0][0][0]
                    trs_1_y = translation_vectors[0][0][1]
                    trs_1_z = translation_vectors[0][0][2]

                    trs_2 = [translation_vectors[1][0][0], translation_vectors[1][0][1], translation_vectors[1][0][2]]
                    trs_2_x = translation_vectors[1][0][0]
                    trs_2_y = translation_vectors[1][0][1]
                    trs_2_z = translation_vectors[1][0][2]
                    trs_1_xx = Cx1
                    trs_1_yy = Cy1
                    trs_1_zz = Cz1
                    trs_2_xx = Cx2
                    trs_2_yy = Cy2
                    trs_2_zz = Cz2
                    # trs_depth = depth_frame_1.get_distance(trs_1[0], trs_1[1])
                    # trs = (convert_depth_to_phys_coord(trs_1[0], trs_1[1], trs_depth, C_Intr))
                    # print(trs)
                    # 정제되지 않은 corner의 평균값 : trs가 corner로부터 나왔으니까 확인해 보기 위해서.
                    # x_mean = (corners[int(i)][0][0][0] + corners[int(i)][0][1][0] + corners[int(i)][0][2][0] + corners[int(i)][0][3][0])*0.25
                    # print((topRight[2] + topLeft[2] + bottomLeft[2] + bottomRight[2]) / 4)
                    # print(x_mean)
                    #
                    # print(trs_1[2])
                    # print(trs_2[2])

                    # rot_x_1 = rotation_vectors[0][0][0]
                    # rot_y_1 = rotation_vectors[0][0][1]
                    # rot_z_1 = rotation_vectors[0][0][2]
                    # rot_x_2 = rotation_vectors[1][0][0]
                    # rot_y_2 = rotation_vectors[1][0][1]
                    # rot_z_2 = rotation_vectors[1][0][2]


                elif len(ids) == 1 and ids[0][0] == 1:
                    rot_2_x = 0
                    rot_2_y = 0
                    rot_2_z = 0
                    trs_2_x = 0
                    trs_2_y = 0
                    trs_2_z = 0
                    rot_1 = [rotation_vectors[0][0][0], rotation_vectors[0][0][1], rotation_vectors[0][0][2]]
                    rot_1_x = rotation_vectors[0][0][0]
                    rot_1_y = rotation_vectors[0][0][1]
                    rot_1_z = rotation_vectors[0][0][2]
                    trs_1 = [translation_vectors[0][0][0], translation_vectors[0][0][1], translation_vectors[0][0][2]]
                    trs_1_x = translation_vectors[0][0][0]
                    trs_1_y = translation_vectors[0][0][1]
                    trs_1_z = translation_vectors[0][0][2]
                    trs_1_xx = Cx1
                    trs_1_yy = Cy1
                    trs_1_zz = Cz1
                    # rot_x_1 = rotation_vectors[0][0][0]
                    # rot_y_1 = rotation_vectors[0][0][1]
                    # rot_z_1 = rotation_vectors[0][0][2]


                elif len(ids) == 1 and ids[0][0] == 2:
                    rot_1_x = 0
                    rot_1_y = 0
                    rot_1_z = 0
                    trs_1_x = 0
                    trs_1_y = 0
                    trs_1_z = 0
                    rot_2 = [rotation_vectors[0][0][0], rotation_vectors[0][0][1], rotation_vectors[0][0][2]]
                    rot_2_x = rotation_vectors[0][0][0]
                    rot_2_y = rotation_vectors[0][0][1]
                    rot_2_z = rotation_vectors[0][0][2]
                    trs_2 = [translation_vectors[0][0][0], translation_vectors[0][0][1], translation_vectors[0][0][2]]
                    trs_2_x = translation_vectors[0][0][0]
                    trs_2_y = translation_vectors[0][0][1]
                    trs_2_z = translation_vectors[0][0][2]
                    trs_2_xx = Cx2
                    trs_2_yy = Cy2
                    trs_2_zz = Cz2

                # tmp = (topRight+topLeft+bottomRight+bottomLeft)/4

                # marker_list.append([c_time, markerID, topRight, topLeft, bottomRight, bottomLeft, rot_1, rot_2, trs_1, trs_2, center])
                # marker_list.append(q
                # [c_time, markerID, topRight, topLeft, bottomRight, bottomLeft])
                marker_list.append(
                    [c_time, markerID, rot_1_x, rot_1_y, rot_1_z, rot_2_x, rot_2_y, rot_2_z, trs_1_xx, trs_1_yy, trs_1_zz,
                     trs_2_xx, trs_2_yy, trs_2_zz])
            marker = np.array(marker_list, dtype=object)

            # 파이썬 zip 내장함수: 리스트 등의 요소 묶어줌.
            for rvec, tvec in zip(rotation_vectors, translation_vectors):
                color_image_1 = cv2.drawFrameAxes(color_image_1, cameraMatrix, distCoeffs, rvec, tvec, 1)
                # color_image_1 = cv2.drawAxis(color_image_1, cameraMatrix, distCoeffs, rvec, tvec, 1)
                # aruco.drawAxis(color_image_1, cameraMatrix, distCoeffs, rvec, tvec, 1)

        # Show images from both cameras
        cv2.namedWindow('RealSense', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('RealSense', 640, 480)
        cv2.imshow('RealSense', color_image_1)
        cv2.waitKey(1)

        # Save images and depth maps from both cameras by pressing 's'
        # 키보드를 누른 후 25ms동안 대기. ch는 사용자가 누른 키의 아스키코드 값을 리턴함.
        ch = cv2.waitKey(25)
        # s의 아스키코드가 115
        if ch == 115:
            suffix = datetime.now().strftime('%y%m%d_%H%M%S')
            fileName = suffix + '.csv'
            with open("marker" + fileName, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(marker)
                marker_list.clear()
                print("Save!!")
            cv2.imwrite("my_image_1.jpg", color_image_1)
            print("Save")
        # esc를 누르거나 q를 누르면 종료시키기. (esc의 아스키 코드가 27)
        elif ch & 0xFF == ord('q') or ch == 27:
            cv2.destroyAllWindows()
            break

finally:
    # Stop streaming
    pipeline_1.stop()
