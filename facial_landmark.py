import cv2
import numpy
import numpy as np
import mediapipe as mp      #fash mash 관련
import pyrealsense2 as rs
import math
import pandas as pd
import os
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import MinMaxScaler

from datetime import datetime
from RealSense_Utilities.realsense_api.realsense_api import RealSenseCamera
from RealSense_Utilities.realsense_api.realsense_api import find_realsense
from RealSense_Utilities.realsense_api.realsense_api import frame_to_np_array
from RealSense_Utilities.realsense_api.realsense_api import mediapipe_detection

# Fixing random state for reproducibility
np.random.seed(19680801)

#face mash 관련
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh_0 = mp.solutions.face_mesh             #이름에 왜 0을 붙였는지는 모르겠으나 fash mesh에 사용되는 변수임.

frame_height, frame_width, channels = (480, 640, 3) #정규화 되어있는 x, y를 위한 값
frame_depth = -4                                   #정규화 되어있는 z를 위한 값


def z_score_normalize(lst):
    normalized = []
    for value in lst:
        normalized_num = (value - np.mean(lst)) / np.std(lst)
        normalized.append(normalized_num)
    return normalized


def min_max_normalize(lst):
    normalized = []

    for value in lst:
        normalized_num = (value - np.mean(lst)) / (max(lst) - min(lst))
        normalized.append(normalized_num)

    return normalized



#one euro filter 관련
def smoothing_factor(t_e, cutoff):
    r = 2 * math.pi * cutoff * t_e
    return r / (r + 1)

#one euro filter 관련
def exponential_smoothing(a, x: np.ndarray, x_prev: np.ndarray) -> np.ndarray:
    return a * x + (1 - a) * x_prev


#3D포인터에서 depth를 phys로 변환하는 함수: 2d로 존재하는 픽셀을 3d로 변환하기 위한 함수
#realsense 카메라 사용하기 위해 존재함.
def convert_depth_to_phys_coord(xp, yp, depth, intr):
    result = rs.rs2_deproject_pixel_to_point(intr, [int(xp), int(yp)], depth)
    return result[0], result[1], result[2]


#inverse intrinsic matrix
def inverse_color(points_set, inverse_intrinsic_matrix, z2):
    #inverse_intrinsic_matrix = np.linalg.inv(np.matrix(inverse_intrinsic_matrix))           #linalg.인수(행렬) : 행렬의 역행렬을 계산.
    # points_set = np.hstack((points_set, np.ones((points_set.shape[0], 1)))).T
    # return np.delete(PI.dot(points_set), 3, axis=0)
    points_set = np.transpose(points_set)
    result = np.dot(z2, inverse_intrinsic_matrix)
    result = np.dot(result, points_set)
    result = np.dot(z2, result)
    li = result.tolist()
    return li[0], li[1], li[2]
    #return li[0][0], li[0][1], li[0][2]


#이미지 줌하는 함수: 실제 사용 x
def zoom(img: np.ndarray, scale, center=None):
    height, width = img.shape[:2]
    rate = height / width

    if center is None:
        center_x = int(width / 2)
        center_y = int(height / 2)
        radius_x, radius_y = int(width / 2), int(height / 2)
    else:
        center_x, center_y = center

    if center_x < width * (1 - rate):
        center_x = width * (1 - rate)
    elif center_x > width * rate:
        center_x = width * rate

    if center_y < height * (1 - rate):
        center_y = height * (1 - rate)
    elif center_y > height * rate:
        center_y = height * rate

    center_x, center_y = int(center_x), int(center_y)
    left_x, right_x = center_x, int(width - center_x)
    up_y, down_y = int(height - center_y), center_y
    radius_x = min(left_x, right_x)
    radius_y = min(up_y, down_y)

    # Actual zoom code
    radius_x, radius_y = int(scale * radius_x), int(scale * radius_y)

    # size calculation
    min_x, max_x = center_x - radius_x, center_x + radius_x
    min_y, max_y = center_y - radius_y, center_y + radius_y

    # Crop image to size
    cropped = img[min_y:max_y, min_x:max_x]
    # Return to original size
    # if scale >= 0:
    #     new_cropped = cv2.resize(cropped, (width, height), interpolation=cv2.INTER_CUBIC)
    # else:
    #     new_cropped = cv2.resize(cropped, (width, height), interpolation=cv2.INTER_CUBIC)

    new_cropped = cv2.resize(cropped, (width, height), interpolation=cv2.INTER_CUBIC)

    return new_cropped





#one euro filter 관련
class OneEuroFilter:
    def __init__(self, t0, x0: np.ndarray, min_cutoff=1.0, beta=0.0,
                 d_cutoff=1.0):
        """Initialize the one euro filter."""
        # The parameters.
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        # Previous values. :: astype: 타입 변경. ndarray타입의 x0을 float 타입으로 변경.
        self.x_prev = x0.astype('float')

        fill_array = np.zeros(x0.shape)
        self.dx_prev = fill_array.astype('float')
        self.t_prev = float(t0)

    def __call__(self, t, x: np.ndarray) -> np.ndarray:
        """Compute the filtered signal."""
        t_e = t - self.t_prev

        # The filtered derivative of the signal.
        a_d = smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = exponential_smoothing(a_d, dx, self.dx_prev)

        # The filtered signal.
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = smoothing_factor(t_e, cutoff)
        x_hat = exponential_smoothing(a, x, self.x_prev)

        # Memorize the previous values.
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t

        return x_hat



def main():
    rs_main = None

    previous_timestamp = 0
    points_3d = None
    points_pixel = None
    points_3d3 = None
    points_pixel3 = None
    points_3d_fin = None
    points_pixel_fin = None
    angle_list = []
    min_cutoff = 0.00001
    beta = 0.0
    first_iter = True

    zoom_scale = 1

    jitter_count = 0
    landmark_iterable = [4, 6, 9, 200]

    cameras = {}
    realsense_device = find_realsense()

    #realsense camera는 import했기 때문에 그냥 쓰면 됨.
    #realsense 카메라의 시리얼 번호에 부합하면
    #if문과 else문의 차이는 인자 넘겨주는 순서가 뒤바뀜. 그리고 디바이스 타입이 다름.
    for serial, devices in realsense_device:
        if serial == '043422251095':
            cameras[serial] = RealSenseCamera(depth_stream_width=640, depth_stream_height=480,
                                              color_stream_width=640, color_stream_height=480,
                                              color_stream_fps=30, depth_stream_fps=90,
                                              device=devices, adv_mode_flag=True, device_type="d455")
        else:
            cameras[serial] = RealSenseCamera(depth_stream_width=640, depth_stream_height=480,
                                              color_stream_width=640, color_stream_height=480,
                                              color_stream_fps=30, depth_stream_fps=90,
                                              device=devices, device_type="d415", adv_mode_flag=True)

    for ser, dev in cameras.items():
        rs_main = dev

    if rs_main is None:
        print("can't initialize realsense cameras")

    s_time = int(round(time.time() * 1000))
    # s_time = round(datetime.now().timestamp(), 3)

    # For static images:
    # face mash 관련: 여기서부터 끝까지 모두 mediapipe
    #with문: 자원을 획득하고 사용 후 반납해야 하는 경우 주로 사용
    with mp_face_mesh_0.FaceMesh(
            # 디폴트 false: 이미지를 비디오 스트림으로 처리하기 위함.
            # true: 얼굴 인식이 모든 입력 이미지에서 실행됨. 정적 이미지 배치를 처리하는 데에 이상적.
            static_image_mode=False,
            # 얼굴 탐지가 성공적인 것이라고 감지되는 신뢰값. 디폴트 0.5
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as face_mesh_0:

        #카메라를 통해 입력된 영상을 받고. 현재 시간으로 time stamp를 설정함.
        #realsense 카메라 사용하기 위해 존재함.
        try:
            while True:
                pixelx = []
                pixely = []
                pixelz = []
                depthz = []
                zz = []
                xx=[]
                yy=[]
                zzz = []
                xxx=[]
                yyy=[]
                diff = []
                fin=[]

                points_3d_iter = np.zeros((0, 3))
                points_pixel_iter = np.zeros((0, 3))

                # points_3d_iter2 = np.zeros((0, 3))
                # points_pixel_iter2 = np.zeros((0, 3))

                points_3d_iter3 = np.zeros((0, 3))
                points_pixel_iter3 = np.zeros((0, 3))



                points_3d_iter_fin = np.zeros((0, 3))
                points_pixel_iter_fin = np.zeros((0, 3))

                rs_main.get_data()

                current_timestamp = datetime.now().timestamp()
                c_time = int(round(time.time() * 1000)) - s_time
                #print(c_time)

                ####frameset관련: 다 알고 이해할 필요 없음. 그냥 realsense 카메라
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


#output
                img_raw = np.copy(img_rs0)
                img_h, img_w, img_c = img_raw.shape

                face_3d = []
                face_2d = []

                #_,ㅡ 사용 이유: 반환된 값 중 하나의 값만 필요
                #mediapipe로부터 detect한 값 중 face landmark값을 변수에 넣는다.
                _, results = mediapipe_detection(img_rs0, face_mesh_0)
                multi_face_landmarks = results.multi_face_landmarks









#facial landmark 추출해서 3D로 변환하기.
################################################  mediapipe: face mesh 관련  #######################
                #인식된 얼굴 중 가장 첫번째 얼굴을 선택하고 랜드마크를 추출한다.
                try:
                    # pixelx = []
                    # pixely = []
                    # pixelz = []
                    # depthz = []
                    # z1 = []
                    # z2_zzz = []

                    if multi_face_landmarks:
                        face_landmarks = results.multi_face_landmarks[0]

                        for i in range(468):

                            pixel_point = face_landmarks.landmark[i]
                            #얼굴 주요 랜드마크 값으로 회전각 계산
                            if i == 33 or i == 263 or i == 1 or i == 61 or i == 291 or i == 199:                #얼굴 주요 랜드마크 값
                                if i == 1:
                                    nose_2d = (pixel_point.x * img_w, pixel_point.y * img_h)
                                    nose_3d = (pixel_point.x * img_w, pixel_point.y * img_h, pixel_point.z * 3000)

                                x, y = int(pixel_point.x * img_w), int(pixel_point.y * img_h)

                                # Get the 2D Coordinates
                                face_2d.append([x, y])

                                # Get the 3D Coordinates
                                face_3d.append([x, y, pixel_point.z])


                            #1) pixel point
                            pixel_x = int(pixel_point.x * frame_width)
                            pixelx.append(pixel_x)
                            pixel_y = int(pixel_point.y * frame_height)
                            pixely.append(pixel_y)
                            pixel_z = float(pixel_point.z)                           #정규화 되어있는 z값
                            pixelz.append(pixel_z)
                            #에서 얻는 pixel 에 pixel_Z = float(pixel_point.z)*220 로 가상의 3D z 좌표 만들어주기
                            #pixel_Z2 = float(pixel_point.z * frame_depth)           #실제 값을 구하기 위해 220을 곱해준 z값




                            #2)  _ = cv2.circle(img_rs0, (pixel_x, pixel_y), 2, (0, 0, 0), -1)
                            temporal_pixel_point = np.array([pixel_x, pixel_y, pixel_z])
                            points_pixel_iter = np.append(points_pixel_iter, temporal_pixel_point[np.newaxis, :], axis=0)

                            # temporal_pixel_point2 = np.array([pixel_x, pixel_y, pixel_Z2])
                            # points_pixel_iter2 = np.append(points_pixel_iter2, temporal_pixel_point2[np.newaxis, :], axis=0)

                            temporal_pixel_point3 = np.array([pixel_x, pixel_y, pixel_z])
                            points_pixel_iter3 = np.append(points_pixel_iter3, temporal_pixel_point[np.newaxis, :], axis=0)

                            temporal_pixel_point_fin = np.array([pixel_x, pixel_y, pixel_z])
                            points_pixel_iter_fin = np.append(points_pixel_iter_fin, temporal_pixel_point[np.newaxis, :], axis=0)


                            #3d point : 픽셀 값
                            depth = rs_main.depth_frame.get_distance(pixel_x, pixel_y)
                            if depth == 0:          #카메라에 문제가 있거나 너무 멀리 있으면 에러 발생.
                                raise ValueError
                            if depth != 0:
                                depthz.append(depth)
                            #depthz.append(depth)

                            #realsense camera를 통해 얻은 depth값을 z값으로 사용한다.
                            x, y, z = convert_depth_to_phys_coord(
                                pixel_x,
                                pixel_y,
                                depth,
                                rs_main.color_intrinsics)
                            xx.append(x)
                            yy.append(y)
                            zz.append(z)


                            #z(depth)
                            temporal_3d_point = np.array([x, y, z])
                            points_3d_iter = np.append(points_3d_iter, temporal_3d_point[np.newaxis, :], axis=0)


                            #보정된 z값을 3D로 변환한다. (intrinsic)
                            #원래 코드는 의미없는 z 에 depth (센서로 얻은 깊이)를 넣어 3D 좌표로 color_intrinsics을 사용해서 처리해주는데,
                            # 아까 220을 곱한 pixel z 를 사용한 xyz 도 color_intrinsics을 사용해서 3D 좌표로 만들고 싶음.
                            # FL_points = np.array([pixel_x, pixel_y, 1])
                            # #print(FL_points)
                            # #intrinsic_matrix = np.array([[rs_main.color_intrinsics.fx, 0, rs_main.color_intrinsics.ppx],[0, rs_main.color_intrinsics.fy, rs_main.color_intrinsics.ppx],[0,0,1]])
                            # inv_intrinsic_matrix = np.array([[1/rs_main.color_intrinsics.fx, 0, 0],[0, 1/rs_main.color_intrinsics.fy, 0],[0, 0, 1]])
                            # x2, y2, z2 = inverse_color(FL_points, inv_intrinsic_matrix, pixel_Z2)
                            # #print(x2, y2, z2)
                            # #
                            # # #3) 그래서 이 z 값을 둘다 저장 필요!
                            # # # 필요 데이터==> 시간 [x, y, z(depth), z(220)] * 468개 점
                            # #
                            # # #z(220)
                            # temporal_3d_point2 = np.array([x2, y2, z2])
                            # points_3d_iter2 = np.append(points_3d_iter2, temporal_3d_point2[np.newaxis, :], axis=0)

                        #여기서 리스트 애들 확인: 매 초마다 얼굴의 468개 점에 대한 정규화 여부 확인 및
                        #print(pixelz)

                        #pixelz: 평균과 표준편차로 정규화 여부 확인 : 정규화 되어있지 않음.
                        pixelz_mean = np.mean(pixelz)
                        print("z_mean: ")
                        print(pixelz_mean)
                        #pixelz_std=np.std(pixelz)
                        # print("z_std: ")
                        # print(pixelz_std)
                        #pixelz_normalize = z_score_normalize(pixelz)
                        # pixelz_normalize = (pixelz - pixelz.min(axis=0)) / (pixelz.max(axis=0) - pixelz.min(axis=0))
                        # print(pixelz_normalize)
                        # pixelz_normalize = MinMaxScaler().fit_transform(pixelz)
                        # print(pixelz_normalize)
                        pixelz_normalize = z_score_normalize(pixelz)
                        #print("pixelz normalize: ")
                        #print(pixelz_normalize)

                        #print("depth: ")
                        #print(depthz)
                        depthz_mean = np.mean(depthz)
                        #print("depth_mean: ")
                        #print(pixelz_mean)
                        for i in range(468):
                            if depthz[i] > 0.2 and depthz[i] < 0.6 :
                                depthz_std=np.std(depthz)
                        #print("depth_std: ")
                        #print(depthz_std)
                        #depth_normalize = z_score_normalize(depthz)
                        print("pixel z: ")
                        print(len(pixelz))

                        #print(depth_normalize)
                        z2_spot = depthz_std*np.array(pixelz_normalize) + np.array(depthz_mean)
                        #z2_spot = (max(pixelz) - min(pixelz)) * 0.8 * np.array(pixelz_normalize) + np.mean(pixelz)
                        #print(z2_spot)

                        print("convert depth to phys coord: z: ")
                        print(zz)

                        for i in range(468):
                            #468개의 점 convert: depth, z2 spot의 값들. 실제 3d에서 얼굴 어떻게 보이는지 비교.
                            x1, y1, z2_s = convert_depth_to_phys_coord(
                                pixelx[i],
                                pixely[i],
                                z2_spot[i],
                                rs_main.color_intrinsics)
                            zzz.append(z2_s)
                            xxx.append(x1)
                            yyy.append(y1)
                            temporal_3d_point3 = np.array([x1, y1, z2_s])
                            points_3d_iter3 = np.append(points_3d_iter3, temporal_3d_point3[np.newaxis, :], axis=0)


                        print("convert depth to phys coord: z2_spot: ")
                        print(zzz)
                        print("depth size except depth == 0 : ")
                        print(len(depthz))


                        # depth, z2 취사 선택하는 코드
                        for i in range(468):
                            diff.append(zzz[i] - zz[i])
                            # diff가 일정 값 이상이면 z2채택, 그렇지 않으면 depth 채택
                            if (zz[i]>0.45 and diff[i]>0.2) or zz[i]> 0.5:
                                fin.append(zzz[i])
                                temporal_3d_point_fin = np.array([xxx[i], yyy[i], zzz[i]])
                                points_3d_iter_fin = np.append(points_3d_iter_fin, temporal_3d_point_fin[np.newaxis, :], axis=0)
                            else:
                                fin.append(zz[i])
                                temporal_3d_point_fin = np.array([xx[i], yy[i], zz[i]])
                                points_3d_iter_fin = np.append(points_3d_iter_fin, temporal_3d_point_fin[np.newaxis, :], axis=0)

                        print("difference between modified z and depth")
                        print(diff)







                        #이 이후부터 계속 VALUE ERROR가 발생. 밑의 3D ITER를 주석 처리 하고 나니까 VALUE ERROR가 발생하지 않음.

                        #3 * 468 = 1404. 총 468개의 점이라고 했고 3개가 있으니까 파일에 1404개를 저장. (x, y, z -> 3개)
                        points_pixel_iter = points_pixel_iter.reshape(1, 1404)
                        points_3d_iter = points_3d_iter.reshape(1, 1404)

                        # points_pixel_iter2 = points_pixel_iter2.reshape(1, 1404)
                        # points_3d_iter2 = points_3d_iter2.reshape(1, 1404)

                        #value error가 발생하는 부분
                        points_pixel_iter3 = points_pixel_iter3.reshape(1, 1404)
                        points_3d_iter3 = points_3d_iter3.reshape(1, 1404)

                        points_pixel_iter_fin = points_pixel_iter_fin.reshape(1, 1404)
                        points_3d_iter_fin = points_3d_iter_fin.reshape(1, 1404)


                        points_pixel_iter = np.concatenate((np.array([c_time])[:, np.newaxis], points_pixel_iter), axis=1)
                        points_3d_iter = np.concatenate((np.array([c_time])[:, np.newaxis], points_3d_iter), axis=1)

                        # points_pixel_iter2 = np.concatenate((np.array([c_time])[:, np.newaxis], points_pixel_iter2), axis=1)
                        # points_3d_iter2 = np.concatenate((np.array([c_time])[:, np.newaxis], points_3d_iter2), axis=1)

                        points_pixel_iter3 = np.concatenate((np.array([c_time])[:, np.newaxis], points_pixel_iter3), axis=1)
                        points_3d_iter3 = np.concatenate((np.array([c_time])[:, np.newaxis], points_3d_iter3), axis=1)

                        points_pixel_iter_fin = np.concatenate((np.array([c_time])[:, np.newaxis], points_pixel_iter_fin), axis=1)
                        points_3d_iter_fin = np.concatenate((np.array([c_time])[:, np.newaxis], points_3d_iter_fin), axis=1)


                        if first_iter:
                            points_3d = points_3d_iter
                            points_pixel = points_pixel_iter

                            # points_3d2 = points_3d_iter2
                            # points_pixel2 = points_pixel_iter2
                            #
                            points_3d3 = points_3d_iter3
                            points_pixel3 = points_pixel_iter3

                            points_3d_fin = points_3d_iter_fin
                            points_pixel_fin = points_pixel_iter_fin

                            first_iter = False



                        else:
                            points_3d = np.concatenate((points_3d, points_3d_iter), axis=0)
                            points_pixel = np.concatenate((points_pixel, points_pixel_iter), axis=0)


                            # points_3d2 = np.concatenate((points_3d2, points_3d_iter2), axis=0)
                            # points_pixel2 = np.concatenate((points_pixel2, points_pixel_iter2), axis=0)

                            points_3d3 = np.concatenate((points_3d3, points_3d_iter3), axis=0)
                            points_pixel3 = np.concatenate((points_pixel3, points_pixel_iter3), axis=0)

                            points_3d_fin = np.concatenate((points_3d_fin, points_3d_iter_fin), axis=0)
                            points_pixel_fin = np.concatenate((points_pixel_fin, points_pixel_iter_fin), axis=0)



                        ###여기서부터 z2의 값 제대로 인식하게 수정하기. : face_3d의 input인 날 것의 x, y, z를 사용하기 때문에
                        #z에 대한 처리를 해 주어야 할 것 같음.
                        face_2d = np.array(face_2d, dtype=np.float64)

                        # Convert it to the NumPy array
                        face_3d = np.array(face_3d, dtype=np.float64)

                        # The camera matrix
                        focal_length = 1 * img_w

                        cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                               [0, focal_length, img_w / 2],
                                               [0, 0, 1]])

                        # The distortion parameters
                        dist_matrix = np.zeros((4, 1), dtype=np.float64)

                        # Solve PnP
                        success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

                        # Get rotational matrix 로드리게스(회전변환행렬)
                        rmat, jac = cv2.Rodrigues(rot_vec)

                        # Get angles
                        angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                        # Get the y rotation degrees -> 이거 y축 회전 아닌 것 같은데
                        x = angles[0] * 360
                        y = angles[1] * 360
                        z = angles[2] * 360

                        #z2 = angles[2] * 360


                        angle_list.append(list([x, y, z]))
#####################################################



                #파이썬 오류 처리 except
                except ValueError:
                    print('value error')
                    continue

                except RuntimeError:
                    print('runtime error')
                    continue

                finally:
                    rs_main.depth_frame.keep()

                elapsed = current_timestamp - previous_timestamp

                # print('FPS:{} / z:{}\r'.format(1 / elapsed, points_3d_iter_hat[0, 0, 2]), end='')
             ##   print('FPS:{}\r'.format(1 / elapsed), end='')

                previous_timestamp = current_timestamp

                angle_array = np.array(angle_list)

                resized_image = cv2.resize(img_rs0, dsize=(0, 0), fx=1, fy=1, interpolation=cv2.INTER_AREA)


                #cv2에서 이미지 출력하기
                cv2.namedWindow('RealSense_front', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('RealSense_front', resized_image.shape[1], resized_image.shape[0])
                cv2.imshow('RealSense_front', resized_image)


                #키 입력 대기
                key = cv2.waitKey(1)


                #모든 이미지 창 닫음
                #ord: 문자를 정수로 받고 유니코드로 반환
                if key & 0xFF == ord('q') or key == 27:
                    cv2.destroyAllWindows()
                    break



                #s누르면 사진 저장됨.
                if key & 0xFF == ord('s'):
                    # for jitter test
                    present_time = datetime.now()
                    if len(str(present_time.month)) == 1:
                        month = '0' + str(present_time.month)
                    else:
                        month = str(present_time.month)

                    if len(str(present_time.day)) == 1:
                        day = '0' + str(present_time.day)
                    else:
                        day = str(present_time.day)

                    if len(str(present_time.hour)) == 1:
                        hour = '0' + str(present_time.hour)
                    else:
                        hour = str(present_time.hour)

                    if len(str(present_time.minute)) == 1:
                        minute = '0' + str(present_time.minute)
                    else:
                        minute = str(present_time.minute)

                    global k
                    k = month + day + hour + minute

                    os.mkdir("./pose_test/{}/".format(month + day + hour + minute))


                    print(points_3d[10:11, 1:1405:3])
                    X = points_3d[10:11, 1:1405:3]
                    Y = points_3d[10:11, 2:1405:3]
                    Z = points_3d[10:11, 3:1405:3]

                    X2 = points_3d3[10:11, 1:1405:3]
                    Y2 = points_3d3[10:11, 2:1405:3]
                    Z2 = points_3d3[10:11, 3:1405:3]

                    X3 = points_3d_fin[10:11, 1:1405:3]
                    Y3 = points_3d_fin[10:11, 2:1405:3]
                    Z3 = points_3d_fin[10:11, 3:1405:3]

                    # fig = plt.figure(figsize=(9, 6))
                    # ax = fig.add_subplot(111)
                    # ax.scatter(X, Z, color='r')
                    # ax.scatter(X2,Z2, color='g')
                    # ax.scatter(X3, Z3, color = 'b')
                    # plt.show()

                    fig = plt.figure(figsize=(9, 6))
                    ax = fig.add_subplot(111, projection='3d')
                    ax.scatter(X, Y, Z, color='r')
                    ax.scatter(X2, Y2, Z2, color='g')
                    ax.scatter(X3,Y3, Z3, color = 'b')
                    plt.show()

                    # ax.scatter(X3,Y3, Z3, color = 'b')
                    # plt.show()





                    pd.DataFrame(points_3d).to_csv(
                        "./pose_test/{}/points_3d.csv".format(month + day + hour + minute))
                    pd.DataFrame(points_pixel).to_csv(
                        "./pose_test/{}/points_pixel.csv".format(month + day + hour + minute)
                    )

                    # pd.DataFrame(points_3d2).to_csv(
                    #     "./pose_test/{}/points_3d2.csv".format(month + day + hour + minute))
                    # pd.DataFrame(points_pixel2).to_csv(
                    #     "./pose_test/{}/points_pixel2.csv".format(month + day + hour + minute)
                    # )
                    pd.DataFrame(points_3d3).to_csv(
                        "./pose_test/{}/points_3d3.csv".format(month + day + hour + minute))
                    pd.DataFrame(points_pixel3).to_csv(
                        "./pose_test/{}/points_pixel3.csv".format(month + day + hour + minute)
                    )
                    pd.DataFrame(diff).to_csv(
                        "./pose_test/{}/diff.csv".format(month + day + hour + minute)
                    )

                    pd.DataFrame(points_3d_fin).to_csv(
                        "./pose_test/{}/points_3d3.csv".format(month + day + hour + minute))
                    pd.DataFrame(points_pixel_fin).to_csv(
                        "./pose_test/{}/points_pixel3.csv".format(month + day + hour + minute)
                    )

                    pd.DataFrame(angle_array).to_csv(
                        "./pose_test/{}/rot_angle.csv".format(month + day + hour + minute)
                    )
                    print("test {} complete and data saved".format(jitter_count))

                    jitter_count += 1
                    first_iter = True


        finally:
            rs_main.stop()

if __name__ == '__main__':
    main()


#data1 = pd.read_csv('./pose_test/03291425/points_3d2.csv')
data1 = pd.read_csv('./pose_test/{}/points_3d.csv'.format(k))
data2 = pd.read_csv('./pose_test/{}/points_3d3.csv'.format(k))
#data2 = pd.read_csv('./pose_test/{}/points_3d2.csv'.format(k))
#data3 = pd.read_csv('./pose_test/{}/points_pixel.csv'.format(k))
#data4 = pd.read_csv('./pose_test/{}/points_pixel2.csv'.format(k))

#열을 기준으로 slicing
X = data1.iloc[10:11, 1:1405:3].values
Y = data1.iloc[10:11, 2:1405:3].values
Z = data1.iloc[10:11, 3:1405:3].values

X2 = data2.iloc[10:11, 1:1405:3].values
Y2 = data2.iloc[10:11, 2:1405:3].values
Z2 = data2.iloc[10:11, 3:1405:3].values
print(Z)
Z = numpy.array(Z)
print(Z)
#a = (X<10 and Y<10 and Z<10 and X2<10 and Y2<10 and Z2<10).all()

# fig = plt.figure(figsize=(9, 6))
# ax = fig.add_subplot(111, projection='3d')



# 시도 1: X, Y, Z 리스트의 모든 값들이 10을 넘지 않는 경우 실행될 코드: 실행 잘 됨. 단 any라서 모든 값들에 대한 검사를 해 주진 않음.
# if np.any(np.array([i < 10 for lst in [X, Y, Z, X2, Y2, Z2] for i in lst])):
#     ax.scatter(X, Y, Z, color='r')
#     ax.scatter(X2, Y2, Z2, color='g')
#     plt.show()


#시도 2:


#원래 코드
# ax.scatter(X, Y, Z, color='r')
# ax.scatter(X2, Y2, Z2, color='g')
# plt.show()
