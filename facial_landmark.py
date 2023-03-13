import cv2
import numpy as np
import mediapipe as mp      #fash mash 관련
import pyrealsense2 as rs
import math
import pandas as pd
import os
import time

from datetime import datetime
from RealSense_Utilities.realsense_api.realsense_api import RealSenseCamera
from RealSense_Utilities.realsense_api.realsense_api import find_realsense
from RealSense_Utilities.realsense_api.realsense_api import frame_to_np_array
from RealSense_Utilities.realsense_api.realsense_api import mediapipe_detection


#face mash 관련
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh_0 = mp.solutions.face_mesh             #이름에 왜 0을 붙였는지는 모르겠으나 fash mesh에 사용되는 변수임.

frame_height, frame_width, channels = (480, 640, 3)


#one euro filter 관련
def smoothing_factor(t_e, cutoff):
    r = 2 * math.pi * cutoff * t_e
    return r / (r + 1)

#one euro filter 관련
def exponential_smoothing(a, x: np.ndarray, x_prev: np.ndarray) -> np.ndarray:
    return a * x + (1 - a) * x_prev


#3D포인터에서 depth를 phys로 변환하는 함수
#realsense 카메라 사용하기 위해 존재함.
def convert_depth_to_phys_coord(xp, yp, depth, intr):
    result = rs.rs2_deproject_pixel_to_point(intr, [int(xp), int(yp)], depth)
    return result[0], result[1], result[2]


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
                points_3d_iter = np.zeros((0, 3))
                points_pixel_iter = np.zeros((0, 3))
                rs_main.get_data()

                current_timestamp = datetime.now().timestamp()
                c_time = int(round(time.time() * 1000)) - s_time
                print(c_time)

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


################################################  mediapipe: face mesh 관련  #######################
                try:
                    if multi_face_landmarks:
                        face_landmarks = results.multi_face_landmarks[0]

                        for i in range(468):
                            pixel_point = face_landmarks.landmark[i]
                            #얼굴 주요 랜드마크 값으로 회전각 계산
                            if i == 33 or i == 263 or i == 1 or i == 61 or i == 291 or i == 199:
                                if i == 1:
                                    nose_2d = (pixel_point.x * img_w, pixel_point.y * img_h)
                                    nose_3d = (pixel_point.x * img_w, pixel_point.y * img_h, pixel_point.z * 3000)

                                x, y = int(pixel_point.x * img_w), int(pixel_point.y * img_h)

                                # Get the 2D Coordinates
                                face_2d.append([x, y])

                                # Get the 3D Coordinates
                                face_3d.append([x, y, pixel_point.z])

                            #pixel point
                            pixel_x = int(pixel_point.x * frame_width)
                            pixel_y = int(pixel_point.y * frame_height)
                            pixel_Z = float(pixel_point.z)

                            # _ = cv2.circle(img_rs0, (pixel_x, pixel_y), 2, (0, 0, 0), -1)
                            temporal_pixel_point = np.array([pixel_x, pixel_y, pixel_Z])
                            points_pixel_iter = np.append(points_pixel_iter, temporal_pixel_point[np.newaxis, :], axis=0)

                            #3d point
                            depth = rs_main.depth_frame.get_distance(pixel_x, pixel_y)
                            if depth == 0:
                                raise ValueError
                            x, y, z = convert_depth_to_phys_coord(
                                pixel_x,
                                pixel_y,
                                depth,
                                rs_main.color_intrinsics)

                            temporal_3d_point = np.array([x, y, z])

                            points_3d_iter = np.append(points_3d_iter, temporal_3d_point[np.newaxis, :], axis=0)

                        points_pixel_iter = points_pixel_iter.reshape(1, 1404)
                        points_3d_iter = points_3d_iter.reshape(1, 1404)
                        points_pixel_iter=np.concatenate((np.array([c_time])[:, np.newaxis],points_pixel_iter),axis=1)
                        points_3d_iter = np.concatenate((np.array([c_time])[:, np.newaxis], points_3d_iter),axis=1)

                        if first_iter:
                            points_3d = points_3d_iter
                            points_pixel = points_pixel_iter
                            first_iter = False
                        else:
                            points_3d = np.concatenate((points_3d, points_3d_iter), axis=0)
                            points_pixel = np.concatenate((points_pixel, points_pixel_iter), axis=0)

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

                        # Get rotational matrix
                        rmat, jac = cv2.Rodrigues(rot_vec)

                        # Get angles
                        angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                        # Get the y rotation degree
                        x = angles[0] * 360
                        y = angles[1] * 360
                        z = angles[2] * 360

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
                print('FPS:{}\r'.format(1 / elapsed), end='')

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

                    os.mkdir("./pose_test/{}/".format(month + day + hour + minute))

                    pd.DataFrame(points_3d).to_csv(
                        "./pose_test/{}/points_3d.csv".format(month + day + hour + minute))
                    pd.DataFrame(points_pixel).to_csv(
                        "./pose_test/{}/points_pixel.csv".format(month + day + hour + minute)
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

