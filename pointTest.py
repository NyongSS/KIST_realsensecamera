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

#csv파일 열기

data = pd.read_csv('./marker230726_164843.csv')
face = pd.read_csv('./points_3d3.csv')

#trans 1 data
X = data.iloc[[0, 2, 4], 8:9].values * 0.01
Y = data.iloc[[0, 2, 4], 9:10].values * 0.01
Z = data.iloc[[0, 2, 4], 10:11].values * 0.01

#trans 2 data
X2 = data.iloc[[1, 3, 5], 11:12].values * 0.01
Y2 = data.iloc[[1, 3, 5], 12:13].values * 0.01
Z2 = data.iloc[[1, 3, 5], 13:14].values * 0.01

X3 = face.iloc[10:11, 1:1405:3].values
Y3 = face.iloc[10:11, 2:1405:3].values
Z3 = face.iloc[10:11, 3:1405:3].values

X6 = face.iloc[1:2, 2:3].values
Y6 = face.iloc[1:2, 3:4].values
Z6 = face.iloc[1:2, 4:5].values

X4 = data.iloc[:, 8:9].values * 0.01
Y4 = data.iloc[:, 9:10].values * 0.01
Z4 = data.iloc[:, 10:11].values * 0.01

X5 = data.iloc[:, 11:12].values * 0.01
Y5 = data.iloc[:, 12:13].values * 0.01
Z5 = data.iloc[:, 13:14].values * 0.01

face = face.drop(face.columns[0], axis=1)

X3 = face.iloc[10:11, 1:1405:3].values
Y3 = face.iloc[10:11, 2:1405:3].values
Z3 = face.iloc[10:11, 3:1405:3].values

print(face.head())
print(Z3)
#plot
fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.scatter(X3, Y3, Z3)
#ax.scatter(X, Y, Z, color='r')
#ax.scatter(X2, Y2, Z2, color='g')

ax.scatter(X3, Y3, Z3, color='b')
ax.scatter(X4, Y4, Z4, color='r')
ax.scatter(X5, Y5, Z5, color='g')

#ax.scatter(X6, Y6, Z6, color='b')
#facial landmark 축 범위 지정하기
# ax.set_zlim(0,0.8)
#plt.axis([-0.5, 0.5, -0.5, 0.5])
plt.show()
