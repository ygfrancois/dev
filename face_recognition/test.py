# coding=utf-8
import face_model
import argparse
import cv2
import sys
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description='face model test')
# general
parser.add_argument('--image-size', default='112,112', help='')
parser.add_argument('--model', default='/home/yangguang/dev/face_recognition/model-r100-arcface-ms1m-refine-v2/model-r100-ii/model,0', help='path to load model.')
parser.add_argument('--ga-model', default='', help='path to load model.')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
args = parser.parse_args()

model = face_model.FaceModel(args)
# img = cv2.imread('Tom_Hanks_54745.png')
# img = model.get_input(img)
# #f1 = model.get_feature(img)
# #print(f1[0:10])
# gender, age = model.get_ga(img)
# print(gender)
# print(age)
# sys.exit(0)


img1 = cv2.imread('Tom_Hanks_54745.png')
img1 = model.get_input(img1)
f1 = model.get_feature(img1)
print(f1[0:10])
img2 = cv2.imread('/home/yangguang/dev/face_recognition/dataset/1.jpg')
img2 = model.get_input(img2)
f2 = model.get_feature(img2)
features2 = pd.DataFrame(f2)
features2.to_csv('/home/yangguang/dev/face_recognition/dataset/yg.csv', index=False)
data = pd.read_csv('/home/yangguang/dev/face_recognition/dataset/yg.csv')
data = data.values.transpose()[0]  # 先取出csv里的为dataframe格式,value变成array,转置一下变成f3形式,再取第一个(读出来是一个列表)
img3 = cv2.imread('/home/yangguang/dev/face_recognition/dataset/2.jpg')
img3 = model.get_input(img3)
f3 = model.get_feature(img3)
print(np.sum(np.square(f3-data)))

img4 = cv2.imread('/home/yangguang/dev/face_recognition/dataset/3.jpg')
img4 = model.get_input(img4)
f4 = model.get_feature(img4)
dist1 = np.sum(np.square(f1-f2))
img5 = cv2.imread('/home/yangguang/dev/face_recognition/dataset/4.jpg')
img5 = model.get_input(img5)
f5 = model.get_feature(img5)
dist3 = np.sum(np.square(f2-f4))
dist4 = np.sum(np.square(f2-f5))
dist1 = np.sum(np.square(f1-f2))
dist2 = np.sum(np.square(f2-f3))
print(dist1, dist2, dist3, dist4)


# sim = np.dot(f1, f2.T)
# print(sim)
#diff = np.subtract(source_feature, target_feature)
#dist = np.sum(np.square(diff),1)



