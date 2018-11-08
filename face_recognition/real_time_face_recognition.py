# coding=utf-8
"""Performs face detection in realtime.

Based on code from https://github.com/shanren7/real_time_face_recognition
"""

import argparse
import face_model
import sys
import time
import numpy as np
import cv2
from mtcnn_detector import MtcnnDetector
import pandas as pd


def to_rgb(img):
    # 把二维变成3维
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret


def add_overlays(frame, faces, frame_rate):
    if faces is not None:
        for face in faces:
            face_bb = face.bounding_box.astype(int)
            cv2.rectangle(frame,
                          (face_bb[0], face_bb[1]), (face_bb[2], face_bb[3]),
                          (0, 255, 0), 2)
            if face.name is not None:
                cv2.putText(frame, face.name, (face_bb[0], face_bb[3]),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                            thickness=2, lineType=2)

    cv2.putText(frame, str(frame_rate) + " fps", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                thickness=2, lineType=2)


def main(args):
    video_capture = cv2.VideoCapture(0)
    c = 0

    while True:
        ret, frame = video_capture.read()

        frame_interval = 2
        timeF = frame_interval

        if (c % timeF == 0):  # capture frame-by-frame
            find_results = []

            face_detector = MtcnnDetector('mtcnn-model/', threshold=[0.5, 0.6, 0.7])
            bounding_boxes, points = face_detector.detect_face(frame)

            nb_faces = bounding_boxes.shape[0]  # number of faces
            print('找到人脸数目为：{}'.format(nb_faces))

            for face_position in bounding_boxes:

                face_position_int = face_position.astype(int)

                # 使用cv2.rectangle来画矩形框
                cv2.rectangle(frame, (face_position_int[0],
                                      face_position_int[1]),
                              (face_position_int[2], face_position_int[3]),
                              (0, 255, 0), 2)

            model = face_model.FaceModel(args)
            img = model.get_input(frame)  # 原来的frame最后还要输出用,不可以破坏
            features = model.get_feature(img)

            features_model = pd.read_csv('/home/yangguang/dev/face_recognition/dataset/yg.csv')
            features_model = features_model.values.transpose()[0]  # 先取出csv里的为dataframe格式,value变成array,转置一下变成f3形式,再取第一个(读出来是一个列表)

            dist = np.sum(np.square(features-features_model))
            if dist < 1.0:  # 样本feature和模板feature之间的距离
                find_results.append('me')
            elif dist >= 1.0:
                find_results.append('others')

            # crop = frame[face_position_int[1]:face_position_int[3], face_position_int[0]:face_position_int[2], ]
                #
                # crop = cv2.resize(crop, (96, 96), interpolation=cv2.INTER_CUBIC)
                #
                # data = crop.reshape(-1, 96, 96, 3)
                #
                # emb_data = sess.run([embeddings],
                #                     feed_dict={images_placeholder: np.array(data),
                #                                phase_train_placeholder: False})[0]
                #
                # predict = model.predict(emb_data)

            cv2.putText(frame, 'detected:{}'.format(find_results), (50, 100),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255, 0, 0),
                        thickness=2, lineType=2)
            print (dist, find_results)

            # print(faces)
        c += 1
        # Draw a rectangle around the faces

        # Display the resulting frame

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()


def parse_arguments(argv):
    parser = argparse.ArgumentParser(description='real time face recognition')
    # general
    parser.add_argument('--image-size', default='112,112', help='')
    parser.add_argument('--model',
                        default='/home/yangguang/dev/face_recognition/model-r100-arcface-ms1m-refine-v2/model-r100-ii/model,0',
                        help='path to load model.')
    parser.add_argument('--ga-model', default='', help='path to load model.')
    parser.add_argument('--gpu', default=0, type=int, help='gpu id')
    parser.add_argument('--det', default=0, type=int,
                        help='mtcnn option, 1 means using R+O, 0 means detect from begining')
    parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
    parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
    args = parser.parse_args()


    parser.add_argument('--debug', action='store_true',
                        help='Enable some debug outputs.')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))