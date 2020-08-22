# this code is to make "./train_data.npy"

import librosa
import numpy as np
import pandas as pd
# for seperating train and test set
from sklearn.model_selection import train_test_split  # for k-Fold Cross Validation
# Seperate train data to train and validation
import os

'''
# 소리 녹음을 위한 
# 변수 설정 및 라이브러리 import 부분
import pyaudio  # 마이크를 사용하기 위한 라이브러리
import wave
import matplotlib.pyplot as plt
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100  # 비트레이트 설정
CHUNK = int(RATE / 10)  # 버퍼 사이즈 1초당 44100비트레이트 이므로 100ms단위
RECORD_SECONDS = 5  # 녹음할 시간 설정
WAVE_OUTPUT_FILENAME = "output.wav"
'''

TRAIN_DATA_PATH = "../../data/train/"

# placeholder는 입력값 그릇, tf.Variable는 업데이트 대상
# 나중에 feeding 되는거 보면 placeholder 대신, python 자체 전역변수를 사용한 듯.
X_train = []  # train_data 저장할 공간
X_test = []
Y_train = []
Y_test = []
tf_classes = 0


def load_wave_generator(path):
    try:  # 혹시
        batch_waves = []
        labels = []
        X_data = []
        Y_label = []
        global X_train, X_test, Y_train, Y_test, tf_classes

        folders = os.listdir(path)  # ['0', '1', '2', '3', '4']

        for folder in folders:
            files = os.listdir(path + "/" + folder)
            # 폴더 이름과 그 폴더에 속하는 파일 갯수 출력
            print("Foldername :", folder, "-", len(files), "파일")
            for wav in files:
                if not wav.endswith(".wav"):  # wave 파일만 읽어들임
                    continue
                else:
                    # print("Filename :",wav)#.wav 파일이 아니면 continue
                    y, sr = librosa.load(path + "/" + folder + "/" + wav)  # 소리파일 읽기
                    # voice feature  by MFCC
                    # ★ I can change it mel-spectrogram : librosa.feature.melspectrogram()
                    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=int(sr * 0.01),
                                                n_fft=int(sr * 0.02)).T

                    print(np.shape(mfcc))  # (502, 20)
                    exit()
                    # label 0 ~ 4 통채로 X_data, Y_label에 넣어버리기
                    X_data.extend(mfcc)  # Q. append와의 차이점? extend는 꺽새를 풀어서 원소만 넣기
                    # print(len(mfcc))

                    '''
                    # labeling 中!
                    # 나중에 학습시킬 data를 이처럼 잘 정리하는 게 좋겠다!
                    # Q. 그리고 생각보다 많은 학습 데이터가 없어도 화자인식은 어느정도의 인식율이 나오는 것인가?

                    0 유인나
                    1 배철수
                    2 이재은
                    3 최일구
                    4 문재인 대통령
                    '''
                    label = [0 for i in range(len(folders))]  # initialize [0 0 0 0 0]
                    label[tf_classes] = 1  # one-hot-vector

                    for i in range(len(mfcc)):
                        Y_label.append(label)  # 꺽새랑 함께
                    # print(Y_label)
            tf_classes = tf_classes + 1  # 다음 training 화자 target으로 이동
    except PermissionError:
        print(path, "를 열수 없습니다.")
        pass
    # end loop
    print("X_data :", np.shape(X_data))
    print("Y_label :", np.shape(Y_label))

    X_train, X_test, Y_train, Y_test = train_test_split(np.array(X_data), np.array(Y_label))

    xy = (X_train, X_test, Y_train, Y_test)
    np.save("./train_data.npy", xy)


load_wave_generator(TRAIN_DATA_PATH)  # input으로 사용될 ./train_data.npy 생성하기

print(tf_classes, "개의 클래스!!")

# n_mfcc(= 음성 feature) 개수가 20, 이 factor는 input X의 shape에 영향을 끼친다.
# 중간 Layer 연산을 통해 20->5 shape down
print("X_train :", np.shape(X_train))  # (37650, 20)
print("Y_train :", np.shape(Y_train))  # (37650,5)
print("X_test :", np.shape(X_test))  # (12550,20)
print("Y_test :", np.shape(Y_test))  # (12550,5)
