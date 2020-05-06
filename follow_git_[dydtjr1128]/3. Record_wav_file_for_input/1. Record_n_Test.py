# after training
# get placeholder : X, (Y), keep_prob
# and model : hypothesis
# only !! 중간과정 생략

import librosa
import numpy as np
import pandas as pd
import tensorflow as tf



sess = tf.Session()
# load meta graph and restore weights
saver = tf.train.import_meta_graph('1. train source/NN_model_100 epochs_trainig.meta')
saver.restore(sess, tf.train.latest_checkpoint('1. train source/'))

# access and create placeholders variables and
graph = tf.get_default_graph()
X = graph.get_tensor_by_name("X:0")
Y = graph.get_tensor_by_name("Y:0")
keep_prob = graph.get_tensor_by_name("keep_prob:0")

num_input = X.shape[-1]  # 20
num_output = Y.shape[-1]  # 5

# access the op to run.
hypothesis = graph.get_tensor_by_name("hypothesis:0")


########## Test 및 프로그램 실행 시작 ##########

# test 데이터 가져오기
Y_labels = ["유인나", "배철수", "이재은", "최일구", "문재인"]
Y_pick = "이재은"
raw_test, sr = librosa.load("../data/test/test_"+Y_pick+".wav")
X_test = librosa.feature.mfcc(y=raw_test, sr=sr, n_mfcc=num_input, hop_length=int(sr*0.01), n_fft=int(sr*0.02)).T

value_counts = pd.value_counts(pd.Series(sess.run(tf.argmax(hypothesis, 1),
                                    feed_dict={X: X_test, keep_prob:1})))
predict_result = value_counts.idxmax()  # 최빈값 가져옴.
        # 혹은 argmax을 사용하여 최대 값의 키를 얻음 : value_coutns.argmax()
print(predict_result)


print("이 화자는 :", Y_labels[predict_result])


correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# No Need to print accuracy
# Because, here there is only True/ False about result
# we need accuracy for NN performance
