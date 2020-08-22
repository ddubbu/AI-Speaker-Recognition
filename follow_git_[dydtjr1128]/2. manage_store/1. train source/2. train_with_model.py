import librosa
import numpy as np
import pandas as pd
import tensorflow as tf

# After read input data.

# 나중에 feeding 되는거 보면 placeholder 대신, python 자체 전역변수를 사용한 듯.
X_train = []  # train_data 저장할 공간
X_test = []
Y_train = []
Y_test = []
# tf_classes = 0


# Now, Model 구축 : 화자인식 NN 버전
X_train, X_test, Y_train, Y_test = np.load("./train_data.npy", allow_pickle=True)
print("X_train :", np.shape(X_train))  # (37650, 20)
print("Y_train :", np.shape(Y_train))  # (37650, 5)

num_input = np.shape(X_train)[1]
num_output = np.shape(Y_train)[1]  # tf_classes


X_train = X_train.astype("float")
X_test = X_test.astype("float")

tf.reset_default_graph()
tf.set_random_seed(777)
learning_rate = 0.001
training_epochs = 100  # <- 100 우선은 조금만 Training
keep_prob = tf.placeholder(tf.float32, name="keep_prob")  # training except some data
sd = 1 / np.sqrt(num_input)  # standard deviation 표준편차(표본표준편차라 1/root(n))

# mfcc의 기본은 20
# 20ms일 때216은 각 mfcc feature의 열이 216
X = tf.placeholder(tf.float32, [None, num_input], name="X")
Y = tf.placeholder(tf.float32, [None, num_output], name="Y")

# W = tf.Variable(tf.random_normal([216, 200]))
# b = tf.Variable(tf.random_normal([200]))

# 1차 히든레이어
W1 = tf.get_variable("w1",
                     # tf.random_normal([216, 180], mean=0, stddev=sd),
                     shape=[num_input, 256],
                     initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([256], mean=0, stddev=sd), name="b1")
L1 = tf.nn.relu(tf.matmul(X, W1) + b1)  # 1차 히든레이어는 'Relu' 함수를 쓴다.
L1 = tf.nn.dropout(L1, keep_prob=keep_prob)

# 2차 히든 레이어
W2 = tf.get_variable("w2",
                     # tf.random_normal([180, 150], mean=0, stddev=sd),
                     shape=[256, 256],
                     initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([256], mean=0, stddev=sd), name="b2")
L2 = tf.nn.tanh(tf.matmul(L1, W2) + b2)  # 2차 히든레이어는 'Relu' 함수를 쓴다.
L2 = tf.nn.dropout(L2, keep_prob=keep_prob)

# 3차 히든 레이어
W3 = tf.get_variable("w3",
                     # tf.random_normal([150, 100], mean=0, stddev=sd),
                     shape=[256, 256],
                     initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([256], mean=0, stddev=sd), name="b3")
L3 = tf.nn.relu(tf.matmul(L2, W3) + b3)  # 3차 히든레이어는 'Relu' 함수를 쓴다.
L3 = tf.nn.dropout(L3, keep_prob=keep_prob)

# 4차 히든 레이어
W4 = tf.get_variable("w4",
                     # tf.random_normal([100, 50], mean=0, stddev=sd),
                     shape=[256, 128],
                     initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([128], mean=0, stddev=sd), name="b4")
L4 = tf.nn.relu(tf.matmul(L3, W4) + b4)  # 4차 히든레이어는 'Relu' 함수를 쓴다.
L4 = tf.nn.dropout(L4, keep_prob=keep_prob)

# 5차 히든 레이어
W5 = tf.get_variable("w5",
                     # tf.random_normal([100, 50], mean=0, stddev=sd),
                     shape=[128, 128],
                     initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([128], mean=0, stddev=sd), name="b5")
L5 = tf.nn.relu(tf.matmul(L4, W5) + b5)  # 5차 히든레이어는 'Relu' 함수를 쓴다.
L5 = tf.nn.dropout(L5, keep_prob=keep_prob)

# 6차 히든 레이어
W6 = tf.get_variable("w6",
                     # tf.random_normal([100, 50], mean=0, stddev=sd),
                     shape=[128, 128],
                     initializer=tf.contrib.layers.xavier_initializer())
b6 = tf.Variable(tf.random_normal([128], mean=0, stddev=sd), name="b6")
L6 = tf.nn.relu(tf.matmul(L5, W6) + b6)  # 6차 히든레이어는 'Relu' 함수를 쓴다.
L6 = tf.nn.dropout(L6, keep_prob=keep_prob)

# 7차 히든 레이어
W7 = tf.get_variable("w7",
                     # tf.random_normal([100, 50], mean=0, stddev=sd),
                     shape=[128, 128],
                     initializer=tf.contrib.layers.xavier_initializer())
b7 = tf.Variable(tf.random_normal([128], mean=0, stddev=sd), name="b7")
L7 = tf.nn.relu(tf.matmul(L6, W7) + b7)  # 7차 히든레이어는 'Relu' 함수를 쓴다.
L7 = tf.nn.dropout(L7, keep_prob=keep_prob)

# 최종 레이어
W8 = tf.get_variable("w8",
                     # tf.random_normal([50, num_output], mean=0, stddev=sd),
                     shape=[128, num_output],
                     initializer=tf.contrib.layers.xavier_initializer())
b8 = tf.Variable(tf.random_normal([num_output], mean=0, stddev=sd), name="b8")

# make it tensor type
hypothesis = tf.matmul(L7, W8)  # tf.matmul(L7, W8) + b8
hypothesis = tf.add(hypothesis, b8, name="hypothesis")


# cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=hypothesis, labels=Y))

# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# is_correct = tf.equal(tf.arg_max(hypothesis, 1), tf.arg_max(Y, 1))

batch_size = 1
x_len = len(X_train)
# 짝수
if (x_len % 2 == 0):
    batch_size = 2
elif (x_len % 3 == 0):
    batch_size = 3
elif (x_len % 4 == 0):
    batch_size = 4
else:
    batch_size = 1

split_X = np.split(X_train, batch_size)
split_Y = np.split(Y_train, batch_size)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epoch in range(training_epochs):
    avg_cost = 0
    for i in range(batch_size):
        batch_xs = split_X[i]
        batch_ys = split_Y[i]
        feed_dict = {X: batch_xs, Y: batch_ys, keep_prob: 0.7}
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost += c / batch_size
        # if(epoch%10==0):
    print('Epoch:', '%04d' % (epoch), 'cost =', '{:.9f}'.format(avg_cost))

correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print("Accuracy: ", sess.run(accuracy, feed_dict={X: X_test, Y: Y_test, keep_prob: 1}))

print('Learning Finished!')

saver = tf.train.Saver()  # 모델 특성이 드러나도록 파일명을 저장하자.
saver.save(sess, './NN_model_'+ str(training_epochs) + ' epochs_trainig')


# ########## Test 및 프로그램 실행 시작 ##########
# y, sr = librosa.load("../../data/test/test_문재인대통령.wav")
#
# X_test = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=num_input, hop_length=int(sr*0.01),n_fft=int(sr*0.02)).T
#
# label = [0 for i in range(5)] # class가 3개이니까 y_test만드는 과정
# label[2] = 1
# Y_test = []
# for i in range(len(X_test)):
#     Y_test.append(label)
#
# # print(np.shape(X_test))
# # print(np.shape(Y_test))
#
# # correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
# # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# #
# #
# # print("predict")
# # print(pd.value_counts(pd.Series(sess.run(tf.argmax(hypothesis, 1),
# #                                     feed_dict={X: X_test, keep_prob:1}))))
# # print("Accuracy: ", sess.run(accuracy, feed_dict={X: X_test, Y:Y_test, keep_prob:1}))
#
# print("======= Result ======== ")
#
# # result = sess.run(hypothesis, feed_dict={X: X_test, keep_prob:1})
# # print("hypothesis :", result)
# # print("======= one-hot? ======== ")
# # hot_vector = sess.run(tf.arg_max(result,1))
# # print(len(hot_vector))
#
#
# # 아직 NN 구조 및 차원 이해는 못했지만, 최빈값이 답인 것을 유추하고
# value_counts = pd.value_counts(pd.Series(sess.run(tf.argmax(hypothesis, 1),
#                                     feed_dict={X: X_test, keep_prob:1})))
# predict_result = value_counts.idxmax()  # 최빈값 가져옴.
#         # 혹은 argmax을 사용하여 최대 값의 키를 얻음 : value_coutns.argmax()
# print(predict_result)
# labels = ["유인나", "배철수", "이재은", "최일구", "문재인"]
#
# print("이 화자는 :", labels[predict_result])