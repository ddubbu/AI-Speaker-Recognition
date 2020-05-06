import tensorflow as tf

sess = tf.Session()
# First let's load meta graph and restore weights
saver = tf.train.import_meta_graph('model-1000.meta')  # 네트워크 생성
saver.restore(sess, tf.train.latest_checkpoint('./'))  # 파라미터 로딩


# Now, let's access and create placeholders variables and
# create feed-dict to feed new data
# Model Netwrok status : (w1+w2)*2 , w4(operator) = A*B
# result : (4+8)*2 = 24
graph = tf.get_default_graph()  # initialize graph
w1 = graph.get_tensor_by_name("w1:0")  # Q. 오른쪽 :0은 뭐지?
w2 = graph.get_tensor_by_name("w2:0")

# Now, access the op that you want to run.
feed_dict ={w1:13.0,w2:17.0}

# get_operation_by_name 안되는뎁...
w4_new = graph.get_operation_by_name("w4:0")  # = (w1+w2)*2
# Q. 근데, placeholders 랑,
# 마지막 tensor(노드) 만 가져오면 되는거 같네?
# 왜냐하면, w3를 생략해서!


print(sess.run(w4_new, feed_dict))
# This will print 60 = (13+17)*2
# using new values of w1 and w2 and saved value of b1.

# 로딩 가능한 graph에서의 operation 종류
print("==============\n현재 로딩 가능한 operation 종류 in this graph")
for op in tf.get_default_graph().get_operations():
    print(op.name)