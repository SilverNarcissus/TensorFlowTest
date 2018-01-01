import tensorflow as tf

# constant input
# node1 = tf.constant(3.0)
# node2 = tf.constant(4.0)
session = tf.Session()
# # print(session.run([node1, node2]))
#
# # constant add
# node3 = tf.add(node1, node2, name="add")
# # print(session.run(node3))
# # print(node3)


# parameter input add
with tf.name_scope("para") as scope:
    a = tf.placeholder(tf.float32, name="a")
    b = tf.placeholder(tf.float32, name="b")
    adder_node = a + b  # + provides a shortcut for tf.add(a, b)
    # writer = tf.summary.FileWriter("./logs", session.graph)
    # session.run(adder_node, {a: [1], b: [2]})

# liner model
# with tf.name_scope("liner_model") as scope:
#     W = tf.Variable([.3], dtype=tf.float32)
#     b = tf.Variable([-.3], dtype=tf.float32)
#     x = tf.placeholder(tf.float32)
#     linear_model = W * x + b
#     init = tf.global_variables_initializer()
#     session.run(init)
#     writer = tf.summary.FileWriter("./logs", session.graph)
#     print(session.run(linear_model, {x: [1, 2, 3, 4]}))
#     # train
#     y = tf.placeholder(tf.float32)
#     squared_deltas = tf.square(linear_model - y)
#     loss = tf.reduce_sum(squared_deltas)
#     print(session.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))
#     # change variable
#     fixW = tf.assign(W, [-1.])
#     fixb = tf.assign(b, [1.])
#     session.run([fixW, fixb])
#     print(session.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))


with tf.name_scope("liner_model_train") as scope:
    # param
    # W = tf.Variable([.3], dtype=tf.float32, name="W")
    W = tf.constant(1.)
    b = tf.Variable([-.3], dtype=tf.float32, name="b")
    # input
    x = tf.placeholder(tf.float32, name="x")
    # model
    linear_model = W * x + b
    # evaluate process
    y = tf.placeholder(tf.float32, name="y")
    squared_deltas = tf.square(linear_model - y, name="deltas")
    loss = tf.reduce_sum(squared_deltas, name="loss")
    # train
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = optimizer.minimize(loss)
    init = tf.global_variables_initializer()
    session.run(init)  # init variable
    for i in range(1000):
        session.run(train, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]})

    writer = tf.summary.FileWriter("./logs", session.graph)
    print(session.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))
