import tensorflow as tf

with tf.Session() as sess:
    model_name = 'model/face_score.pb'
    with tf.gfile.FastGFile(model_name, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        g_in = tf.import_graph_def(graph_def)
logdir = 'tensor_board'
train_writer = tf.summary.FileWriter(logdir)
train_writer.add_graph(sess.graph)
