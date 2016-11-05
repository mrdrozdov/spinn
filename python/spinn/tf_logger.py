import tensorflow as tf


class TFLogger(object):
    """ Creates an "empty model" that writes Tensorflow summaries. Can
        visualize these summaries with Tensorboard.
    """
    def __init__(self, summary_dir):
        super(TFLogger, self).__init__()
        self.summary_dir = summary_dir
        self.__initialize()

    def __initialize(self):
        sess = tf.Session()
        loss = tf.Variable(0.0, name="loss", trainable=False)
        acc = tf.Variable(0.0, name="accuracy", trainable=False)
        loss_summary = tf.scalar_summary("loss", loss)
        acc_summary = tf.scalar_summary("accuracy", acc)
        summary_op = tf.merge_summary([loss_summary, acc_summary])
        summary_writer = tf.train.SummaryWriter(self.summary_dir, sess.graph)
        saver = tf.train.Saver(tf.all_variables())
        sess.run(tf.initialize_all_variables())

        self.sess = sess
        self.summary_op = summary_op
        self.summary_writer = summary_writer
        self.loss = loss
        self.acc = acc

    def log(self, step, loss, accuracy):
        feed_dict = {
            self.loss: loss,
            self.acc: accuracy,
        }

        summaries, _, _ = self.sess.run(
            [self.summary_op, self.loss, self.acc],
            feed_dict)
        self.summary_writer.add_summary(summaries, step)
