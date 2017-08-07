import tensorflow as tf

from inputs import Inputs
from model import Model


class Evaluator(object):

    def __init__(self, path_to_eval_log_dir):
        self.summary_writer = tf.summary.FileWriter(path_to_eval_log_dir)

    def evaluate(self, path_to_checkpoint, path_to_tfrecords_file, num_examples, global_step):
        batch_size = 128
        num_batches = int(num_examples / batch_size)

        with tf.Graph().as_default():
            input_ops = Inputs(path_to_tfrecords_file=path_to_tfrecords_file,
                               batch_size=batch_size,
                               shuffle=False,
                               min_queue_examples=5000,
                               num_preprocess_threads=4,
                               num_reader_threads=1)
            images, input_seqs, target_seqs, mask = input_ops.build_batch()

            mymodel = Model(vocab_size=12,
                        mode='evaluate',
                        embedding_size=512,
                        num_lstm_units=128,
                        lstm_dropout_keep_prob=0.7,
                        cnn_drop_rate=0.2,
                        initializer_scale=0.08)

            logits = mymodel.inference(images, input_seqs, mask)
            digit_predictions = tf.argmax(logits, axis=1)

            labels = tf.reshape(target_seqs, [-1])
            weights = tf.to_float(tf.reshape(mask, [-1]))
            predictions = tf.reshape(digit_predictions, [-1])

            accuracy, update_accuracy = tf.metrics.accuracy(
                labels=labels,
                predictions=predictions,
                weights=weights
            )

            tf.summary.image('image', images)
            tf.summary.scalar('accuracy', accuracy)
            tf.summary.histogram('variables',
                                 tf.concat([tf.reshape(var, [-1]) for var in tf.trainable_variables()], axis=0))
            summary = tf.summary.merge_all()

            with tf.Session() as sess:
                sess.run([tf.global_variables_initializer(),
                          tf.local_variables_initializer()])
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess=sess, coord=coord)

                restorer = tf.train.Saver()
                restorer.restore(sess, path_to_checkpoint)

                for _ in range(num_batches):
                    sess.run(update_accuracy)

                accuracy_val, summary_val = sess.run([accuracy, summary])
                self.summary_writer.add_summary(
                    summary_val, global_step=global_step)

                coord.request_stop()
                coord.join(threads)

        return accuracy_val
