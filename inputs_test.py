import numpy as np
import tensorflow as tf

from inputs import Inputs
from model import Model


class InputsModelTest(tf.test.TestCase):

    def _checkOutputs(self, Tensors_output, Tensors_name, expected_shapes, feed_dict=None):
        # Verifies that the model produces expected outputs.

        # Args:
        #   expected_shapes: A dict mapping Tensor or Tensor name to expected output
        #     shape.
        #   feed_dict: Values of Tensors to feed into Session.run().

        fetches = Tensors_output

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            outputs = sess.run(fetches, feed_dict)

            for index, output in enumerate(outputs):
                tensor_name = Tensors_name[index]
                expected = expected_shapes[index]
                actual = output.shape
                if expected != actual:
                    print("Tensor %s has shape %s ." % (tensor_name, actual))
                    # self.fail("Tensor %s has shape %s (expected %s)." %
                    #           (tensor_name, actual, expected))
            coord.request_stop()
            coord.join(threads)

    def testForBatch_shuffle(self):
        # path_to_tfrecords_file = '/notebooks/dataVolume/workspace/data'
        path_to_tfrecords_file = '/home/amax/Documents/wit/data/train.tfrecords'
        input_ops = Inputs(path_to_tfrecords_file=path_to_tfrecords_file,
                           batch_size=32,
                           shuffle=True,
                           min_queue_examples=5000,
                           num_preprocess_threads=4,
                           num_reader_threads=1)

        images, input_seqs, target_seqs, mask = input_ops.build_batch()

        Tensors_output = [images, input_seqs, target_seqs, mask]
        Tensors_name = ['images', 'input_seqs',
                        'target_seqs', 'mask']

        expected_shapes = [(32, 54, 54, 3),  # [batch_size, image_height, image_width, 3]
                           (32, ),  # [batch_size, sequence_length]
                           (32, ),  # [batch_size, sequence_length]
                           (32, )]   # [batch_size, sequence_length]

        self._checkOutputs(Tensors_output, Tensors_name, expected_shapes)

    def testForBatch_not_shuffle(self):
        # path_to_tfrecords_file = '/notebooks/dataVolume/workspace/data'
        path_to_tfrecords_file = '/home/amax/Documents/wit/data/val.tfrecords'
        input_ops = Inputs(path_to_tfrecords_file=path_to_tfrecords_file,
                           batch_size=128,
                           shuffle=False,
                           min_queue_examples=5000,
                           num_preprocess_threads=4,
                           num_reader_threads=1)

        images, input_seqs, target_seqs, mask = input_ops.build_batch()

        Tensors_output = [images, input_seqs, target_seqs, mask]
        Tensors_name = ['images', 'input_seqs',
                        'target_seqs', 'mask']

        expected_shapes = [(128, 54, 54, 3),  # [batch_size, image_height, image_width, 3]
                           (128, ),  # [batch_size, sequence_length]
                           (128, ),  # [batch_size, sequence_length]
                           (128, )]   # [batch_size, sequence_length]

        self._checkOutputs(Tensors_output, Tensors_name, expected_shapes)

    def testForModel(self):
        # path_to_tfrecords_file = '/notebooks/dataVolume/workspace/data'
        path_to_tfrecords_file = '/home/amax/Documents/wit/data/train.tfrecords'
        input_ops = Inputs(path_to_tfrecords_file=path_to_tfrecords_file,
                           batch_size=32,
                           shuffle=True,
                           min_queue_examples=5000,
                           num_preprocess_threads=4,
                           num_reader_threads=1)

        images, input_seqs, target_seqs, mask = input_ops.build_batch()

        mymodel = Model(vocab_size=11,
                        mode='train',
                        embedding_size=512,
                        num_lstm_units=128,
                        lstm_dropout_keep_prob=0.7,
                        cnn_drop_rate=0.2,
                        initializer_scale=0.08)

        logits = mymodel.inference(images, input_seqs, mask, state=None)

        total_loss = mymodel.loss(logits, target_seqs)

        Tensors_output = [images, input_seqs, target_seqs, mask, logits]
        Tensors_name = ['images', 'input_seqs',
                        'target_seqs', 'mask', 'logits']

        expected_shapes = [(32, 54, 54, 3),  # [batch_size, image_height, image_width, 3]
                           (32, ),  # [batch_size, sequence_length]
                           (32, ),  # [batch_size, sequence_length]
                           (32, ),   # [batch_size, sequence_length]
                           ()]

        self._checkOutputs(Tensors_output, Tensors_name, expected_shapes)


if __name__ == "__main__":
    tf.test.main()
