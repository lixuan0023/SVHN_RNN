import tensorflow as tf

from inputs import Inputs
from model import Model


class InferenceWrapper(object):
    """Model wrapper class for performing inference."""

    def __init__(self):
        super(InferenceWrapper, self).__init__()
        self.build_model()

    def build_model(self):
        mymodel = Model(vocab_size=12,
                        mode='inference',
                        embedding_size=512,
                        num_lstm_units=64,
                        lstm_dropout_keep_prob=0.7,
                        cnn_drop_rate=0.2,
                        initializer_scale=0.08)
        mymodel.build()

    def feed_image(self, sess, encoded_image):
        initial_state = sess.run(fetches="lstm/initial_state:0",
                                 feed_dict={"image_feed:0": encoded_image})
        return initial_state

    def inference_step(self, sess, input_feed, state_feed):
        softmax_output, state_output = sess.run(
            fetches=["softmax:0", "lstm/state:0"],
            feed_dict={
                "input_feed:0": input_feed,
                "lstm/state_feed:0": state_feed,
            })
        return softmax_output, state_output
