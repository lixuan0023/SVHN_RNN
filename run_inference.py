import tensorflow as tf
import os
import math
import numpy as np
from PIL import Image

from digits_inference import DigitsInference
from inference_wrapper import InferenceWrapper


def main(_):
    path_to_image_file = '/home/amax/Documents/wit/workspace/SVNH_RNN/train1.png'
    path_to_restore_checkpoint_file = '/home/amax/Documents/wit/logs/train/latest.ckpt'
    raw_image = Image.open(path_to_image_file)
    raw_image = raw_image.resize([54, 54])
    encoded_image = np.array(raw_image, dtype=np.float32)
    image = encoded_image / 256.0
    image = (image - 0.5) * 2

    g = tf.Graph()
    with g.as_default():

        model = InferenceWrapper()
        restorer = tf.train.Saver()

    with tf.Session(graph=g) as sess:

        restorer.restore(sess, path_to_restore_checkpoint_file)

        inference_model = DigitsInference(model)
        inference_list = inference_model.beam_search(sess, image)

        for i, result in enumerate(inference_list):
            print(i, result.numbers, math.exp(result.logprob))

if __name__ == "__main__":
    tf.app.run(main=main)

    # g.finalize()
