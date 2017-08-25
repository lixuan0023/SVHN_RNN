import tensorflow as tf
import os
import math
import numpy as np
from PIL import Image

from digits_inference import DigitsInference
from inference_wrapper import InferenceWrapper


class Inference(object):

    def __init__(self, path_to_checkpoint_file):
        g = tf.Graph()
        with g.as_default():
            model = InferenceWrapper()
            restorer = tf.train.Saver()

        sess = tf.Session(graph=g)
        restorer.restore(sess, path_to_checkpoint_file)
        inference_model = DigitsInference(model)

        self.sess = sess
        self.inference_model = inference_model
        self.image = None
        self.raw_image = None
        self.path_to_image_file = None

    def feed_image(self):
        path_to_image_file = self.path_to_image_file
        assert os.path.exists(path_to_image_file), 'The file %s does not exist' % path_to_image_file

        raw_image = Image.open(path_to_image_file)
        image = raw_image.resize([54, 54])
        image = np.array(image, dtype=np.float32)
        image = image / 256.0
        image = (image - 0.5) * 2
        self.image = image
        self.raw_image = raw_image

    def run(self, path_to_image_file):
        self.path_to_image_file = path_to_image_file
        self.feed_image()

        inference_list = self.inference_model.beam_search(
            self.sess, self.image)

        return self.raw_image, inference_list

        def output(self, inference_list):
            print('The candidate result as follow.')
            print('No.\tresult\tprobability')
            for i, result in enumerate(inference_list):
                numbers = result.numbers[1:-1]
                ns = [str(n) for n in numbers]
                strn = ''.join(ns)
                prob = math.exp(result.logprob)
                print(' %d\t %s\t% 0.3f' % (i + 1, strn, prob))

if __name__ == '__main__':
    path_to_checkpoint_file = '/notebooks/dataVolume/workspace/logs/train/latest.ckpt'
    path_to_image_file = '/notebooks/dataVolume/workspace/test/10.png'
    inf = Inference(path_to_checkpoint_file)
    image, inference_list = inf.run(path_to_image_file)

    imshow(image)
    inf.output(inference_list)
