import tensorflow as tf


class Inputs(object):

    def __init__(self,
                 path_to_tfrecords_file,
                 batch_size=32,
                 shuffle=True,
                 min_queue_examples=5000,
                 num_preprocess_threads=4,
                 num_reader_threads=1):

        super(Inputs, self).__init__()

        self.reader = tf.TFRecordReader()
        self.path_to_tfrecords_file = path_to_tfrecords_file
        self.batch_size = batch_size
        self.min_queue_examples = min_queue_examples
        self.num_preprocess_threads = num_preprocess_threads
        self.num_reader_threads = num_reader_threads
        self.shuffle = shuffle

        assert tf.gfile.Exists(
            path_to_tfrecords_file), '%s not found' % path_to_tfrecords_file
        assert num_preprocess_threads % 2 == 0

    def prefetch_input_data(self):
        reader = self.reader
        path_to_tfrecords_file = self.path_to_tfrecords_file
        min_queue_examples = self.min_queue_examples
        batch_size = self.batch_size
        num_reader_threads = self.num_reader_threads

        filename_queue = tf.train.string_input_producer(
            [path_to_tfrecords_file], shuffle=True, capacity=16, name='filename_queue')

    # Recommendation:
    # capacity = min_queue_examples + (num_threads+a small safety margin)*
    # batch_size
        capacity = min_queue_examples + 100 * batch_size

        values_queue = tf.RandomShuffleQueue(
            capacity=capacity,
            min_after_dequeue=min_queue_examples,
            dtypes=[tf.string],
            name="random_input_queue_name")

        enqueue_ops = []
        for _ in range(num_reader_threads):
            _, value = reader.read(filename_queue)
            enqueue_ops.append(values_queue.enqueue([value]))

        tf.train.queue_runner.add_queue_runner(tf.train.queue_runner.QueueRunner(
            values_queue, enqueue_ops))
        tf.summary.scalar("queue/%s/fraction_of_%d_full" % (values_queue.name, capacity),
                          tf.cast(values_queue.size(), tf.float32) * (1. / capacity))

        return values_queue

    def parse_sequence_example(self, serialized_example, thread_id):
        features = tf.parse_single_example(
            serialized_example,
            features={
                'image': tf.FixedLenFeature([], tf.string),
                'length': tf.FixedLenFeature([], tf.int64),
                'digits': tf.FixedLenFeature([7], tf.int64)
            })

        # data process
        image = Inputs.image_process(features['image'], thread_id)
        length = tf.cast(features['length'], tf.int32)
        digits = tf.cast(features['digits'], tf.int32)

        input_seq, target_seq, indicator = Inputs.digits_process(
            digits, length)

        return image, input_seq, target_seq, indicator

    def batch_with_dynamic_pad_shuffled(self):
        input_queue = self.prefetch_input_data()
        batch_size = self.batch_size
        num_preprocess_threads = self.num_preprocess_threads
        queue_capacity = 2 * num_preprocess_threads * batch_size

        enqueue_list = []
        for thread_id in range(num_preprocess_threads):
            serialized_sequence_example = input_queue.dequeue()
            image, input_seq, target_seq, indicator = self.parse_sequence_example(
                serialized_sequence_example, thread_id)
            enqueue_list.append([image, input_seq, target_seq, indicator])

        images, input_seqs, target_seqs, mask = tf.train.batch_join(
            enqueue_list,
            batch_size=batch_size,
            capacity=queue_capacity,
            dynamic_pad=True,
            name="batch_and_pad")

        lengths = tf.add(tf.reduce_sum(mask, 1), 1)
        tf.summary.scalar("digits_length/batch_min", tf.reduce_min(lengths))
        tf.summary.scalar("digits_length/batch_max", tf.reduce_max(lengths))
        tf.summary.scalar("digits_length/batch_mean", tf.reduce_mean(lengths))

        return images, input_seqs, target_seqs, mask

    def batch_with_dynamic_pad_not_shuffled(self):
        def im_process(raw_image):
            image = tf.decode_raw(raw_image, tf.uint8)
            image = tf.reshape(image, [64, 64, 3])
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
            image = tf.div(image, 256.0)
            image = tf.multiply(tf.subtract(image, 0.5), 2)
            return image

        reader = self.reader
        path_to_tfrecords_file = self.path_to_tfrecords_file
        batch_size = self.batch_size
        num_reader_threads = self.num_reader_threads
        num_preprocess_threads = self.num_preprocess_threads
        min_queue_examples = self.min_queue_examples

        filename_queue = tf.train.string_input_producer(
            [path_to_tfrecords_file], shuffle=False, name='filename_queue')

        capacity = min_queue_examples + 3 * batch_size
        values_queue = tf.FIFOQueue(
            capacity=capacity, dtypes=[tf.string], name="fifo_input_queue_name")

        enqueue_ops = []
        for _ in range(num_reader_threads):
            _, value = reader.read(filename_queue)
            enqueue_ops.append(values_queue.enqueue([value]))

        tf.train.queue_runner.add_queue_runner(tf.train.queue_runner.QueueRunner(
            values_queue, enqueue_ops))

        enqueue_list = []
        for _ in range(num_preprocess_threads):
            serialized_example = values_queue.dequeue()

            features = tf.parse_single_example(
                serialized_example,
                features={
                    'image': tf.FixedLenFeature([], tf.string),
                    'length': tf.FixedLenFeature([], tf.int64),
                    'digits': tf.FixedLenFeature([7], tf.int64)
                })
            # data process
            image = im_process(features['image'])
            length = tf.cast(features['length'], tf.int32)
            digits = tf.cast(features['digits'], tf.int32)
            input_seq, target_seq, indicator = Inputs.digits_process(
                digits, length)
            enqueue_list.append([image, input_seq, target_seq, indicator])

        queue_capacity = 2 * num_preprocess_threads * batch_size  # 2->10
        images, input_seqs, target_seqs, mask = tf.train.batch_join(
            enqueue_list,
            batch_size=batch_size,
            capacity=queue_capacity,
            dynamic_pad=True,
            name="batch_not_shuffled")
        images = tf.image.resize_images(images, [54, 54])

        return images, input_seqs, target_seqs, mask

    def build_batch(self):
        shuffle = self.shuffle
        if shuffle:
            return self.batch_with_dynamic_pad_shuffled()
        else:
            return self.batch_with_dynamic_pad_not_shuffled()

    @staticmethod
    def image_process(raw_image, thread_id=0):
        # Helper function to log an image summary to the visualizer. Summaries are
        # only logged in thread 0.
        def image_summary(name, image):
            if not thread_id:
                tf.summary.image(name, tf.expand_dims(image, 0))

        with tf.name_scope("decode", values=[raw_image]):
            image = tf.decode_raw(raw_image, tf.uint8)
            image = tf.reshape(image, [64, 64, 3])
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image_summary("original_image", image)

        color_ordering = thread_id % 2
        with tf.name_scope("distort_color", values=[image]):
            if color_ordering == 0:
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.032)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            elif color_ordering == 1:
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.032)
        image_summary("distort_color", image)

        with tf.name_scope("random_crop", values=[image]):
            image = tf.random_crop(image, [54, 54, 3])
            image = tf.div(image, 256.0)
            image = tf.multiply(tf.subtract(image, 0.5), 2)
        image_summary("final_image", image)

        return image

    @staticmethod
    def digits_process(digits, length):
        # equal to input_length = [length+1]
        input_length = tf.expand_dims(tf.add(length, 1), 0)

        input_seq = tf.slice(digits, [0], input_length)
        target_seq = tf.slice(digits, [1], input_length)
        indicator = tf.ones(input_length, dtype=tf.int32)

        return input_seq, target_seq, indicator
