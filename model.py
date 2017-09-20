import tensorflow as tf


class Model(object):

    def __init__(self,
                 vocab_size=12,
                 mode='train',
                 embedding_size=512,
                 num_lstm_units=16,
                 lstm_dropout_keep_prob=0.7,
                 cnn_drop_rate=0.2,
                 initializer_scale=0.08):
        super(Model, self).__init__()
        # global variable
        self.input_mask = None
        self.vocab_size = vocab_size
        self.mode = mode
        # CNN parameters
        self.embedding_size = embedding_size
        self.cnn_drop_rate = cnn_drop_rate
        # RNN parameters
        self.num_lstm_units = num_lstm_units
        self.lstm_dropout_keep_prob = lstm_dropout_keep_prob
        self.initializer_scale = initializer_scale
        self.initializer = tf.random_uniform_initializer(
            minval=-self.initializer_scale,
            maxval=self.initializer_scale)

    def cnn_layer(self, x):
        if self.mode == 'train':
            drop_rate = self.cnn_drop_rate
        else:  # inference or evaluate
            drop_rate = 0.0

        with tf.variable_scope('hidden1'):
            conv = tf.layers.conv2d(x, filters=48, kernel_size=[
                                    5, 5], padding='same')
            norm = tf.layers.batch_normalization(conv)
            activation = tf.nn.relu(norm)
            pool = tf.layers.max_pooling2d(
                activation, pool_size=[2, 2], strides=2, padding='same')
            dropout = tf.layers.dropout(pool, rate=drop_rate)
            hidden1 = dropout

        with tf.variable_scope('hidden2'):
            conv = tf.layers.conv2d(hidden1, filters=64, kernel_size=[
                                    5, 5], padding='same')
            norm = tf.layers.batch_normalization(conv)
            activation = tf.nn.relu(norm)
            pool = tf.layers.max_pooling2d(
                activation, pool_size=[2, 2], strides=1, padding='same')
            dropout = tf.layers.dropout(pool, rate=drop_rate)
            hidden2 = dropout

        with tf.variable_scope('hidden3'):
            conv = tf.layers.conv2d(hidden2, filters=128, kernel_size=[
                                    5, 5], padding='same')
            norm = tf.layers.batch_normalization(conv)
            activation = tf.nn.relu(norm)
            pool = tf.layers.max_pooling2d(
                activation, pool_size=[2, 2], strides=2, padding='same')
            dropout = tf.layers.dropout(pool, rate=drop_rate)
            hidden3 = dropout

        with tf.variable_scope('hidden4'):
            conv = tf.layers.conv2d(hidden3, filters=160, kernel_size=[
                                    5, 5], padding='same')
            norm = tf.layers.batch_normalization(conv)
            activation = tf.nn.relu(norm)
            pool = tf.layers.max_pooling2d(
                activation, pool_size=[2, 2], strides=1, padding='same')
            dropout = tf.layers.dropout(pool, rate=drop_rate)
            hidden4 = dropout

        with tf.variable_scope('hidden5'):
            conv = tf.layers.conv2d(hidden4, filters=192, kernel_size=[
                                    5, 5], padding='same')
            norm = tf.layers.batch_normalization(conv)
            activation = tf.nn.relu(norm)
            pool = tf.layers.max_pooling2d(
                activation, pool_size=[2, 2], strides=2, padding='same')
            dropout = tf.layers.dropout(pool, rate=drop_rate)
            hidden5 = dropout

        with tf.variable_scope('hidden6'):
            conv = tf.layers.conv2d(hidden5, filters=192, kernel_size=[
                                    5, 5], padding='same')
            norm = tf.layers.batch_normalization(conv)
            activation = tf.nn.relu(norm)
            pool = tf.layers.max_pooling2d(
                activation, pool_size=[2, 2], strides=1, padding='same')
            dropout = tf.layers.dropout(pool, rate=drop_rate)
            hidden6 = dropout

        with tf.variable_scope('hidden7'):
            conv = tf.layers.conv2d(hidden6, filters=192, kernel_size=[
                                    5, 5], padding='same')
            norm = tf.layers.batch_normalization(conv)
            activation = tf.nn.relu(norm)
            pool = tf.layers.max_pooling2d(
                activation, pool_size=[2, 2], strides=2, padding='same')
            dropout = tf.layers.dropout(pool, rate=drop_rate)
            hidden7 = dropout

        with tf.variable_scope('hidden8'):
            conv = tf.layers.conv2d(hidden7, filters=192, kernel_size=[
                                    5, 5], padding='same')
            norm = tf.layers.batch_normalization(conv)
            activation = tf.nn.relu(norm)
            pool = tf.layers.max_pooling2d(
                activation, pool_size=[2, 2], strides=1, padding='same')
            dropout = tf.layers.dropout(pool, rate=drop_rate)
            hidden8 = dropout

        flatten = tf.reshape(hidden8, [-1, 4 * 4 * 192])

        with tf.variable_scope('hidden9'):
            dense = tf.layers.dense(flatten, units=3072, activation=tf.nn.relu)
            hidden9 = dense

        with tf.variable_scope('hidden10'):
            dense = tf.layers.dense(hidden9, units=3072, activation=tf.nn.relu)
            hidden10 = dense

        with tf.variable_scope('image_embedding') as scope:
            initializer = self.initializer
            image_embeddings = tf.contrib.layers.fully_connected(
                inputs=hidden10,
                num_outputs=self.embedding_size,#512
                activation_fn=None,
                weights_initializer=initializer,
                biases_initializer=None,
                scope=scope)

        return image_embeddings

    def build_seq_embeddings(self, input_seqs):
        with tf.variable_scope('seq_embedding'), tf.device('/cpu:0'):
            initializer = self.initializer

            embedding_map = tf.get_variable(
                name='map',
                shape=[self.vocab_size, self.embedding_size],
                initializer=initializer)
        seq_embeddings = tf.nn.embedding_lookup(embedding_map, input_seqs)

        return seq_embeddings

    def last_fully_connected(self, lstm_outputs):
        initializer = self.initializer
        with tf.variable_scope('logits') as logits_scope:
            logits = tf.contrib.layers.fully_connected(
                inputs=lstm_outputs,
                num_outputs=self.vocab_size,
                activation_fn=None,
                weights_initializer=initializer,
                scope=logits_scope)
        return logits

    def rnn_layer(self, inputs_embeddings):
        lstm_dropout_keep_prob = self.lstm_dropout_keep_prob  # 0.7
        num_lstm_units = self.num_lstm_units  # 64
        initializer = self.initializer

        lstm_cell = tf.contrib.rnn.BasicLSTMCell(
            num_units=num_lstm_units, state_is_tuple=True)
        image_embeddings = inputs_embeddings['images']
        seq_embeddings = inputs_embeddings['seq']

        if self.mode == 'train':
            lstm_cell = tf.contrib.rnn.DropoutWrapper(
                lstm_cell,
                input_keep_prob=lstm_dropout_keep_prob,
                output_keep_prob=lstm_dropout_keep_prob)

        with tf.variable_scope("lstm", initializer=self.initializer) as lstm_scope:

            zero_state = lstm_cell.zero_state(
                batch_size=image_embeddings.get_shape()[0], dtype=tf.float32)
            _, initial_state = lstm_cell(image_embeddings, zero_state)

            lstm_scope.reuse_variables()

            if self.mode == 'train' or self.mode == 'evaluate':
                input_mask = self.input_mask
                sequence_length = tf.reduce_sum(input_mask, 1)
                lstm_outputs, _ = tf.nn.dynamic_rnn(
                    cell=lstm_cell,
                    inputs=seq_embeddings,
                    sequence_length=sequence_length,
                    initial_state=initial_state,
                    dtype=tf.float32,
                    scope=lstm_scope)

            else:  # inference
                tf.concat(axis=1, values=initial_state, name="initial_state")
                state_feed = tf.placeholder(dtype=tf.float32,
                                            shape=[
                                                None, sum(lstm_cell.state_size)],
                                            name="state_feed")
                state_tuple = tf.split(
                    value=state_feed, num_or_size_splits=2, axis=1)

                lstm_outputs, state_tuple = lstm_cell(
                    inputs=tf.squeeze(seq_embeddings, axis=[1]),
                    state=state_tuple)
                tf.concat(axis=1, values=state_tuple, name="state")

            lstm_outputs = tf.reshape(
                lstm_outputs, [-1, lstm_cell.output_size])

        logits = self.last_fully_connected(lstm_outputs)

        return logits

    def inference(self, input_image, input_seqs, input_mask=None):
        self.input_mask = input_mask

        image_embeddings = self.cnn_layer(input_image)
        seq_embeddings = self.build_seq_embeddings(input_seqs)
        inputs_embeddings = {'images': image_embeddings,
                             'seq': seq_embeddings}
        logits = self.rnn_layer(inputs_embeddings)

        return logits

    def build(self):
        input_feed = tf.placeholder(dtype=tf.int64,
                                    shape=[None],  # batch_size
                                    name="input_feed")
        input_seqs = tf.expand_dims(input_feed, 1)

        image = tf.placeholder(dtype=tf.float32,
                                    shape=[54, 54, 3], 
                                    name="image_feed")
        input_image = tf.reshape(image, [1, 54, 54, 3])

        image_embeddings = self.cnn_layer(input_image)
        seq_embeddings = self.build_seq_embeddings(input_seqs)
        inputs_embeddings = {'images': image_embeddings,
                             'seq': seq_embeddings}
        logits = self.rnn_layer(inputs_embeddings)
        tf.nn.softmax(logits, name="softmax")

    def loss(self, logits, target_seqs):

        input_mask = self.input_mask
        targets = tf.reshape(target_seqs, [-1])
        weights = tf.to_float(tf.reshape(input_mask, [-1]))

        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=targets, logits=logits)
        batch_loss = tf.div(
            tf.reduce_sum(tf.multiply(losses, weights)),
            tf.reduce_sum(weights),
            name='batch_loss')

        tf.losses.add_loss(batch_loss)
        total_loss = tf.losses.get_total_loss()

        # Add summaries.
        tf.summary.scalar('losses/batch_loss', batch_loss)
        tf.summary.scalar('losses/total_loss', total_loss)
        for var in tf.trainable_variables():
            tf.summary.histogram('parameters/' + var.op.name, var)

        return total_loss
