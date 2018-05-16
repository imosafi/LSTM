import tensorflow as tf
import numpy as np
import math
from model_configuration import use_droupout
class LSTM:

    def __init__(self, num_units, num_layers, batch_size, learning_rate,
                 training_seq_len, vocab_size, is_test_mode=False):
        self.num_units, self.num_layers, self.vocab_size, self.learning_rate = (num_units, num_layers, vocab_size,
                                                                                learning_rate)
        self.batch_size, self.training_seq_len = (batch_size, training_seq_len) if not is_test_mode else (1, 1)

        self.lstm_cell = tf.contrib.rnn.MultiRNNCell([self.get_lstm_cell(use_dropout=use_droupout) for _ in range(self.num_layers)])
        self.initial_state = self.lstm_cell.zero_state(self.batch_size, tf.float32)

        self.x_data = tf.placeholder(tf.int32, [self.batch_size, self.training_seq_len])
        self.y_output = tf.placeholder(tf.int32, [self.batch_size, self.training_seq_len])

        with tf.variable_scope('model_variables'):
            # Softmax Output Weights
            weights = tf.get_variable('weights', [self.num_units, self.vocab_size], tf.float32, tf.random_normal_initializer())
            bias = tf.get_variable('bias', [self.vocab_size], tf.float32, tf.constant_initializer(0.0))

            embedding_matrix = tf.get_variable('embedding_matrix', [self.vocab_size, self.num_units],
                                            tf.float32, tf.random_normal_initializer())

            embedding_output = tf.nn.embedding_lookup(embedding_matrix, self.x_data)
            rnn_inputs = tf.split(axis=1, num_or_size_splits=self.training_seq_len, value=embedding_output)
            rnn_inputs_trimmed = [tf.squeeze(x, [1]) for x in rnn_inputs]

        outputs, last_state = tf.contrib.legacy_seq2seq.rnn_decoder(rnn_inputs_trimmed, self.initial_state, self.lstm_cell)

        # RNN outputs
        output = tf.reshape(tf.concat(axis=1, values=outputs), [-1, self.num_units])
        # Logits and output
        self.logit_output = tf.matmul(output, weights) + bias
        self.model_output = tf.nn.softmax(self.logit_output)

        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([self.logit_output], [tf.reshape(self.y_output, [-1])],
                        [tf.ones([self.batch_size * self.training_seq_len])],
                        self.vocab_size)
        self.cost = tf.reduce_sum(loss) / (self.batch_size * self.training_seq_len)
        self.final_state = last_state
        gradients, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tf.trainable_variables()), 4.5)
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = optimizer.apply_gradients(zip(gradients, tf.trainable_variables()))

    def get_lstm_cell(self, use_dropout=use_droupout, dropout_rate=0.1):
        if use_dropout:
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.num_units)
            return tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=1 - dropout_rate)
        else:
            return tf.contrib.rnn.BasicLSTMCell(self.num_units)

    def test_model(self, sess, text, number_to_char_dict, char_to_number_dict):
        state = sess.run(self.lstm_cell.zero_state(1, tf.float32))
        char_list = list(text)
        correct_predications = 0
        cross_entropy = 0
        for i in range(len(char_list) - 1):
            char = char_list[i]
            x = np.zeros((1, 1))
            x[0, 0] = char_to_number_dict[char]
            feed_dict = {self.x_data: x, self.initial_state: state}
            [model_output, state] = sess.run([self.model_output, self.final_state], feed_dict=feed_dict)
            sample = np.argmax(model_output[0])
            out_letter = number_to_char_dict[sample]
            if out_letter == char_list[i + 1]:
                correct_predications += 1
            probability_index = char_to_number_dict[char_list[i + 1]]
            prob = model_output[0][probability_index]
            cross_entropy = cross_entropy - math.log(prob, 2)
        return (correct_predications / len(char_list)), cross_entropy / len(char_list) - 1

    def generate_text(self, sess, pregenerate_text, number_to_char_dict, char_to_number_dict, num_to_generate):
        state = sess.run(self.lstm_cell.zero_state(1, tf.float32))
        char_list = list(pregenerate_text)
        for char in char_list[:-1]:
            x = np.zeros((1, 1))
            x[0, 0] = char_to_number_dict[char]
            feed_dict = {self.x_data: x, self.initial_state: state}
            [state] = sess.run([self.final_state], feed_dict=feed_dict)
        generated_text = ''
        char = char_list[-1]
        for n in range(num_to_generate):
            x = np.zeros((1, 1))
            x[0, 0] = char_to_number_dict[char]
            feed_dict = {self.x_data: x, self.initial_state: state}
            [model_output, state] = sess.run([self.model_output, self.final_state], feed_dict=feed_dict)
            char_index = np.random.choice(np.arange(self.vocab_size), p=model_output[0])
            # sample = np.argmax(model_output[0])
            char = number_to_char_dict[char_index]
            generated_text = generated_text + char
        return (generated_text)