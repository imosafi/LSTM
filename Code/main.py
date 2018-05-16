import tensorflow as tf
import numpy as np
from pathlib import Path
import collections
from tensorflow.contrib import rnn
import pickle
import os
import datetime
import unicodedata
from tensorflow.python.client import device_lib
import matplotlib.pyplot as plt
import re
# from sklearn.utils import shuffle as shuffle
import random
import string
from FileManager import FileManager, TextType
from LSTM import LSTM
from model_configuration import *
import sys

def build_dictionaries(words):
    count = collections.Counter(words).most_common()
    char_to_number_dict = dict()
    for char, _ in count:
        char_to_number_dict[char] = len(char_to_number_dict)
    number_to_char_dict = dict(zip(char_to_number_dict.values(), char_to_number_dict.keys()))
    return char_to_number_dict, number_to_char_dict

print('Tensorflow Version: ' + tf.__version__)
load_model_flag = sys.argv[1]
print ("Load train model " , load_model_flag)

file_manager = FileManager()
training_data = file_manager.get_cleaned_text(text_type=TextType.TRAIN)
testing_data = file_manager.get_cleaned_text(text_type=TextType.TEST)
print('Number of characters for training: {}'.format(len(training_data)))
print('Number of characters for testing: {}'.format(len(testing_data)))


char_list = list(training_data)
char_to_number_dict, number_to_char_dict = build_dictionaries(char_list)
num_of_unique_chars = len(number_to_char_dict)
print('Number of unique characters: {}'.format(num_of_unique_chars))


text_as_numbers = []
for x in char_list:
    text_as_numbers.append(char_to_number_dict[x])
text_as_numbers = np.array(text_as_numbers)

print('Configuration : epocs %s layers %s hidden %s dropout %s' % (num_of_epochs , num_layers, n_hidden, use_droupout))

lstm_model = LSTM(n_hidden, num_layers, batch_size, learning_rate,
                        sequence_length, num_of_unique_chars)

with tf.variable_scope(tf.get_variable_scope(), reuse=True):
    test_lstm_model = LSTM(n_hidden, num_layers, batch_size, learning_rate,
                                 sequence_length, num_of_unique_chars, is_test_mode=True)

num_batches = int(len(text_as_numbers)/(batch_size * sequence_length)) + 1
# Split up text indices into subarrays, of equal size
batches = np.array_split(text_as_numbers, num_batches)
# Reshape each split into [batch_size, training_seq_len]
batches = [np.resize(x, [batch_size, sequence_length]) for x in batches]

skip_training = False


saver = tf.train.Saver(tf.global_variables())

# Initialize all variables

init = tf.global_variables_initializer()

train_loss = []
with tf.Session() as sess:

    if not load_model_flag:
        sess.run(init)
        pretrain_time = datetime.datetime.now()
        print('current time: ' + str(pretrain_time) + ' start training:')
    else:
        saver.restore(sess, file_manager.trained_model_path)
        skip_training = True
        print('trained model was loaded, skipped training')

    if not skip_training:
        for epoch in range(num_of_epochs):
            random.shuffle(batches)
            targets = [np.roll(x, -1, axis=1) for x in batches]
            print('Epoch {}/{}:'.format(epoch + 1, num_of_epochs))
            state = sess.run(lstm_model.initial_state)
            for j, batch in enumerate(batches):
                training_dict = {lstm_model.x_data: batch, lstm_model.y_output: targets[j]}
                for i, (c, h) in enumerate(lstm_model.initial_state):
                    training_dict[c] = state[i].c
                    training_dict[h] = state[i].h

                temp_loss, state, _ = sess.run([lstm_model.cost, lstm_model.final_state, lstm_model.train_op],
                                       feed_dict=training_dict)
                train_loss.append(temp_loss)

                if j % 20 == 0:
                    print('Epoch: {}, Batch: {}/{}, Loss: {:.2f}'.format(epoch + 1, j+1, num_batches + 1, temp_loss))

        if not load_model_flag:
            save_path = saver.save(sess, file_manager.trained_model_path)
            print('Done! training took: ' + str(datetime.datetime.now() - pretrain_time))

    pretrain_time = datetime.datetime.now()
    print('current time: ' + str(pretrain_time)+' start Testing:')
    accuracy, cross_entropy = test_lstm_model.test_model(sess, testing_data,  number_to_char_dict, char_to_number_dict)
    print('accuracy: {}, cross entropy: {}'.format(accuracy, cross_entropy))
    print('Done! testing took: ' + str(datetime.datetime.now() - pretrain_time))

    pretrain_time = datetime.datetime.now()
    print('current time: ' + str(pretrain_time)+' start generating data:')
    # play with the size of training data you use
    generated_text = test_lstm_model.generate_text(sess, testing_data[:50], number_to_char_dict, char_to_number_dict,
                                                   text_length_to_generate)
    #print(generated_text)
    print('Done! generating took: ' + str(datetime.datetime.now() - pretrain_time))

    file_manager.save_results(accuracy, cross_entropy, generated_text)


plt.plot(train_loss, 'k-')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.show()