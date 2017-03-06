import tensorflow as tf
import numpy as np
from bregman.suite import *

filenames = tf.train.match_filenames_once('/Users/Nick/Music/iTunes/iTunes Media/Music/mixMatch/*')

count_num_files = tf.size(filenames)
filename_queue = tf.train.string_input_producer(filenames)
reader  = tf.WholeFileReader()
filename, file_contents = reader.read(filename_queue)

chromo = tf.placeholder(tf.float32)
max_freqs = tf.argmax(chromo, 0)

def get_next_chromogram(sess):
    audio_file = sess.run(filename)
    F = Chromogram(audio_file, nfft=16384, wfft=8192, nhop=2205)
    return F.X

def extract_feature_vector(sess, chromo_data):
    num_features, num_samples = np.shape(chromo_data)
    freq_vals = sess.run(max_freqs, feed_dict={chromo: chromo_data})
    hist, bins = np.histogram(freq_vals, bins=range(num_features + 1))
    return  hist.astype(float) / num_samples

def get_dataset(sess):
    num_files = sess.run(count_num_files)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    xs = []
    for i in range(num_files):
        chromo_data = get_next_chromogram(sess)
        x = [extract_feature_vector(sess, chromo_data)]
        x = np.matrix(x)
        if len(xs) == 0:
            xs = x
        else:
            xs = np.vstack((xs, x))
        return xs
