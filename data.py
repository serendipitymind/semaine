import csv
import numpy as np
import random
import glob
import os.path
import sys
import operator
import threading
from keras.utils import to_categorical
from keras.preprocessing.image import img_to_array, load_img
# from extractor import Extractor
from tqdm import tqdm
import pandas as pd
import time
import tensorflow as tf
# import videoto3d


class threadsafe_iterator:
    def __init__(self, iterator):
        self.iterator = iterator
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return next(self.iterator)


def threadsafe_generator(func):
    """ Decorator """
    def gen(*a, **kw):
        return threadsafe_iterator(func(*a, **kw))
    return gen


class Dataset():
    def __init__(self,  image_size=(96, 96), frames=20, model='audio'):
        # Get the data
        self.max_features_text = 100000
        self.max_len_text = 100
        self.audio_max_sequence_length = 683
        self.data = self.get_data()
        self.image_shape = image_size
        self.train, self.validate, self.test = self.split_train_test()              # will del data
        self.image_frame = frames
        # self.vid3d = videoto3d.Videoto3D(image_size[0], image_size[1], frames)
        self.model = model
        self.image_max_sequence_length = frames


    # @staticmethod
    def get_data(self):
        """ Load data from csv file"""
        # data:  0: train/validate/test  1: video_path   2: audio_path   3: label
        data = pd.read_csv(os.path.join('data', 'data_file.csv')).values
        df = pd.read_csv(os.path.join('data', 'data_file.csv'))

        # with open(os.path.join('data', 'data_file.csv'), 'r') as fin:
        #     reader = csv.reader(fin)
        #     data = list(reader)
        scipt = [item[0] for item in pd.DataFrame(df['transcriptions']).values]
        from keras.preprocessing import text, sequence
        tokenizer = text.Tokenizer(num_words=self.max_features_text)
        tokenizer.fit_on_texts(list(scipt))
        list_tokenized = tokenizer.texts_to_sequences(scipt)
        X_t = sequence.pad_sequences(list_tokenized, maxlen=self.max_len_text)
        self.word_index = tokenizer.word_index
        X_t = X_t[:, np.newaxis, :]
        # for index in range(len(X_t)):
        #     X_t[index] = X_t[index].tolist()
        data = np.concatenate((data[:, :, np.newaxis], X_t), axis=1)
        return data

    @staticmethod
    def get_frames_for_sample(sample):
        """Given a sample row from the data file, get all the corresponding frame
        filenames."""
        def sort_by_number(elem):
            a = str(elem).split('/')[-1][:-4]
            return int(a)
        each_video_save_full_path = os.path.dirname(sample[1])
        list_picture = [os.path.join(each_video_save_full_path, item) for item in os.listdir(each_video_save_full_path)
                        if item.endswith('.jpg')]
        list_picture.sort(key=sort_by_number)
        return list_picture

    @staticmethod
    def rescale_list(input_list, size):
        """Given a list and a size, return a rescaled/samples list. For example,
        if we want a list of size 5 and we have a list of size 25, return a new
        list of size five which is every 5th element of the origina list."""
        assert len(input_list) >= size

        # Get the number to skip between iterations.
        skip = len(input_list) // size

        # Build our new output.
        output = [input_list[i] for i in range(0, len(input_list), skip)]

        # Cut off the last one if needed.
        return output[:size]

    def opensmile_features(self, wav_file):
        name = os.path.basename(wav_file)[:-4] + '.csv'
        out_file = os.path.join(os.path.dirname(wav_file), name)
        pathExcuteFile = os.path.join(sys.path[0], "openSMILE-2.1.0/inst/bin/SMILExtract ")
        pathConfig = os.path.join(sys.path[0], 'config/emobase2010.conf')
        if not os.path.exists(out_file):
            cmd = pathExcuteFile + " -C " + pathConfig + " -I " + wav_file + " -O " + out_file
            os.system(cmd)
            time.sleep(0.2)
        csv_file = pd.read_csv(filepath_or_buffer=out_file, sep=';')
        data = csv_file.values
        (l, r) = data.shape
        audio_frame_number = l
        while l < self.audio_max_sequence_length:
            data = np.row_stack((data, np.zeros([r])))
            (l, _) = data.shape
        return audio_frame_number, data

    def opensmile_features2(self, wav_file):
        name = os.path.basename(wav_file)[:-4] + '_988.csv'
        out_file = os.path.join(os.path.dirname(wav_file), name)
        pathExcuteFile = os.path.join(sys.path[0], "openSMILE-2.1.0/inst/bin/SMILExtract ")
        pathConfig = os.path.join(sys.path[0], 'config/emobase.conf')
        if not os.path.exists(out_file):
            cmd = pathExcuteFile + " -C " + pathConfig + " -I " + wav_file + " -O " + out_file
            os.system(cmd)
            time.sleep(0.2)
        csv_file = pd.read_csv(filepath_or_buffer=out_file, sep=';')
        data = csv_file.values
        data = np.array(data[-1][0].split(',')[2:-1], dtype=np.float32)
        return data

    def split_train_test(self):
        import gc
        train = []
        validate =[]
        test = []
        for item in self.data:
            if item[0] == 'train':
                train.append(item)
            elif item[0] == 'validate':
                validate.append(item)
            elif item[0] == 'test':
                test.append(item)
        del self.data
        gc.collect()
        return train, validate, test

    @threadsafe_generator
    def frame_generator(self, batch_size,  train_validate, color=False, skip=True):
        """Return a generator that we can use to train on. There are
               a couple different things we can return:

               data_type: 'features', 'images'
               """
        # Get the right dataset for the generator.
        data = self.train                   # default
        if train_validate == 'train':
            data = self.train
            print("creating {} generator with {} samples".format(train_validate, len(data)))
        elif train_validate == 'validate':
            data = self.validate
            print("creating {} generator with {} samples".format(train_validate, len(data)))
        elif train_validate == 'test':
            data = self.test
            print("creating {} generator with {} samples".format(train_validate, len(data)))
            for sample in data:
                x, y, audio_f = [], [], []
                if self.model == 'image' or 'both':
                    video = self.vid3d.video3d(sample[1], color=color, skip=skip)
                    # fr = self.get_frames_for_sample(sample)
                    # video = self.build_image_sequence(fr)
                    x.append(video)
                if self.model == 'audio' or 'both':
                    _, audio_features = self.opensmile_features(sample[2])
                    audio_f.append(audio_features)

                ca = sample[3]
                labels_dic = {"anger": 0, "disgust": 1, "fear": 2, "happiness": 3, "sadness": 4,
                              "surprise": 5}  ##### 注意这里的移植性
                label = np.zeros(shape=(len(labels_dic)), dtype=np.int)
                label[labels_dic[ca]] = 1
                y.append(label)
                if self.model == 'image':
                    yield np.array(x), np.array(y)
                elif self.model == 'audio':
                    yield np.array(audio_f), np.array(y)                        # audio : np.array(audio_f), np.array(y)
                else:
                    yield [np.array(x), np.array(audio_f)], np.array(y)
        if not train_validate == 'test':
            while 1:
                x, y, audio_f = [], [], []
                # generate batch_size samples
                for _ in range(batch_size):
                    sample = random.choice(data)
                    if self.model == 'image' or 'both':
                        video = self.vid3d.video3d(sample[1], color=color, skip=skip)
                        # fr = self.get_frames_for_sample(sample)
                        # video = self.build_image_sequence(fr)
                        x.append(video)
                    if self.model == 'audio' or 'both':
                        _, audio_features = self.opensmile_features(sample[2])
                        # audio_features = self.opensmile_features2(sample[2])
                        audio_f.append(audio_features)
                    ca = sample[3]
                    labels_dic = {"anger": 0, "disgust": 1, "fear": 2, "happiness": 3, "sadness": 4, "surprise": 5}         ##### 注意这里的移植性
                    label = np.zeros(shape=(len(labels_dic)), dtype=np.int)
                    label[labels_dic[ca]] = 1
                    y.append(label)
                # if color:
                #     # x = np.array(x).transpose((0, 2, 3, 4, 1))
                #     pass
                # else:
                #     x = np.array(x).transpose((0, 2, 3, 1))
                if self.model == 'image':
                    yield np.array(x), np.array(y)
                elif self.model == 'audio':
                    yield np.array(audio_f), np.array(y)                        # audio : np.array(audio_f), np.array(y)
                else:
                    yield [np.array(x), np.array(audio_f)], np.array(y)

    def get_extracted_sequence(self, data_type, sample):
        """Get the saved extracted features."""
        filename = sample[1][:-4]
        path = filename + '-' + str(self.image_max_sequence_length) + '-' + data_type + '.npy'
        if os.path.isfile(path):
            return np.load(path)
        else:
            return None

    def build_image_sequence(self, frames):
        """Given a set of frames (filenames), build our sequence."""
        def process_image(image, target_shape):
            """Given an image, process it and return the array."""
            # Load the image.
            h, w = target_shape
            image = load_img(image, target_size=(h, w))

            # Turn it into numpy, normalize and return.
            img_arr = img_to_array(image)
            x = (img_arr / 255.).astype(np.float32)
            return x
        a = [process_image(x, self.image_shape) for x in frames]
        while len(a) < self.image_max_sequence_length:
            a.append(np.zeros(shape=(self.image_shape[0], self.image_shape[1], 3)))
        return a

    def extract_image_features(self):
        extract_features_model = Extractor()
        pbar = tqdm(total=len(self.data))
        for sample in self.data:
            filename = sample[1][:-4]
            path = filename + '-' + str(self.image_max_sequence_length) + '-' + 'features' + '.npy'
            if os.path.isfile(path + '.npy'):
                pbar.update(1)
                continue
            # Get the frames for this video.
            # print(sample[1])
            frames = self.get_frames_for_sample(sample)
            # frames = self.rescale_list(frames, self.seq_length)
            # Now loop through and extract features to build the sequence.
            sequence = []
            features_dim = 0
            for image in frames:
                features = extract_features_model.extract(image)
                features_dim = len(features)
                sequence.append(features)
            while len(sequence) < self.image_max_sequence_length:
                sequence.append(np.zeros(shape=(features_dim,)))
            # Save the sequence.
            np.save(path, sequence)
            pbar.update(1)
        pbar.close()

    def extract_audio_features(self):
        print("extracting audio features")
        pbar = tqdm(total=len(self.data))
        for item in self.data:
            wav_file = item[2]
            name = os.path.basename(wav_file)[:-4] + '.csv'
            out_file = os.path.join(os.path.dirname(wav_file), name)
            if os.path.exists(out_file):
               continue
            pathExcuteFile = os.path.join(sys.path[0], "openSMILE-2.1.0/inst/bin/SMILExtract ")
            pathConfig = os.path.join(sys.path[0], 'config/emobase2010.conf')
            cmd = pathExcuteFile + " -C " + pathConfig + " -I " + wav_file + " -O " + out_file
            os.system(cmd)
            time.sleep(0.2)
            csv_file = pd.read_csv(filepath_or_buffer=out_file, sep=';')
            data = csv_file.values
            (l, r) = data.shape
            self.audio_max_sequence_length = max((l, self.audio_max_sequence_length))
            pbar.update(1)
        pbar.close()
        print("audio_max_sequence_length : {}".format(self.audio_max_sequence_length))

    def generate_tfrecorder(self):
        def _int64_feature(value):
            return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

        def _bytes_feature(value):
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

        def write_data(data, name_of_data):
            name = os.path.join('data', '{}.tfrecoreder'.format(name_of_data))
            writer = tf.python_io.TFRecordWriter(name)
            pbar = tqdm(total=len(data))
            for sample in data:
                frames = self.get_frames_for_sample(sample)
                # frames = self.rescale_list(frames, self.seq_length)
                image_real_sequence_length = len(frames)
                if self.is_raw_image:
                    # Build the image sequence
                    sequence = self.build_image_sequence(frames)
                else:
                    # Get the sequence from disk.
                    sequence = self.get_extracted_sequence('features', sample)
                    if sequence is None:
                        raise ValueError("Can't find {} sequence. Did you generate them?".format(sample[1]))
                audio_frame_num, audio_features = self.opensmile_features(sample[2])
                ca = sample[3]
                labels_dic = {"anger": 0, "disgust": 1, "fear": 2, "happiness": 3, "sadness": 4,
                              "surprise": 5}  ##### 注意这里的移植性
                label = labels_dic[ca]
                image_raw = np.array(sequence, dtype=np.float32).tostring()
                audio_raw = np.array(audio_features,  dtype=np.float32).tostring()
                example = tf.train.Example(features=tf.train.Features(feature={
                    'image': _bytes_feature(image_raw),
                    'audio': _bytes_feature(audio_raw),
                    'label': _int64_feature(label),
                    'audio_real_sequence_length': _int64_feature(audio_frame_num),
                    'image_real_sequence_length': _int64_feature(image_real_sequence_length),
                }))
                writer.write(example.SerializeToString())
                pbar.update(1)
            pbar.close()
            writer.close()

        train, validate, test = self.split_train_test()
        write_data(train, 'train')
        write_data(validate, 'validate')
        write_data(test, 'test')

    def get_tf_data(self, split_name='train', is_training=True, num_epochs=50, batch_size=32):
        if split_name not in ('train', 'validate', 'test'):
            raise ValueError("Unexpected splite_name !!! ")
        path = os.path.join('data', '{}.tfrecoreder'.format(split_name))
        filename_queue = tf.train.string_input_producer([path], num_epochs=num_epochs, shuffle=is_training)
        # if split_name == 'train':
        #     path1 = os.path.join('data', '{}.tfrecoreder'.format('train'))
        #     path2 = os.path.join('data', '{}.tfrecoreder'.format('validate'))
        #     filename_queue = tf.train.string_input_producer([path1, path2], num_epochs=num_epochs, shuffle=is_training)
        # else:
        #     path2 = os.path.join('data', '{}.tfrecoreder'.format('test'))
        #     filename_queue = tf.train.string_input_producer([path2], num_epochs=num_epochs, shuffle=is_training)
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(
            serialized_example,
            features={
                'label': tf.FixedLenFeature([], tf.int64),
                'image_real_sequence_length': tf.FixedLenFeature([], tf.int64),
                'audio_real_sequence_length': tf.FixedLenFeature([], tf.int64),
                'audio': tf.FixedLenFeature([], tf.string),
                'image': tf.FixedLenFeature([], tf.string),
            }
        )
        label = tf.cast(features['label'], tf.int64)
        real_audio_sequence_length = tf.cast(features['audio_real_sequence_length'], tf.int64)
        real_image_sequence_length = tf.cast(features['image_real_sequence_length'], tf.int64)
        audio_feature = tf.decode_raw(features['audio'], tf.float32)
        audio_feature = tf.reshape(audio_feature, (self.audio_max_sequence_length, self.audio_feature_dim))
        image = tf.decode_raw(features['image'], tf.float32)
        if not self.is_raw_image:
            image = tf.reshape(image, (self.image_max_sequence_length, self.image_feature_dim))
        else:
            image = tf.reshape(image, (self.image_max_sequence_length, self.image_shape[0], self.image_shape[1], self.image_shape[2]))
        # capacity=(min_after_dequeue+(num_threads+a small safety margin∗batchsize)
        min_after_dequeue = 200
        capacity = min_after_dequeue + (4 + 10 * batch_size)
        if is_training:
            labels, audio_real_sequence_lengths, image_real_sequence_lengths, audio_features, images = tf.train.shuffle_batch(
                [label, real_audio_sequence_length, real_image_sequence_length, audio_feature, image], batch_size, capacity, min_after_dequeue, num_threads=4)
        else:
            labels, audio_real_sequence_lengths, image_real_sequence_lengths, audio_features, images = tf.train.batch(
                [label, real_audio_sequence_length,real_image_sequence_length, audio_feature, image], batch_size, num_threads=1, capacity=capacity)
        return labels, audio_real_sequence_lengths, image_real_sequence_lengths, audio_features, images