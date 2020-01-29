# -*- coding: utf-8 -*-

# python imports
from __future__ import print_function
import numpy as np
import pickle
import warnings
from random import random

# keras imports
import keras
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Activation, Dropout
from keras.utils import to_categorical

# project imports
from saving import save_model, load_model
from features import get_features
from config import (TRAINER_BATCH_SIZE, TRAINER_EPOCHS, FER_MODEL_PATH, FER_DATASET_PATH,
                    FER_ALL_EMOTIONS, FER_SELECTED_EMOTIONS, FER_NEUTRAL_EFFECT, FER_USAGES)


class Trainer:

    def __init__(self):
        self.batch_size = TRAINER_BATCH_SIZE
        self.epochs = TRAINER_EPOCHS
        self.labels = FER_SELECTED_EMOTIONS

        self.x_train = []
        self.y_train = []
        self.x_dev = []
        self.y_dev = []
        self.x_test = []
        self.y_test = []

        self.model = None


    @property
    def x_all(self):
        if (self.x_dev == self.x_test).all():
            return np.concatenate([self.x_train, self.x_test])
        return np.concatenate([self.x_train, self.x_dev, self.x_test])


    @property
    def y_all(self):
        if (self.y_dev == self.y_test).all():
            return np.concatenate([self.y_train, self.y_test])
        return np.concatenate([self.y_train, self.y_dev, self.y_test])


    def train(self):
        # np.random.seed(1234)
        self.model = model = self._create_model()

        # checkpointer = keras.callbacks.ModelCheckpoint(
        #     FER_MODEL_PATH,
        #     save_best_only = True,
        #     monitor = 'val_acc',
        #     mode = 'auto',
        #     verbose = 1
        # )
        checkpointer = ModelCheckpoint(FER_MODEL_PATH, verbose=1)

        evaluator = ModelEvaluate(
            self.x_train, self.y_train,
            self.x_dev, self.y_dev,
            self.x_test, self.y_test,
            self.x_all, self.y_all
        )

        model.fit(
            self.x_train, self.y_train,
            batch_size=self.batch_size,
            epochs = self.epochs,
            validation_data = (self.x_dev, self.y_dev),
            callbacks = [checkpointer, evaluator],
            verbose = 1
        )


    def _create_model(self):
        model = Sequential()
        model.labels = self.labels

        model.add(Dense(256, input_dim=self.x_train.shape[1]))
        model.add(BatchNormalization())
        model.add(Activation('tanh'))
        model.add(Dropout(0.5))

        model.add(Dense(256))
        model.add(BatchNormalization())
        model.add(Activation('tanh'))
        model.add(Dropout(0.5))

        model.add(Dense(self.y_train.shape[1]))
        model.add(BatchNormalization())
        model.add(Activation('softmax'))

        model.compile(
            optimizer = keras.optimizers.RMSprop(),
            loss = keras.losses.categorical_crossentropy,
            metrics = ['accuracy']
        )

        model.summary()
        return model


    def load_data(self):
        self._clear_data()
        images, landmarks, details = pickle.load(open(FER_DATASET_PATH, "rb"))

        for i, img in enumerate(images):
            emotion = details[i][3]
            emotion_name = FER_ALL_EMOTIONS[emotion]
            if not emotion_name in self.labels:
                continue
            if emotion_name == 'neutral' and random() > FER_NEUTRAL_EFFECT:
                continue

            emotion = self.labels.index(emotion_name)
            usage = details[i][0]
            features = get_features(img, landmarks[i])

            if usage == FER_USAGES.index('train'):
                self.x_train.append(features)
                self.y_train.append(emotion)
            elif usage == FER_USAGES.index('test'):
                self.x_test.append(features)
                self.y_test.append(emotion)

        self.x_dev = self.x_test
        self.y_dev = self.y_test

        self.x_train = np.array(self.x_train)
        self.y_train = np.array(self.y_train)
        self.x_dev = np.array(self.x_dev)
        self.y_dev = np.array(self.y_dev)
        self.x_test = np.array(self.x_test)
        self.y_test = np.array(self.y_test)

        self.y_train = to_categorical(self.y_train, len(self.labels))
        self.y_dev = to_categorical(self.y_dev, len(self.labels))
        self.y_test = to_categorical(self.y_test, len(self.labels))


    def report_train(self):
        return self.report(self.x_train, self.y_train, 'Train')


    def report_dev(self):
        return self.report(self.x_dev, self.y_dev, 'Dev')


    def report_test(self):
        return self.report(self.x_test, self.y_test, 'Test')


    def report_all(self):
        return self.report(self.x_all, self.y_all, 'All')


    def report(self, x, y, title=''):
        model = load_model(FER_MODEL_PATH)
        y_predicted = model.predict(x)
        y_predicted = (y_predicted == y_predicted.max(axis=1)[:, None]).astype(np.int8)
        result = np.zeros((len(self.labels), len(self.labels)))

        for i in range(y.shape[0]):
            desired_emotion_index = np.where(y[i] == 1)[0][0]
            predicted_emotion_index = np.where(y_predicted[i] == 1)[0][0]
            result[desired_emotion_index][predicted_emotion_index] += 1

        row_format = "{:>15}" * (len(self.labels) + 1)
        s = '| ' + row_format.format("", *self.labels) + ' |\n'
        for emo, row in zip(self.labels, result):
            row_precent = ((row * 100.) / np.sum(row)) if row.any() else np.zeros_like(row)
            final_row = ["%d/%5.2f" % (row[i], row_precent[i]) for i in range(len(row))]
            s += '| ' + row_format.format(emo, *final_row) + ' |\n'

        table_width = len(s.split('\n')[0])
        header_text = " Report: %s " % title
        half_header_width = (table_width - len(header_text)) // 2
        header = ('-' * half_header_width) + header_text + ('-' * half_header_width)
        header += '-' * (table_width - len(header))
        footer = '=' * table_width
        s = header + '\n' + s + footer

        return s


    def _clear_data(self):
        self.x_train = []
        self.y_train = []
        self.x_dev = []
        self.y_dev = []
        self.x_test = []
        self.y_test = []



class ModelCheckpoint(keras.callbacks.Callback):

    def __init__(self, filepath, verbose=0, save_weights_only=False):
        super(ModelCheckpoint, self).__init__()
        self.filepath = filepath
        self.verbose = verbose
        self.save_weights_only = save_weights_only

        self.best_acc = -np.inf
        self.best_loss = np.inf
        self.best_val_acc = -np.inf
        self.best_val_loss = np.inf
        self.best_epoch = 0


    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        acc = logs.get('acc')
        loss = logs.get('loss')
        val_acc = logs.get('val_acc')
        val_loss = logs.get('val_loss')
        metrics = [acc, loss, val_acc, val_loss]

        if None in metrics:
            warnings.warn(
                'Can save best model only with acc, loss, val_acc, and val_loss available, skipping.',
                RuntimeWarning
            )
        else:
            # save_current = val_acc >= self.best_val_acc and \
            #     val_loss <= self.best_val_loss and \
            #     acc >= self.best_acc and \
            #     loss <= self.best_loss

            save_current = False
            if val_acc > self.best_val_acc:
                save_current = True
            elif val_acc == self.best_val_acc and val_loss < self.best_val_loss:
                save_current = True
            elif val_acc == self.best_val_acc and val_loss == self.best_val_loss and acc > self.best_acc:
                save_current = True
            elif val_acc == self.best_val_acc and val_loss == self.best_val_loss and acc == self.best_acc and loss < self.best_loss:
                save_current = True

            if save_current:
                self.best_acc = acc
                self.best_loss = loss
                self.best_val_acc = val_acc
                self.best_val_loss = val_loss
                self.best_epoch = epoch + 1

                filepath = self.filepath.format(epoch=epoch + 1, **logs)
                if self.verbose > 0:
                    print('Epoch %05d: saving model to %s' % (epoch + 1, filepath))

                save_model(self.model, filepath, overwrite=True, save_weights_only=self.save_weights_only)
            else:
                if self.verbose > 0:
                    print('Epoch %05d: Some metrics did not improve from epoch %05d' % (epoch + 1, self.best_epoch))



class ModelEvaluate(keras.callbacks.Callback):

    def __init__(self, x_train, y_train, x_dev, y_dev, x_test, y_test, x_all, y_all):
        super(ModelEvaluate, self).__init__()
        self.x_train = x_train
        self.y_train = y_train
        self.x_dev = x_dev
        self.y_dev = y_dev
        self.x_test = x_test
        self.y_test = y_test
        self.x_all = x_all
        self.y_all = y_all


    def on_epoch_end(self, epoch, logs=None):
        self._evaluate(self.model, self.x_test, self.y_test)
        print()


    def on_train_end(self, logs=None):
        model = load_model(FER_MODEL_PATH)
        print('Best result on train set:')
        self._evaluate(model, self.x_train, self.y_train)
        print('Best result on dev set:')
        self._evaluate(model, self.x_dev, self.y_dev)
        print('Best result on test set:')
        self._evaluate(model, self.x_test, self.y_test)
        print('Best result on all sets:')
        self._evaluate(model, self.x_all, self.y_all)


    def _evaluate(self, model, x, y):
        score = model.evaluate(x, y, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])



if __name__ == '__main__':
    t = Trainer()
    t.load_data()
    t.train()

    print(t.report_train())
    print(t.report_test())
    print(t.report_all())
