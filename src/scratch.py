#!/usr/bin/python
from keras.layers import Embedding
import pandas as pd
import gzip
import os
import numpy as np
import re
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.externals import joblib
from keras import layers
from keras.models import Sequential, Model
from keras.layers.embeddings import Embedding
from keras import regularizers
from keras.layers import Activation, Dense, Permute, Bidirectional, Dropout, Conv1D, MaxPooling1D,GlobalMaxPooling1D, dot, concatenate
from keras.layers import LSTM
from customlayers import negexp_manhattan, NegExpManhattan, absdiff, AbsDiff
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping

DATA_DIR = os.path.abspath(os.path.dirname(os.path.realpath(__file__))+"/../data/")
EMBEDDING_DIM=300
"""
https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html

"""
MAX_SEQUENCE_LENGTH = 37
BATCH_SIZE = 32
EPOCHS = 10

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^A-Za-z0-9^,!.\'+-=]", " ", text)
    text = re.sub(r"\'s", " 's ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", " cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text

def load_kaggle_data(path):
    data = pd.read_csv(path, sep=",", quotechar='"')
    data.question1 = data.question1.astype(str).apply(clean_text)
    data.question2 = data.question2.astype(str).apply(clean_text)
    return data

def load_and_checkpoint_tokenized_train_test():
    train_q1_pkl = os.path.join(DATA_DIR, 'train_q1.pkl')
    train_q2_pkl = os.path.join(DATA_DIR, 'train_q2.pkl')
    true_labels_pkl = os.path.join(DATA_DIR, 'labels.pkl')
    test_q1_pkl = os.path.join(DATA_DIR, 'test_q1.pkl')
    test_q2_pkl = os.path.join(DATA_DIR, 'test_q2.pkl')
    word_index_pkl = os.path.join(DATA_DIR, 'word_idx.pkl')
    if any([not os.path.isfile(f) for f in [train_q1_pkl, train_q2_pkl, test_q1_pkl, test_q2_pkl,
                                            true_labels_pkl, word_index_pkl]]):
        train = load_kaggle_data(os.path.join(DATA_DIR, 'train.csv'))
        test = load_kaggle_data(os.path.join(DATA_DIR, 'test.csv'))
        train_q1_q2_test_q1_q2 = train.question1.astype(str).tolist() + \
                                 train.question2.astype(str).tolist() + \
                                 test.question1.astype(str).tolist() + \
                                 test.question2.astype(str).tolist()
        tokenizer = Tokenizer(num_words=200000)
        print("Fitting tokenizer")
        tokenizer.fit_on_texts(train_q1_q2_test_q1_q2)
        print("Storing tokenizer")
        print("texts to sequences")
        sequences = tokenizer.texts_to_sequences(train_q1_q2_test_q1_q2)
        word_index = tokenizer.word_index
        print('Found %s unique tokens.' % len(word_index))
        print("Padding sequences")
        data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
        print("Splitting train/test data")
        train_q1_q2 = data[:len(train)*2]
        true_labels = train['is_duplicate'].tolist()
        test_q1_q2 = data[-(len(test)*2):]
        train_q1 = train_q1_q2[:len(train)]
        train_q2 = train_q1_q2[-len(train):]
        test_q1 = test_q1_q2[:len(test)]
        test_q2 = test_q1_q2[-len(test):]
        assert train_q1.shape[0] == len(true_labels)
        assert train_q2.shape[0] == len(true_labels)
        assert test_q1.shape[0] == len(test)
        assert test_q2.shape[0] == len(test)
        joblib.dump(true_labels, true_labels_pkl)
        joblib.dump(word_index, word_index_pkl)
        joblib.dump(train_q1, train_q1_pkl)
        joblib.dump(train_q2, train_q2_pkl)
        joblib.dump(test_q1, test_q1_pkl)
        joblib.dump(test_q2, test_q2_pkl)
    else:
        true_labels = joblib.load(true_labels_pkl)
        train_q1 = joblib.load(train_q1_pkl)
        train_q2 = joblib.load(train_q2_pkl)
        word_index = joblib.load(word_index_pkl)
        test_q1 = joblib.load(test_q1_pkl)
        test_q2 = joblib.load(test_q2_pkl)

    return train_q1, train_q2, true_labels, test_q1, test_q2, word_index

def load_embeddings(word_index):
    print("load embeddings")
    embeddings_index = {}
    with gzip.open(os.path.join(DATA_DIR, 'glove.840B.300d.txt.gz'), 'rt') as f:
        for line in f:
            try:
                values = line.split()
                word = values[0]
                if word in word_index:
                    coefs = np.asarray(values[1:], dtype='float32')
                    embeddings_index[word] = coefs
            except Exception as e:
                print("ERROR PROCESSING LINE: {}\n{}".format(line,e))
    embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
    print('Found %s word vectors.' % len(embeddings_index))

    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector


    embedding_layer = Embedding(len(word_index) + 1,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)
    return embedding_layer


def pop_layer(model):
    if not model.outputs:
        raise Exception('Sequential model cannot be popped: model is empty.')

    model.layers.pop()
    if not model.layers:
        model.outputs = []
        model.inbound_nodes = []
        model.outbound_nodes = []
    else:
        model.layers[-1].outbound_nodes = []
        try:
            model.outputs = [model.layers[-1].output]
        except:
            pass
    model.built = False


def retrain_last_output_layer_as_lr(prev_model, train_q1, train_q2, true_labels, redo_model=False):
    model_fit_h5 = DATA_DIR + "/mueller_2016_with_refined_output.h5"
    if os.path.exists(model_fit_h5) and not redo_model:
        model = load_model(model_fit_h5, custom_objects={'NegExpManhattan': NegExpManhattan,
                                                         "AbsDiff": AbsDiff})
    else:
        # pop layers off prev_model
        base_model = prev_model
        pop_layer(base_model)
        pop_layer(base_model)

        for l in base_model.layers:
            l.trainable = False

        single_branch_model = base_model.layers[-1]
        for l in single_branch_model.layers:
            l.trainable = False
        ## Inputs
        question1 = layers.Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
        question2 = layers.Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')

        m1 = single_branch_model(question1)
        m2 = single_branch_model(question2)


        #output_layers
        mmult = layers.multiply([m1, m2])
        mabs = absdiff([m1, m2])

        merged = layers.concatenate([mmult, mabs])
        merged = Dropout(0.2)(merged)
        merged = Dense(20, activation='relu')(merged)
        merged = Dropout(0.2)(merged)
        predictions = Dense(1, activation='sigmoid')(merged)
        model = Model([question1, question2], predictions)
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        print('Training')
        save_best_weights = 'refine_weights.h5'

        callbacks = [ModelCheckpoint(save_best_weights, monitor='val_loss', save_best_only=True),
                     EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')]

        model.fit([train_q1, train_q2],
                  true_labels,
                  batch_size=32,
                  epochs=50,
                  validation_split=0.1,
                  verbose=True,
                  shuffle=True,
                  callbacks=callbacks)
        model.load_weights(save_best_weights)
        model.save(model_fit_h5, overwrite=True)
    return model

def train_mueller_2016(train_q1, train_q2, true_labels,
                              embedding_layer, redo_model=False):
    # http://www.mit.edu/~jonasm/info/MuellerThyagarajan_AAAI16.pdf
    model_fit_h5 = DATA_DIR + "/mueller_2016.h5"
    if os.path.exists(model_fit_h5) and not redo_model:
        model = load_model(model_fit_h5, custom_objects={'NegExpManhattan': NegExpManhattan})
    else:
        ## Inputs
        question1 = layers.Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
        question2 = layers.Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')

        ## Encoders
        #see https://keras.io/getting-started/functional-api-guide/#shared-layers
        embeder = Sequential()
        embeder.add(embedding_layer)
        shared_cnn_embeder = Sequential()
        shared_cnn_embeder.add(embedding_layer)
        shared_cnn_embeder.add(LSTM(75))
        encoded_q1 = shared_cnn_embeder(question1)
        encoded_q2 = shared_cnn_embeder(question2)
        #merged_vector = layers.merge([encoded_q1, encoded_q2], mode='cos', dot_axes=1)
        predictions = negexp_manhattan([encoded_q1, encoded_q2], axes=1)
        predictions = layers.Reshape((1,))(predictions)
        model = Model([question1, question2], predictions)
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        print('Training')
        save_best_weights = 'initial_train_weights.h5'

        callbacks = [ModelCheckpoint(save_best_weights, monitor='val_loss', save_best_only=True),
                     EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')]

        model.fit([train_q1, train_q2],
                            true_labels,
                            batch_size=32,
                            epochs=50,
                            validation_split=0.1,
                            verbose=True,
                            shuffle=True,
                            callbacks=callbacks)
        model.load_weights(save_best_weights)
        model.save(model_fit_h5, overwrite=True)
    return model



def simple_submission(preds, fname):
    with open(fname, 'w') as handle:
        print("test_id,is_duplicate", file=handle)
        for i,p in enumerate(np.nditer(preds)):
            print(",".join(map(str,[i, p])), file=handle)


if __name__ == '__main__':
    train_q1, train_q2, true_labels, test_q1, test_q2, word_index = load_and_checkpoint_tokenized_train_test()
    glove_pkl = DATA_DIR+"/glove_embeddings.pkl"
    if os.path.exists(glove_pkl):
        embedding_layer = joblib.load(glove_pkl)
    else:
        embedding_layer = load_embeddings(word_index)
        joblib.dump(embedding_layer, glove_pkl)

    # Model 1: epoch 8 s: 0.3725 - acc: 0.8269 - val_loss: 0.4696 - val_acc: 0.7901
    # Model 2: Epoch 10/10
    #384075/384075 [==============================] - 3235s - loss: 0.4116 - acc: 0.8455 - val_loss: 0.4474 - val_acc: 0.8301
    fit_model = train_mueller_2016(train_q1, train_q2, true_labels,
                                  embedding_layer)
    refined_model = retrain_last_output_layer_as_lr(fit_model, train_q1, train_q2, true_labels)
    preds = refined_model.predict([test_q1, test_q2], batch_size=BATCH_SIZE, verbose=1)
    simple_submission(preds, DATA_DIR+"/mueller_2016_refined_fit.csv")
    # See following for notes on taking an existing model, removing layers, and adding new layers.
    #https://github.com/fchollet/keras/issues/871
    #print('Test loss / test accuracy = {:.4f} / {:.4f}'.format(loss, acc))
