import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Dense, SimpleRNN, LSTM, GRU, Bidirectional, TimeDistributed
from tensorflow.keras.utils import get_custom_objects
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import hls4ml 
import argparse
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph
import time

def focal_loss(gamma=2, alpha=0.5):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        bce = tf.keras.backend.binary_crossentropy(y_true, y_pred)
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        return tf.reduce_mean(alpha * tf.pow(1. - pt, gamma) * bce)
    return loss

def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    yt = tf.reshape(y_true, [-1])
    yp = tf.reshape(y_pred, [-1])
    inter = tf.reduce_sum(yt * yp)
    return 1 - (2 * inter + smooth) / (tf.reduce_sum(yt) + tf.reduce_sum(yp) + smooth)

def combined_loss(alpha=0.5, gamma=2.0):
    fl = focal_loss(gamma=gamma)
    def loss_fn(y_true, y_pred):
        return alpha * fl(y_true, y_pred) + (1 - alpha) * dice_loss(y_true, y_pred)
    return loss_fn

def normalize(images):
    mean = np.mean(images)[np.newaxis]
    sigma = np.std(images)[np.newaxis]
    images_normalized = (images - mean) / sigma
    return images_normalized

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CNN based on NPZ and save results to NPZ.")
    parser.add_argument('-i','--input', type=str, required=True, help="Path to the validation input NPZ file.")
    parser.add_argument('-m', '--model', type=str, default=None, required=False, help="Path to the model file for results.")
    parser.add_argument('-o', '--output', type=str, default=None, required=False, help="Path to the output model file.")
    args = parser.parse_args()
    
    input_file = args.input
    output_file = args.output
    saved_model = args.model
    print("Model - ",saved_model," Validation sample - ", input_file)

    data=np.load(input_file, allow_pickle=True)
    events = data['events']
    events = events.astype(np.float32)
    events_labels = data['events_labels']
    events_labels = events_labels.astype('int')
    
    loss_fn = combined_loss(alpha=0.4, gamma=2.)
    model = tf.keras.models.load_model(saved_model, custom_objects={'loss_fn': combined_loss(alpha=0.4, gamma=2.)})
    model.summary()

    events[events == -1] = 0
    print(events.shape, events_labels.shape)

    X_train, X_test, y_train, y_test = train_test_split(events, events_labels, test_size=0.99, random_state=42)
    
    X_train = np.nan_to_num(normalize(X_train))
    X_test = np.nan_to_num(normalize(X_test))
    
    start = time.time()
    test_output = model.predict(X_test)
    keras_latency = time.time() - start
    test_output = test_output.reshape(test_output.shape[0],-1)
    threshold = 0.5
    ypred_binary = np.zeros(len(test_output.flatten()), dtype=int)
    ypred_binary[test_output.flatten() > threshold] = 1
    y_test_flat = y_test.flatten()
    y_test_flat[y_test_flat > 0] = 1
    print("Accuracy score - ", accuracy_score(y_test_flat, ypred_binary))

    cm = confusion_matrix(y_test.flatten(), ypred_binary, normalize='true')
    cm = pd.DataFrame(cm, range(2),range(2))
    print("CM - ", cm)

    config = hls4ml.utils.config_from_keras_model(model)
    hls_model = hls4ml.converters.convert_from_keras_model(
        model=model,
        hls_config=config,
        backend='Vitis'
    )
    hls_model.compile()
    start = time.time()
    hls_test_output = hls_model.predict(X_test)
    print(hls_test_output.shape)
    print(test_output.shape)
    hls_latency = time.time() - start

    hls_model.write()

    threshold = 0.5
    ypred_binary = np.zeros(len(hls_test_output.flatten()), dtype=int)
    ypred_binary[hls_test_output.flatten() > threshold] = 1
    y_test_flat = y_test.flatten()
    y_test_flat[y_test_flat > 0] = 1
    print("hls Accuracy score - ", accuracy_score(y_test_flat, ypred_binary))

    cm = confusion_matrix(y_test.flatten(), ypred_binary, normalize='true')
    cm = pd.DataFrame(cm, range(2),range(2))
    print("hls CM - ", cm)

    print(f"Keras latency: {keras_latency:.6f} seconds")
    print(f"HLS latency: {hls_latency:.6f} seconds")

    if args.output!=None:
        y_test = y_test.reshape(y_test.shape[0],-1)
        np.savez(output_file, labels = y_test,RNNscore = test_output, features=X_test, HLSscore = hls_test_output)
        hls_output = output_file.split(".npz", 1)[0] + "_hls"
        np.savez(hls_output, labels = y_test,RNNscore = hls_test_output, features=X_test, HLSscore = test_output)
