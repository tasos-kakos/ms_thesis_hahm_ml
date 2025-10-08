import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, SimpleRNN, LSTM, GRU, Bidirectional, TimeDistributed
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
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import time

### Imports a RNN model saved in h5 format

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RNN based on NPZ and save results to NPZ.")
    parser.add_argument('-i','--input', type=str, required=True, help="Path to the input NPZ file.")
    parser.add_argument('-m', '--model', type=str, default=None, required=False, help="Path to the model file for results.")
    parser.add_argument('-o', '--output', type=str, default=None, required=False, help="Path to the output model file.")
    args = parser.parse_args()
    
    input_file = args.input
    output_file = args.output
    saved_model = args.model
    print("Model - ",saved_model," Validation sample - ", input_file)

    data=np.load(input_file, allow_pickle=True)
    events = np.asarray(data['events']).astype('float32')
    events_labels = np.asarray(data['events_labels']).astype('float32')

    model = tf.keras.models.load_model(saved_model)
    model.summary()

    events[events == -1] = 0
    print(events.shape, events_labels.shape)

    X_train, X_test, y_train, y_test = train_test_split(events, events_labels, test_size=0.99, random_state=42)
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
    #print(test_output.reshape(test_output.shape[0],-1).shape)
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


#    print("Flops RNN python - ", get_flops(model))
#    print("Flops RNN HLS - ", get_flops(hls_model))
#    hls_model.build()
