import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, SimpleRNN, LSTM, GRU, Bidirectional, TimeDistributed, Dropout, Reshape
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
import argparse
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plotting(y_test,y_pred):
    threshold = 0.5
    ypred_binary = np.zeros(len(y_pred.flatten()), dtype=int)
    ypred_binary[y_pred.flatten() > threshold] = 1
    y_test_flat = y_test.flatten()
    y_test_flat[y_test_flat > 0] = 1
    print("Accuracy score - ", accuracy_score(y_test_flat, ypred_binary))

    #print("Lengths - ", len(ypred_binary), len(y_test.flatten))

    cm = confusion_matrix(y_test.flatten(), ypred_binary, normalize='true')
    cm = pd.DataFrame(cm, range(2),range(2))
    print("CM - ", cm)
    plt.figure(figsize = (10,10))
    sns.heatmap(cm, annot=True, annot_kws={"size": 12}) # font size
    plt.xlabel('Truth')
    plt.ylabel('Predictions')
    plt.title('Muon identification confusion matrix')
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')  # Save as PNG
    plt.show()


    fpr, tpr, _ = roc_curve(y_test.flatten(), ypred_binary.flatten())
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Random classifier baseline
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig("ROC_curve.png")
    plt.show()

    plt.figure(figsize=(10, 8))
    smask = y_test == 1
    plt.hist(y_pred[smask], log=True, alpha=0.5)
    plt.hist(y_pred[~smask], log=True, alpha=0.5)
    plt.xlabel('CNN output value')
    plt.ylabel('Count')
    plt.title('CNN result repartition')
    plt.savefig('result_histogram.png', dpi=300, bbox_inches='tight')  # Save as PNG
    plt.show()

def create_rnn_model(input_shape, labels_shape):
    input_layer = Input(shape=input_shape)
    x = GRU(8, return_sequences=True)(input_layer)
    x = Dropout(0.2)(x)
    x = GRU(8, return_sequences=True)(x)
    x = Dropout(0.2)(x)

    
    output = Dense(labels_shape[1], activation='sigmoid', name='rnn_output')(x)

#    output = TimeDistributed(Dense(labels_shape[1], activation='sigmoid'), name='rnn_output')(x)


    model = Model(inputs=input_layer, outputs=output)

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RNN based on NPZ and save results to NPZ.")
    parser.add_argument('-i','--input', type=str, required=True, help="Path to the input NPZ file.")
    parser.add_argument('-o', '--output', type=str, default=None, required=False, help="Path to the output model file.")
    parser.add_argument('-r', '--results', type=str, default=None, required=False, help="Path to the output NPZ file for results.")
    args = parser.parse_args()
    
    input_file = args.input
    output_file = args.output

    data = np.load(input_file)
    events = data['events']
    events_labels = data['events_labels']
    input_shape = events[0].shape
    labels_shape=events_labels[0].shape
    print("input_shape = ", input_shape)
    print("labels_shape = ", labels_shape)
    print("labels dtype = ", np.dtype(events_labels[0][0][0]))

    model = create_rnn_model(input_shape, labels_shape)
    model.summary()

    fname = "best_model.keras"
    mcp_save = tf.keras.callbacks.ModelCheckpoint(fname, save_best_only=True, monitor='val_loss', mode='min')

    #Options for EarlyStopping and ReduceLROnPlateau
    reduce_on_p = tf.keras.callbacks.ReduceLROnPlateau(factor=0.67, patience=3, verbose=1, min_lr=1E-5),
    stopping = tf.keras.callbacks.EarlyStopping(patience=10, verbose=1)
    def get_sample_weights(y, pos_weight=5.0):
        """
        y: array of shape (batch_size, seq_len), with values 0.0 or 1.0
        pos_weight: how much more we want to weight the positive class
        """
        y = np.array(y)
        return np.where(y == 1.0, pos_weight, 1.0)

    X_train, X_test, y_train, y_test = train_test_split(events, events_labels, test_size=0.2, random_state=42)

    X_train_full, X_val, y_train_full, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

    model.compile(optimizer='adam',
                loss = {'rnn_output': 'binary_crossentropy'},metrics=["binary_accuracy"],
                 weighted_metrics=[
                tf.keras.metrics.Precision(name='weighted_precision'),
                tf.keras.metrics.Recall(name='weighted_recall')
                ])

    history = model.fit(X_train_full,{'rnn_output': y_train_full},
                        epochs=10,
                        validation_data=(X_val, {'rnn_output': y_val}),
                        verbose=1,
                        batch_size=32,
                        callbacks=[reduce_on_p, stopping])
    if output_file!=None:
        model.save(output_file, save_format="h5")
        # model.save_weights(output_file)
        print(f"Model saved to {output_file}")
    print(X_test.shape)
    print(y_test.shape)
    
    results = model.evaluate(X_test,y_test)
#    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(results)
    
    #print(f"Validation Loss: {test_loss}")
    #print(f"Validation Accuracy: {test_accuracy}")

    y_pred = model.predict(X_test)
    print("Y validation shape -", y_test.shape)

    #plotting(y_test,y_pred)
    threshold = 0.5
    ypred_binary = np.zeros(len(y_pred.flatten()), dtype=int)
    ypred_binary[y_pred.flatten() > threshold] = 1
    y_test_flat = y_test.flatten()
    y_test_flat[y_test_flat > 0] = 1
    print("Accuracy score - ", accuracy_score(y_test_flat, ypred_binary))

    #print("Lengths - ", len(ypred_binary), len(y_test.flatten))

    cm = confusion_matrix(y_test.flatten(), ypred_binary, normalize='true')
    cm = pd.DataFrame(cm, range(2),range(2))
    print("CM - ", cm)

    if args.results != None:
        np.savez(args.results,test=y_test, pred=y_pred)
