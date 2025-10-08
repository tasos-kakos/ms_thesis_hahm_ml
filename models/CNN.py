import numpy as np
import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras.regularizers import l1
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import backend as K
from tensorflow.keras.utils import get_custom_objects
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import argparse
import copy

def create_cnn_model(input_shape):
    
    kernel_regularizer= None # tf.keras.regularizers.l1_l2(l1=0.000001, l2=0.00001)
    p_drop = 0.1
    if barrel_index == 3:
        model = tf.keras.models.Sequential(name = 'CNN_BMS')
    elif barrel_index == 5:
        model = tf.keras.models.Sequential(name = 'CNN_BOS')
    elif barrel_index == 2:
        model = tf.keras.models.Sequential(name = 'CNN_BML')
    elif barrel_index == 4:
        model = tf.keras.models.Sequential(name = 'CNN_BOL')
    elif barrel_index == 1:
        model = tf.keras.models.Sequential(name = 'CNN_BIS')
    elif barrel_index == 0:
        model = tf.keras.models.Sequential(name = 'CNN_BIL')
    model.add(tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same', kernel_regularizer = kernel_regularizer, input_shape=input_shape))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(p_drop))
    
    model.add(tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same', kernel_regularizer = kernel_regularizer, input_shape=input_shape))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(p_drop))
    
    model.add(tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid', padding='same', kernel_regularizer = kernel_regularizer))
    
    return model

positive_weight = tf.Variable(1.0, trainable=False, dtype=tf.float32)

def weighted_binary_crossentropy_with_scheduler(y_true, y_pred):
    """
    Computes the weighted binary crossentropy using a dynamic positive_weight variable.
    Negative weight is fixed at 1.0.
    The weights are normalized so that they sum to 1.
    """
    neg_weight = 1.0
    # Normalize the weights: norm_neg + norm_pos == 1.
    norm_neg = neg_weight / (neg_weight + positive_weight)
    norm_pos = positive_weight / (neg_weight + positive_weight)
    
    # Compute the binary crossentropy loss.
    y_true = tf.cast(y_true, tf.float32)
    bce = tf.keras.backend.binary_crossentropy(y_true, y_pred)
    
    # Create a weight vector based on y_true: if y_true == 1, use norm_pos, else use norm_neg.
    weight_vector = y_true * norm_pos + (1 - y_true) * norm_neg
    weighted_bce = bce * weight_vector
    
    return tf.reduce_mean(weighted_bce)

# Custom callback to update the positive_weight over epochs.
class WeightScheduler(tf.keras.callbacks.Callback):
    def __init__(self, initial_weight, final_weight, start_epoch, end_epoch):
        """
        Args:
          initial_weight: Starting weight for the positive class.
          final_weight: Target weight after end_epoch.
          start_epoch: Epoch to start changing the weight.
          end_epoch: Epoch by which to reach final_weight.
        """
        super().__init__()
        self.initial_weight = initial_weight
        self.final_weight = final_weight
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch

    def on_epoch_begin(self, epoch, logs=None):
        # Before start_epoch, keep the weight at initial_weight.
        if epoch < self.start_epoch:
            new_weight = self.initial_weight
        # After end_epoch, set the weight to final_weight.
        elif epoch >= self.end_epoch:
            new_weight = self.final_weight
        # Otherwise, linearly interpolate between initial_weight and final_weight.
        else:
            fraction = (epoch - self.start_epoch) / (self.end_epoch - self.start_epoch)
            new_weight = self.initial_weight + fraction * (self.final_weight - self.initial_weight)
        
        # Update the variable.
        positive_weight.assign(new_weight)
        print(f"Epoch {epoch + 1}: positive weight set to {new_weight:.4f}")

def focal_loss(gamma=2, alpha=0.25):
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

def combined_loss(alpha=0.4, gamma=2.0):
    fl = focal_loss(gamma=gamma)
    def loss_fn(y_true, y_pred):
        return alpha * fl(y_true, y_pred) + (1 - alpha) * dice_loss(y_true, y_pred)
    return loss_fn

def read_model(fileName):
        model = tf.keras.models.load_model(fileName)
        return model
"""   
def normalize(images):
    nonzero_mask = images != 0  # Identify nonzero pixels
    mean = np.mean(images[nonzero_mask], axis = 0)  # Mean of nonzero pixels
    std = np.std(images[nonzero_mask], axis = 0) + 1e-8  # Std deviation (avoid division by 0)
    
    images_norm = np.zeros_like(images, dtype=np.float32)  # Keep zero values unchanged
    images_norm[nonzero_mask] = (images[nonzero_mask] - mean) / std  # Normalize only nonzero
    
    return images_norm

"""
def normalize(images):
    mean = np.mean(images)[np.newaxis]
    sigma = np.std(images)[np.newaxis]
    images_normalized = (images - mean) / sigma
    return images_normalized

def displacement_downsampling(events, events_labels, plot_labels, eta_station, phi_sector, event_number, eta_region_id, displacement, threshold, target_ratio, random_state=30):
    """
    Zero out any muon where displacement is below the provided threshold.
    Find which images still have labelled hits (pos) vs. all-zero (neg).
    Randomly drop events from the larger class so that
        n_pos / (n_pos + n_neg) == target_ratio
    Returns:
      All the downsampled fed arrays and the new label array to be used for training 
    """
    rng = np.random.default_rng(random_state)

    # Apply threshold in-place on a copy
    labels = np.copy(events_labels)
    too_small_dis = (labels == 1) & (displacement < threshold)
    labels[too_small_dis] = 0

    # Figure out pos vs. neg image‐indices
    summed = np.any(labels == 1, axis=(1,2,3))
    pos_idx = np.nonzero(summed)[0]
    neg_idx = np.nonzero(~summed)[0]

    n_pos, n_neg = len(pos_idx), len(neg_idx)
    curr_ratio = n_pos / (n_pos + n_neg)

    # If we’re already at the target (or can’t change it), just return
    if np.isclose(curr_ratio, target_ratio):
        keep_idx = np.arange(len(labels))
        print("Target ratio is already met or can't be changed")
        return events, labels, plot_labels, eta_station, phi_sector, event_number, eta_region_id, displacement

    # Decide which class to down‐sample
    if curr_ratio < target_ratio:
        # Too few positives --> drop some negatives
        # Target:  n_pos / (n_pos + N_keep) = target_ratio
        N_keep = int(np.floor((n_pos * (1 - target_ratio)) / target_ratio))
        assert N_keep < n_neg
        neg_keep = rng.choice(neg_idx, size=N_keep, replace=False)
        pos_keep = pos_idx

    else:
        # Too many positives --> drop some positives
        # Target:  P_keep / (P_keep + n_neg) = target_ratio
        P_keep = int(np.floor((target_ratio * n_neg) / (1 - target_ratio)))
        assert P_keep < n_pos
        pos_keep = rng.choice(pos_idx, size=P_keep, replace=False)
        neg_keep = neg_idx

    # combine to get final keep‐list
    keep_idx = np.concatenate([pos_keep, neg_keep])
    
    events_filtered = events[keep_idx]
    events_labels_filtered = labels[keep_idx]
    plot_labels_filtered = plot_labels[keep_idx]
    eta_station_filtered = eta_station[keep_idx]
    phi_sector_filtered = phi_sector[keep_idx]
    event_number_filtered = event_number[keep_idx]
    eta_region_id_filtered = eta_region_id[keep_idx]
    displacement_filtered = displacement[keep_idx]
    
    print(f"Kept {len(keep_idx)} event images, of which "f"{(labels.shape[0]*target_ratio):.0f} ""should contain displaced muon hits above displacement threshold ("f"{(target_ratio*100):.0f}""%)")
    return events_filtered, events_labels_filtered, plot_labels_filtered, eta_station_filtered, phi_sector_filtered, event_number_filtered, eta_region_id_filtered, displacement_filtered

def muongun_cleaner(events, events_labels, plot_labels, eta_station, phi_sector, event_number, eta_region_id, displacement, target_ratio, random_state=30):
    rng = np.random.default_rng(random_state)
    mask = (plot_labels == 2)
    events[mask] = 0
    events_labels[mask] = 0
    eta_station[mask] = 0
    plot_labels[mask] = 0
    
    # Figure out pos vs. neg image‐indices
    summed = np.any(events_labels == 1, axis=(1,2,3))
    pos_idx = np.nonzero(summed)[0]
    neg_idx = np.nonzero(~summed)[0]

    n_pos, n_neg = len(pos_idx), len(neg_idx)
    curr_ratio = n_pos / (n_pos + n_neg)

    # If we’re already at the target (or can’t change it), just return
    if np.isclose(curr_ratio, target_ratio):
        keep_idx = np.arange(len(events_labels))
        print("Target ratio is already met or can't be changed")
        return events, events_labels, plot_labels, eta_station, phi_sector, event_number, eta_region_id, displacement

    # Decide which class to down‐sample
    if curr_ratio < target_ratio:
        # Too few positives --> drop some negatives
        # Target:  n_pos / (n_pos + N_keep) = target_ratio
        N_keep = int(np.floor((n_pos * (1 - target_ratio)) / target_ratio))
        assert N_keep < n_neg
        neg_keep = rng.choice(neg_idx, size=N_keep, replace=False)
        pos_keep = pos_idx

    else:
        # Too many positives --> drop some positives
        # Target:  P_keep / (P_keep + n_neg) = target_ratio
        P_keep = int(np.floor((target_ratio * n_neg) / (1 - target_ratio)))
        assert P_keep < n_pos
        pos_keep = rng.choice(pos_idx, size=P_keep, replace=False)
        neg_keep = neg_idx

    # combine to get final keep‐list
    keep_idx = np.concatenate([pos_keep, neg_keep])
    
    events_filtered = events[keep_idx]
    events_labels_filtered = events_labels[keep_idx]
    plot_labels_filtered = plot_labels[keep_idx]
    eta_station_filtered = eta_station[keep_idx]
    phi_sector_filtered = phi_sector[keep_idx]
    event_number_filtered = event_number[keep_idx]
    eta_region_id_filtered = eta_region_id[keep_idx]
    displacement_filtered = displacement[keep_idx]
    
    print(f"Kept {len(keep_idx)} event images, of which "f"{(events_labels.shape[0]*target_ratio):.0f} ""should contain displaced muon hits above displacement threshold ("f"{(target_ratio*100):.0f}""%)")
    return events_filtered, events_labels_filtered, plot_labels_filtered, eta_station_filtered, phi_sector_filtered, event_number_filtered, eta_region_id_filtered, displacement_filtered
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process HDF5 files and save to NPZ.")
    parser.add_argument('-i','--input', type=str, required=True, help="Path to the input NPZ file.")
    parser.add_argument('-m', '--model', type=str, required=False, default="", help="Path to the output hdf5 model file.")
    parser.add_argument('-o', '--output', type=str, required=False, default="", help="Path to the output npz file.")
    parser.add_argument('-n', '--numbarrel', type=int, choices=range(0, 6), default=3, help="Optional barrel index (0-5), default is 3.")
    parser.add_argument('-t', '--threshold', type=float, default=0, help="Optional displacement threshold (in mm) for displaced muon hits to be identified")
    parser.add_argument('-r', '--ratio', type=float, default=0.75, help="Optional signal to background ratio.")
    parser.add_argument('-rm_p', '--remove_prompt', action="store_true", default = False, help="Optional ability to remove prompt muons from input")
    args = parser.parse_args()

    retrainModel=False
    
    # Load the data
    data=np.load(args.input, allow_pickle=True)
    events = data['events']
    events = events.astype('float32')
    print(events.shape)
    
    events_labels = data['events_labels']
    events_labels = events_labels.astype('int')
    
    displacement = data['displacement']
    displacement = displacement.astype('float32')
    
    eta_station = data['eta_station']
    eta_station = eta_station.astype('int')
    
    plot_labels = data['plot_labels']
    plot_labels = plot_labels.astype('int')
    
    phi_sector = data['phi_sector']
    phi_sector.astype('int')
    
    event_number = data['event_number']
    event_number = event_number.astype('int')
    
    eta_region_id = data['eta_region_id']
    eta_region_id = eta_region_id.astype('float32')
    
    # Downsample input data according to displacement threshold and keep desired ratio of total images
    threshold = args.threshold
    target_ratio = args.ratio
    remove_prompt = args.remove_prompt
    print("Image shape before applying threshold: ", events.shape)
    
    if threshold != 0:
        events, events_labels, plot_labels, eta_station, phi_sector, event_number, eta_region_id, displacement = displacement_downsampling(events, events_labels, plot_labels, eta_station, phi_sector, event_number, eta_region_id, displacement, threshold, target_ratio)

    if remove_prompt:
        events, events_labels, plot_labels, eta_station, phi_sector, event_number, eta_region_id, displacement = muongun_cleaner(events, events_labels, plot_labels, eta_station, phi_sector, event_number, eta_region_id, displacement, target_ratio)

    if threshold != 0 or remove_prompt == True:
        print("Downsampled images shape: ", events.shape)
    
    input_shape = events.shape[1:]
    
    barrel_index = args.numbarrel

    if retrainModel:
        model = read_model(args.model)
        model.build(input_shape=input_shape)
        model.summary()

    model = create_cnn_model(input_shape)
    
    #loss_fn = weighted_binary_crossentropy_with_scheduler    
    loss_fn = combined_loss(alpha=0.4, gamma=2.)
    #loss_fn = focal_loss()
    #loss_fn = 'binary_crossentropy'
    
    get_custom_objects()["loss_fn"] = loss_fn

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-2),
                loss = loss_fn,
                metrics=['accuracy'])

    model.summary()

    #norm_events = np.nan_to_num(normalize(events))

    X_train, X_test, y_train, y_test, eta_station_train, eta_station_test, displacement_train, displacement_test, plot_labels_train, plot_labels_test, phi_sector_train, phi_sector_test, event_number_train, event_number_test, eta_region_id_train, eta_region_id_test = train_test_split(events, events_labels, eta_station, displacement, plot_labels, phi_sector, event_number, eta_region_id, test_size=0.3, random_state=42)
    X_test, X_val, y_test, y_val, eta_station_test, eta_station_val, displacement_test, displacement_val, plot_labels_test, plot_labels_val, phi_sector_test, phi_sector_val, event_number_test, event_number_val, eta_region_id_test, eta_region_id_val = train_test_split(X_test, y_test, eta_station_test, displacement_test, plot_labels_test, phi_sector_test, event_number_test, eta_region_id_test, test_size = 1/3, random_state=42)
    
    X_test_norm = np.nan_to_num(normalize(X_test))
    X_val = np.nan_to_num(normalize(X_val))
    
    # Focus mask on hits (non-zero driftR)
    focus_mask = (X_train > 0).astype(np.float32)
    X_train = np.nan_to_num(normalize(X_train))
    # Add per pixel weights
    alpha = 5.0
    pixel_weights = 1.0 + (alpha - 1.0) * focus_mask
    
    # Normalize so mean weight is 1.0
    pixel_weights *= np.prod(pixel_weights.shape) / pixel_weights.sum()
    
    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
    scheduler = WeightScheduler(initial_weight=1, final_weight=12, start_epoch=0, end_epoch=49)

    history = model.fit(X_train, y_train,
                        sample_weight=pixel_weights,
                        validation_data=(X_val, y_val),
                        epochs=50, batch_size=64, verbose=1,
                        callbacks=[tf.keras.callbacks.CSVLogger("history_{}.csv".format(model.name)),
                                   early_stopping,
                                   reduce_lr,
                                   scheduler
                                   ])
    if args.model != "":
        model.save(args.model)
    
    predictions = model.predict(X_test_norm)
    predictions = np.squeeze(predictions, axis=-1)
    history = np.genfromtxt("history_{}.csv".format(model.name), delimiter=",", names=True)

    y_test = np.squeeze(y_test,axis = -1)

    flat_preds= predictions.flatten()
    ypred_binary = np.zeros(len(flat_preds), dtype=int)
    ypred_binary[flat_preds > 0.5] = 1

    print(np.corrcoef(y_test.flatten(),ypred_binary))
    if args.output != "" :
       np.savez(args.output,predictions=predictions, y_test=y_test, x_test = X_test, x_test_norm = X_test_norm, y_val=y_val, x_val=X_val, eta_station_test = eta_station_test, displacement_test=displacement_test, plot_labels_test=plot_labels_test, phi_sector_test=phi_sector_test, event_number_test=event_number_test, eta_region_id_test=eta_region_id_test, history=history)
