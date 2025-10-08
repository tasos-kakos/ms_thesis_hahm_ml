import h5py
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from utils.display_events import plot_event, display
import argparse

# loading hdf5 file data
def load_data_from_hdf5(file_name, background_filename = None):
    events = []
    labels = []
    event_order = []
    displacement_labels = []
    truth_labels = []
    phi_stations = [1,2,3,4,5,8] 
    eta_stations = [1,2,3,4,5,6] 
    with h5py.File(file_name, 'r') as h5file:   
        if background_filename!= None:
            h5bgfile_list = []
            for bg_index in range(len(background_filename)):
                h5bgfile_list.append(h5py.File(background_filename[bg_index], 'r'))
            for i, evt_n in enumerate(h5file.keys()):
                bg_n = []
                for h5bgfile in h5bgfile_list:
                    if len(list(h5bgfile.keys())) <= i:
                           continue
                    bg_n.append(list(h5bgfile.keys())[i])
                if i > 100000:
                    break # Limit of 100000 values
                if not (i%100): # Print index every 1k values
                    print("___________________________________ NEW EVENT MDT = ", i, evt_n, " _____________________________________")
                grp = h5file[evt_n]
                grp_bg = []
                j = 0
                for h5bgfile in h5bgfile_list:
                    if len(list(h5bgfile.keys())) <= i:
                        continue
                    grp_bg.append(h5bgfile[bg_n[j]])
                    j = j+1
                for eta_station in eta_stations:
                    for phi_station in phi_stations:
                        event, label, displacement_label, truth_label = preprocess_data(grp,phi_station,eta_station,grp_bg)
                        events.append(np.array(event))
                        labels.append(np.array(label))
                        displacement_labels.append(displacement_label)
                        truth_labels.append(truth_label)
                        order = int(evt_n.split("_")[-1])
                        event_order.append(np.array(order))
        else:
                print(background_filename)
                for i, evt_n in enumerate(h5file.keys()):
                    if i > 100000:
                        break # Limit of 1000 values
                    if not (i%100): # Print index every 1k values
                        print("___________________________________ NEW EVENT MDT = ", i, evt_n, " _____________________________________")
                    grp = h5file[evt_n]
                    for eta_station in eta_stations:
                        for phi_station in phi_stations:
                            event, label, displacement_label, truth_label = preprocess_data(grp,phi_station,eta_station)
                            events.append(np.array(event))
                            labels.append(np.array(label))
                            displacement_labels.append(displacement_label)
                            truth_labels.append(truth_label)
                            order = int(evt_n.split("_")[-1])
                            event_order.append(order)

    return np.array(events, dtype=object), np.array(labels, dtype=object), np.array(event_order), np.array(displacement_labels), np.array(truth_labels)

## creates arrays of hits organized per layer, normalizes and applies padding 
def preprocess_data(event, phi_station, eta_station, bg_event=None):
    norm_event = []
    labels = []
    max_length = 5
    multiLayer = np.asarray(event['multiLayer'][()])
    layer = np.asarray(event['tubeLayer'][()])
    stationIndex = np.asarray(event['stationIndex'][()])
    stationEta = np.asarray(event['stationEta'][()])
    stationPhi = np.asarray(event['stationPhi'][()])
    tube = np.asarray(event['tube'])
    driftDist = np.asarray(event['driftR'])
    displacement = np.asarray(event['muonDisplacement'])

    isMuon = event['isMuon'][()]
    if bg_event!=None:
        for bg in bg_event:  
            multiLayer = np.concatenate((multiLayer, np.asarray(bg['multiLayer'][()]))).astype(int)
            layer = np.concatenate((layer, np.asarray(bg['tubeLayer'][()]))).astype(int)
            stationIndex = np.concatenate((stationIndex, np.asarray(bg['stationIndex'][()]))).astype(int)
            stationEta = np.concatenate((stationEta, np.asarray(bg['stationEta'][()]))).astype(int)
            stationPhi = np.concatenate((stationPhi, np.asarray(bg['stationPhi'][()]))).astype(int)
            tube = np.concatenate((tube, np.asarray(bg['tube'][()]))).astype(int)
            driftDist = np.concatenate((driftDist, np.asarray(bg['driftR'][()])))
            isMuon = np.concatenate((isMuon, np.asarray(bg['isMuon'][()])*2)) # Background -> 0 - Muon gun -> 2
            displacement = np.concatenate((displacement, np.zeros(np.asarray(bg['isMuon'][()]).shape)))
    isStation_phi = np.isin(np.asarray(stationPhi), [phi_station])
    isBarrel = (stationIndex < 6).astype(int)
    isStation_eta = np.isin(np.asarray(stationEta), [eta_station]) 
    stationIndex = stationIndex//2 + 1
    layer2 = (
    np.where(stationIndex  == 1, (multiLayer - 1) * 4 + layer, 0) +
    np.where(stationIndex  == 2, 8 + (multiLayer - 1) * 3 + layer, 0) +
    np.where(stationIndex  == 3, 14 + (multiLayer - 1) * 3 + layer, 0)
    )
    Nlayers = 20
    data = [[] for _ in range(Nlayers)]
    labels=[[] for _ in range(Nlayers)]
    truth_labels= [[] for _ in range(Nlayers)]
    displacement_labels = [[] for _ in range(Nlayers)]

    for i in range (0, len(tube)):
        if not (isBarrel[i] and isStation_phi[i] and isStation_eta[i]):
            continue
        hit=[tube[i], driftDist[i]]
        data[layer2[i]-1].append(hit)
        labels[layer2[i]-1].append(isMuon[i])
        truth_labels[layer2[i]-1].append(isMuon[i])
        displacement_labels[layer2[i]-1].append(displacement[i])
    for i in range (0, Nlayers):
        if len(data[i]) > 0 and len(data[i]) < max_length:
            data_size = len(data[i])
            padding = np.zeros((max_length - data_size, 2))
            padding = padding - 1
            data[i] = np.concatenate([data[i], padding])
            padding = np.zeros(max_length - data_size)
            labels[i] = np.concatenate([labels[i], padding])
            truth_labels[i]=np.concatenate([truth_labels[i], padding])
            displacement_labels[i] = np.concatenate([displacement_labels[i], padding])
        if len(data[i]) == 0:
            padding = np.full((max_length,2), -1)
            data[i]=padding
            padding = np.full((max_length,), 0)
            labels[i]=padding
            truth_labels[i]=padding
            displacement_labels[i] = padding
        if len(data[i]) > max_length :
            data[i] = data[i][:max_length]
            labels[i] = labels[i][:max_length]
            truth_labels[i] = truth_labels[i][:max_length]
            displacement_labels[i] = displacement_labels[i][:max_length]
        data[i]=np.asarray(data[i]).reshape(-1)
    return np.array(data), np.array(labels), np.array(displacement_labels), np.array(truth_labels)

def downsample(events, event_labels, desired_ratio, order, displacement,truth_label):
    """
    Downsamples provided events by removing images containing no signal to fit the desired ratio (#signal_images/total_images)
    events - array containing all event images
    event_labels - array containing even labels
    desired_ratio - float from 0 to 1 corresponding to the desired signal/total images ratio
    order - array containing the order of events
    """
    summed_labels = np.array([np.sum(event_labels[i]) for i in range(len(event_labels))])
    nb_muons = (summed_labels >= 1).sum()
    current_ratio = nb_muons/len(event_labels)
    print("Current ratio - ", current_ratio)
    if current_ratio >= desired_ratio: # Return if current ratio is higher than what is demanded
        return events, event_labels, order, displacement
    
    muons = events[summed_labels>=1]
    muon_labels = event_labels[summed_labels>=1]
    muon_order = order[summed_labels>=1]
    muon_displacement = displacement[summed_labels>=1]
    muon_truth = truth_label[summed_labels>=1]

    non_muons = events[summed_labels==0]
    non_muon_labels = event_labels[summed_labels==0]
    non_muon_order = order[summed_labels==0]
    non_muon_displacement = displacement[summed_labels==0]
    non_muon_truth = truth_label[summed_labels==0]

    randomize_background = np.random.permutation(len(non_muon_labels))
    non_muons_shuffled = non_muons[randomize_background] # Shuffle non muons to vary the output
    new_non_muon_number = np.floor((1-desired_ratio)*len(muons)/desired_ratio).astype(int)

    new_non_muons = non_muons_shuffled[:new_non_muon_number]
    new_non_muon_labels = non_muon_labels[:new_non_muon_number]
    new_events = np.concatenate((muons,new_non_muons))
    new_labels = np.concatenate((muon_labels,new_non_muon_labels))
    new_order = np.concatenate((muon_order, non_muon_order[:new_non_muon_number]))
    new_displacement = np.concatenate((muon_displacement, non_muon_displacement[:new_non_muon_number]))
    new_truth_label = np.concatenate((muon_truth, non_muon_truth[:new_non_muon_number]))
    return new_events, new_labels, new_order, new_displacement, new_truth_label


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process HDF5 files and save to NPZ.")
    parser.add_argument('-i','--input', type=str, required=True, help="Path to the input HDF5 file.")
    parser.add_argument('-o', '--output', type=str, required=True, help="Path to the output NPZ file.")
    parser.add_argument('-b', '--background', nargs='*', type=str, required=False, default ="", help="Path to the input background NPZ file.")
    parser.add_argument('-r', '--ratio', type=float, default=0.75, help="Optional signal to background ratio.")
    parser.add_argument('-e', '--excludeprompt', action="store_true", help="Exclude prompt muons from label image.")

    args = parser.parse_args()
    output_file = args.output
    print(args.input)

    background_file = None
    if args.background != "":
        background_file = args.background
    events, events_labels, order, displacement, truth_event_labels = load_data_from_hdf5(args.input, background_file)

    events, events_labels, order, displacement, truth_event_labels = downsample(events=events, event_labels=events_labels, desired_ratio=args.ratio,order=order, displacement= displacement, truth_label=truth_event_labels)

    events = events.astype('float32')
    events_labels = events_labels.astype('float32')

    sorted_indices = np.argsort(order)

    events = events[sorted_indices]
    events_labels = events_labels[sorted_indices]
    displacement = displacement[sorted_indices]
    truth_event_labels =  truth_event_labels[sorted_indices]


    # This code unlabels all prompt muons
    plot_output = output_file.split(".npz", 1)[0] + "_plot"
    if args.excludeprompt :
        print("Exclude muon gun data from label")
        events_labels[(displacement <= 200) & (events_labels == 1)] = 3
        np.savez(plot_output,events=events, events_labels=events_labels, displacement=displacement, truth_label = truth_event_labels)
        events_labels[events_labels == 3 ] = 0
        events_labels[events_labels == 2 ] = 0
        #events, events_labels, order, displacement, truth_event_labels = downsample(events=events, event_labels=events_labels, desired_ratio=args.ratio,order=order,displacement=displacement, truth_label=truth_event_labels)
    else :
        np.savez(plot_output,events=events, events_labels=events_labels, displacement=displacement)
        events_labels[events_labels == 2 ] = 1

    sorted_indices = np.argsort(order)

    events = events[sorted_indices]
    events_labels = events_labels[sorted_indices]
    displacement = displacement[sorted_indices]
    truth_event_labels =  truth_event_labels[sorted_indices]

    # End of code unlabelling prompt muons
    print("Final image shape: ",events.shape)

    np.savez(args.output,events=events, events_labels=events_labels, displacement=displacement, truth_label = truth_event_labels)
