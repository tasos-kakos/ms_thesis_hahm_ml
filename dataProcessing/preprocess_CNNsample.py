import h5py
import numpy as np
import argparse
import copy

def layer_rate(image):
    """
    Create an array containing the number of hits per layer for the given image
    """
    barrel_index = args.numbarrel
    if barrel_index > 1:
        n_layers = 6
    else:
        n_layers = 8
    hits_per_layer = np.zeros(n_layers, dtype=int)
    for layer in range(n_layers):
        hits_per_layer[layer] = np.sum(image[layer] == 1)
    return hits_per_layer

# loading hdf5 file data
def load_data_from_hdf5(file_name, barrel_index):
    """
    Load data from hdf5 file and convert it into images
    """
    events = []
    event_number = []
    labels = []
    image_format = args.format   # s --> single eta, h --> half eta
    eta_pos_index = args.eta_pos # 0 --> Negative eta stations, 1 --> Positive eta stations
    eta_regions = [1, 1.5, 2, 2.5, 3, 3.5, 4,5, 5, 5.5, 6, 6.5, 7]
    eta_stations = []
    displacements = []
    phi_stations = args.phi_stations
    eta_region_id = []
    phi_sector = []
    with h5py.File(file_name, 'r') as h5file:
        i = 0
        for evt_n in h5file.keys():
            grp = h5file[evt_n]
            for phi_station in phi_stations:
                if image_format == 'h':
                    image_event, image_label, hits_rates, eta_station, event_id, displacement = preprocess_data(grp, barrel_index, phi_station, eta_pos_index, eta_region = eta_pos_index)
                    events.append(image_event)
                    labels.append(image_label)
                    eta_stations.append(eta_station)
                    phi_sector.append(phi_station)
                    event_number.append(event_id)
                    eta_region_id.append(eta_pos_index) # Eta region is equivalent to eta_pos_index for half eta range image format
                    displacements.append(displacement)
                if image_format == 's':
                    for eta_region in eta_regions:
                        if barrel_index == 2 and phi_station == 7 and eta_region >= 5.5:
                            continue # No MDTs for eta station 6, phi station 7 in BML
                        if barrel_index != 4 and eta_region > 6:
                            continue # No MDTs in eta station 7 except BOL
                        if barrel_index == 4 and eta_region > 6 and phi_station != 7:
                            continue # Only BOL phi station 7 has MDTs in eta station 7
                        image_event, image_label, hits_rates, eta_station, event_id, displacement = preprocess_data(grp, barrel_index, phi_station, eta_pos_index, eta_region)                
                        events.append(image_event)
                        labels.append(image_label)
                        eta_stations.append(eta_station)
                        phi_sector.append(phi_station)
                        event_number.append(event_id)
                        eta_region_id.append(eta_region)
                        displacements.append(displacement)
    events = np.array(events)
    events = np.expand_dims(events, axis=-1)
    event_number = np.array(event_number)
    event_number = np.expand_dims(event_number, axis=-1)
    labels = np.array(labels)
    labels = np.expand_dims(labels, axis=-1)
    eta_stations = np.array(eta_stations) 
    eta_stations = np.expand_dims(eta_stations, axis=-1) 
    displacements = np.array(displacements)
    displacements = np.expand_dims(displacements, axis=-1) 
    eta_region_id = np.array(eta_region_id)
    eta_region_id = np.expand_dims(eta_region_id, axis=-1)
    phi_sector = np.array(phi_sector)
    phi_sector = np.expand_dims(phi_sector, axis=-1)
    print("CNN image input shape:", events.shape)
    return np.array(events, dtype=object), np.array(labels, dtype=object), eta_stations, phi_sector, event_number, eta_region_id, displacements

def create_event_image(tube_hits, layer_hits, driftDist_hits, num_bins, eta_index):
    """
    Create an image following the Small Middle or Inner Barrel layout
    tube_hits - tubes for each layer
    layer_hits - layers for each chamber
    driftDist_hits - list of hits in the desired selection containing drift distance
    num_bins - number of tubes per layer in total selection
    eta_index - eta station values corresponding to each hit
    """
    barrel_index = args.numbarrel
    if barrel_index > 1:
        image = np.zeros((6, num_bins))
        eta_station = np.zeros((6, num_bins))
    elif barrel_index <= 1:
        image = np.zeros((8, num_bins))
        eta_station = np.zeros((8, num_bins))
    for tube, driftDist, layer, eta in zip(tube_hits, driftDist_hits, layer_hits, eta_index):
        tube_bin = int(tube) - 1
        if tube_bin>=(num_bins) or tube_bin<0:
            continue
        image[int(layer)-1, tube_bin] = driftDist
        eta_station[int(layer)-1, tube_bin] = eta
    return image, eta_station

def create_label_image(tube_hits, layer_hits, label_hits, num_bins):
    """
    Create an image following the Small Middle Barrel layout
    tube_hits - tubes for each layer
    layer_hits - layers for each chamber
    driftDist_hits - list of hits in the desired selection containing drift distance
    num_bins - number of tubes per layer in total selection
    eta_index - eta station values corresponding to each hit
    """
    barrel_index = args.numbarrel
    if barrel_index > 1:
        image = np.zeros((6, num_bins))
    elif barrel_index <= 1:
        image = np.zeros((8, num_bins))
    for tube, layer, label in zip(tube_hits, layer_hits, label_hits):
        tube_bin = int(tube) - 1
        if tube_bin>=(num_bins) or tube_bin<0:
            continue
        image[int(layer)-1, tube_bin] = int(label)
    return image

def create_displacement_image(tube_hits, layer_hits, label_displacement, num_bins):
    """
    Create a label image following the Small Middle Barrel layout
    tube_hits - tubes for each layer
    layer_hits - layers for each chamber
    label_hits - list of hits in the desired selection
    num_bins - number of tubes per layer in total selection
    """
    barrel_index = args.numbarrel
    if barrel_index > 1:
        image = np.zeros((6, num_bins))
    elif barrel_index <= 1:
        image = np.zeros((8, num_bins))
    for tube, layer, label in zip(tube_hits, layer_hits, label_displacement):
        tube_bin = int(tube) - 1
        if tube_bin>=(num_bins)  or tube_bin<0:
            continue
        image[int(layer)-1, tube_bin] = int(label)
    return image

def preprocess_data(event, barrel_index, phi_station, eta_pos_index, eta_region):
    """
    Load data from event at given barrel index and phi station and create associated event image and label
    event - targeted event
    barrel_index - station ID / targeted muon spectrometer barrel (0 - 5)
    phi_station - targeted phi station (1 - 8)
    eta_pos_index - decides whether to focus on processing positive or negative eta station data (0 is negative, 1 is positive)
    eta_region - integer or half integer determining the eta station region currently looped.
                 Either a single eta station (integer value) or the overlap region between two (half-integer value) 
    """
    image_format = args.format
    if image_format == 's':
        if barrel_index == 3 or barrel_index == 2:
            num_bins = 56  # largest tube number seen across all eta stations is 56.
        elif barrel_index == 1:
            num_bins = 72 # largest eta station is 1 with 71 tubes, but we use 72, to make the tube dimension even.
        elif barrel_index == 5 or barrel_index == 4: 
            num_bins = 72 # largest tube number seen across all eta stations is 72.
        elif barrel_index == 0:
            num_bins = 36 # largest tube number seen across all eta stations is 36.
    
    if image_format == 'h':
        if barrel_index == 3:
            num_bins = 280  # Eta stations -1,1 have 56 tubes. Eta stations -5, 5 have 32. The rest have 48 per layer.
        elif barrel_index == 1:
            num_bins = 360 # Eta stations -1,1 have 70 tubes. The rest have 58.
        elif barrel_index == 5: 
            num_bins = 424 # Eta stations -6,6 have 64 tubes. The rest have 72.
        elif barrel_index == 0:
            num_bins = 204 # Max total number of tubes seen across all phi sectors, specifically in sector 3.
        elif barrel_index == 2:
            num_bins = 296 # Max total number of tubes seen in an individual phi sector.
        elif barrel_index == 4:
            num_bins = 400 # Max total number of tubes seen in an individual phi sector. Eta stations [-6,6]
    
    hits_rates = []
    layer = event['tubeLayer'][()]
    multiLayer = event['multiLayer'][()]
    stationIndex = event['stationIndex'][()]
    stationEta = event['stationEta'][()]
    stationPhi = event['stationPhi'][()]
    isMuon = event['isMuon'][()]
    displacement = event['muonDisplacement']
    driftR = event['driftR'][()] 
    eventNumber = event['eventNumber'][()]
    isBI = np.isin(np.asarray(stationIndex), [barrel_index])
    
    if image_format == 'h':
        if eta_pos_index == 1:
            isStation_eta = np.isin(np.asarray(stationEta), [1, 2, 3, 4, 5, 6])
       
        elif eta_pos_index == 0:
            isStation_eta = np.isin(np.asarray(stationEta), [-6, -5, -4, -3, -2, -1])

    if image_format == 's':
        if eta_pos_index == 1:
            if (2*eta_region) % 2 == 0:          
                isStation_eta = np.isin(np.asarray(stationEta), [int(eta_region)])        
            else:
                isStation_eta = np.isin(np.asarray(stationEta), [int(eta_region-0.5),int(eta_region+0.5)])
       
        if eta_pos_index == 0:
            if (2*eta_region) % 2 == 0:
                isStation_eta = np.isin(np.asarray(stationEta), [int(-eta_region)])        
            else:
                isStation_eta = np.isin(np.asarray(stationEta), [int(-(eta_region+0.5)),int(-(eta_region-0.5))])
    
    isStation_phi = np.isin(np.asarray(stationPhi), [phi_station])
    stationIndex = stationIndex//2 + 1
    if barrel_index > 1:
        layer2 = (multiLayer - 1) * 3 + layer
    elif barrel_index <= 1:
        layer2 = (multiLayer - 1) * 4 + layer
    tube = np.asarray(event['tube'])
    
    if image_format == 'h':
        if barrel_index == 3:
            if eta_pos_index == 1:
                conditions = [stationEta == i for i in range(1, 7)]
                offsets = [0, 56, 104, 152, 200, 232]
            elif eta_pos_index == 0:
                conditions = [stationEta == i for i in range(-6, 0)]
                offsets = [0, 48, 80, 128, 176, 224]
    
        elif barrel_index == 1:
            if eta_pos_index == 1:
                conditions = [stationEta == i for i in range(1, 7)]
                offsets = [0, 71, 129, 187, 245, 303]
            elif eta_pos_index == 0:
                conditions = [stationEta == i for i in range(-6, 0)]
                offsets = [0, 58, 116, 174, 232, 290]
        
        elif barrel_index == 5:
            if eta_pos_index == 1:
                if phi_station != 4:
                    conditions = [stationEta == i for i in range(1, 7)]
                    offsets = [0, 72, 144, 216, 288, 360]
            
                else:
                    conditions = [stationEta == i for i in range(1, 7)]
                    offsets = [0, 72, 144, 216, 288, 336]
            
            elif eta_pos_index == 0:
                if phi_station != 4:
                    conditions = [stationEta == i for i in range(-6, 0)]
                    offsets = [0, 64, 136, 208, 280, 352]
                
                else:
                    conditions = [stationEta == i for i in range(-6, 0)]
                    offsets = [0, 64, 112, 184, 256, 328]
                    
        elif barrel_index == 0:
            if phi_station == 1 or phi_station == 2 or phi_station == 5:
                if eta_pos_index == 1:
                    conditions = [stationEta == i for i in range(1, 7)]
                    offsets = [0, 30, 66, 96, 132, 162]
            
                if eta_pos_index == 0:
                    conditions = [stationEta == i for i in range(-6, 0)]
                    offsets = [0, 36, 66, 102, 132, 168]
        
            if phi_station == 3:
                if eta_pos_index == 1:
                    conditions = [stationEta == i for i in range(1, 7)]
                    offsets = [0, 30, 66, 102, 138, 168]
            
                if eta_pos_index == 0:
                    conditions = [stationEta == i for i in range(-6, 0)]
                    offsets = [0, 36, 66, 102, 138, 174]
        
            if phi_station == 4:
                if eta_pos_index == 1:
                    conditions = [stationEta == i for i in range(1, 7)]
                    offsets = [0, 24, 60, 90, 126, 156]
            
                if eta_pos_index == 0:
                    conditions = [stationEta == i for i in range(-6, 0)]
                    offsets = [0, 30, 60, 96, 126, 162]
        
            if phi_station == 7:
                if eta_pos_index == 1:
                    conditions = [stationEta == i for i in range(1, 7)]
                    offsets = [0, 30, 60, 90, 120, 150]
            
                if eta_pos_index == 0:
                    conditions = [stationEta == i for i in range(-6, 0)]
                    offsets = [0, 30, 60, 90, 120, 150]
    
        elif barrel_index == 2:
            if eta_pos_index == 1:
                if phi_station <= 2:
                    conditions = [stationEta == i for i in range(1, 7)]
                    offsets = [0, 56, 112, 168, 208, 248]
            
                if phi_station == 3 or phi_station == 7:
                    conditions = [stationEta == i for i in range(1, 7)]
                    offsets = [0, 48, 104, 160, 200, 240]
            
                if phi_station == 4 or phi_station == 5:
                    conditions = [stationEta == i for i in range(1, 7)]
                    offsets = [0, 32, 88, 144, 184, 224]
            
                if phi_station == 6:
                    conditions = [stationEta == i for i in range(1, 7)]
                    offsets = [0, 50, 106, 162, 202, 242]
            
                if phi_station == 8:
                    conditions = [stationEta == i for i in range(1, 7)]
                    offsets = [0, 40, 96, 152, 192, 232]
                
            if eta_pos_index == 0:
                if phi_station != 7:
                    conditions = [stationEta == i for i in range(-6, 0)]
                    offsets = [0, 48, 88, 128, 184, 240]
            
                else:
                    conditions = [stationEta == i for i in range(-6, 0)]
                    offsets = [0, 0, 40, 80, 136, 192]
    
        elif barrel_index == 4:
            if eta_pos_index == 1:
                if phi_station <=2:
                    conditions = [stationEta == i for i in range(1, 7)]
                    offsets = [0, 72, 144, 200, 272, 344]
            
                if phi_station == 3:
                    conditions = [stationEta == i for i in range(1, 7)]
                    offsets = [0, 64, 136, 192, 264, 336]
            
                if phi_station == 4 or phi_station == 5:
                    conditions = [stationEta == i for i in range(1, 7)]
                    offsets = [0, 48, 120, 176, 248, 320]
                
                if phi_station == 6:
                    conditions = [stationEta == i for i in range(1, 7)]
                    offsets = [0, 56, 128, 184, 256, 328]
            
                if phi_station == 7:
                    conditions = [stationEta == i for i in range(1, 7)]
                    offsets = [0, 56, 104, 152, 224, 296]
            
                if phi_station == 8:
                    conditions = [stationEta == i for i in range(1, 7)]
                    offsets = [0, 56, 128, 176, 248, 320]
        
            if eta_pos_index == 0:
                if phi_station <=6 and phi_station != 3:
                    conditions = [stationEta == i for i in range(-6, 0)]
                    offsets = [0, 56, 128, 200, 256, 328]
            
                if phi_station == 3:
                    conditions = [stationEta == i for i in range(-6, 0)]
                    offsets = [0, 56, 128, 200, 256, 304]
          
                if phi_station == 7:
                    conditions = [stationEta == i for i in range(-6, 0)]
                    offsets = [0, 56, 128, 200, 248, 296]
            
                if phi_station == 8:
                    conditions = [stationEta == i for i in range(-6, 0)]
                    offsets = [0, 56, 128, 200, 248, 320]
        
        tube2 = tube + np.select(conditions, offsets, default=-1)
    
    if image_format == 's':
        if barrel_index == 3:
            if eta_pos_index == 1:
                if (2*eta_region) % 2 == 0:
                    tube2 = tube
                elif eta_region == 1.5:
                    conditions = [
                        stationEta ==  1,
                        stationEta ==  2]
                
                    offsets = [-28,28]
                    tube2 = tube + np.select(conditions, offsets, default=-1)
            
                elif eta_region > 1.5 and eta_region < 5.5 and (2*eta_region) % 2 != 0:
                    conditions = [
                        stationEta ==  int(eta_region - 0.5),
                        stationEta ==  int(eta_region + 0.5)]
                
                    offsets = [-20,28]
                    tube2 = tube + np.select(conditions, offsets, default=-1)
            
                else:
                    conditions = [
                        stationEta ==  5,
                        stationEta ==  6]
                
                    offsets = [-4,28]
                    tube2 = tube + np.select(conditions, offsets, default=-1)
            
                
            elif eta_pos_index == 0:
                if (2*eta_region) % 2 == 0:
                    tube2 = tube
                elif eta_region != 4.5 and (2*eta_region) % 2 != 0:
                    conditions = [
                        stationEta ==  int(-(eta_region + 0.5)),
                        stationEta ==  int(-(eta_region - 0.5))]
                
                    offsets = [-20,28]
                    tube2 = tube + np.select(conditions, offsets, default=-1)
                else:
                    conditions = [
                        stationEta ==  -5,
                        stationEta ==  -4]
                
                    offsets = [-4,28]
                    tube2 = tube + np.select(conditions, offsets, default=-1)
    
        elif barrel_index == 1:
            if eta_pos_index == 1:
                if (2*eta_region) % 2 == 0:
                    tube2 = tube
                elif eta_region == 1.5:
                    conditions = [
                        stationEta ==  1,
                        stationEta ==  2]
                
                    offsets = [-35,36]
                    tube2 = tube + np.select(conditions, offsets, default=-1)
            
                elif eta_region > 1.5 and (2*eta_region) % 2 != 0:
                    conditions = [
                        stationEta ==  int(eta_region - 0.5),
                        stationEta ==  int(eta_region + 0.5)]
                
                    offsets = [-22,36]
                    tube2 = tube + np.select(conditions, offsets, default=-1)
            
            elif eta_pos_index == 0:
                if (2*eta_region) % 2 == 0:
                    tube2 = tube
                else:
                    conditions = [
                        stationEta ==  int(-(eta_region + 0.5)),
                        stationEta ==  int(-(eta_region - 0.5))]
                
                    offsets = [-22,36]
                    tube2 = tube + np.select(conditions, offsets, default=-1)
        
        elif barrel_index == 5:
            if eta_pos_index == 1:
                if phi_station != 4:
                    if (2*eta_region) % 2 == 0:
                        tube2 = tube
                    else:
                        conditions = [
                            stationEta ==  int(eta_region - 0.5),
                            stationEta ==  int(eta_region + 0.5)]
                
                        offsets = [-36,36]
                        tube2 = tube + np.select(conditions, offsets, default=-1)
            
                else:
                    if (2*eta_region) % 2 == 0:
                        tube2 = tube
                    elif eta_region == 1.5:
                        conditions = [
                            stationEta ==  1,
                            stationEta ==  2]
                
                        offsets = [-12,36]
                        tube2 = tube + np.select(conditions, offsets, default=-1)
            
                    elif eta_region > 1.5 and (2*eta_region) % 2 != 0:
                        conditions = [
                            stationEta ==  int(eta_region - 0.5),
                            stationEta ==  int(eta_region + 0.5)]
                
                        offsets = [-36,36]
                        tube2 = tube + np.select(conditions, offsets, default=-1)
            
            elif eta_pos_index == 0:
                if (2*eta_region) % 2 == 0:
                    tube2 = tube
                elif eta_region != 5.5 and (2*eta_region) % 2 != 0:
                    conditions = [
                        stationEta ==  int(-(eta_region + 0.5)),
                        stationEta ==  int(-(eta_region - 0.5))]
                
                    offsets = [-36,36]
                    tube2 = tube + np.select(conditions, offsets, default=-1)
                else:
                    conditions = [
                        stationEta ==  -6,
                        stationEta ==  -5]
                
                    offsets = [-28,36]
                    tube2 = tube + np.select(conditions, offsets, default=-1)
            
        elif barrel_index == 0:
            if eta_pos_index == 1:
                if (2*eta_region) % 2 == 0:
                    tube2 = tube
                else:            
                    if phi_station == 1 or phi_station == 2 or phi_station == 5:
                        if eta_region == 2.5 or eta_region == 4.5:
                            conditions = [
                                stationEta ==  int(eta_region - 0.5),
                                stationEta ==  int(eta_region + 0.5)]
                
                            offsets = [-18,18]
                            tube2 = tube + np.select(conditions, offsets, default=-1)
                        else:
                            conditions = [
                                stationEta ==  int(eta_region - 0.5),
                                stationEta ==  int(eta_region + 0.5)]
                
                            offsets = [-12,18]
                            tube2 = tube + np.select(conditions, offsets, default=-1)
        
                    if phi_station == 3:
                        if eta_region > 1.5 and eta_region < 5.5:
                            conditions = [
                                stationEta ==  int(eta_region - 0.5),
                                stationEta ==  int(eta_region + 0.5)]
                
                            offsets = [-18,18]
                            tube2 = tube + np.select(conditions, offsets, default=-1)
                        else:
                            conditions = [
                                stationEta ==  int(eta_region - 0.5),
                                stationEta ==  int(eta_region + 0.5)]
                
                            offsets = [-12,18]
                            tube2 = tube + np.select(conditions, offsets, default=-1)
        
                    if phi_station == 4:
                        if eta_region == 2.5 or eta_region == 4.5:
                            conditions = [
                                stationEta ==  int(eta_region - 0.5),
                                stationEta ==  int(eta_region + 0.5)]
                
                            offsets = [-18,18]
                            tube2 = tube + np.select(conditions, offsets, default=-1)
                        elif eta_region == 3.5 or eta_region == 5.5:
                            conditions = [
                                stationEta ==  int(eta_region - 0.5),
                                stationEta ==  int(eta_region + 0.5)]
                
                            offsets = [-12,18]
                            tube2 = tube + np.select(conditions, offsets, default=-1)
                        else:
                            conditions = [
                                stationEta ==  1,
                                stationEta ==  2]
                
                            offsets = [-6,18]
                            tube2 = tube + np.select(conditions, offsets, default=-1)
        
                    if phi_station == 7:
                        if eta_region != 2.5:
                            conditions = [
                                stationEta ==  int(eta_region - 0.5),
                                stationEta ==  int(eta_region + 0.5)]
                
                            offsets = [-12,18]
                            tube2 = tube + np.select(conditions, offsets, default=-1)
                        else:
                            conditions = [
                                stationEta ==  2,
                                stationEta ==  3]
                
                            offsets = [-18,18]
                            tube2 = tube + np.select(conditions, offsets, default=-1)
            
            if eta_pos_index == 0:
                if (2*eta_region) % 2 == 0:
                    tube2 = tube
                else:            
                    if phi_station == 1 or phi_station == 2 or phi_station == 5:
                        if eta_region == 5.5 or eta_region == 3.5 or eta_region == 1.5:
                            conditions = [
                                stationEta ==  int(-(eta_region + 0.5)),
                                stationEta ==  int(-(eta_region - 0.5))]
                
                            offsets = [-18,18]
                            tube2 = tube + np.select(conditions, offsets, default=-1)
                        else:
                            conditions = [
                                stationEta ==  int(-(eta_region + 0.5)),
                                stationEta ==  int(-(eta_region - 0.5))]
                
                            offsets = [-12,18]
                            tube2 = tube + np.select(conditions, offsets, default=-1)
        
                    if phi_station == 3:
                        if eta_region != 4.5:
                            conditions = [
                                stationEta ==  int(-(eta_region + 0.5)),
                                stationEta ==  int(-(eta_region - 0.5))]
                
                            offsets = [-18,18]
                            tube2 = tube + np.select(conditions, offsets, default=-1)
                        else:
                            conditions = [
                                stationEta ==  -5,
                                stationEta ==  -4]
                
                            offsets = [-12,18]
                            tube2 = tube + np.select(conditions, offsets, default=-1)
        
                    if phi_station == 4:
                        if eta_region == 1.5 or eta_region == 3.5:
                            conditions = [
                                stationEta ==  int(-(eta_region + 0.5)),
                                stationEta ==  int(-(eta_region - 0.5))]
                
                            offsets = [-18,18]
                            tube2 = tube + np.select(conditions, offsets, default=-1)
                        else:
                            conditions = [
                                stationEta ==  int(-(eta_region + 0.5)),
                                stationEta ==  int(-(eta_region - 0.5))]
                
                            offsets = [-12,18]
                            tube2 = tube + np.select(conditions, offsets, default=-1)
        
                    if phi_station == 7:
                        if eta_region != 1.5:
                            conditions = [
                                stationEta ==  int(-(eta_region + 0.5)),
                                stationEta ==  int(-(eta_region - 0.5))]
                
                            offsets = [-12,18]
                            tube2 = tube + np.select(conditions, offsets, default=-1)
                        else:
                            conditions = [
                                stationEta ==  -2,
                                stationEta ==  -1]
                
                            offsets = [-18,18]
                            tube2 = tube + np.select(conditions, offsets, default=-1)
    
        elif barrel_index == 2:
            if eta_pos_index == 1:
                if (2*eta_region) % 2 == 0:
                    tube2 = tube
                else:
                    if phi_station <= 2:
                        if eta_region <= 3.5:
                            conditions = [
                                stationEta ==  int(eta_region - 0.5),
                                stationEta ==  int(eta_region + 0.5)]
                
                            offsets = [-28,28]
                            tube2 = tube + np.select(conditions, offsets, default=-1)
                        else:
                            conditions = [
                                stationEta ==  int(eta_region - 0.5),
                                stationEta ==  int(eta_region + 0.5)]
                
                            offsets = [-12,28]
                            tube2 = tube + np.select(conditions, offsets, default=-1)
            
                    if phi_station == 3 or phi_station == 7:
                        if eta_region == 1.5:
                            conditions = [
                                stationEta ==  1,
                                stationEta ==  2]
                
                            offsets = [-20,28]
                            tube2 = tube + np.select(conditions, offsets, default=-1)
                        elif eta_region == 2.5 or eta_region == 3.5:
                            conditions = [
                                stationEta ==  int(eta_region - 0.5),
                                stationEta ==  int(eta_region + 0.5)]
                
                            offsets = [-28,28]
                            tube2 = tube + np.select(conditions, offsets, default=-1)
                        else:
                            conditions = [
                                stationEta ==  int(eta_region - 0.5),
                                stationEta ==  int(eta_region + 0.5)]
                
                            offsets = [-12,28]
                            tube2 = tube + np.select(conditions, offsets, default=-1)
            
                    if phi_station == 4 or phi_station == 5:
                        if eta_region == 1.5:
                            conditions = [
                                stationEta ==  1,
                                stationEta ==  2]
                
                            offsets = [-4,28]
                            tube2 = tube + np.select(conditions, offsets, default=-1)
                        elif eta_region == 2.5 or eta_region == 3.5:
                            conditions = [
                                stationEta ==  int(eta_region - 0.5),
                                stationEta ==  int(eta_region + 0.5)]
                
                            offsets = [-28,28]
                            tube2 = tube + np.select(conditions, offsets, default=-1)
                        else:
                            conditions = [
                                stationEta ==  int(eta_region - 0.5),
                                stationEta ==  int(eta_region + 0.5)]
                
                            offsets = [-12,28]
                            tube2 = tube + np.select(conditions, offsets, default=-1)
            
                    if phi_station == 6:
                        if eta_region == 1.5:
                            conditions = [
                                stationEta ==  1,
                                stationEta ==  2]
                
                            offsets = [-22,28]
                            tube2 = tube + np.select(conditions, offsets, default=-1)
                        elif eta_region == 2.5 or eta_region == 3.5:
                            conditions = [
                                stationEta ==  int(eta_region - 0.5),
                                stationEta ==  int(eta_region + 0.5)]
                
                            offsets = [-28,28]
                            tube2 = tube + np.select(conditions, offsets, default=-1)
                        else:
                            conditions = [
                                stationEta ==  int(eta_region - 0.5),
                                stationEta ==  int(eta_region + 0.5)]
                
                            offsets = [-12,28]
                            tube2 = tube + np.select(conditions, offsets, default=-1)
            
                    if phi_station == 8:
                        if eta_region == 2.5 or eta_region == 3.5:
                            conditions = [
                                stationEta ==  int(eta_region - 0.5),
                                stationEta ==  int(eta_region + 0.5)]
                
                            offsets = [-28,28]
                            tube2 = tube + np.select(conditions, offsets, default=-1)
                        else:
                            conditions = [
                                stationEta ==  int(eta_region - 0.5),
                                stationEta ==  int(eta_region + 0.5)]
                
                            offsets = [-12,28]
                            tube2 = tube + np.select(conditions, offsets, default=-1)
                
            if eta_pos_index == 0:
                if (2*eta_region) % 2 == 0:
                    tube2 = tube
                else:
                    if eta_region == 5.5:
                        conditions = [
                            stationEta ==  -6,
                            stationEta ==  -5]
                
                        offsets = [-20,28]
                        tube2 = tube + np.select(conditions, offsets, default=-1)
                    elif eta_region == 3.5 or eta_region == 4.5:
                        conditions = [
                            stationEta ==  int(-(eta_region + 0.5)),
                            stationEta ==  int(-(eta_region - 0.5))]
                
                        offsets = [-12,28]
                        tube2 = tube + np.select(conditions, offsets, default=-1)
                    else:
                        conditions = [
                            stationEta ==  int(-(eta_region + 0.5)),
                            stationEta ==  int(-(eta_region - 0.5))]
                
                        offsets = [-28,28]
                        tube2 = tube + np.select(conditions, offsets, default=-1)
    
        elif barrel_index == 4:
            if eta_pos_index == 1:
                if (2*eta_region) % 2 == 0:
                    tube2 = tube
                else:
                    if phi_station <=2:
                        if eta_region == 3.5:
                            conditions = [
                                stationEta ==  3,
                                stationEta ==  4]
                
                            offsets = [-20,36]
                            tube2 = tube + np.select(conditions, offsets, default=-1)
                        else:
                            conditions = [
                                stationEta ==  int(eta_region - 0.5),
                                stationEta ==  int(eta_region + 0.5)]
                
                            offsets = [-36,36]
                            tube2 = tube + np.select(conditions, offsets, default=-1)
            
                    if phi_station == 3:
                        if eta_region == 1.5:
                            conditions = [
                                stationEta ==  1,
                                stationEta ==  2]
                
                            offsets = [-28,36]
                            tube2 = tube + np.select(conditions, offsets, default=-1)
                        elif eta_region == 3.5:
                            conditions = [
                                stationEta ==  3,
                                stationEta ==  4]
                
                            offsets = [-20,36]
                            tube2 = tube + np.select(conditions, offsets, default=-1)
                        else:
                            conditions = [
                                stationEta ==  int(eta_region - 0.5),
                                stationEta ==  int(eta_region + 0.5)]
                
                            offsets = [-36,36]
                            tube2 = tube + np.select(conditions, offsets, default=-1)
            
                    if phi_station == 4 or phi_station == 5:
                        if eta_region == 1.5:
                            conditions = [
                                stationEta ==  1,
                                stationEta ==  2]
                
                            offsets = [-12,36]
                            tube2 = tube + np.select(conditions, offsets, default=-1)
                        elif eta_region == 3.5:
                            conditions = [
                                stationEta ==  3,
                                stationEta ==  4]
                
                            offsets = [-20,36]
                            tube2 = tube + np.select(conditions, offsets, default=-1)
                        else:
                            conditions = [
                                stationEta ==  int(eta_region - 0.5),
                                stationEta ==  int(eta_region + 0.5)]
                
                            offsets = [-36,36]
                            tube2 = tube + np.select(conditions, offsets, default=-1)
                
                    if phi_station == 6:
                        if eta_region == 1.5 or eta_region == 3.5:
                            conditions = [
                                stationEta ==  int(eta_region - 0.5),
                                stationEta ==  int(eta_region + 0.5)]
                
                            offsets = [-20,36]
                            tube2 = tube + np.select(conditions, offsets, default=-1)
                        else:
                            conditions = [
                                stationEta ==  int(eta_region - 0.5),
                                stationEta ==  int(eta_region + 0.5)]
                
                            offsets = [-36,36]
                            tube2 = tube + np.select(conditions, offsets, default=-1)
            
                    if phi_station == 7:
                        if eta_region == 1.5 or eta_region == 6.5:
                            conditions = [
                                stationEta ==  int(eta_region - 0.5),
                                stationEta ==  int(eta_region + 0.5)]
                
                            offsets = [-20,36]
                            tube2 = tube + np.select(conditions, offsets, default=-1)
                        elif eta_region == 2.5 or eta_region == 3.5:
                            conditions = [
                                stationEta ==  int(eta_region - 0.5),
                                stationEta ==  int(eta_region + 0.5)]
                
                            offsets = [-12,36]
                            tube2 = tube + np.select(conditions, offsets, default=-1)
                        else:
                            conditions = [
                                stationEta ==  int(eta_region - 0.5),
                                stationEta ==  int(eta_region + 0.5)]
                
                            offsets = [-36,36]
                            tube2 = tube + np.select(conditions, offsets, default=-1)
            
                    if phi_station == 8:
                        if eta_region == 1.5:
                            conditions = [
                                stationEta ==  1,
                                stationEta ==  2]
                
                            offsets = [-20,36]
                            tube2 = tube + np.select(conditions, offsets, default=-1)
                        elif eta_region == 3.5:
                            conditions = [
                                stationEta ==  3,
                                stationEta ==  4]
                
                            offsets = [-12,36]
                            tube2 = tube + np.select(conditions, offsets, default=-1)
                        else:
                            conditions = [
                                stationEta ==  int(eta_region - 0.5),
                                stationEta ==  int(eta_region + 0.5)]
                
                            offsets = [-36,36]
                            tube2 = tube + np.select(conditions, offsets, default=-1)
        
            if eta_pos_index == 0:
                if (2*eta_region) % 2 == 0:
                    tube2 = tube
                else:
                    if phi_station <=6 and phi_station != 3:
                        if eta_region == 2.5 or eta_region == 5.5:
                            conditions = [
                                stationEta ==  int(-(eta_region + 0.5)),
                                stationEta ==  int(-(eta_region - 0.5))]
                
                            offsets = [-20,36]
                            tube2 = tube + np.select(conditions, offsets, default=-1)
                        else:
                            conditions = [
                                stationEta ==  int(-(eta_region + 0.5)),
                                stationEta ==  int(-(eta_region - 0.5))]
                
                            offsets = [-36,36]
                            tube2 = tube + np.select(conditions, offsets, default=-1)
            
                    if phi_station == 3:
                        if eta_region == 2.5 or eta_region == 5.5:
                            conditions = [
                                stationEta ==  int(-(eta_region + 0.5)),
                                stationEta ==  int(-(eta_region - 0.5))]
                
                            offsets = [-20,36]
                            tube2 = tube + np.select(conditions, offsets, default=-1)
                        elif eta_region == 1.5:
                            conditions = [
                                stationEta ==  -2,
                                stationEta ==  -1]
                
                            offsets = [-12,36]
                            tube2 = tube + np.select(conditions, offsets, default=-1)
                        else:
                            conditions = [
                                stationEta ==  int(-(eta_region + 0.5)),
                                stationEta ==  int(-(eta_region - 0.5))]
                
                            offsets = [-36,36]
                            tube2 = tube + np.select(conditions, offsets, default=-1)
            
                    if phi_station == 7:
                        if eta_region == 6.5 or eta_region == 4.5 or eta_region == 3.5:
                            conditions = [
                                stationEta ==  int(-(eta_region + 0.5)),
                                stationEta ==  int(-(eta_region - 0.5))]
                
                            offsets = [-36,36]
                            tube2 = tube + np.select(conditions, offsets, default=-1)
                        elif eta_region == 5.5:
                            conditions = [
                                stationEta ==  -6,
                                stationEta ==  -5]
                
                            offsets = [-20,36]
                            tube2 = tube + np.select(conditions, offsets, default=-1)
                        else:
                            conditions = [
                                stationEta ==  int(-(eta_region + 0.5)),
                                stationEta ==  int(-(eta_region - 0.5))]
                
                            offsets = [-12,36]
                            tube2 = tube + np.select(conditions, offsets, default=-1)
            
                    if phi_station == 8:
                        if eta_region == 4.5 or eta_region == 3.5 or eta_region == 1.5:
                            conditions = [
                                stationEta ==  int(-(eta_region + 0.5)),
                                stationEta ==  int(-(eta_region - 0.5))]
                
                            offsets = [-36,36]
                            tube2 = tube + np.select(conditions, offsets, default=-1)
                        elif eta_region == 5.5:
                            conditions = [
                                stationEta ==  -6,
                                stationEta ==  -5]
                
                            offsets = [-20,36]
                            tube2 = tube + np.select(conditions, offsets, default=-1)
                        else:
                            conditions = [
                                stationEta ==  -3,
                                stationEta ==  -2]
                
                            offsets = [-12,36]
                            tube2 = tube + np.select(conditions, offsets, default=-1)
    
    driftDist = np.asarray(event['driftR'])
    
    mask = isBI & isStation_eta & isStation_phi
    
    event_id = eventNumber
    tube = np.asarray(tube)[mask]
    tube2 = np.asarray(tube2)[mask]
    layer2=np.asarray(layer2)[mask]
    isMuon=np.asarray(isMuon)[mask]
    displacement=np.asarray(displacement)[mask]
    driftDist=np.asarray(driftDist)[mask]
    eta_station = np.asarray(stationEta)[mask]
    
    image, eta_index=create_event_image(tube2, layer2, driftDist, num_bins, eta_station)
    image_labels=create_label_image(tube2, layer2, isMuon, num_bins)
    image_displacement= create_displacement_image(tube2,layer2,displacement, num_bins)
      
    hits_per_layer=layer_rate(image)
    hits_rates.append(hits_per_layer)
    return image, image_labels, np.asarray(hits_rates), eta_index, event_id, image_displacement

def merge_data_and_background(events, background):
    """
    Merges the data from real background events into the data of simulated events.
    events - array containing simulated events images
    background - array containing real background images
    """
    if len(background) < len(events):
        length = len(background)
    else: 
        length = len(events)
    for i in range(length):
        events[i][events[i]==0] = background[i][events[i]==0]
    events = np.array(events)
    return events

def merge_data_and_muons(events, muons, events_labels, muon_labels, eta_station, muon_eta_station, label_single = False):
    """
    Merges the data from single muons coming from muon gun samples into the data of simulated events.
    events - array containing simulated events images
    muons - array containing single muon events images
    events_labels - array conatining label images
    muon_labels - array containing label images for single muons
    eta_station - array containing the eta station information corresponding to each labelled hit
    muon_eta_station - array containing eta station information corresponding to labelled single muons
    label_single - If true, label also single muons when merging with event images
    """
    plot_labels = np.copy(events_labels) # a copy of the label images, but to be filled with entries of 2 for the single-muons mainly for plotting reasons
    if len(muons) < len(events):
        length = len(muons)
    else: 
        length = len(events)
    for i in range(len(events)):
        events[i][events[i]==0] = muons[i][events[i]==0]
        eta_station[i][eta_station[i]==0] = muon_eta_station[i][eta_station[i]==0]
        if label_single:
            events_labels[i][events_labels[i]==0] = muon_labels[i][events_labels[i]==0]
        plot_labels[i][plot_labels[i]==0] = muon_labels[i][plot_labels[i]==0]*2
    return events, events_labels, eta_station, plot_labels

def downsample(events, event_labels, desired_ratio, eta_station, phi_sector, event_number, eta_region_id, displacements, plot_labels):
    """
    Downsamples provided events by removing images containing no signal to fit the desired ratio (#signal_images/total_images)
    events - array containing all event images
    event_labels - array containing even labels
    desired_ratio - float from 0 to 1 corresponding to the desired signal/total images ratio
    eta_station - array containing the eta station index of identified displaced muons
    phi_sector - array containing the phi station of identified hits
    event_number - array containing the event ID number associated to each event image
    eta_region_id - array containing the eta station region each event image covers
    displacements - array containing the displacement value calculated for each displaced muon hit
    plot_labels - array containing seperate labels for displaced muons and prompt muons, useful for analysis.
    """
    summed_labels = np.array([np.sum(event_labels[i]) for i in range(len(event_labels))])
    nb_muons = (summed_labels >= 1).sum()
   
    current_ratio = nb_muons/len(event_labels)

    if current_ratio >= desired_ratio: # Return if current ratio is higher than what is demanded
        return events, event_labels, eta_station, phi_sector, event_number, eta_region_id, displacements, plot_labels
    
    muons = events[summed_labels>=1]
    muon_labels = event_labels[summed_labels>=1]
    muon_eta_stations = eta_station[summed_labels>=1]
    muon_phi_sectors = phi_sector[summed_labels>=1]
    muon_event_numbers = event_number[summed_labels>=1]
    muon_eta_region_id = eta_region_id[summed_labels>=1]
    muon_displacement = displacements[summed_labels>=1]
    muon_plot_labels = plot_labels[summed_labels>=1]
  
    non_muons = events[summed_labels==0]
    non_muon_labels = event_labels[summed_labels==0]
    non_muon_eta_stations = eta_station[summed_labels==0]
    non_muon_phi_sectors = phi_sector[summed_labels==0]
    non_muon_event_numbers = event_number[summed_labels==0]
    non_muon_eta_region_id = eta_region_id[summed_labels==0]
    non_muon_displacement = displacements[summed_labels==0]
    non_muon_plot_labels = plot_labels[summed_labels==0]
    
    randomize_background = np.random.permutation(len(non_muon_labels))
    non_muons_shuffled = non_muons[randomize_background] # Shuffle non muons to vary the output
    non_muon_labels_shuffled = non_muon_labels[randomize_background]
    non_muon_eta_stations_shuffled = non_muon_eta_stations[randomize_background]
    non_muon_phi_sectors_shuffled = non_muon_phi_sectors[randomize_background]
    non_muon_event_numbers_shuffled = non_muon_event_numbers[randomize_background]
    non_muon_eta_region_id_shuffled = non_muon_eta_region_id[randomize_background]
    non_muon_displacement_shuffled = non_muon_displacement[randomize_background]
    non_muon_plot_labels_shuffled = non_muon_plot_labels[randomize_background]
    
    new_non_muon_number = np.floor((1-desired_ratio)*len(muons)/desired_ratio).astype(int)

    new_non_muons = non_muons_shuffled[:new_non_muon_number]
    new_non_muon_labels = non_muon_labels_shuffled[:new_non_muon_number]
    new_events = np.concatenate((muons,new_non_muons))
    new_labels = np.concatenate((muon_labels,new_non_muon_labels))
    new_eta_stations = np.concatenate((muon_eta_stations, non_muon_eta_stations_shuffled[:new_non_muon_number]))
    new_phi_sectors = np.concatenate((muon_phi_sectors, non_muon_phi_sectors_shuffled[:new_non_muon_number]))
    new_event_numbers = np.concatenate((muon_event_numbers, non_muon_event_numbers_shuffled[:new_non_muon_number]))
    new_eta_region_id = np.concatenate((muon_eta_region_id, non_muon_eta_region_id_shuffled[:new_non_muon_number]))
    new_displacement = np.concatenate((muon_displacement, non_muon_displacement_shuffled[:new_non_muon_number]))
    new_plot_labels = np.concatenate((muon_plot_labels, non_muon_plot_labels_shuffled[:new_non_muon_number]))
    randomize_events = np.random.permutation(len(new_labels)) #Shuffle again to prevent muon images being grouped together
    
    new_events_shuffled = new_events[randomize_events]
    new_labels_shuffled = new_labels[randomize_events]
    new_eta_shuffled = new_eta_stations[randomize_events]
    new_phi_shuffled = new_phi_sectors[randomize_events]
    new_displacement_shuffled = new_displacement[randomize_events]
    new_event_numbers_shuffled = new_event_numbers[randomize_events]
    new_eta_region_id_shuffled = new_eta_region_id[randomize_events]
    new_plot_labels_shuffled = new_plot_labels[randomize_events]
    return new_events_shuffled, new_labels_shuffled, new_eta_shuffled, new_phi_shuffled, new_event_numbers_shuffled, new_eta_region_id_shuffled, new_displacement_shuffled, new_plot_labels_shuffled

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process HDF5 files and save to NPZ.")
    parser.add_argument('-i','--input', type=str, required=True, help="Path to the input HDF5 file.")
    parser.add_argument('-o', '--output', type=str, required=True, help="Path to the output NPZ file.")
    parser.add_argument('-b', '--background', type=str, required=False, default ="", help="Path to the background HDF5 file.")
    parser.add_argument('-m', '--muon_gun', type=str, nargs='+', required=False, default =[], help="Paths to the muon gun HDF5 files.")
    parser.add_argument('-n', '--numbarrel', type=int, choices=range(0, 6), default=3, help="Optional barrel index (0-5), default is 3.")
    parser.add_argument('-r', '--ratio', type=float, default=0.5, help="Optional signal to background ratio.")
    parser.add_argument('-e', '--eta_pos', type=int, choices=[0,1], default=1, help="Ability to obtain images for either positive or negative eta stations. 0 is for negative, 1 (default) is for positive.")
    parser.add_argument('-f', '--format', type=str, choices=['s','h'],required=True, help="Output image format (s --> single eta region, h --> half eta range images (either positive or negative eta stations))")
    parser.add_argument('-phi', '--phi_stations', type=int, nargs='+', choices=[1, 2, 3, 4, 5, 6, 7, 8], required = True, help='List of phi station indices (e.g., --p 1 2 3 4 5 8)')
    parser.add_argument('-l', '--label_single', action="store_true", default = False, help="Enable single muon labelling")
    args = parser.parse_args()

    events, events_labels, eta_station, phi_sector, event_number, eta_region_id, displacements  = load_data_from_hdf5(args.input, args.numbarrel)

    # Add background noise to signal, if provided
    if args.background != "":
        events_bg, events_labels_bg, eta_station_bg, phi_sector_bg, event_number_bg, eta_region_id_bg, displacement_bg = load_data_from_hdf5(args.background, args.numbarrel)
        events = merge_data_and_background(events, events_bg)
    
    label_single = args.label_single # If TRUE single muons are ALSO labelled!
    
    # Add single muons to images, non-displaced, if provided
    if args.muon_gun != []:
        muon_guns = args.muon_gun
        for muon_gun in muon_guns:
            events_mg, events_labels_mg, eta_station_mg, phi_sector_mg, event_number_mg, eta_region_id_mg, displacement_mg = load_data_from_hdf5(muon_gun, args.numbarrel)
            events, events_labels, eta_station, plot_labels = merge_data_and_muons(events, events_mg, events_labels, events_labels_mg, eta_station, eta_station_mg, label_single)   
    # Downsample the events
    events, events_labels, eta_station, phi_sector, event_number, eta_region_id, displacements, plot_labels = downsample(events=events, event_labels=events_labels, desired_ratio=args.ratio, eta_station=eta_station, phi_sector=phi_sector, event_number=event_number, eta_region_id=eta_region_id, displacements=displacements, plot_labels=plot_labels)
    
    print("Final image shape: ",events.shape)
    
    station_index = np.full(events.shape[0], args.numbarrel)
    image_format = np.full(events.shape[0], args.format)

    if image_format[0] == 'h':
        print("Image format: Half-eta range")
    if image_format[0] == 's':
        print("Image format: Single-eta range")
    print("Station index: ", station_index[0])

    np.savez(args.output,events=events, events_labels=events_labels, eta_station=eta_station, phi_sector=phi_sector, event_number=event_number, eta_region_id=eta_region_id, station_index = station_index, image_format = image_format, displacement=displacements, plot_labels=plot_labels)
