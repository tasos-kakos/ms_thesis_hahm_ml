import ROOT
ROOT.gSystem.Load('libRVecSignedChar.so') # Load necessary dictionaries
import numpy as np
import h5py
import cppyy
import argparse


def save_events_to_hdf5(events, file_name):
    with h5py.File(file_name, 'w') as h5file:
        for i, event in enumerate(events):
            grp = h5file.create_group(f'event_{i}')
            for key, data in event.items():
                grp.create_dataset(key, data=data)

def process_events(input_path, output_path, max_events=10, onlyOneSector=True):
    tree_name = 'BasicTesterTree'

    vector_names = [
        'eventNumber','trkMdt_driftR', 'MdtChamHitId_multiLayer', 'MdtChamHitId_stationEta', 'MdtChamHitId_stationPhi', 'MdtChamHitId_stationIndex', 'MdtChamHitId_tubeLayer', 'MdtChamHitId_tube', 'MdtChamHit_Adc', 'MdtChamHit_Tdc', 'MdtChamHit_ChamberLink', 'MdtChamHit_driftR', 'MdtChamHit_uncertDriftR', 'MdtChamId_multiLayer', 'MdtChamId_stationEta', 'MdtChamId_stationPhi', 'MdtChamId_stationIndex', 'MdtChamId_tube', 'MdtChamId_tubeLayer', 'MdtCham_NTubesPerML', 'MdtCham_hitN', 'MdtCham_matchedCaloTag', 'MdtCham_matchedCaloTag_IPcut', 'MdtCham_matchedMuon', 'MdtCham_matchedMuon_IPcut', 
    ]


    df = ROOT.RDataFrame(tree_name, input_path)
    data = df.AsNumpy(columns=vector_names)
    events = []

    for i, rvec in enumerate(data['MdtChamHitId_stationEta']):
        if i > 100000:
            break
        if not (i%1000): # Print index every 1k values
            print("___________________________________ NEW EVENT MDT = ", i, " _____________________________________")
        
        event={}
        mask = np.ones(len(data['MdtChamHitId_tube'][i]), dtype=bool)
        mask[np.isin(data['MdtChamHit_driftR'][i], data['trkMdt_driftR'][i])] = False
    
        station_eta = data['MdtChamHitId_stationEta'][i]
        arr = np.array(list(station_eta), dtype=np.int8)
        station_eta_signed_int = arr.tolist()
        mask_station_eta = np.isin(np.asarray(station_eta_signed_int), [-7, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 7])
        
        station_index = data['MdtChamHitId_stationIndex'][i]
        station_index = station_index.view(np.uint8)
        mask_station_index = np.isin(np.asarray(station_index), [0, 1, 2, 3, 4, 5])

        station_phi = data['MdtChamHitId_stationPhi'][i]
        station_phi = station_phi.view(np.uint8)  
        mask_station_phi = np.isin(np.asarray(station_phi), [1, 2, 3, 4, 5, 6, 7, 8])

        final_mask = mask_station_eta & mask_station_phi & mask_station_index & mask
        
        event_variables = ['MdtChamHitId_multiLayer', 'MdtChamHitId_stationEta', 'MdtChamHitId_stationPhi', 'MdtChamHitId_stationIndex', 'MdtChamHitId_tubeLayer', 'MdtChamHitId_tube']

        for name in event_variables:
            name_dict = name.split("_")[-1]
            values = data[name][i]
            if type(values) == type(data['MdtChamHitId_stationEta'][i]):
                arr_values = np.array(list(values), dtype=np.int8)
                values = arr_values.tolist()
            if type(values) == type(data['MdtChamHitId_stationIndex'][i]):
                values = values.view(np.uint8)
            if onlyOneSector:
                filtered_values = np.asarray(values)[final_mask]
                event[name_dict] = np.asarray(filtered_values)
            else:
                event[name_dict] = np.asarray(values)

        driftR = data['MdtChamHit_driftR'][i]
        filtered_values = np.asarray(driftR)[final_mask]
        event['driftR'] = np.asarray(filtered_values)
        event['eventNumber'] = data['eventNumber'][i]
    
        isMuon = np.zeros(len(filtered_values), dtype=int)
        event["isMuon"]=np.asarray(isMuon)
        event["muonDisplacement"]=np.zeros(isMuon.shape,dtype=float)
        events.append(event)

    print("---------------------------------------------------------------------------------------------------------------------------------------------")
    print("len(events) = ", len(events))
    save_events_to_hdf5(events, output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process ROOT files and save to HDF5.")
    parser.add_argument('-i','--input', type=str, required=True, help="Path to the input ROOT file.")
    parser.add_argument('-o','--output', type=str, required=True, help="Path to the output HDF5 file.")
    args = parser.parse_args()
    process_events(args.input, args.output)
