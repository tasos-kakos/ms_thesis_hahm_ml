import ROOT
ROOT.gSystem.Load('libRVecSignedChar.so') # Load necessary RVec dictionaries
import numpy as np
import h5py
import sys
import random
import cppyy
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import argparse
import struct

print(sys.version)
print(np.__version__)

import numpy as np

class MdtDigitizationTool:
    def __init__(self, ns2TDC, resTDC):
        self.m_ns2TDC = ns2TDC
        self.m_resTDC = resTDC #from ATHENA MDT_DigitizationTool.h

    def digitizeTime(self, times):
        tmpCounts = times / self.m_ns2TDC
        tdcCounts = np.random.normal(tmpCounts, self.m_resTDC)
        tdcCounts = np.rint(tdcCounts).astype(int)
        return tdcCounts

    def r_t_transform(self, distancesDrift):
        distances = np.array([0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.])
        times = np.array([0., 20., 40., 60., 80., 110., 140., 180., 230., 280., 350., 420., 510., 600., 690., 780.])
        interp_function = interp1d(distances, times, kind='linear', fill_value='extrapolate')
        timesDrift = interp_function(distancesDrift)
        return timesDrift

def rvec_signedchar_to_numpy(rvec):
    """
    Convert ROOT::VecOps::RVec<signed char> to numpy int8. 
    Used for muon gun sample data conversion.     
    """
    n = len(rvec)
    arr = []
    for i in range(n):
        v = rvec[i]
        if isinstance(v, (bytes, bytearray)):
            # interpret raw byte
            arr.append(int(np.int8(v[0])))
        elif isinstance(v, str):
            # take char code, then cast to int8
            arr.append(int(np.int8(ord(v))))
        else:
            # already a number
            arr.append(int(np.int8(v)))
    return np.array(arr, dtype=np.int8)

def save_events_to_hdf5(events, file_name):
    with h5py.File(file_name, 'w') as h5file:
        for i, event in enumerate(events):
            grp = h5file.create_group(f'event_{i}')
            for key, data in event.items():
                grp.create_dataset(key, data=data)


def get_maxTubesValue(station_name, station_phi, station_eta):
    key = (station_name, station_phi, station_eta)
    maxTubesValue = {            ####### (stationName, stationPhi, stationEta): max number of tubes
        (0, 3, 1): 30,           ####### stationName: 0->BIL, 2->BML, 4->BOL
        (0, 3, 2): 36,
        (0, 3, 3): 36,
        (0, 3, 4): 36,
        (0, 3, 5): 30,
        (0, 3, 6): 36,
        (2, 3, 1): 48,
        (2, 3, 2): 56,
        (2, 3, 3): 56,
        (2, 3, 4): 40,
        (2, 3, 5): 40,
        (2, 3, 6): 48,
        (4, 3, 1): 64,
        (4, 3, 2): 72,
        (4, 3, 3): 56,
        (4, 3, 4): 72,
        (4, 3, 5): 72,
        (4, 3, 6): 56,
    }
    return maxTubesValue.get(key)

def generate_bkg_hit(event):
    bkg_rate = 0.1
    stationPhi = -99
    stationEta = -99
    for stationIndex in [0, 2, 4]:
        if (stationIndex == 0):
            Nlayers = 4
        else:
            Nlayers = 3
        for stationEta in range(1,7):
            nMaxTubes = get_maxTubesValue(stationIndex, 3, stationEta)
            for multilayer in [1,2]:
                for layer in range (1, Nlayers+1):
                    N = np.random.binomial(nMaxTubes, bkg_rate)
                    bkgTubes=np.random.uniform(1, 72, N)
                    bkgTubes=np.full(N, -1)
                    event["tube"] = np.append(event["tube"], np.asarray(bkgTubes))
                    for tube in bkgTubes:
                        tube = getTubeFromCenter(tube, stationIndex, 3, stationEta)
                    distances = [random.uniform(0, 14.6) for _ in range(N)]
                    event["driftR"] = np.append(event["driftR"], np.asarray(distances))
                    times = tool.r_t_transform(distances)
                    event["Tdc"] = np.append(event["Tdc"], np.asarray(times))
                    tubeLayers = np.full(N, int(layer))
                    event["tubeLayer"] = np.append(event["tubeLayer"], np.asarray(tubeLayers))
                    tubeMultiLayers = np.full(N, int(multilayer))
                    event["multiLayer"] = np.append(event["multiLayer"], np.asarray(tubeMultiLayers))
                    tubeStationIndex = np.full(N, int(stationIndex))
                    event["stationIndex"] = np.append(event["stationIndex"], np.asarray(tubeStationIndex))
                    tubeStationPhi = np.full(N, int(stationPhi))
                    event["stationPhi"] = np.append(event["stationPhi"], np.asarray(tubeStationPhi))
                    tubeStationEta = np.full(N, int(stationEta))
                    event["stationEta"] = np.append(event["stationEta"], np.asarray(tubeStationEta))
                
    return event


def getTubeFromCenter(tubeNumber, stationIndex, stationPhi, stationEta):
    nMaxTubes = get_maxTubesValue(stationIndex, stationPhi, stationEta)
    center = (nMaxTubes + 1) / 2
    centeredTube = tubeNumber - center
    return centeredTube

def process_events(input_path, output_path, generate_bkg, max_events=10, onlyOneSector=True, dark_photon_mask=True):
    tree_name = 'MuonHitTest'
    vector_names = [
        'MdtSimHits_multiLayer', 'MdtSimHits_stationEta', 'MdtSimHits_stationPhi', 'MdtSimHits_stationIndex',
        'MdtSimHits_tubeLayer', 'MdtSimHits_tube', 'MdtSimHitsLocalPosX', 'MdtSimHitsLocalPosY',
        'MdtSimHitsLocalPosZ', 'MdtSimHitsGlobPosX', 'MdtSimHitsGlobPosY', 'MdtSimHitsGlobPosZ', 'MdtSimHitsPdgId',
        'MdtSimHitsGlobDirX', 'MdtSimHitsGlobDirY', 'MdtSimHitsGlobDirZ', 'MdtSimHitsKinericEnergy', 'MdtSimHitsMass',
        'MdtSimHitsMotherBarcode', 'MdtSimHitsMotherPdgId', 'MdtSimHitsPartBarcode', 'eventNumber', 'TruthParticle_Barcode','TruthParticle_Pdg',
        'TruthParticle_End_vertex_id','TruthParticle_Production_vertex_id','TruthVertex_Id', 'TruthVertexX','TruthVertexY','TruthVertexZ',
        'RpcSimHits_doubletPhi', 'RpcSimHits_doubletR', 'RpcSimHits_doubletZ', 'RpcSimHits_gasGap',
        'RpcSimHits_stationEta', 'RpcSimHits_stationPhi', 'RpcSimHits_stationIndex', 'RpcSimHitsLocalPosX',
        'RpcSimHitsLocalPosY', 'RpcSimHitsLocalPosZ', 'RpcSimHitsGlobalTime', 'RpcSimHitsGlobPosX',
        'RpcSimHitsGlobPosY', 'RpcSimHitsGlobPosZ', 'RpcSimHits_measuresPhi', 'RpcSimHits_strip', 'RpcSimHitsPdgId'
    ]

    df = ROOT.RDataFrame(tree_name, input_path)
    data = df.AsNumpy(columns=vector_names)
    events=[]
    event_displacement = []

    # Go through each data point
    for i, rvec in enumerate(data['eventNumber']):
        if i > 100000:
            break # Limit of 100k values
        if not (i%1000): # Print index every 1k values
            print("___________________________________ NEW EVENT MDT = ", i, " _____________________________________")
        event={}
        distances = []
        isMuon=[]
        muon_displacement=[]

        event_number = data['eventNumber'][i]
        mother_barcode = np.asarray(data['MdtSimHitsMotherBarcode'][i])
        mother_pdg_id = np.asarray(data['MdtSimHitsMotherPdgId'][i])
        truth_particle_barcode = np.asarray(data['TruthParticle_Barcode'][i])
        truth_particle_pdg = np.asarray(data['TruthParticle_Pdg'][i])
        mask_dark_photon = np.isin(mother_pdg_id,3000001)
        
        truth_end_vertex = np.asarray(data['TruthParticle_End_vertex_id'][i])
        truth_production_vertex = np.asarray(data['TruthParticle_Production_vertex_id'][i])
        truth_vertex_id = np.asarray(data['TruthVertex_Id'][i])
        truth_vertex_X = np.asarray(data['TruthVertexX'][i])
        truth_vertex_Y = np.asarray(data['TruthVertexY'][i])
        truth_vertex_Z = np.asarray(data['TruthVertexZ'][i])

        station_eta = data['MdtSimHits_stationEta'][i]
        try:
            arr = np.array(list(station_eta), dtype=np.int8) # working conversion for dark photon samples
        except TypeError as err:
            arr = rvec_signedchar_to_numpy(station_eta) # working conversion for muon gun samples
        station_eta = arr.tolist()
        mask_station_eta = np.isin(np.asarray(station_eta), [-7, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 7])
        station_index = data['MdtSimHits_stationIndex'][i]
        station_index = station_index.view(np.uint8)
        station_phi = data['MdtSimHits_stationPhi'][i]
        station_phi = station_phi.view(np.uint8)
        mask_station_index = np.isin(np.asarray(station_index), [0, 1, 2, 3, 4, 5])
        mask_signal = (np.abs(np.asarray(data['MdtSimHitsPdgId'][i])) == 13)
        final_mask = mask_station_eta & mask_station_index & mask_signal 
        if dark_photon_mask:
            final_mask =  final_mask & mask_dark_photon

        mother_particles = mother_barcode[final_mask & mask_dark_photon]

        mask_truth_particles = np.isin(truth_particle_barcode, mother_particles)
        production_vertex = truth_production_vertex[mask_truth_particles]
        end_vertex = truth_end_vertex[mask_truth_particles]
        production_vertex_id = np.isin(truth_vertex_id,production_vertex)
        end_vertex_id = np.isin(truth_vertex_id,end_vertex)

        production_vertex_x = truth_vertex_X[production_vertex_id]
        production_vertex_y = truth_vertex_Y[production_vertex_id]
        production_vertex_z = truth_vertex_Z[production_vertex_id]

        end_vertex_x = truth_vertex_X[end_vertex_id]
        end_vertex_y = truth_vertex_Y[end_vertex_id]
        end_vertex_z = truth_vertex_Z[end_vertex_id]

        displacement = np.sqrt((end_vertex_x - production_vertex_x)**2 + (end_vertex_y - production_vertex_y)**2 +(end_vertex_z - production_vertex_z)**2)

        event_variables=['MdtSimHits_stationEta', 'MdtSimHits_stationIndex', 'MdtSimHits_stationPhi', 'MdtSimHits_multiLayer','MdtSimHits_tubeLayer','MdtSimHits_tube']
        for name in event_variables:
            name_dict = name.split("_")[-1]
            values = data[name][i]
            if type(values) == cppyy.gbl.ROOT.VecOps.RVec['signed char']:
                try:
                    arr_values = np.array(list(values), dtype=np.int8)
                except TypeError as err:
                    arr_values = rvec_signedchar_to_numpy(values)
                values = arr_values.tolist()
            if type(values) == cppyy.gbl.ROOT.VecOps.RVec['unsigned char']:
                values = values.view(np.uint8)
            if onlyOneSector:
                filtered_values = np.asarray(values)[final_mask]
                event[name_dict] = np.asarray(filtered_values)
            else:
                event[name_dict] = np.asarray(values)
        
        event['eventNumber'] = event_number

        tubeNumber = data['MdtSimHits_tube'][i]
        tubeNumber = tubeNumber.view(np.uint8)
        stationIndex = station_index
        stationEta = station_eta
        stationPhi = station_phi
        localPosX = data['MdtSimHitsLocalPosX'][i]
        localPosY = data['MdtSimHitsLocalPosY'][i]
        globDirX = data['MdtSimHitsGlobDirX'][i]
        globDirY = data['MdtSimHitsGlobDirY'][i]
        globDirZ = data['MdtSimHitsGlobDirZ'][i]
        globPosX = data['MdtSimHitsGlobPosX'][i]
        globPosY = data['MdtSimHitsGlobPosY'][i]
        globPosZ = data['MdtSimHitsGlobPosZ'][i]
        kinEn = data['MdtSimHitsKinericEnergy'][i]
        mass = data['MdtSimHitsMass'][i]
        
        # Identify displacement coordinates and calculate displacement for muons originating from a dark photon
        for j in range (0, len(mask_signal)):
            if not final_mask[j]:
                continue
            driftDist =  np.sqrt(localPosX[j]*localPosX[j] + localPosY[j]*localPosY[j])
            distances=np.append(distances, np.array(driftDist))
            mother_particle = mother_barcode[j]

            mask_truth_particle = np.isin(truth_particle_barcode, mother_particle)
            production_vertex = truth_production_vertex[mask_truth_particle]
            end_vertex = truth_end_vertex[mask_truth_particle]
            production_vertex_id = np.isin(truth_vertex_id,production_vertex)
            end_vertex_id = np.isin(truth_vertex_id,end_vertex)

            production_vertex_x = truth_vertex_X[production_vertex_id]
            production_vertex_y = truth_vertex_Y[production_vertex_id]
            production_vertex_z = truth_vertex_Z[production_vertex_id]

            end_vertex_x = truth_vertex_X[end_vertex_id]
            end_vertex_y = truth_vertex_Y[end_vertex_id]
            end_vertex_z = truth_vertex_Z[end_vertex_id]

            displacement_particle = np.sqrt((end_vertex_x - production_vertex_x)**2 + (end_vertex_y - production_vertex_y)**2 +(end_vertex_z - production_vertex_z)**2)
            if displacement_particle.shape[0]==0:
                displacement_particle=0
            isMuon=np.append(isMuon,1)
            muon_displacement=np.append(muon_displacement,displacement_particle)

        timesDrift = tool.r_t_transform(distances)
        tdcCounts = tool.digitizeTime(timesDrift)
        event["isMuon"]=np.asarray(isMuon)
        event["muonDisplacement"]=np.asarray(muon_displacement)
        event["driftR"]=np.asarray(distances)
        event["Tdc"]=np.asarray(tdcCounts)
        
        if (generate_bkg):
            event = generate_bkg_hit(event)
        events.append(event)
        event_displacement.append(displacement)
 
    print("---------------------------------------------------------------------------------------------------------------------------------------------")
    print("len(events) = ", len(events))
    save_events_to_hdf5(events, output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process ROOT files and save to HDF5.")
    parser.add_argument('-i','--input', type=str, required=True, help="Path to the input ROOT file.")
    parser.add_argument('-o', '--output', type=str, required=True, help="Path to the output HDF5 file.")
    parser.add_argument("--generate_bkg", action="store_true", help="Ability to generate background.")
    parser.add_argument('-m',"--mask_photon", action="store_false", help="Disable dark photon mother filtering")
    args = parser.parse_args()
    MDT_ns2TDC = 25/32
    MDT_resTDC = 0.5
    tool = MdtDigitizationTool(MDT_ns2TDC, MDT_resTDC)
    process_events(args.input, args.output, args.generate_bkg, dark_photon_mask=args.mask_photon)
