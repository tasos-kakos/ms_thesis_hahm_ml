import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, VPacker, HPacker, TextArea, DrawingArea
from matplotlib.patches import Circle
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredOffsetbox
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process HDF5 files and save to NPZ.")
    parser.add_argument('-i','--input', type=str, required=True, help="Path to the input NPZ file.")
    parser.add_argument('-m', '--mass', type=float, required=True, help="Dark photon mass in GeV")
    parser.add_argument('-l', '--lifetime', type=float, required=True, help="Dark photon lifetime in mm")
    parser.add_argument('-evs', '--events', type=int, nargs='+', default=[], required = False, help='List of events to save (e.g., --evs 1 2 3 4 5 132 256)')
    parser.add_argument('-n_gr', '--num_graphs', type=int, required=False, default = -1, help="Number of images to plot. A negative or zero value plots all available images")
    args = parser.parse_args()

# Load the .npz file
file_name = args.input
train_data = np.load(file_name, allow_pickle=True)

dark_photon_mass = args.mass # in GeV --> Change to the corresponding dark photon mass, depending on the sample
lifetime = args.lifetime # in mm --> Change to the corresponding dark photon lifetime, depending on the sample

key = list(train_data.keys())[0]  # Get the first key in the .npz file
array = train_data[key].astype(float)  # Loads the event images
label_array = train_data[list(train_data.keys())[1]] # Loads the label images
eta = list(train_data.keys())[2] 
eta_array = train_data[list(train_data.keys())[2]] # Loads the eta images
phi_array = train_data[list(train_data.keys())[3]] # Loads the phi sector information
event_array = train_data[list(train_data.keys())[4]] # Loads the event id of each image
eta_region_array = train_data[list(train_data.keys())[5]] # Loads the eta region of each image
station_index_array = train_data[list(train_data.keys())[6]] # Loads the barrel station of each image
image_format_array = train_data[list(train_data.keys())[7]] # Loads the type of format of each image (single eta or half eta range)
displacement_array = train_data[list(train_data.keys())[8]] # Loads the displacement images
plot_label_array = train_data[list(train_data.keys())[9]] # Loads the images used to distinguish muons and displaced muons

tube_x_index = array.shape[2] # Number of tubes per image

# Ensure the data shape matches the expected format (x, 6, tube_x_index, 1)

if len(array.shape) != 4 or (array.shape[1:] != (6, tube_x_index, 1) and array.shape[1:] != (8, tube_x_index, 1)):
    raise ValueError(f"Unexpected data shape: {array.shape}. Expected (x, 6 or 8, tube_x_index, 1).")

# Reshape the array to (x, n_layers, tube_x_index) by removing the last dimension
array = array.squeeze(-1)

# Keep only images containing signal hits.
mask = np.array([np.sum(lbl) >= 1 for lbl in label_array])

array = array[mask]
label_array = label_array[mask]
eta_array = eta_array[mask]
phi_array = phi_array[mask]
event_array = event_array[mask]
eta_region_array = eta_region_array[mask]
station_index_array = station_index_array[mask]
image_format_array = image_format_array[mask]
displacement_array = displacement_array[mask]
plot_label_array = plot_label_array[mask]

print('Total images containing signal hits: ', label_array.shape[0])

flat_plot_labels = plot_label_array.flatten()
mask_1 = flat_plot_labels == 1
mask_2 = flat_plot_labels == 2
mask_full = mask_1 & mask_2
displaced_muon_hits = flat_plot_labels[mask_1]
prompt_muon_hits = flat_plot_labels[mask_2]
print('displaced muon hits: ', displaced_muon_hits.shape[0])
print('prompt muon hits: ', prompt_muon_hits.shape[0])

# Maximum number of graphs to plot (x)
if args.num_graphs <= 0:
    num_graphs = array.shape[0]
else:
    num_graphs = args.num_graphs

events_to_save = args.events

for i in range(num_graphs): # Choose number of graphs to plot (1 - to num_graphs) 
    eta_index = np.unique(eta_array[i][eta_array[i]!=0])
    event_id = event_array[i][0]
    phi_index = phi_array[i][0]
    eta_region = eta_region_array[i][0]
    
    station_index = station_index_array[i]
    
    if station_index == 0:
        SI = 'BIL'
    elif station_index == 1:
        SI = 'BIS'
    elif station_index == 2:
        SI = 'BML'
    elif station_index == 3:
        SI = 'BMS'
    elif station_index == 4:
        SI = 'BOL'
    elif station_index == 5:
        SI = 'BOS'
    
    image_format = image_format_array[i]
    
    fig, ax = plt.subplots(figsize=(10, 3))
    
    plt.imshow(array[i], aspect='auto', cmap='cividis')
    
    # Set x and y ticks to start at 1
    num_layers = array[i].shape[0]
    num_tubes = array[i].shape[1]
    
    # Set tick frequency
    if image_format == 's':
        tube_ticks = np.arange(-1, num_tubes, 5)
    else:
        tube_ticks = np.arange(-1, num_tubes, 20)
    
    ax.set_xticks(tube_ticks)
    ax.set_yticks(np.arange(num_layers))
    ax.set_xticklabels(tube_ticks + 1)
    ax.set_yticklabels(np.arange(1, num_layers + 1))
    
    plt.colorbar(label="Drift Distance (mm)")
    
    # Overlay signal hits in a different color. 
    plot_label_slice = np.squeeze(plot_label_array[i])
    if plot_label_slice.ndim != 2:
        raise ValueError(f"Expected 2D array after squeezing, got shape {plot_label_slice.shape}")
    
    hits_y, hits_x = np.where(plot_label_slice == 2)
    ax.scatter(hits_x, hits_y, color='cyan', s=70, edgecolor='k', label='Muons')
    
    dis_y, dis_x = np.where(plot_label_slice == 1)
    ax.scatter(dis_x, dis_y, color='red', s=70, edgecolor='k', label='Displaced $μ$ from $Z_d$')
    
    # ============================== Create a combined info box with a legend marker ===============================================#
    # Top text with signal eta station, process info etc.
    top_text = (
        f"$μ$ hits η station(s): {eta_index}\n"
        f"Station Index = {station_index_array[i]} ({SI})\n"
        f"$φ$ sector = {phi_index}\n"
        f"$H \\rightarrow Z_d \\; Z_d$\n"
        r"$m_{Z_d}=$"f" ${dark_photon_mass}$ GeV, $c \\tau={lifetime}$ mm"
    )
    # Create a TextArea for the top text.
    ta_top = TextArea(top_text, textprops=dict(color="black", size=10))
    
    # Create a small drawing area with a cyan circle as the legend marker for prompt muons.
    da = DrawingArea(15, 15, 0, 0)
    circle = Circle((8, 8), 5, fc="cyan", ec="k")
    da.add_artist(circle)
    
    # Create a small drawing area with a red circle as the legend marker for displaced muons.
    da_dis = DrawingArea(15, 15, 0, 0)
    circle_dis = Circle((8, 8), 5, fc="red", ec="k")
    da_dis.add_artist(circle_dis)
    
    # Create a TextArea for the label of displaced muon and prompt muon hits
    ta_hit = TextArea("Prompt $μ$", textprops=dict(color="black", size=10))
    ta_dis = TextArea("Displaced $μ$ from $Z_d$", textprops=dict(color="black", size=10))
    # Pack the circles and their labels horizontally.
    hbox = HPacker(children=[da, ta_hit], align="center", pad=0, sep=5)
    hbox_dis = HPacker(children=[da_dis, ta_dis], align="center", pad=0, sep=5)
    # Pack the top text and the hboxes vertically.
    vbox = VPacker(children=[ta_top, hbox], align="left", pad=0, sep=5)
    vbox_dis = VPacker(children=[vbox, hbox_dis], align="left", pad=0, sep=5)
    # ---- End combined info box ----
    
    # Determine the optimal placement for the info box based on the blank quadrant in the data
    data = array[i]
    nrows, ncols = data.shape
    mid_row, mid_col = nrows // 2, ncols // 2
    
    quad_means = {
        'tl': np.mean(data[:mid_row, :mid_col]),
        'tr': np.mean(data[:mid_row, mid_col:]),
        'bl': np.mean(data[mid_row:, :mid_col]),
        'br': np.mean(data[mid_row:, mid_col:])
    }
    best_quad = min(quad_means, key=quad_means.get)
    
    # Map quadrant to anchored location strings.
    loc_dict = {
        'tl': 'upper left',
        'tr': 'upper right',
        'bl': 'lower left',
        'br': 'lower right'
    }
    anchor_loc = loc_dict[best_quad]
    
    # Create an anchored offset box that will always be fully within the axes.
    anchored_box = AnchoredOffsetbox(loc=anchor_loc, child=vbox_dis, pad=0.5,
                                    frameon=True, borderpad=0.5,
                                    bbox_to_anchor=None,
                                    bbox_transform=ax.transAxes)
    # Configure the box appearance using its patch.
    anchored_box.patch.set_boxstyle("round")
    anchored_box.patch.set_facecolor("white")
    anchored_box.patch.set_alpha(0.5)  
    ax.add_artist(anchored_box)
    #================================================= End of info box ===============================================================#
    plt.title(f"MDT hits (Single event image {i + 1}, Event ID: {event_id})")
    plt.xlabel(f"Tube number ({tube_x_index})")
    plt.ylabel(f"Layer number ({array.shape[1]})")
    
    #===========================================Single Eta Region Plotting ===========================================================#
    
    if station_index == 3 and image_format == 's':
        #Draw limit lines according to event eta region
        if (2*eta_region)%2 != 0:
            plt.axvline(x = tube_x_index/2, color = 'cyan', ls='--')
        else:
            if np.abs(eta_region) != 1 and np.abs(eta_region) != 5:
                plt.axvline(x = 48, color = 'cyan', ls='--')
                plt.axvspan(48, 56, color='black', alpha=0.3, lw=0, label = "Dead space (no tubes here)")
                plt.xlim(0,tube_x_index)
            elif np.abs(eta_region) == 5:
                plt.axvline(x = 32, color = 'cyan', ls='--')
                plt.axvspan(32, 56, color='black', alpha=0.3, lw=0, label = "Dead space (no tubes here)")
                plt.xlim(0,tube_x_index)
                
    if station_index == 1 and image_format == 's':
        if (2*eta_region)%2 != 0:
            plt.axvline(x = tube_x_index/2, color = 'cyan', ls='--')
        else:
            if np.abs(eta_region) == 1:
                plt.axvline(x = 71, color = 'cyan', ls='--')
                plt.axvspan(71, tube_x_index, color='black', alpha=0.3, lw=0, label = "Dead space (no tubes here)")
            else:
                plt.axvline(x = 58, color = 'cyan', ls='--')
                plt.axvspan(58, tube_x_index, color='black', alpha=0.3, lw=0, label = "Dead space (no tubes here)")
            plt.xlim(0,tube_x_index)
            
    if station_index == 5 and image_format == 's':
        if (2*eta_region)%2 != 0:
            plt.axvline(x = tube_x_index/2, color = 'cyan', ls='--')
        else:
            if eta_region == 1 and phi_index == 4:
                plt.axvline(x = 48, color = 'cyan', ls='--')
                plt.axvspan(48, tube_x_index, color='black', alpha=0.3, lw=0, label = "Dead space (no tubes here)")
                plt.xlim(0,tube_x_index)
            if eta_region == 6:
                plt.axvline(x = 64, color = 'cyan', ls='--')
                plt.axvspan(64, tube_x_index, color='black', alpha=0.3, lw=0, label = "Dead space (no tubes here)")
                plt.xlim(0,tube_x_index)
                
    if station_index == 0 and image_format == 's':
        if (2*eta_region)%2 != 0:
            plt.axvline(x = tube_x_index/2, color = 'cyan', ls='--')
        else:
            if eta_region == 1 and phi_index == 4:
                plt.axvline(x = 24, color = 'cyan', ls='--')
                plt.axvspan(24, tube_x_index, color='black', alpha=0.3, lw=0, label = "Dead space (no tubes here)")
                plt.xlim(0,tube_x_index)
            elif eta_region == 1 and phi_index == 7:    
                if np.any(np.isin(eta_array[i], [1, 2, 3, 4, 5, 6, 7])) == True:
                    plt.axvline(x = 30, color = 'cyan', ls='--')
                    plt.axvspan(30, tube_x_index, color='black', alpha=0.3, lw=0, label = "Dead space (no tubes here)")
                    plt.xlim(0,tube_x_index)
            elif eta_region == 1 and phi_index != 4 and phi_index != 7:
                plt.axvline(x = 30, color = 'cyan', ls='--')
                plt.axvspan(30, tube_x_index, color='black', alpha=0.3, lw=0, label = "Dead space (no tubes here)")
                plt.xlim(0,tube_x_index)
            elif eta_region == 3:
                if phi_index != 3:
                    plt.axvline(x = 30, color = 'cyan', ls='--')
                    plt.axvspan(30, tube_x_index, color='black', alpha=0.3, lw=0, label = "Dead space (no tubes here)")
                    plt.xlim(0,tube_x_index)
            elif (eta_region == 4 and phi_index == 7) or eta_region == 5 or (eta_region == 6 and (phi_index == 4 or phi_index == 7)):
                plt.axvline(x = 30, color = 'cyan', ls='--')
                plt.axvspan(30, tube_x_index, color='black', alpha=0.3, lw=0, label = "Dead space (no tubes here)")
                plt.xlim(0,tube_x_index)
    
    if station_index == 2 and image_format == 's':
        if (2*eta_region)%2 != 0:
            plt.axvline(x = tube_x_index/2, color = 'cyan', ls='--')
        else:
            if eta_region == 1:
                if phi_index == 6:
                    plt.axvline(x = 50, color = 'cyan', ls='--')
                    plt.axvspan(50, tube_x_index, color='black', alpha=0.3, lw=0, label = "Dead space (no tubes here)")
                    plt.xlim(0,tube_x_index)
                if phi_index == 8:
                    plt.axvline(x = 40, color = 'cyan', ls='--')
                    plt.axvspan(40, tube_x_index, color='black', alpha=0.3, lw=0, label = "Dead space (no tubes here)")
                    plt.xlim(0,tube_x_index)
                if phi_index == 3:
                    plt.axvline(x = 48, color = 'cyan', ls='--')
                    plt.axvspan(48, tube_x_index, color='black', alpha=0.3, lw=0, label = "Dead space (no tubes here)")
                    plt.xlim(0,tube_x_index)
                if phi_index == 4:
                    plt.axvline(x = 32, color = 'cyan', ls='--')
                    plt.axvspan(32, tube_x_index, color='black', alpha=0.3, lw=0, label = "Dead space (no tubes here)")
                    plt.xlim(0,tube_x_index)
                elif np.any(np.isin(eta_array[i], [1, 2, 3, 4, 5, 6, 7])) == True:
                    if phi_index == 7:
                        plt.axvline(x = 48, color = 'cyan', ls='--')
                        plt.axvspan(48, tube_x_index, color='black', alpha=0.3, lw=0, label = "Dead space (no tubes here)")
                        plt.xlim(0,tube_x_index)
                    if phi_index == 5:
                        plt.axvline(x = 32, color = 'cyan', ls='--')
                        plt.axvspan(32, tube_x_index, color='black', alpha=0.3, lw=0, label = "Dead space (no tubes here)")
                        plt.xlim(0,tube_x_index)
                else:
                    if phi_index == 1:
                        plt.axvline(x = 32, color = 'cyan', ls='--')
                        plt.axvspan(32, tube_x_index, color='black', alpha=0.3, lw=0, label = "Dead space (no tubes here)")
                        plt.xlim(0,tube_x_index)
                    if phi_index == 2:
                        plt.axvline(x = 48, color = 'cyan', ls='--')
                        plt.axvspan(48, tube_x_index, color='black', alpha=0.3, lw=0, label = "Dead space (no tubes here)")
                        plt.xlim(0,tube_x_index)
                    if phi_index == 5:
                        plt.axvline(x = 40, color = 'cyan', ls='--')
                        plt.axvspan(40, tube_x_index, color='black', alpha=0.3, lw=0, label = "Dead space (no tubes here)")
                        plt.xlim(0,tube_x_index)
            if eta_region == 4 or eta_region == 5:
                plt.axvline(x = 40, color = 'cyan', ls='--')
                plt.axvspan(40, tube_x_index, color='black', alpha=0.3, lw=0, label = "Dead space (no tubes here)")
                plt.xlim(0,tube_x_index)
            if eta_region == 6:
                plt.axvline(x = 48, color = 'cyan', ls='--')
                plt.axvspan(48, tube_x_index, color='black', alpha=0.3, lw=0, label = "Dead space (no tubes here)")
                plt.xlim(0,tube_x_index)
    
    if station_index == 4 and image_format == 's':
        #Draw limit lines according to event eta region
        if (2*eta_region)%2 != 0:
            plt.axvline(x = tube_x_index/2, color = 'cyan', ls='--')
        else:
            if np.any(np.isin(eta_array[i], [1, 2, 3, 4, 5, 6, 7])) == True:
                if np.abs(eta_region) != 4 or np.abs(eta_region) != 5 or np.abs(eta_region) != 7:
                    if np.abs(eta_region) == 1:
                        if phi_index == 3:
                            plt.axvline(x = 64, color = 'cyan', ls='--')
                            plt.axvspan(64, tube_x_index, color='black', alpha=0.3, lw=0, label = "Dead space (no tubes here)")
                            plt.xlim(0,tube_x_index)
                        elif phi_index == 4 or phi_index == 5:
                            plt.axvline(x = 48, color = 'cyan', ls='--')
                            plt.axvspan(48, tube_x_index, color='black', alpha=0.3, lw=0, label = "Dead space (no tubes here)")
                            plt.xlim(0,tube_x_index)
                        elif phi_index == 6 or phi_index == 7 or phi_index == 8:
                            plt.axvline(x = 56, color = 'cyan', ls='--')
                            plt.axvspan(56, tube_x_index, color='black', alpha=0.3, lw=0, label = "Dead space (no tubes here)")
                            plt.xlim(0,tube_x_index)
                    elif np.abs(eta_region) == 2:
                        if phi_index == 7:
                            plt.axvline(x = 48, color = 'cyan', ls='--')
                            plt.axvspan(48, tube_x_index, color='black', alpha=0.3, lw=0, label = "Dead space (no tubes here)")
                            plt.xlim(0,tube_x_index)
                    elif np.abs(eta_region) == 3:
                        if phi_index == 7 or phi_index == 8:
                            plt.axvline(x = 48, color = 'cyan', ls='--')
                            plt.axvspan(48, tube_x_index, color='black', alpha=0.3, lw=0, label = "Dead space (no tubes here)")
                            plt.xlim(0,tube_x_index)
                        else:
                            plt.axvline(x = 56, color = 'cyan', ls='--')
                            plt.axvspan(56, tube_x_index, color='black', alpha=0.3, lw=0, label = "Dead space (no tubes here)")
                            plt.xlim(0,tube_x_index)
                    elif np.abs(eta_region) == 6:
                        plt.axvline(x = 56, color = 'cyan', ls='--')
                        plt.axvspan(56, tube_x_index, color='black', alpha=0.3, lw=0, label = "Dead space (no tubes here)")
                        plt.xlim(0,tube_x_index)
            else:
                if np.abs(eta_region) != 4 or np.abs(eta_region) != 5 or np.abs(eta_region) != 7:
                    if np.abs(eta_region) == 1:
                        if phi_index >= 1 and phi_index <= 4:
                            plt.axvline(x = 48, color = 'cyan', ls='--')
                            plt.axvspan(48, tube_x_index, color='black', alpha=0.3, lw=0, label = "Dead space (no tubes here)")
                            plt.xlim(0,tube_x_index)
                        elif phi_index == 5 or phi_index == 6 or phi_index == 8:
                            plt.axvline(x = 56, color = 'cyan', ls='--')
                            plt.axvspan(56, tube_x_index, color='black', alpha=0.3, lw=0, label = "Dead space (no tubes here)")
                            plt.xlim(0,tube_x_index)
                    elif np.abs(eta_region) == 2:
                        if phi_index == 7 or phi_index == 3:
                            plt.axvline(x = 48, color = 'cyan', ls='--')
                            plt.axvspan(48, tube_x_index, color='black', alpha=0.3, lw=0, label = "Dead space (no tubes here)")
                            plt.xlim(0,tube_x_index)
                    elif np.abs(eta_region) == 3:
                        if phi_index == 7 or phi_index == 8:
                            plt.axvline(x = 48, color = 'cyan', ls='--')
                            plt.axvspan(48, tube_x_index, color='black', alpha=0.3, lw=0, label = "Dead space (no tubes here)")
                            plt.xlim(0,tube_x_index)
                        else:
                            plt.axvline(x = 56, color = 'cyan', ls='--')
                            plt.axvspan(56, tube_x_index, color='black', alpha=0.3, lw=0, label = "Dead space (no tubes here)")
                            plt.xlim(0,tube_x_index)
                    elif np.abs(eta_region) == 6:
                        plt.axvline(x = 56, color = 'cyan', ls='--')
                        plt.axvspan(56, tube_x_index, color='black', alpha=0.3, lw=0, label = "Dead space (no tubes here)")
                        plt.xlim(0,tube_x_index)
    
    #================================================ End of Single Eta region plotting ==============================================#
    
    #==================================================== Half eta range plotting ====================================================#
    
    if station_index == 3 and image_format == 'h':
        #----------------------------------------- Eta coverage [1,6] ----------------------------------------------------------------#
        if np.any(np.isin(eta_array[i], [1, 2, 3, 4, 5, 6])) == True:
            for eta_line in range(1,6):
                if  eta_line == 1:
                    plt.axvline(x = 56, color = 'cyan', ls='--',label = f"Eta station {eta_line} limit")
                elif eta_line != 5:
                    plt.axvline(x = (eta_line - 1)*48 + 56, color = 'cyan', ls='--',label = f"Eta station {eta_line} limit")
                else:
                    plt.axvline(x = 3*48 + 32 + 56, color = 'cyan', ls='--',label = f"Eta station {eta_line} limit")

        else: 
            #------------------------------------ Eta coverage [-6, -1] --------------------------------------------------------------#
            for eta_line in range(-6,-1):
                if  eta_line == -6:
                    plt.axvline(x = 48, color = 'cyan', ls='--',label = f"Eta station {eta_line} limit")
                elif eta_line == -5:
                    plt.axvline(x = 48 + 32, color = 'cyan', ls='--',label = f"Eta station {eta_line} limit")
                else:
                    plt.axvline(x = (eta_line + 6)*48 + 32, color = 'cyan', ls='--',label = f"Eta station {eta_line} limit")
            #-------------------------------------------------------------------------------------------------------------------------#   
        
    elif station_index == 1 and image_format == 'h':
        if np.any(np.isin(eta_array[i], [1, 2, 3, 4, 5, 6])) == True:
            for eta_line in range(1,6):
                plt.axvline(x = 71 + (eta_line-1)*58, color = 'cyan', ls='--',label = f"Eta station {eta_line} limit")
        else: 
            for eta_line in range(-6,-1):
                plt.axvline(x = (eta_line+7)*58, color = 'cyan', ls='--',label = f"Eta station {eta_line} limit")
    
    elif station_index == 5 and image_format == 'h':
        if np.any(np.isin(eta_array[i], [1, 2, 3, 4, 5, 6])) == True:
            for eta_line in range(1,6):
                plt.axvline(x = eta_line*72, color = 'cyan', ls='--',label = f"Eta station {eta_line} limit")
        else:
            for eta_line in range(-6,-1):
                plt.axvline(x = 64 +(eta_line+6)*72, color = 'cyan', ls='--',label = f"Eta station {eta_line} limit")
                
    elif station_index == 0 and image_format == 'h':
        if np.any(np.isin(eta_array[i], [1, 2, 3, 4, 5, 6])) == True:
            if phi_index == 1 or phi_index == 2 or phi_index == 5:
                for eta_line in range(1,7):
                    if eta_line % 2 != 0:
                        plt.axvline(x = (eta_line//2 + 1)*30 + (eta_line//2)*36, color = 'cyan', ls='--',label = f"Eta station {eta_line} limit")
                    else:
                        plt.axvline(x = (eta_line/2)*30 + (eta_line/2)*36, color = 'cyan', ls='--',label = f"Eta station {eta_line} limit")
                plt.axvspan(3*30+3*36, tube_x_index, color='black', alpha=0.3, lw=0, label = "Dead space (no tubes here)")
                plt.xlim(0,tube_x_index)
            if phi_index == 3:
                for eta_line in range(1,6):
                    if eta_line < 3:
                        plt.axvline(x = 30 + (eta_line//2)*36, color = 'cyan', ls='--',label = f"Eta station {eta_line} limit")
                    elif eta_line == 3:
                        plt.axvline(x = (eta_line-1)*36 + 30, color = 'cyan', ls='--',label = f"Eta station {eta_line} limit")
                    else:
                        plt.axvline(x = 3*36 + (eta_line//2.1)*30, color = 'cyan', ls='--',label = f"Eta station {eta_line} limit")

            if phi_index == 4:
                for eta_line in range(1,7):
                    if eta_line % 2 != 0:
                        plt.axvline(x = 24 + (eta_line//2)*30 + (eta_line//2)*36, color = 'cyan', ls='--',label = f"Eta station {eta_line} limit")
                    elif eta_line % 2 == 0 and eta_line != 6:
                        plt.axvline(x = 24 +(eta_line/2 - 1)*30 + (eta_line/2)*36, color = 'cyan', ls='--',label = f"Eta station {eta_line} limit")
                    else:
                        plt.axvline(x = 24 +(eta_line/2)*30 + 2*36, color = 'cyan', ls='--',label = f"Eta station {eta_line} limit")
                plt.axvspan(3*30+2*36+24, tube_x_index, color='black', alpha=0.3, lw=0, label = "Dead space (no tubes here)")
                plt.xlim(0,tube_x_index)
            
            if phi_index == 7:
                for eta_line in range(1,7):
                    plt.axvline(x = eta_line*30, color = 'cyan', ls='--',label = f"Eta station {eta_line} limit")
                plt.axvspan(180, tube_x_index, color='black', alpha=0.3, lw=0, label = "Dead space (no tubes here)")
                plt.xlim(0,tube_x_index)
                
        else:
            if phi_index == 1 or phi_index == 2 or phi_index == 5:
                for eta_line in range(-6,0):
                    if eta_line % 2 == 0:
                        plt.axvline(x = ((eta_line+7)//2 + 1)*36 + ((eta_line+7)//2)*30, color = 'cyan', ls='--',label = f"Eta station {eta_line} limit")
                    else:
                        plt.axvline(x = ((eta_line+7)/2)*30 + ((eta_line+7)/2)*36, color = 'cyan', ls='--',label = f"Eta station {eta_line} limit")
                plt.axvspan(3*30+3*36, tube_x_index, color='black', alpha=0.3, lw=0, label = "Dead space (no tubes here)")
                plt.xlim(0,tube_x_index)
            if phi_index == 3:
                for eta_line in range(-6,-1):
                    if eta_line < -3:
                        plt.axvline(x = ((eta_line+7)//2)*30 + ((eta_line+7)//2.1+1)*36, color = 'cyan', ls='--',label = f"Eta station {eta_line} limit")
                    elif eta_line == -3:
                        plt.axvline(x = np.abs(eta_line)*36 + 30, color = 'cyan', ls='--',label = f"Eta station {eta_line} limit")
                    else:
                        plt.axvline(x = 4*36 + 30, color = 'cyan', ls='--',label = f"Eta station {eta_line} limit")

            if phi_index == 4:
                for eta_line in range(-6,0):
                    if eta_line % 2 != 0  and eta_line != -1:
                        plt.axvline(x = ((eta_line+7)//2 + 1)*30 + ((eta_line+7-1)//2)*36, color = 'cyan', ls='--',label = f"Eta station {eta_line} limit")
                    elif eta_line % 2 == 0:
                        plt.axvline(x = ((eta_line+7)//2 + 1)*30 + ((eta_line+7)//2)*36, color = 'cyan', ls='--',label = f"Eta station {eta_line} limit")
                    else:
                        plt.axvline(x = 24 + 3*30 + 2*36, color = 'cyan', ls='--',label = f"Eta station {eta_line} limit")
                plt.axvspan(3*30+2*36+24, tube_x_index, color='black', alpha=0.3, lw=0, label = "Dead space (no tubes here)")
                plt.xlim(0,tube_x_index)
                
            if phi_index == 7:
                for eta_line in range(-6,0):
                    if eta_line != -1:
                        plt.axvline(x = (eta_line+7)*30, color = 'cyan', ls='--',label = f"Eta station {eta_line} limit")
                    else:
                        plt.axvline(x = 5*30 + 36, color = 'cyan', ls='--',label = f"Eta station {eta_line} limit")
                plt.axvspan(186, tube_x_index, color='black', alpha=0.3, lw=0, label = "Dead space (no tubes here)")
                plt.xlim(0,tube_x_index)
                
    elif station_index == 2 and image_format == 'h':
        if np.any(np.isin(eta_array[i], [1, 2, 3, 4, 5, 6])) == True:
            if phi_index <= 2:
                for eta_line in range(1,6):
                    if eta_line <= 3:
                        plt.axvline(x = eta_line*56, color = 'cyan', ls='--',label = f"Eta station {eta_line} limit")
                    else:
                        plt.axvline(x = 3*56 + (eta_line-3)*40, color = 'cyan', ls='--',label = f"Eta station {eta_line} limit")
            
            if phi_index == 3:
                for eta_line in range(1,7):
                    if eta_line == 1:
                        plt.axvline(x = 48, color = 'cyan', ls='--',label = f"Eta station {eta_line} limit")
                    elif eta_line > 1 and eta_line < 4:
                        plt.axvline(x = (eta_line-1)*56 + 48, color = 'cyan', ls='--',label = f"Eta station {eta_line} limit")
                    elif eta_line >= 4 and eta_line < 6:
                        plt.axvline(x = 2*56 + 48 + (eta_line - 3)*40, color = 'cyan', ls='--',label = f"Eta station {eta_line} limit")
                    else:
                        plt.axvline(x = 2*56 + 2*48 + 2*40, color = 'cyan', ls='--',label = f"Eta station {eta_line} limit")
                plt.axvspan(2*56 + 2*48 + 2*40, tube_x_index, color='black', alpha=0.3, lw=0, label = "Dead space (no tubes here)")
                plt.xlim(0,tube_x_index)
            
            if phi_index == 4 or phi_index == 5:
                for eta_line in range(1,7):
                    if eta_line == 1:
                        plt.axvline(x = 32, color = 'cyan', ls='--',label = f"Eta station {eta_line} limit")
                    elif eta_line > 1 and eta_line < 4:
                        plt.axvline(x = (eta_line-1)*56 + 32, color = 'cyan', ls='--',label = f"Eta station {eta_line} limit")
                    elif eta_line >= 4 and eta_line < 6:
                        plt.axvline(x = 2*56 + 32 + (eta_line - 3)*40, color = 'cyan', ls='--',label = f"Eta station {eta_line} limit")
                    else:
                        plt.axvline(x = 2*56 + 32 + 2*40 + 48, color = 'cyan', ls='--',label = f"Eta station {eta_line} limit")
                plt.axvspan(2*56 + 32 + 2*40 + 48, tube_x_index, color='black', alpha=0.3, lw=0, label = "Dead space (no tubes here)")
                plt.xlim(0,tube_x_index)
            if phi_index == 6:
                for eta_line in range(1,7):
                    if eta_line == 1:
                        plt.axvline(x = 50, color = 'cyan', ls='--',label = f"Eta station {eta_line} limit")
                    elif eta_line > 1 and eta_line < 4:
                        plt.axvline(x = (eta_line-1)*56 + 50, color = 'cyan', ls='--',label = f"Eta station {eta_line} limit")
                    elif eta_line >= 4 and eta_line < 6:
                        plt.axvline(x = 2*56 + 50 + (eta_line - 3)*40, color = 'cyan', ls='--',label = f"Eta station {eta_line} limit")
                    else:
                        plt.axvline(x = 2*56 + 50 + 2*40 + 48, color = 'cyan', ls='--',label = f"Eta station {eta_line} limit")
                plt.axvspan(2*56 + 50 + 2*40 + 48, tube_x_index, color='black', alpha=0.3, lw=0, label = "Dead space (no tubes here)")
                plt.xlim(0,tube_x_index)
            if phi_index == 7:
                for eta_line in range(1,6):
                    if eta_line <= 3:
                        plt.axvline(x = eta_line*56, color = 'cyan', ls='--',label = f"Eta station {eta_line} limit")
                    else:
                        plt.axvline(x = 3*56 + (eta_line-3)*40, color = 'cyan', ls='--',label = f"Eta station {eta_line} limit")
                plt.axvspan(3*56 + 2*40, tube_x_index, color='black', alpha=0.3, lw=0, label = "Dead space (no tubes here)")
                plt.xlim(0,tube_x_index)
            if phi_index == 8:
                for eta_line in range(1,7):
                    if eta_line == 1:
                        plt.axvline(x = 40, color = 'cyan', ls='--',label = f"Eta station {eta_line} limit")
                    elif eta_line > 1 and eta_line < 4:
                        plt.axvline(x = (eta_line-1)*56 + 40, color = 'cyan', ls='--',label = f"Eta station {eta_line} limit")
                    elif eta_line >= 4 and eta_line < 6:
                        plt.axvline(x = 2*56 + 40 + (eta_line - 3)*40, color = 'cyan', ls='--',label = f"Eta station {eta_line} limit")
                    else:
                        plt.axvline(x = 2*56 + 40 + 2*40 + 48, color = 'cyan', ls='--',label = f"Eta station {eta_line} limit")
                plt.axvspan(2*56 + 40 + 2*40 + 48, tube_x_index, color='black', alpha=0.3, lw=0, label = "Dead space (no tubes here)")
                plt.xlim(0,tube_x_index)
        else:
            if phi_index == 1 or phi_index == 4:
                for eta_line in range(-6,0):
                    if eta_line == -6:
                        plt.axvline(x = 48, color = 'cyan', ls='--',label = f"Eta station {eta_line} limit")
                    elif eta_line > -6 and eta_line < -3:
                        plt.axvline(x = 48 + (eta_line+6)*40, color = 'cyan', ls='--',label = f"Eta station {eta_line} limit")
                    elif eta_line >= -3 and eta_line != -1:
                        plt.axvline(x = 48 + 2*40 + (eta_line + 4)*56, color = 'cyan', ls='--',label = f"Eta station {eta_line} limit")
                    else:
                        plt.axvline(x = 48 + 2*40 + 2*56 + 32, color = 'cyan', ls='--',label = f"Eta station {eta_line} limit")
                plt.axvspan(2*56 + 32 + 2*40 + 48, tube_x_index, color='black', alpha=0.3, lw=0, label = "Dead space (no tubes here)")
                plt.xlim(0,tube_x_index)
            if phi_index == 2 or phi_index == 3:
                for eta_line in range(-6,0):
                    if eta_line == -6:
                        plt.axvline(x = 48, color = 'cyan', ls='--',label = f"Eta station {eta_line} limit")
                    elif eta_line > -6 and eta_line < -3:
                        plt.axvline(x = 48 + (eta_line+6)*40, color = 'cyan', ls='--',label = f"Eta station {eta_line} limit")
                    elif eta_line >= -3 and eta_line != -1:
                        plt.axvline(x = 48 + 2*40 + (eta_line + 4)*56, color = 'cyan', ls='--',label = f"Eta station {eta_line} limit")
                    else:
                        plt.axvline(x = 2*48 + 2*40 + 2*56, color = 'cyan', ls='--',label = f"Eta station {eta_line} limit")
                plt.axvspan(2*56 + 2*40 + 2*48, tube_x_index, color='black', alpha=0.3, lw=0, label = "Dead space (no tubes here)")
                plt.xlim(0,tube_x_index)
            if phi_index == 5 or phi_index == 8:
                for eta_line in range(-6,0):
                    if eta_line == -6:
                        plt.axvline(x = 48, color = 'cyan', ls='--',label = f"Eta station {eta_line} limit")
                    elif eta_line > -6 and eta_line < -3:
                        plt.axvline(x = 48 + (eta_line+6)*40, color = 'cyan', ls='--',label = f"Eta station {eta_line} limit")
                    elif eta_line >= -3 and eta_line != -1:
                        plt.axvline(x = 48 + 2*40 + (eta_line + 4)*56, color = 'cyan', ls='--',label = f"Eta station {eta_line} limit")
                    else:
                        plt.axvline(x = 48 + 3*40 + 2*56, color = 'cyan', ls='--',label = f"Eta station {eta_line} limit")
                plt.axvspan(2*56 + 3*40 + 48, tube_x_index, color='black', alpha=0.3, lw=0, label = "Dead space (no tubes here)")
                plt.xlim(0,tube_x_index)
            if phi_index == 6:
                for eta_line in range(-6,0):
                    if eta_line == -6:
                        plt.axvline(x = 48, color = 'cyan', ls='--',label = f"Eta station {eta_line} limit")
                    elif eta_line > -6 and eta_line < -3:
                        plt.axvline(x = 48 + (eta_line+6)*40, color = 'cyan', ls='--',label = f"Eta station {eta_line} limit")
                    elif eta_line >= -3 and eta_line != -1:
                        plt.axvline(x = 48 + 2*40 + (eta_line + 4)*56, color = 'cyan', ls='--',label = f"Eta station {eta_line} limit")
                    else:
                        plt.axvline(x = 48 + 2*40 + 2*56 + 50, color = 'cyan', ls='--',label = f"Eta station {eta_line} limit")
                plt.axvspan(2*56 + 2*40 + 48 + 50, tube_x_index, color='black', alpha=0.3, lw=0, label = "Dead space (no tubes here)")
                plt.xlim(0,tube_x_index)
            if phi_index == 7:
                for eta_line in range(-5,0):
                    if eta_line < -3:
                        plt.axvline(x = (eta_line+6)*40, color = 'cyan', ls='--',label = f"Eta station {eta_line} limit")
                    elif eta_line >= -3:
                        plt.axvline(x = 2*40 + (eta_line + 4)*56, color = 'cyan', ls='--',label = f"Eta station {eta_line} limit")
                plt.axvspan(3*56 + 2*40, tube_x_index, color='black', alpha=0.3, lw=0, label = "Dead space (no tubes here)")
                plt.xlim(0,tube_x_index)
    
    elif station_index == 4 and image_format == 'h':
        
        if np.any(np.isin(eta_array[i], [1, 2, 3, 4, 5, 6])) == True:
            if phi_index <= 2:
                for eta_line in range(1,6):
                    if eta_line < 3:
                        plt.axvline(x = eta_line*72, color = 'cyan', ls='--',label = f"Eta station {eta_line} limit")
                    elif eta_line == 3:
                        plt.axvline(x = 56 + 2*72, color = 'cyan', ls='--',label = f"Eta station {eta_line} limit")
                    else:
                        plt.axvline(x = 56 + (eta_line -1)*72, color = 'cyan', ls='--',label = f"Eta station {eta_line} limit")
            if phi_index == 3:
                for eta_line in range(1,7):
                    if eta_line <= 2:
                        plt.axvline(x = 64 + (eta_line//2)*72, color = 'cyan', ls='--',label = f"Eta station {eta_line} limit")
                    elif eta_line == 3:
                        plt.axvline(x = 56 + 64 + 72, color = 'cyan', ls='--',label = f"Eta station {eta_line} limit")
                    elif eta_line >= 4 and eta_line < 6:
                        plt.axvline(x = 56 + 64 + (eta_line - 2)*72, color = 'cyan', ls='--',label = f"Eta station {eta_line} limit")
                    else:
                        plt.axvline(x = 2*56 + 64 + 3*76, color = 'cyan', ls='--',label = f"Eta station {eta_line} limit")
                plt.axvspan(2*56 + 64 + 3*76, tube_x_index, color='black', alpha=0.3, lw=0, label = "Dead space (no tubes here)")
                plt.xlim(0,tube_x_index)
            if phi_index == 4 or phi_index == 5:
                for eta_line in range(1,7):
                    if eta_line <= 2:
                        plt.axvline(x = 48 + (eta_line//2)*72, color = 'cyan', ls='--',label = f"Eta station {eta_line} limit")
                    elif eta_line == 3:
                        plt.axvline(x = 56 + 48 + 72, color = 'cyan', ls='--',label = f"Eta station {eta_line} limit")
                    elif eta_line >= 4 and eta_line < 6:
                        plt.axvline(x = 56 + 48 + (eta_line - 2)*72, color = 'cyan', ls='--',label = f"Eta station {eta_line} limit")
                    else:
                        plt.axvline(x = 2*56 + 48 + 3*76, color = 'cyan', ls='--',label = f"Eta station {eta_line} limit")
                plt.axvspan(2*56 + 48 + 3*76, tube_x_index, color='black', alpha=0.3, lw=0, label = "Dead space (no tubes here)")
                plt.xlim(0,tube_x_index)
            if phi_index == 6:
                for eta_line in range(1,7):
                    if eta_line <= 2:
                        plt.axvline(x = 56 + (eta_line//2)*72, color = 'cyan', ls='--',label = f"Eta station {eta_line} limit")
                    elif eta_line == 3:
                        plt.axvline(x = 2*56 + 72, color = 'cyan', ls='--',label = f"Eta station {eta_line} limit")
                    elif eta_line >= 4 and eta_line < 6:
                        plt.axvline(x = 2*56 + (eta_line - 2)*72, color = 'cyan', ls='--',label = f"Eta station {eta_line} limit")
                    else:
                        plt.axvline(x = 3*56 + 3*76, color = 'cyan', ls='--',label = f"Eta station {eta_line} limit")
                plt.axvspan(3*56 + 3*76, tube_x_index, color='black', alpha=0.3, lw=0, label = "Dead space (no tubes here)")
                plt.xlim(0,tube_x_index)
            if phi_index == 7:
                for eta_line in range(1,7):
                    if eta_line <= 3:
                        plt.axvline(x = 56 + (eta_line-1)*48, color = 'cyan', ls='--',label = f"Eta station {eta_line} limit")
                    else:
                        plt.axvline(x = (eta_line//6 + 1)*56 + 2*48 + (eta_line//5 +1)*72, color = 'cyan', ls='--',label = f"Eta station {eta_line} limit")
                plt.axvspan(2*56 + 2*48 + 2*72, tube_x_index, color='black', alpha=0.3, lw=0, label = "Dead space (no tubes here)")
                plt.xlim(0,tube_x_index)
            if phi_index == 8:
                for eta_line in range(1,7):
                    if eta_line <= 2:
                        plt.axvline(x = 56 + (eta_line//2)*72, color = 'cyan', ls='--',label = f"Eta station {eta_line} limit")
                    elif eta_line == 3:
                        plt.axvline(x = 56 + 72 + 48, color = 'cyan', ls='--',label = f"Eta station {eta_line} limit")
                    elif eta_line >= 4 and eta_line < 6:
                        plt.axvline(x = 56 + (eta_line - 2)*72 + 48, color = 'cyan', ls='--',label = f"Eta station {eta_line} limit")
                    else:
                        plt.axvline(x = 2*56 + 3*76 + 48, color = 'cyan', ls='--',label = f"Eta station {eta_line} limit")
                plt.axvspan(2*56 + 3*76 + 48, tube_x_index, color='black', alpha=0.3, lw=0, label = "Dead space (no tubes here)")
                plt.xlim(0,tube_x_index)
        else: 
            if phi_index <= 2 or phi_index == 4:
                for eta_line in range(-6,0):
                    if eta_line == -6:
                        plt.axvline(x = 56, color = 'cyan', ls='--',label = f"Eta station {eta_line} limit")
                    elif eta_line > -6 and eta_line < -3:
                        plt.axvline(x = 56 + (eta_line+6)*72, color = 'cyan', ls='--',label = f"Eta station {eta_line} limit")
                    elif eta_line >= -3 and eta_line != -1:
                        plt.axvline(x = ((eta_line + 4)//2 + 2)*72  + 2*56, color = 'cyan', ls='--',label = f"Eta station {eta_line} limit")
                    else:
                        plt.axvline(x = 48 + 3*72 + 2*56 , color = 'cyan', ls='--',label = f"Eta station {eta_line} limit")
                plt.axvspan(48 + 3*72 + 2*56, tube_x_index, color='black', alpha=0.3, lw=0, label = "Dead space (no tubes here)")
                plt.xlim(0,tube_x_index)
            if phi_index == 3:
                for eta_line in range(-6,0):
                    if eta_line == -6:
                        plt.axvline(x = 56, color = 'cyan', ls='--',label = f"Eta station {eta_line} limit")
                    elif eta_line > -6 and eta_line < -3:
                        plt.axvline(x = 56 + (eta_line+6)*72, color = 'cyan', ls='--',label = f"Eta station {eta_line} limit")
                    elif eta_line >= -3:
                        plt.axvline(x = (eta_line + 3)*48 + 2*72  + 2*56, color = 'cyan', ls='--',label = f"Eta station {eta_line} limit")
                plt.axvspan(2*56 + 2*72 + 2*48, tube_x_index, color='black', alpha=0.3, lw=0, label = "Dead space (no tubes here)")
                plt.xlim(0,tube_x_index)
            if phi_index == 5 or phi_index == 6:
                for eta_line in range(-6,0):
                    if eta_line == -6:
                        plt.axvline(x = 56, color = 'cyan', ls='--',label = f"Eta station {eta_line} limit")
                    elif eta_line > -6 and eta_line < -3:
                        plt.axvline(x = 56 + (eta_line+6)*72, color = 'cyan', ls='--',label = f"Eta station {eta_line} limit")
                    elif eta_line >= -3 and eta_line != -1:
                        plt.axvline(x = ((eta_line + 4)//2 + 2)*72  + 2*56, color = 'cyan', ls='--',label = f"Eta station {eta_line} limit")
                    else:
                        plt.axvline(x = 3*72 + 3*56 , color = 'cyan', ls='--',label = f"Eta station {eta_line} limit")
                plt.axvspan(3*72 + 3*56, tube_x_index, color='black', alpha=0.3, lw=0, label = "Dead space (no tubes here)")
                plt.xlim(0,tube_x_index)
            if phi_index == 7:
                for eta_line in range(-6,0):
                    if eta_line == -6:
                        plt.axvline(x = 56, color = 'cyan', ls='--',label = f"Eta station {eta_line} limit")
                    elif eta_line > -6 and eta_line < -3:
                        plt.axvline(x = 56 + (eta_line+6)*72, color = 'cyan', ls='--',label = f"Eta station {eta_line} limit")
                    elif eta_line >= -3 and eta_line != -1:
                        plt.axvline(x = (eta_line+4)*48 + 2*72 + 56, color = 'cyan', ls='--',label = f"Eta station {eta_line} limit")
                    else:
                        plt.axvline(x = 2*48 + 3*72 + 56 , color = 'cyan', ls='--',label = f"Eta station {eta_line} limit")
                plt.axvspan(2*48 + 3*72 + 56, tube_x_index, color='black', alpha=0.3, lw=0, label = "Dead space (no tubes here)")
                plt.xlim(0,tube_x_index)
            if phi_index == 8:
                for eta_line in range(-6,0):
                    if eta_line == -6:
                        plt.axvline(x = 56, color = 'cyan', ls='--',label = f"Eta station {eta_line} limit")
                    elif eta_line > -6 and eta_line < -3:
                        plt.axvline(x = 56 + (eta_line+6)*72, color = 'cyan', ls='--',label = f"Eta station {eta_line} limit")
                    elif eta_line >= -3 and eta_line != -1:
                        plt.axvline(x = 48 + ((eta_line + 4)//2 + 2)*72 + 56, color = 'cyan', ls='--',label = f"Eta station {eta_line} limit")
                    else:
                        plt.axvline(x = 2*56 + 3*72 + 48 , color = 'cyan', ls='--',label = f"Eta station {eta_line} limit")
                plt.axvspan(48 + 3*72 + 2*56, tube_x_index, color='black', alpha=0.3, lw=0, label = "Dead space (no tubes here)")
                plt.xlim(0,tube_x_index)
    
    plt.tight_layout()
    if (i + 1) in events_to_save:
        plt.savefig(f"MDTmZd_{int(dark_photon_mass*1000)}_avgtau_{int(lifetime)}_{SI}_event_image{i+1}.png", format="png", dpi=300)
    plt.show()
train_data.close()
