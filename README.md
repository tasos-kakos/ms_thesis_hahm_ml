# Displaced Muon Detection Using Machine Learning: <br>ATLAS L0 Muon Trigger Upgrade</br>
A repository with the main python scripts used for the master thesis: <br><a href="https://uu.diva-portal.org/smash/record.jsf?aq2=%5B%5B%5D%5D&c=4&af=%5B%5D&searchType=LIST_LATEST&sortOrder2=title_sort_asc&language=sv&pid=diva2%3A1979429&aq=%5B%5B%5D%5D&sf=all&aqe=%5B%5D&sortOrder=author_sort_asc&onlyFullText=false&noOfRows=50&dswid=8170">Displaced Muon Detection Using Machine Learning: ATLAS L0 Muon Trigger Upgrade</a></br>

These files are parts of the broader repository managed and maintained by the <a href="https://nextgentriggers.web.cern.ch/t22/">NextGenTriggers WP2 T2.2: Enhancing the Level-0 Muon Trigger</a> project group at CERN.
<br>In this repository only the scripts written and maintained by the author of the aforementioned thesis are provided due to confidentiality reasons. Scripts contained are used for the following practices:</br>
- Ntuple data dumping
- Data preprocessing
- Data Visualization
- Machine Learning model training and testing
- Result Extraction
- Conversion to High Level Synthesis (HLS)

## Content description

| Script             | Description                                                                                  |
|--------------------|----------------------------------------------------------------------------------------------|
| `MdtNtupleDumper_sim.py`       | Takes as input a `.root` ntuple file and dumps the raw **HAHM** or **muon-gun simulation** data to a `.h5` file to be used for CNN and RNN input data preprocessing               |
| `MdtNtupleDumper_muonTester.py`| Takes as input a `.root` ntuple file and dumps the raw **REAL background** data to a `.h5` file to be used as noise in the CNN and RNN input data preprocessing                   |
| `preprocess_CNNsample.py`      | Takes as inputs `.h5` files (HAHM, muon-gun and background data) obtained by the ntuple dumpers and **produces the input images** to the **CNN** model in the form of `.npz` files|
| `preprocess_RNNsample.py`      | Takes as inputs `.h5` files (HAHM, muon-gun and background) obtained by the ntuple dumpers and **produces the input** to the **RNN** model in the form of `.npz` files            |
| `CNN_image_plotting.py`        | Takes as input a `.npz` file produced by the `preprocess_CNNsample.py` script and visualizes the CNN input images contained in the file            |
| `CNN.py`                       | Takes as input a `.npz` file produced by the `preprocess_CNNsample.py` script, splits the data into train, cross-validation and test sets, trains the CNN and outputs a trained model as a `.h5` file, as well as a results `.npz` file |
| `RNN.py`                       | Takes as input a `.npz` file produced by the `preprocess_RNNsample.py` script, splits the data into train, cross-validation and test sets, trains the CNN and outputs a trained model as a `.h5` file, as well as a results `.npz` file along with useful evaluation plots            |
| `Plot_CNN_results`             | Takes as input a `.npz` file produced by the `CNN.py` script and produces various model evaluation plots    |                                    
