import os
import gdown

this_file_path = os.path.dirname(__file__)
data_file_path = os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(__file__))), 'data')

def download():
    # Make sure the data folder exists
    if not os.path.exists(data_file_path):
        os.makedirs(data_file_path)

    # Download the data files
    MVC_url = 'https://drive.google.com/uc?id=1ZR7HfJO3Ul5Ir1Y_ARQoa-nWwpRL_geD'
    gdown.download(MVC_url, os.path.join(data_file_path, 'MVC.tsv'), quiet=False)
    SVC_url = 'https://drive.google.com/uc?id=12jIbe0FBTnb7XGK3xhvgvdRHCsEUTqfj'
    gdown.download(SVC_url, os.path.join(data_file_path, 'SVC.tsv'), quiet=False)
