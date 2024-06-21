import os
import pandas as pd
import numpy as np
import warnings
from src.feature_extraction import *


warnings.filterwarnings("ignore")

if __name__ == '__main__':
    filepath = 'data/PreAttentiveGaze'
    block = "Block3.tsv"
    for task in ["MultipleVC", "SingleVC"]:
        whole_dataframe = pd.DataFrame()
        session_list = os.listdir(filepath)
        for session in session_list:
            participant_list = os.listdir(os.path.join(filepath, session))
            for participant in participant_list:
                stimuliIndex_list = os.listdir(os.path.join(filepath, session, participant, task))
                for stimuliIndex in stimuliIndex_list:
                    dataName = f"{session}_{participant}_{task}_{stimuliIndex}_{block}"
                    data = pd.read_csv(os.path.join(filepath, session, participant, task, stimuliIndex, dataName), sep='\t')
                    data = fill_nan(data)