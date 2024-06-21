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
                    
                    this_dataframe = pd.DataFrame()
                    x_data, y_data = get_gazeXY(data)
                    eye_type_data = get_eye_type(data)
                    velocity_data = velocity(data)
                    angle_data = angular(data)

                    path_length_data = get_path_length(data)
                    rt_data = reaction_time(data)
                    fixation_duration_data = Fixation_Duration(data)
                    fixation_dispersion_data = Fixation_Dispersion(data)
                    fixation_count_data = Fixation_Count(data)
                    
                    saccade_duration_data = Saccade_Duration(data)
                    saccade_dispersion_data = Saccade_Dispersion(data)
                    saccade_velocity_data, saccade_amplitude_data = Saccade_Velocity_Amplitude(data)
                    saccade_count_data = Saccade_Count(data)

                    mfcc_data = MFCC(velocity_data)
                    left_diameter_data = pupilLeft(data)
                    right_diameter_data = pupilRight(data)
                    filtered_diameter_data = pupil(data)
                    
                    this_dataframe['x'] = [x_data]
                    this_dataframe['y'] = [y_data]
                    this_dataframe['eye_type'] = [eye_type_data]
                    this_dataframe['velocity'] = [velocity_data]
                    this_dataframe['angle'] = [angle_data]
                    this_dataframe['path_length'] = [path_length_data]
                    this_dataframe['reaction_time'] = [rt_data]
                    this_dataframe['fixation_duration'] = [fixation_duration_data]
                    this_dataframe['fixation_dispersion'] = [fixation_dispersion_data]
                    this_dataframe['fixation_count'] = [fixation_count_data]
                    this_dataframe['saccade_duration'] = [saccade_duration_data]
                    this_dataframe['saccade_dispersion'] = [saccade_dispersion_data]
                    this_dataframe['saccade_velocity'] = [saccade_velocity_data]
                    this_dataframe['saccade_amplitude'] = [saccade_amplitude_data]
                    this_dataframe['saccade_count'] = [saccade_count_data]
                    this_dataframe['mfcc'] = [mfcc_data]
                    this_dataframe['left_pupil'] = [left_diameter_data]
                    this_dataframe['right_pupil'] = [right_diameter_data]
                    this_dataframe['filtered_pupil'] = [filtered_diameter_data]
                    this_dataframe['participant'] = [int(participant.split('_')[1])]
                    this_dataframe['session'] = [int(session.split('_')[1])]
                    this_dataframe['stimuli_index'] = [int(stimuliIndex.split('_')[1])]
                    whole_dataframe = pd.concat([whole_dataframe, this_dataframe], ignore_index=True)

        whole_dataframe.to_csv(f"data/{task}.tsv", sep='\t', index=False)