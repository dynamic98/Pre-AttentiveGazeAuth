import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from .ML_util import statistic_calculation

def data_preprocessing(data: pd.DataFrame):
    data_x = pd.DataFrame(data['x'].tolist(), columns=naming_columns("x", 84))
    data_y = pd.DataFrame(data['y'].tolist(), columns=naming_columns("y", 84))
    data_velocity = pd.DataFrame(data['velocity'].tolist(), columns=naming_columns("velocity", 83))
    data_angle = pd.DataFrame(data['angle'].tolist(), columns=naming_columns("angle", 83))
    data_mfcc = pd.DataFrame(data['mfcc'].tolist(), columns = naming_columns("mfcc", 12))
    data_left_pupil = pd.DataFrame(data['left_pupil'].tolist(), columns=naming_columns("left_pupil", 84))
    data_right_pupil = pd.DataFrame(data['right_pupil'].tolist(), columns=naming_columns("right_pupil", 84))
    data_filtered_pupil = pd.DataFrame(data['filtered_pupil'].tolist(), columns=naming_columns("filtered_pupil", 84))
    data_eyetype = pd.DataFrame(data['eye_type'].tolist(), columns=naming_columns("eye_type", 84))
    data_fixation_duration = pd.DataFrame(data["fixation_duration"].apply(statistic_calculation).tolist(), columns=naming_columns("fixation_duration", 4))
    data_fixation_dispersion = pd.DataFrame(data["fixation_dispersion"].apply(statistic_calculation).tolist(), columns=naming_columns("fixation_dispersion", 4))
    data_saccade_duration = pd.DataFrame(data["saccade_duration"].apply(statistic_calculation).tolist(), columns=naming_columns("saccade_duration", 4))
    data_saccade_dispersion = pd.DataFrame(data["saccade_dispersion"].apply(statistic_calculation).tolist(), columns=naming_columns("saccade_dispersion", 4))
    data_saccade_velocity = pd.DataFrame(data["saccade_velocity"].apply(statistic_calculation).tolist(), columns=naming_columns("saccade_velocity", 4))
    data_saccade_amplitude = pd.DataFrame(data["saccade_amplitude"].apply(statistic_calculation).tolist(), columns=naming_columns("saccade_amplitude", 4))
    data_scalar = data[['path_length', 'fixation_count', 'saccade_count','reaction_time', 'session', 'participant', 'stimuli_index']]
    
    processed_data = pd.concat([data_x, data_y, data_velocity, data_angle, data_mfcc, data_left_pupil, data_right_pupil, 
                                data_filtered_pupil, data_eyetype, data_fixation_duration, data_fixation_dispersion, data_saccade_duration, 
                                data_saccade_dispersion, data_saccade_velocity, data_saccade_amplitude, data_scalar], axis=1)
    
    processed_data = processed_data.apply(pd.to_numeric, errors='coerce')
    processed_data = processed_data.dropna(axis=0)
    processed_data = processed_data.sample(frac=1, random_state=5).reset_index(drop=True)
    return processed_data

def naming_columns(name, length):
    return [name + str(i) for i in range(length)]

import pandas as pd
from sklearn.preprocessing import StandardScaler

def z_score_normalization(data: pd.DataFrame, mean_std: dict = None):
    # Define the groups of columns based on the specified patterns
    groups = {
        'x': [f'x{i}' for i in range(84)],
        'y': [f'y{i}' for i in range(84)],
        'velocity': [f'velocity{i}' for i in range(83)],
        'angle': [f'angle{i}' for i in range(83)],
        'left_pupil': [f'left_pupil{i}' for i in range(84)],
        'right_pupil': [f'right_pupil{i}' for i in range(84)],
        'filtered_pupil': [f'filtered_pupil{i}' for i in range(84)],
        'eye_type': [f'eye_type{i}' for i in range(84)]
    }

    # List of all columns in the dataset
    all_columns = data.columns.tolist()

    # Flatten the group columns into a single list
    group_columns = [col for group in groups.values() for col in group]

    # Identify remaining columns that are not in any group and exclude participant, session, stimuli_index
    remaining_columns = [col for col in all_columns if col not in group_columns and col not in ['participant', 'session', 'stimuli_index']]

    # Initialize the scaler and the normalized data
    normalized_data = data.copy()
    mean_std = mean_std or {}
    
    # Function to normalize and store mean/std
    def normalize_columns(columns, group):
        if columns:
            if group not in mean_std:
                scaler = StandardScaler()
                # Fit the scaler on the combined columns to ensure the same mean and std are used
                combined_data = data[columns].values.flatten().reshape(-1, 1)
                scaler.fit(combined_data)
                mean = scaler.mean_[0]
                std = scaler.scale_[0]
                mean_std[group] = {'mean': mean, 'std': std}
                # Transform the data
                for col in columns:
                    normalized_data[col] = (data[col] - mean) / std
            else:
                mean = mean_std[group]['mean']
                std = mean_std[group]['std']
                # Transform the data
                for col in columns:
                    normalized_data[col] = (data[col] - mean) / std

    # Normalize each group of columns together
    for group, columns in groups.items():
        columns = [col for col in columns if col in data.columns]
        normalize_columns(columns, group)

    # Normalize each remaining column individually
    for col in remaining_columns:
        normalize_columns([col], col)

    return normalized_data, mean_std



if __name__ == "__main__":
    pass