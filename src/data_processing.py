import os
import pandas as pd
import numpy as np

this_file_path = os.path.dirname(__file__)
data_file_path = os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(__file__))), 'data')


def row_processing(row):
    
    left_pupil = row[['left_diameter' + str(i) for i in range(1, 85)]].to_list()
    right_pupil = row[['right_diameter' + str(i) for i in range(1, 85)]].to_list()
    avg_pupil = row[['together_diameter' + str(i) for i in range(1, 85)]].to_list()
    x_rawgaze = row[['x' + str(i) for i in range(1, 85)]].to_list()
    y_rawgaze = row[['y' + str(i) for i in range(1, 85)]].to_list()
    velocity = row[['velocity' + str(i) for i in range(1, 84)]].to_list()
    angular = row[['angular' + str(i) for i in range(1, 84)]].to_list()
    mfcc = row[['mfcc' + str(i) for i in range(1, 13)]].to_list()
    eyetype = row[['eye_type' + str(i) for i in range(1, 85)]].to_list()
    eye_type_array = row[['eye_type' + str(i) for i in range(1, 85)]].to_numpy()
    # fixation
    fixation_duration = []
    fixation_dispersion = []
    fixation_num = 1
    
    # saccade
    saccade_duration = []
    saccade_dispersion = []
    saccade_velocity = []
    saccade_amplitude = []
    saccade_num = 1
    
    changes = np.where(eye_type_array[1:] != eye_type_array[:-1])[0]
    changed_values = eye_type_array[changes]
    last_index_value = (len(eye_type_array) - 1, eye_type_array[-1])
    optimized_change_points = list(zip(changes + 1, changed_values))
    optimized_change_points.append(last_index_value)
    for _, num in optimized_change_points:
        if num == 1:
            fixation_duration.append(row['fixation_duration'+str(fixation_num)])
            fixation_dispersion.append(row['fixation_dispersion'+str(fixation_num)])
            fixation_num += 1
        elif num == 2:
            saccade_duration.append(row['saccade_duration'+str(saccade_num)])
            saccade_dispersion.append(row['saccade_dispersion'+str(saccade_num)])
            try:
                saccade_velocity.append(row['saccade_velocity'+str(saccade_num)])
                saccade_amplitude.append(row['saccade_amplitude'+str(saccade_num)])
            except:
                pass
            saccade_num += 1

    return x_rawgaze, y_rawgaze, velocity, angular, mfcc, left_pupil, right_pupil, avg_pupil, eyetype, fixation_duration, fixation_dispersion, saccade_duration, saccade_dispersion, saccade_velocity, saccade_amplitude


data_MVC = pd.read_csv(os.path.join(data_file_path, 'MVC.csv'))
df = data_MVC.apply(row_processing, axis=1)
new_df = pd.DataFrame(df.tolist(), columns=['x', 'y', 'velocity', 'angle', 'mfcc', 'left_pupil', 'right_pupil', 'avg_pupil', 'eyetype', 'fixation_duration', 'fixation_dispersion', 'saccade_duration', 'saccade_dispersion', 'saccade_velocity', 'saccade_amplitude'])
scalar_df = data_MVC[['path_length','fixation_count', 'saccade_count', 'rt', 'session', 'participant', 'stimuli_index']]
new_df = pd.concat([new_df, scalar_df], axis=1)
new_df.to_csv(os.path.join(data_file_path, 'MVC_processed.tsv'), sep='\t', index=False)

data_SVC = pd.read_csv(os.path.join(data_file_path, 'SVC.csv'))
df = data_SVC.apply(row_processing, axis=1)
new_df = pd.DataFrame(df.tolist(), columns=['x', 'y', 'velocity', 'angle', 'mfcc', 'left_pupil', 'right_pupil', 'avg_pupil', 'eyetype', 'fixation_duration', 'fixation_dispersion', 'saccade_duration', 'saccade_dispersion', 'saccade_velocity', 'saccade_amplitude'])
scalar_df = data_SVC[['path_length','fixation_count', 'saccade_count', 'rt', 'session', 'participant', 'stimuli_index']]
new_df = pd.concat([new_df, scalar_df], axis=1)
new_df.to_csv(os.path.join(data_file_path, 'SVC_processed.tsv'), sep='\t', index=False)