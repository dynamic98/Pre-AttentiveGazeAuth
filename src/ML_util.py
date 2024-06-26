import itertools
import numpy as np
import math
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from scipy.special import softmax
from tqdm import tqdm
import os
import re
import cv2
import statistics

from .stimuli.util import cartesian

import logging

logger = logging.getLogger('root')

def task_encoding(task):
    task_dict = {'shape':1, 'size':2, 'orientation':3, 'hue':4, 'brightness':5}
    encoded_task = task_dict[task]
    return encoded_task

def similarity_encoding(similarity):
    if '#' in similarity:
        similarity = similarity.split('#')[1]
    similarity = list(map(int,similarity.split('-')))
    difference = abs(similarity[0]-similarity[1])
    return difference

def normalize_cm(cm: np.array):
    new_cm = np.zeros_like(cm, dtype=np.float_)
    cm_sum_list = cm.sum(axis=1)
    cm_length = len(cm_sum_list)
    for i in range(cm_length):
        for j in range(cm_length):
            new_cm[i,j] = (float(cm[i,j])/float(cm_sum_list[i]))
    return new_cm

def average(datalist):
    data = datalist.strip("[]")
    if len(data)!= 0:
        data = list(map(float, data.split(",")))
        return sum(data)/len(data)
    else:
        return 0

def stack_ydata_from_same(y_data, stack):
    # Allocate y_data to label dictionary
    # to stack by same label

    label_dict = {k:[] for k in list(set(y_data))}
    for index,label in enumerate(y_data):
        label_dict[label].append(index)
    
    stack_y = []
    stack_index = [[] for _ in range(stack)]

    #
    for label in label_dict:
        index_list = label_dict[label]
        stack_size = len(index_list)//stack
        for i in range(stack):
            stack_index[i].extend(index_list[i*stack_size:(i+1)*stack_size])
        stack_y.extend([label for _ in range(stack_size)])

    stack_index = np.array(stack_index)
    return stack_index, stack_y

def stack_ydata_from_same_combinations(y_data, stack):
    label_dict = {k:[] for k in list(set(y_data))}
    for index,label in enumerate(y_data):
        label_dict[label].append(index)
    stack_y = []
    for n, label in enumerate(label_dict):
        index_list = label_dict[label]
        stack_combinations = itertools.combinations(index_list, stack)
        stack_array = np.array(list(stack_combinations)).T
        _, stack_size = stack_array.shape
        stack_y.extend([label for _ in range(stack_size)])
        if n == 0:
            stack_index = stack_array
        else:
            stack_index = np.concatenate((stack_index, stack_array),axis=1)
    return stack_index, stack_y

def stack_ydata_from_stride(y_data, stack, samplesize=None):
    # sample construction by stacking n continuous blocks
    label_dict = {k:[] for k in list(set(y_data))}
    for index,label in enumerate(y_data):
        label_dict[label].append(index)
    stack_y = []
    stack_index = None
    history = []
    for n, label in enumerate(label_dict):
        index_list = label_dict[label]
        num_blocks = int(len(index_list)/stack)
        for i in range(0, num_blocks):
            stacked = index_list[i*stack:(i+1)*stack]
            history.append(stacked)
            stacked = np.array(stacked)
            stack_y.extend([label])
            if stack_index is None:
                stack_index = stacked
            else:
                stack_index = np.vstack([stack_index, stacked])

    # if the number of stacked samples is less than the set sample size, randomly select n and add a stack
    
    stack_samplesize, _ = stack_index.shape    
    logger.info(f"stack_samplesize: {stack_samplesize}")
    
    if (samplesize != None) and (stack_samplesize < samplesize) :
        logger.info(f"add random samples")        
        import random    
        remain = samplesize-stack_samplesize
        amount = int(round(remain/len(label_dict),0))
        for n, label in enumerate(label_dict):

            index_list = label_dict[label]
            for _ in range(amount):
                while True:
                    random_stacked = random.sample(index_list,stack)
                    if random_stacked not in history:
                        break  
                history.append(random_stacked)
                random_stacked = np.array(random_stacked)
                stack_y.extend([label])
                stack_index = np.vstack([stack_index, random_stacked])

    
    stack_index = stack_index.T # (stack, num_samples) -> (num_samples, stack) e.g., (2000,3)
    logger.info(f"stack_index.shape: {stack_index.shape}")        
    logger.info(f"stack_y.shape: {len(stack_y)}")
    return stack_index, stack_y


def stack_ydata_from_each_combinations(y_data, index1, index2, index3):
    # stack = 3
    label1_dict = {k:[] for k in list(set(y_data))}
    label2_dict = {k:[] for k in list(set(y_data))}
    label3_dict = {k:[] for k in list(set(y_data))}
    for index,label in enumerate(y_data):
        if index in index1:
            label1_dict[label].append(index)

        elif index in index2:
            label2_dict[label].append(index)
        
        elif index in index3:
            label3_dict[label].append(index)

    stack_y = []
    for n in range(13):
        label = n
        index_list = [label1_dict[label], label2_dict[label], label3_dict[label]]
        stack_array = cartesian(index_list).T
        _, stack_size = stack_array.shape
        stack_y.extend([label for _ in range(stack_size)])
        if n == 0:
            stack_index = stack_array
        else:
            stack_index = np.concatenate((stack_index, stack_array),axis=1)
    return stack_index, stack_y

def stack_ydata_from_anything(y_data, index1, index2, index3):
    # stack = 3
    label1_dict = {k:[] for k in list(set(y_data))}
    label2_dict = {k:[] for k in list(set(y_data))}
    label3_dict = {k:[] for k in list(set(y_data))}
    for index,label in enumerate(y_data):
        if index in index1:
            label1_dict[label].append(index)

        if index in index2:
            label2_dict[label].append(index)
        
        if index in index3:
            label3_dict[label].append(index)

    firstOne = label1_dict[0][0]
    secondOne = label2_dict[0][0]
    thirdOne = label3_dict[0][0]
    stack_y = []

    if (firstOne == secondOne) & (firstOne != thirdOne):
        for n in range(13):
            stackCombination = np.array(list(itertools.combinations(label1_dict[n], 2)))
            stackCombination = cartesian2Stack([stackCombination, label3_dict[n]]).T
            _, stack_size = stackCombination.shape
            stack_y.extend([n for _ in range(stack_size)])
            if n == 0:
                stack_index = stackCombination
            else:
                stack_index = np.concatenate((stack_index, stackCombination),axis=1)
    elif (firstOne == thirdOne) & (firstOne != secondOne):
        for n in range(13):
            stackCombination = np.array(list(itertools.combinations(label1_dict[n], 2)))
            stackCombination = cartesian2Stack([stackCombination, label2_dict[n]]).T
            stackCombination = np.vstack([stackCombination[0], stackCombination[2], stackCombination[1]])
            _, stack_size = stackCombination.shape
            stack_y.extend([n for _ in range(stack_size)])
            if n == 0:
                stack_index = stackCombination
            else:
                stack_index = np.concatenate((stack_index, stackCombination),axis=1)
    elif (thirdOne == secondOne) & (thirdOne != firstOne):
        for n in range(13):
            stackCombination = np.array(list(itertools.combinations(label3_dict[n], 2)))
            stackCombination = cartesian2Stack([stackCombination, label1_dict[n]]).T
            stackCombination = np.vstack([stackCombination[2], stackCombination[1], stackCombination[0]])
            _, stack_size = stackCombination.shape
            stack_y.extend([n for _ in range(stack_size)])
            if n == 0:
                stack_index = stackCombination
            else:
                stack_index = np.concatenate((stack_index, stackCombination),axis=1)
    elif (firstOne != secondOne) & (firstOne != secondOne):
        for n in range(13):
            stackCombination = cartesian([label1_dict[n], label2_dict[n], label3_dict[n]]).T
            _, stack_size = stackCombination.shape
            stack_y.extend([n for _ in range(stack_size)])
            if n == 0:
                stack_index = stackCombination
            else:
                stack_index = np.concatenate((stack_index, stackCombination),axis=1)
    elif (firstOne == secondOne == thirdOne):
        for n in range(13):
            stackCombination = np.array(list(itertools.combinations(label1_dict[n], 3))).T
            _, stack_size = stackCombination.shape
            stack_y.extend([n for _ in range(stack_size)])
            if n == 0:
                stack_index = stackCombination
            else:
                stack_index = np.concatenate((stack_index, stackCombination),axis=1)
    return stack_index, stack_y


def latefusion(clf, x_data, stack_index:np.array, stack_y):
    stack, total_stack_size = stack_index.shape
    multiply_predict = []
    # We actually tested plus and max fusion, but the result of multiply fusion was the best.
    probabilities = clf.predict_proba(x_data)
    yLabel = list(set(stack_y))
    fusion_proba = []

    for i in range(total_stack_size):
        multiply = np.ones((1,len(yLabel)))
        for j in range(stack):
            this_probabilities = probabilities[stack_index[j,i]]
            multiply = multiply * np.array(this_probabilities)
        multiply_predict.append(yLabel[np.argmax(multiply)])
        fusion_proba.append(multiply)

    cm_multiply = confusion_matrix(stack_y, multiply_predict)
    acc_multiply = accuracy_score(stack_y, multiply_predict)
    recall_multiply = recall_score(stack_y, multiply_predict, average='weighted')
    precision_multiply = precision_score(stack_y, multiply_predict, average='weighted')
    f1_multiply = f1_score(stack_y, multiply_predict, average='weighted')

    results = {'cm_multiply':cm_multiply,'acc_multiply': acc_multiply,'precision_multiply':precision_multiply, 
                'recall_multiply':recall_multiply,'f1_multiply': f1_multiply, 'multiply_proba':fusion_proba}

    return results

def visualize_cm(cm, clf_name:str, title:str, path='', iv=False):
    length_cm = cm.shape[0]
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.array(list(range(1,length_cm+1))))
    disp.plot(include_values=iv)
    plt.title(f'{title}_{clf_name}')
    # plt.show()
    if len(path)==0:
        plt.show()
    else:
        plt.savefig(os.path.join('ml-results','latefusion','from cm', path, f'{title}_{clf_name}.png'))
        plt.close()


def cartesian2Stack(arrays, out=None):
    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype
    n = np.prod([x.shape[0] for x in arrays])
    if out is None:
        out = np.zeros([n, 3], dtype=dtype)

    m = int(n / arrays[0].shape[0])
    # print(np.repeat(arrays[0], m, axis=0))
    out[:,:2] = np.repeat(arrays[0], m, axis=0)
    for j in range(0, arrays[0].shape[0]):
        out[j*m:(j+1)*m, 2] = arrays[1]

    return out


def bracket2array(arraylike_string:str):
    result = []
    line_slice = arraylike_string.split(']\n')
    for i in line_slice:
        result.append([*map(float, re.findall('-?[\d]+\.?[\d]*', i))])
    return np.array(result)

def convert2binaryCM(cm:np.array, participant):
    array_line = participant-1
    tp = cm[array_line, array_line]
    fp = cm[:,array_line].sum() - tp
    fn = cm[array_line,:].sum() - tp
    tn = cm.sum() - (tp+fp+fn)
    out = np.array([[tp, fp],[fn,tn]])

    return out

def resize_img(img):
    img = img[200:880,620:1300]
    resize_img = cv2.resize(img, dsize=(200,200), interpolation=cv2.INTER_AREA )
    img_arr = np.array(resize_img,dtype=np.float_)
    return img_arr

def BiometricEvaluation(confusionMatrix, targetLabel, metric:str):
    totalValue = np.sum(confusionMatrix)
    tp = confusionMatrix[targetLabel, targetLabel]
    fn = confusionMatrix[targetLabel,:].sum() - tp
    fp = confusionMatrix[:,targetLabel].sum() - tp
    tn = totalValue - (tp+fn+fp)
    if metric == 'FAR':
        return fp/(fp+tn)
    elif metric == 'FRR':
        return fn/(fn+tp)

def accuracyMeasurementForVerification(confusionMatrix, targetLabel):
    totalValue = np.sum(confusionMatrix)
    tp = confusionMatrix[targetLabel, targetLabel]
    fn = confusionMatrix[targetLabel,:].sum() - tp
    fp = confusionMatrix[:,targetLabel].sum() - tp
    tn = totalValue - (tp+fn+fp)
    return (tp+tn)/totalValue

def accuracyMeasurement(confusionMatrix):
    labelAmount,_ = confusionMatrix.shape
    diagonalValue = 0
    totalValue = np.sum(confusionMatrix)
    for i in range(labelAmount):
        diagonalValue += confusionMatrix[i,i]
    return diagonalValue/totalValue

def statistic_calculation(data:list):
    data = [x for x in data if x!='']
    data = list(map(float, data))
    data = [x for x in data if math.isnan(x) == False]
    if len(data) == 0:
        return [np.nan, np.nan, np.nan, np.nan]
    elif len(data) == 1:
        return [data[0],0,data[0],data[0]]
    else:
        mean_value = statistics.mean(data)
        std_value = statistics.stdev(data)
        max_value = max(data)
        min_value = min(data)
        return [mean_value, std_value, max_value, min_value]


if __name__ == '__main__':
    pass