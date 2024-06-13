import sys, os
import json
import matplotlib.pyplot as plt
from preattentive import PreattentiveObject


this_file_path = os.path.dirname(__file__)
preattentive_object = PreattentiveObject(1980, 1080, 'black')
with open(os.path.join(this_file_path, 'LevelDictionary_MVC.json')) as f:
    level_dict_MVC = json.load(f)
with open(os.path.join(this_file_path, 'LevelDictionary_SVC.json')) as f:
    level_dict_SVC = json.load(f)


def index2image(index: int, type: str):
    """
    This function takes an index and a type as input and returns an image based on the given index and type.
    
    Parameters:
        index (int): The index of the stimuli.
        type (str): The type of the stimuli. Can be either 'MVC' or 'SVC'.
    
    Returns:
        numpy.ndarray: The image represented as a NumPy array.
    """
    if type == 'MVC':
        info = level_dict_MVC[str(index)]
        info_level = info['level']
        info_target_list = info['target_list']
        bg = preattentive_object.draw_MVC(info_level, info_target_list)
        bg = BGR2RGB(bg)
        return bg
    
    elif type == 'SVC':
        info = level_dict_SVC[str(index)]
        visual_component = info['visual_component']
        info_level = info['level']
        info_target_list = info['target_list']
        bg = preattentive_object.draw_SVC(visual_component, info_level, info_target_list)
        bg = BGR2RGB(bg)
        return bg
    
    else:
        raise ValueError('Invalid type, please use either "MVC" or "SVC" as type.')
    

def BGR2RGB(image):
    return image[...,::-1]
    
    
if __name__ == "__main__":
    bg = index2image(5, 'SVC')
    plt.imshow(bg)
    plt.show()