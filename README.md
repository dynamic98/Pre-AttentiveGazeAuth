# Pre-AttentiveGazeAuth

### Index 
- [1. Project Introduction](#1-project-introduction)
- [2. Task and Gaze Stimuli](#2-task-and-gaze-stimuli)
- [3. Data Collection Envs](#3-data-collection-envs)
- [4. Collected Gaze Features](#4-collected-gaze-features)
- [5. Dev Envs](#5-dev-envs)
- [6. Understandig Dataset Structure](#6-understandig-dataset-structure)
- [7. Codes for Analysis](#7-codes-for-analysis)


---
### 1. Project Introduction 
- Category: `Security and Privacy` `Authentication` `Human-centered computing`
- Keywords: `Gaze` `Authentication` `Person Identification` `Pre-attentive processing` 
- Data: [google drvie download🔗](https://drive.google.com/drive/folders/12H32y8S0DhlHZcCObwhHYpgGESu4KD1w?usp=sharing)

---
### 2. Task and Gaze Stimuli
<img width="800" alt="스크린샷 2024-06-17 오후 6 36 38" src="https://github.com/dynamic98/Pre-AttentiveGazeAuth/assets/98831107/75048462-8a36-42fd-9f8c-a87e1b49dea9">


- **A visual search task** where targets (black circles) and distractors (dashed circles) are arranged in a circular pattern. 
- Each level has a specific arrangement of targets and distractors.

---
### 3. Data Collection Envs
<img width="800" alt="스크린샷 2024-06-17 오후 4 07 07" src="https://github.com/dynamic98/Pre-AttentiveGazeAuth/assets/98831107/99e41256-74f8-4690-a394-c927af37ab05">


---
### 4. Collected Gaze Features
**(1) Raw Gaze**
 - Coordinate of gaze point (x1, y1, …, x84, y84), (120 Hz × 0.7 s = 84 samples)

   
**(2) Eye Movement**
 - Path length: Length of path traveled in screen
 - Gaze velocity: Velocity of path traveled in screen
 - Gaze angle: Angle with centroid of previous fixation

   
**(3) Eye Movement**
 - Path length: Length of path traveled in screen
 - Gaze velocity: Velocity of path traveled in screen
 - Gaze angle: Angle with centroid of prev

   
**(4) Fixation**
 - Reaction time: Time until the first fixation is made outside the cross point (equal to the first fixation time)
 - Fixation duration: Duration per fixation
 - Fixation dispersion: Spatial spread during a fixation
 - Fixation count: Number of fixations
   
   
**(5) Saccade**
 - Saccade duration: Duration per saccade
 - Saccade velocity: Angular velocity per saccade
 - Saccade amplitude: Angular distance per saccade
 - Saccade dispersion: Spatial spread during a saccade
 - Saccade count: Numbers of saccades
   
**(6) MFCC**
 - 12 Mel-frequency ceptral coefficients for overall stimuli


**(7) Pupil**
 - Left pupil diameter: Pupil diameter of left eye
 - Right pupil diameter: Pupil diameter of right eye
 - Average pupil diameter: Average of left and right pupil diameter


**Additional Explanation for fixation and saccade**
<img width="800" alt="스크린샷 2024-06-17 오후 5 09 11" src="https://github.com/dynamic98/Pre-AttentiveGazeAuth/assets/98831107/980e21ce-ebd8-4474-a09e-08e201e09a07">


In fixation section, 'duration' and 'dispersion' value is extracted. 
In saccade section, 'duration', 'velocity', 'amplitude', and 'dispersion' value is extracted. 
The number of each section is a 'count' value

---
### 5. Dev Envs
##### python version <img src="https://img.shields.io/badge/Python-3776AB?style=plastic&logo=Python&logoColor=white"/>
    python version = 3.10
##### install package
    pip install -r requirements.txt
---

### 6. Understandig Dataset Structure
##### Directory
```bash
Pre-AttentiveGazeAuth/
├── data/
│   ├── MVC.tsv
│   └── SVC.tsv
├── LICENSE
├── main.ipynb
├── README.md
├── requirements.txt
└── src/
    ├── download.py
    ├── load_data.py
    ├── metric_revised.py
    ├── ML_util.py
    ├── model.py
    ├── preprocessing.py
    └── stimuli/
        ├── LevelDictionary_MVC.json
        ├── LevelDictionary_SVC.json
        ├── preattentive.py
        ├── stimuli.py
        └── util.py
```
##### JSON Files
The repository contains two JSON files, `LevelDictionary_MVC.json` and `LevelDictionary_SVC.json`, which define different levels for the visual search task.

##### LevelDictionary_MVC.json
- **`level`**: Specifies the shape, size, hue, and brightness of the targets.
  - The array contains four elements representing:
    - Shape (e.g., 3)
    - Size (e.g., 50)
    - Hue (e.g., 3)
    - Brightness (e.g., 2)
- **`target_list`**: Lists the positions (0 to 15) where the targets are located.

Example:
```json
"0": {
    "level": [3, 50, 3, 2],
    "target_list": [6, 5, 1, 10]
}
```
This means for level 0, the targets are at positions 6, 5, 1, and 10 with shape 3, size 50, hue 3, and brightness 2.

##### LevelDictionary_SVC.json

- **`visual_component`**: Describes the visual feature used in the level (e.g., "brightness", "hue", "shape", "size").
- **`level`**: Specifies the configuration for the visual component.
- **`target_list`**: Lists the positions where the targets are located.

Example:
```json
"0": {
    "visual_component": "brightness",
    "level": [1, 1, 1, 1],
    "target_list": [9, 10, 14, 5]
}
```
This means for level 0, the targets are at positions 9, 10, 14, and 5, and the task focuses on the brightness visual component with the specified configuration.

##### Example Explanation

For level 0 in `LevelDictionary_MVC.json`:
- **`level`**: `[3, 50, 3, 2]` (shape, size, hue, brightness)
- **`target_list`**: `[6, 5, 1, 10]` (positions of targets)
  - Targets are at positions 6, 5, 1, and 10 with shape 3, size 50, hue 3, and brightness 2.

For level 0 in `LevelDictionary_SVC.json`:
- **`visual_component`**: `"brightness"`
- **`level`**: `[1, 1, 1, 1]` (configuration for brightness)
- **`target_list`**: `[9, 10, 14, 5]`
  - Targets are at positions 9, 10, 14, and 5 focusing on the brightness visual component with the specified configuration.

---


### 7. Codes for Analysis 
1. Heatmap (code location: ___)
2. Scanpath (code location: ___) 







