# Pre-AttentiveGazeAuth

### 1. Project Introduction 
- category: `Security and Privacy` `Authentication` `Human-centered computing`
- keywords: `Gaze` `Authentication` `Person Identification` `Pre-attentive processing` 
- 🔗: [download link](https://drive.google.com/drive/folders/12H32y8S0DhlHZcCObwhHYpgGESu4KD1w?usp=sharing)

---
### 2. Gaze Stimuli
<img width="455" alt="스크린샷 2024-06-17 오후 6 08 59" src="https://github.com/dynamic98/Pre-AttentiveGazeAuth/assets/98831107/314f1c1f-c323-49f9-9f58-3c4043484e71">


---
### 3. Data Collection Envs
<img width="723" alt="스크린샷 2024-06-17 오후 4 07 07" src="https://github.com/dynamic98/Pre-AttentiveGazeAuth/assets/98831107/b8193ea3-65db-4011-a901-615a7ade6239">


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


---
### 5. Additional Explanation for fixation and saccade  

<img width="1000" alt="스크린샷 2024-06-17 오후 5 09 11" src="https://github.com/dynamic98/Pre-AttentiveGazeAuth/assets/98831107/980e21ce-ebd8-4474-a09e-08e201e09a07">

Duration, dispersion, and count are extracted for each section

---
### 6. Dev Envs <img src="https://img.shields.io/badge/Python-3776AB?style=plastic&logo=Python&logoColor=white"/>
##### python version
    python version = 3.10
##### install package
    pip install -r requirements.txt

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
    ├── __pycache__/
    │   ├── download.cpython-310.pyc
    │   ├── load_data.cpython-310.pyc
    │   ├── ML_util.cpython-310.pyc
    │   ├── model.cpython-310.pyc
    │   └── preprocessing.cpython-310.pyc
    ├── download.py
    ├── load_data.py
    ├── metric_revised.py
    ├── ML_util.py
    ├── model.py
    ├── preprocessing.py
    └── stimuli/
        ├── __pycache__/
        │   ├── preattentive.cpython-310.pyc
        │   └── util.cpython-310.pyc
        ├── LevelDictionary_MVC.json
        ├── LevelDictionary_SVC.json
        ├── preattentive.py
        ├── stimuli.py
        └── util.py
```







