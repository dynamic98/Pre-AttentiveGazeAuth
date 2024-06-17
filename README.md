# Pre-AttentiveGazeAuth

### Project Introduction 
- category: `Security and Privacy` `Authentication` `Human-centered computing`
- keywords: `Gaze` `Authentication` `Person Identification` `Pre-attentive processing` 
- 🔗: [download link](https://drive.google.com/uc?id=1ZR7HfJO3Ul5Ir1Y_ARQoa-nWwpRL_geD)

---
### Gaze Stimuli
<img width="555" alt="image" src="https://github.com/dynamic98/Pre-AttentiveGazeAuth/assets/98831107/78930fe7-b545-4b9d-bd8e-13f5a8e6bc8a">

---
### Data Collection Envs
<img width="923" alt="스크린샷 2024-06-17 오후 4 07 07" src="https://github.com/dynamic98/Pre-AttentiveGazeAuth/assets/98831107/b8193ea3-65db-4011-a901-615a7ade6239">


---
### Collected Gaze Features
**Raw Gaze**
 - Coordinate of gaze point (x1, y1, …, x84, y84), (120 Hz × 0.7 s = 84 samples)

   
**Eye Movement**
 - Path length: Length of path traveled in screen
 - Gaze velocity: Velocity of path traveled in screen
 - Gaze angle: Angle with centroid of previous fixation

   
**Eye Movement**
 - Path length: Length of path traveled in screen
 - Gaze velocity: Velocity of path traveled in screen
 - Gaze angle: Angle with centroid of prev

   
**Fixation**
 - Reaction time: Time until the first fixation is made outside the cross point (equal to the first fixation time)
 - Fixation duration: Duration per fixation
 - Fixation dispersion: Spatial spread during a fixation
 - Fixation count: Number of fixations
   
   
**Saccade**
 - Saccade duration: Duration per saccade
 - Saccade velocity: Angular velocity per saccade
 - Saccade amplitude: Angular distance per saccade
 - Saccade dispersion: Spatial spread during a saccade
 - Saccade count: Numbers of saccades
   
**MFCC**
 - 12 Mel-frequency ceptral coefficients for overall stimuli


**Pupil**
 - Left pupil diameter: Pupil diameter of left eye
 - Right pupil diameter: Pupil diameter of right eye
 - Average pupil diameter: Average of left and right pupil diameter


---
### Dev Envs <img src="https://img.shields.io/badge/Python-3776AB?style=plastic&logo=Python&logoColor=white"/>
##### python version
    python version = 3.10
##### install package
    pip install -r requirements.txt
