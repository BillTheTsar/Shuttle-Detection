<p style="font-size: 2.0em; font-weight: bold; margin-bottom: 0;">Badminton Shuttle Head Tracking from Video</p>

*A machine learning project by Bill Zhang*

## Abstract

> This report fleshes out a two-stage computer vision system capable of localizing the position of a badminton shuttle in 
> video footage, frame by frame. Over two months, I developed 11 model versions which incorporate contextual motion 
> modeling and heatmap-based inference. The final system achieves real-time inference speeds (86 FPS) and 
> sub-pixel precision on standard 1080p footage, with practical applications in sports analytics and even shot 
> anticipation.

---

# 1. Introduction

### Problem statement

**Given a video of a badminton game, identify the precise position of the badminton shuttle head in all frames in which it is present and in play.** 

This task is deceptively simple and yet rich with challenges that make it both an academically and practically worthwhile problem:

- **Miniscule object size:** Badminton shuttles are extremely small, occupying roughly 1/30 of both the width and height
    of the frame (roughly 0.11% of the total frame area). The shuttle head is a small part of an already tiny object, 
    making it extraordinarily difficult to localize with precision.
- **Complex object shape:** Badminton shuttles exhibit a wide variety of silhouettes and patterns depending on the 
  viewing angle. This is in contrast to most ball sports, where the silhouette of the “ball” is a circle, lending easily
  to traditional convolutional filters.
- **High speed and occlusion:** Upon impact, badminton shuttles, albeit for a split second, become the fastest objects 
  among all ball sports. There is significant blur and occlusion due to the racket, making precise positional 
  predictions very challenging.



### Motivations

Whether we want to analyze the shot selection of professionals, anticipate the shot played a split second before impact,
or provide an alternative to the Hawkeye technology, determining the precise position of the shuttle head is the 
necessary foundation for any high-level automation task. Also, it is a fun project that I thought would be 
computationally feasible and highly rewarding with a reasonable development time of 1-2 months.

---

### Assumptions and definitions

- We use only the frames of a 1920x1080 resolution video from as input.
- The frame rate of the video is at least 30 FPS, with no theoretical upper limit, though much beyond 60 FPS is impractical.
- A shuttle is in play from the moment it leaves the player’s hand during serves and stops being in play when it stops 
  moving on the ground (not when it lands).
- We use present interchangeably with visible. Thus, a shuttle head is present if and only if it is not fully occluded 
  (occlusion by the racket does not qualify as fully occluded, as it is partial) and in the frame.



### Ideal result

A model which can perfectly predict whether a shuttle is in play and present. If it is, return (x_pred, y_pred, 1) 
where x_pred and y_pred are the normalized coordinates in [0, 1] x [0, 1]. Otherwise, return (0, 0, 0) for that frame 
(the last entry represents visibility). Furthermore, the model should be fast enough to achieve real-time inference 
on consumer GPUs, which we define to be 60 FPS.



### Extensions considered

- **Shot prediction:** I originally wanted to predict the shot a player makes moments before impact. Such a prediction 
   model would require not only video data, but features such as the shuttle head position, racket orientation and pose 
   estimation of the players. I soon realized that the time required for such a project would be infeasible. Worse, the 
   existence of such a model is not guaranteed; if the best players in the world are frequently fooled by deceptions, 
   what chances does a model trained by an undergraduate student have?
- **Commentary:** Combined with a bounding box + multi-object tracking model for the players, the ideal-result model 
   could comment on the game, offering simple remarks such as “player 1 hit a cross-court drop from the back-court left 
   corner” and “player 3 played a winning smash straight down the line”.

<br><br>