# Tennis Shot Recognition

## Dataset
The project is already provided with a dataset, however, it is possible to expand it by hand.

### Tennis video

Firstly it is required to have a video of a tennis rally. It is possible to download it from youtube with a video converter

### Body keypoints

Run the script to note all the body parts keypoints poisitions during the entire video with the command:
```
python Landmarker.py Video/Video_Name.mp4
```

### Annotate shots

Afterwards, you need to manually note the types of shots 
To do so you can run the following command:
```
python Labeler.py Video/Video_Name.mp4
```
After the video starts it is possible to annotate the shot in the exact frame where the player hits the ball, by using your keyborad:
* **Left arrow** : Go back 5 frames
* **Right arrow** : Go forward 5 frames
* **Up arrow** : Annotate Forehand
* **Down arrow** : Annotate Backhand
* **Space** : Annotate Serve
* **Esc** : Exit and save annotations

### Adding shots to the dataset

Lastly, you can use the shots annotation made to create the csv file for the dataset by the following command:
```
python .\Shots_Annotation.py ".\video\Video_Name.mp4" ".\Shots_Annotation\Video_Name_Labeled.csv" ".\DataSet\Player_Name"
```
For example
> python Shots_Annotation.py ".video\Nadal_2min.mp4" ".\Shots_Annotation\Nadal_2min_Labeled.csv" ".\DataSet\Nadal"

Objectives:

- Extract body key point from a video
- Create a dataset for the 3 main shots in tennis (forehand, backhand, service) and neutral position
- Train a model predict and classify the shots in a video
- Use diffferent model (LSTM, GRU, Transformer) in order to compare the results and performances
  
<img width="1092" height="309" alt="image" src="https://github.com/user-attachments/assets/16932434-a100-4bef-8637-7d19ff4ed7eb" />

<img width="982" height="332" alt="image" src="https://github.com/user-attachments/assets/c9b49ba8-31b9-4d78-b2c0-98d543d1ba2e" />
