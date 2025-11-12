# Tennis Shot Recognition

## Dataset
The project is already provided with a dataset, however, it is possible to expand it by hand.

### Tennis video

Firstly it is required to have a video of a tennis rally. It is possible to download it from youtube with a video converter.

### Body keypoints

Run the script to note all the body parts keypoints poisitions during the entire video with the command:
```
python Landmarker.py Video/Video_Name.mp4
```

### Annotate shots

Afterwards, you need to manually note the types of shots 
To do so you have to run the following command:
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

Lastly, you must use the shots annotation made to create the csv file for the dataset by the following command:
```
python Shots_Annotation.py ".\video\Video_Name.mp4" ".\Shots_Annotation\Video_Name_Labeled.csv" ".\DataSet\Player_Name"
```
For example
> python Shots_Annotation.py ".video\Nadal_2min.mp4" ".\Shots_Annotation\Nadal_2min_Labeled.csv" ".\DataSet\Nadal"

# Training the model

# Display results and real time shot classification

To display the model training results and watch the calssification in action, run the **Landarker.py** script with ```--infer``` and ```--model_path``` tags as a command as follow: 
```
python Landmarker.py ".\video\Video_Name.mp4" --infer --model_path "Model_Name.keras"
```
For example:
> python Landmarker.py ".\video\Federer_2min.mp4" --infer --model_path "Shot_Classification_SimpleRNN.keras

or:

> python Landmarker.py ".\video\Federer_2min.mp4" --infer --model_path "Shot_Classification_GRU.keras
