#USAGE

#CREATE DATASET
connet to hsr then
$ cd ...to your workplace
$ mkidr dataset
$ python get_data.py
(press 's' to save an image, num of image should > 200)

separate the dataset to train and test (leave about 15 image for test)

#ANNOTATION
$ python label_tool.py
(after annotated an image, press 'n' to save the label and turn to the next image)

#TRAIN
$ python train.py
(train loss should lower than 100, test loss may around 8000?)

#EVAL
$ python eval.py (modify eval.py   img_file = ("location to test image")




If everything is OK, then start move the robot
build the robot package
then roscd to the directory that contains the trained model (for example, params.meta, params.index...)
$ python force_listener_wrs.py
open another terminal (in the same directory)
$ python send_tf_wrs.py 
open another terminal (in the same directory)
$ python detect_holes_wrs.py (after the object can be seen, press 's' to detect holes, should wait a moment until the model is loaded, otherwise you will get an error)

check the hole locations using rosrun rviz , show tf 

open another terminal (in the same directory)
$ python robot_execute_wrs.py


hand over a leg to the robot, and press its hand, it should start assembly, after it finishs one leg, hand over another leg and press its hand....