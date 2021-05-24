# Tressette-card-detector
This is a python program that uses openCV and Tensorflow 2.x to detect tresette playing cards on an image or webcam feed. It is created using Object Detection API with TensorFlow 2 on Windows.
Tressette card detector was trained on a pretrained efficientdet_d1_coco17_tpu-32 downloaded from [Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md)
To 



## Usage
After the pull request it is necessary to extract the '.rar' files from inference_graph and inference_graph/saved_model which are compressed due to the upload size restrictions.



## Files
objectDetection-webcam.py - webcam card detector

objectDetection-image.py - image card detector

images/test1 - directory containing test images loaded into objectDetection-image.py script

training - directory containing 'test.record' and 'train.record' files which are used to feed 'ssd_efficientdet_d1_640x640_coco17_tpu-8.config' file. It also contains 'labelmap.pbtxt' file which labels 10 different classes from the card deck. 

inference_graph - contains the data for a trained card detector placed in **saved_model** folder. The **checkpoint** folder was provided in order to allow the user to further train the model from the last checkpoint by changing the checkpoint path in the .config file placed in training directory.
