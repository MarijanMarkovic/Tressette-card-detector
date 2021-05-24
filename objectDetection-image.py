import os
import cv2
import numpy as np
import tensorflow as tf


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging (1)
tf.get_logger().setLevel('ERROR')  # Suppress TensorFlow logging (2)

# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


# PROVIDE PATH TO SAVED MODEL DIRECTORY
PATH_TO_SAVED_MODEL = 'object_detection/inference_graph/saved_model'

# PROVIDE PATH TO LABEL MAP
PATH_TO_LABELS = 'object_detection/training/labelmap.pbtxt'

# PROVIDE THE MINIMUM CONFIDENCE THRESHOLD
MIN_CONF_THRESH = 0.5

# LOAD THE MODEL
import time
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

print('Loading model...', end='')
start_time = time.time()

# LOAD SAVED MODEL AND BUILD DETECTION FUNCTION
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)

end_time = time.time()
elapsed_time = end_time - start_time
print('Model loaded! Took {} seconds'.format(elapsed_time))

# LOAD LABEL MAP DATA FOR PLOTTING
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

TEST_DIR_PATH = 'object_detection/images/test1'
mylist = os.listdir(TEST_DIR_PATH)


for count, value in enumerate(mylist):

    # PROVIDE PATH TO IMAGE DIRECTORY
    IMAGE_PATHS = 'object_detection/images/test/' + value
    print('Running inference for {}... '.format(IMAGE_PATHS), end='')

    image = cv2.imread(IMAGE_PATHS)

    # THE INPUT NEEDS TO BE A TENSOR, CONVERT IT WITH `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    # THE MODEL EXPECTS A BATCH OF IMAGES, SO ADD AN AXIS WITH `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]
    print(input_tensor.shape)

    detections = detect_fn(input_tensor)

    # ALL OUTPUTS ARE BATCHES TENSORS,
    # CONVERT TO NUMPY ARRAY AND TAKE THE INDEX [0] TO REMOVE THE BATCH DIMENSION.
    # THE FIRST num_detections IS ONLY OF INTEREST
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # DETECTION CLASSES SHOULD BE INTS
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    image_with_detections = image.copy()

    # SET MIN_SCORE_THRESH BASED ON YOU MINIMUM THRESHOLD FOR DETECTIONS
    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_with_detections,
        detections['detection_boxes'],
        detections['detection_classes'],
        detections['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=50,
        min_score_thresh=MIN_CONF_THRESH,
        agnostic_mode=False)

    # DISPLAYS OUTPUT IMAGE
    cv2.imshow('Object Detector', image_with_detections)
    # CLOSES WINDOW ONCE KEY IS PRESSED
    cv2.waitKey(0)
    # CLEANUP
    cv2.destroyAllWindows()
