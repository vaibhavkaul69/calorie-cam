import cv2
import numpy as np
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
import keras
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from keras.applications.vgg16 import preprocess_input
from flask import Flask, render_template, request
import requests
import os
from volume_estimator import VolumeEstimator
import sys
import json
from keras.models import Model, model_from_json
from food_volume_estimation.depth_estimation.custom_modules import *
from food_volume_estimation.food_segmentation.food_segmentator import FoodSegmentator
import matplotlib.pyplot as plt
from pyntcloud import PyntCloud

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
    print("FOLDER CREATED")


@app.route('/')
def index():
    return "hi SACHIN SANS"


@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return 'No file uploaded'

    file = request.files['file']
    filepath = (os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
    # filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    # CONFIG
    # Define configuration for the model
    class InferenceConfig(Config):
        # Give the configuration a recognizable name
        NAME = "segmentation_of_food"

        # Set batch size to 1 since we're running inference on a single image
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        # Number of classes (including background)
        NUM_CLASSES = 2  # 1 Background + 1 Object

    # Create config object
    config = InferenceConfig()

    # Create Mask R-CNN model in inference mode
    model_seg = MaskRCNN(mode="inference", config=config, model_dir="./logs")

    # Load pre-trained weights
    model_seg.load_weights('mask_rcnn_food_segmentation.h5', by_name=True)
    clusters = ['food']
    class_names = ['bg'] + clusters

    # MOdels
    model_classifier = load_model('model.h5')
    # START
    path = filepath
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model_seg.detect([image], verbose=0)
    r = results[0]
    out_put = []
    for i in range(r["rois"].shape[0]):
        masked_image = cv2.bitwise_and(image, image, mask=r["masks"][:, :, i].astype(np.uint8))
        masked_image = cv2.resize(masked_image, (224, 224))
        out_put.append(masked_image)
    lable = []
    for i in out_put:
        img_array = np.expand_dims(i, axis=0)
        img_array = preprocess_input(img_array)
        result = model_classifier.predict(img_array)
        classes = ['alloo_matar', 'apple', 'banana', 'bhindi', 'carrot', 'cucumber', 'dal makhni', 'fried rice',
                   'Jalebi', 'orange', 'pizza', 'roti', 'samosa']
        res = np.argmax(result)
        lable.append(classes[res])
    # Paths to model archiecture/weights
    depth_model_architecture = 'monovideo_fine_tune_food_videos.json'
    depth_model_weights = 'monovideo_fine_tune_food_videos.h5'
    segmentation_model_weights = 'mask_rcnn_food_segmentation.h5'

    # Create estimator object and intialize
    estimator = VolumeEstimator(arg_init=False)
    with open(depth_model_architecture, 'r') as read_file:
        custom_losses = Losses()
        objs = {'ProjectionLayer': ProjectionLayer,
                'ReflectionPadding2D': ReflectionPadding2D,
                'InverseDepthNormalization': InverseDepthNormalization,
                'AugmentationLayer': AugmentationLayer,
                'compute_source_loss': custom_losses.compute_source_loss}
        model_architecture_json = json.load(read_file)
        estimator.monovideo = model_from_json(model_architecture_json, custom_objects=objs)
    estimator._VolumeEstimator__set_weights_trainable(estimator.monovideo, False)
    estimator.monovideo.load_weights(depth_model_weights)
    estimator.model_input_shape = estimator.monovideo.inputs[0].shape.as_list()[1:]
    depth_net = estimator.monovideo.get_layer('depth_net')
    estimator.depth_model = Model(inputs=depth_net.inputs, outputs=depth_net.outputs, name='depth_model')
    print('[*] Loaded depth estimation model.')

    # Depth model configuration
    MIN_DEPTH = 0.01
    MAX_DEPTH = 10
    estimator.min_disp = 1 / MAX_DEPTH
    estimator.max_disp = 1 / MIN_DEPTH
    estimator.gt_depth_scale = 0.35  # Ground truth expected median depth

    # Create segmentator object
    estimator.segmentator = FoodSegmentator(segmentation_model_weights)

    # Set plate adjustment relaxation parameter
    estimator.relax_param = 0.01
    outputs_list = estimator.estimate_volume(path, fov=70, plate_diameter_prior=0,
                                             plot_results=False, lable=lable)

    # FINAL OUTPUT
    out = outputs_list
    Calories = {}
    weights = {}
    calaroies = pd.read_csv("Calories.csv")

    for i in range(len(out)):
        vol = out[i] * 1000000
        ind = list(calaroies.loc[calaroies.name == lable[i]].index)[0]
        wei = calaroies.density[ind] * vol
        calory = (wei / 100) * calaroies.cal[ind]
        if wei > 10:
            if lable[i] in Calories.keys():
                Calories[lable[i]] += calory
                weights[lable[i]] += wei
            else:
                Calories[lable[i]] = calory
                weights[lable[i]] = wei

    total_cal = np.sum(list(Calories.values()))
    total_weight = np.sum(list(weights.values()))
    print("Total Calories your are going to intake is", round(total_cal, 2), "cal")
    print("Total weight of food is", round(total_weight, 2), "gms")
    outpu = {}
    outpu['1'] = "Total Calories your are goint to intake is" + " " + str(round(total_cal, 2)) + "cal"
    outpu['2'] = "Total weight of food is" + " " + str(round(total_weight, 2)) + "gms"
    print(filepath)

    return render_template('result.html', out=outpu)


if __name__ == '__main__':
    app.run()
