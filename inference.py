import cv2
import os
import re
import glob
import random
import shutil
import warnings
import numpy as np 
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.image as mpimg
from platform import python_version
from IPython.display import Image, display
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf
import keras
from tensorflow import keras
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.applications import vgg16
from keras.utils import img_to_array, array_to_img, load_img
from keras.preprocessing.image import ImageDataGenerator
import matplotlib

SHAPE = (224, 224, 3)
classes = ["NORMAL", "PNEUMONIA"]
last_conv_layer_name = "block5_conv3"


def set_seed(seed):
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)

def load_model():
    set_seed(33)
    
    vgg = vgg16.VGG16(weights=None, include_top=False, input_shape=SHAPE)

    for layer in vgg.layers[:-8]:
        layer.trainable = False

    x = vgg.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(2, activation="softmax")(x)

    model = Model(vgg.input, x)
    model.compile(loss="categorical_crossentropy", 
                    optimizer=SGD(learning_rate=0.0001, momentum=0.9), metrics=["accuracy"])
    
    model.load_weights('vgg16_model_weights_10.h5')  # Load custom weights here
    
    return model

model = load_model()

def preprocess_input(img_path, size):
    img = keras.preprocessing.image.load_img(img_path, target_size=size)
    array = keras.preprocessing.image.img_to_array(img)
    array = np.expand_dims(array, axis=0)
    return array

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def gradcam_inference(img_path, output_path="static/inference_gradcam"):
    img_array = preprocess_input(img_path, size=SHAPE[:2])
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)

    img = cv2.imread(img_path)
    og_dim = (img.shape[1], img.shape[0])
    zoom_dim = tuple([og_dim[0]*2,og_dim[1]*2])
    img = cv2.resize(img, (SHAPE[1], SHAPE[0]))

    heatmap = cv2.resize(heatmap, (SHAPE[1], SHAPE[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = heatmap * 0.4 + img
    superimposed_img = cv2.resize(superimposed_img, zoom_dim)

    output_img_path = output_path + "/grad_cam_" + os.path.basename(img_path)
    cv2.imwrite(output_img_path, superimposed_img)


    pred_array = model.predict(img_array)
    predicted_class = classes[np.argmax(pred_array)]

    return predicted_class,output_img_path

# # Example usage:
# gradcam_inference("chest_xray/test/PNEUMONIA/person24_virus_58.jpeg")
