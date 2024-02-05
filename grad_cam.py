import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import argparse

import matplotlib.cm as cm
import glob
import tensorflow.keras

'''
Gradcam source code adapted from:
https://keras.io/examples/vision/grad_cam/
For further information, please visit the link.
'''
model_builder = keras.applications.xception.Xception
preprocess_input = keras.applications.xception.preprocess_input
decode_predictions = keras.applications.xception.decode_predictions
last_conv_layer_name = "block4_pool"

def get_img_array(img_path, size):
    img = keras.preprocessing.image.load_img(img_path, target_size=size)
    array = keras.preprocessing.image.img_to_array(img)
    array = np.expand_dims(array, axis=0)
    return array


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()
    
    
def save_gradcam(img_path, heatmap, cam_path, alpha=0.4):
    # Load the original image
    img = keras.preprocessing.image.load_img(img_path)
    img = keras.preprocessing.image.img_to_array(img)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img * (1-alpha)
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)
    
    # Save the superimposed image
    superimposed_img.save(cam_path)
    # Save heatmap image only
    heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    heatmap.save(cam_path[0:-4] + "heatmap" + cam_path[-4:])
    #save xray image only
    xray = keras.preprocessing.image.array_to_img(img)
    xray.save(cam_path[0:-4] + "xray" + cam_path[-4:])
    
    
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="001", help="Name of model to be loaded / tested.")
parser.add_argument("--path_model", type=str, default=None, help="Path to main folder where models are saved.")
parser.add_argument("--path_dataset", type=str, default=None, help="Path to dataset.")
parser.add_argument("--path_output", type=str, default=None, help="Path to output gradcam images.")

args = parser.parse_args()
print("List of arguments:")
print(args)

#load the model
model_name = args.model_name
path_model = args.path_model
model_path = os.path.join(path_model, model_name)
model_checkpoint_path = os.path.join(path_model, model_name + "_checkpoint")
loaded_model = tensorflow.keras.models.load_model(model_checkpoint_path)
# Remove last layer's sigmoid
loaded_model.layers[-1].activation = None

#load test files
img_size = (224, 224)
path_to_test = args.path_dataset
path_neg = os.path.join(path_to_test, "neg")
path_pos = os.path.join(path_to_test, "pos")
files_neg = sorted(glob.glob(path_neg + "/*.png"))
files_pos = sorted(glob.glob(path_pos + "/*.png"))
files = files_neg + files_pos

#specify output location
path_output = args.path_output
path_output = os.path.join(path_output, model_name)
os.makedirs(path_output, exist_ok=True)



    
for image in files:
    #converts image to an image array
	img_path = image
	img_array = preprocess_input(get_img_array(img_path, size=img_size))
    #grabs image filename for heatmap saving
	image_filename, image_extension = os.path.splitext(os.path.basename(img_path))
	img_path_destination = os.path.join(path_output, image_filename + ".png")

	# Generate class activation heatmap
	heatmap = make_gradcam_heatmap(img_array, loaded_model, last_conv_layer_name)
    
	#save image with activation heatmap
	save_gradcam(img_path, heatmap, cam_path=img_path_destination)
