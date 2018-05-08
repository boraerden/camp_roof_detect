import sys
sys.path.insert(0, '../../ssd_keras')
sys.path.insert(0, '../../ssd_keras/ssd_encoder_decoder')
sys.path.insert(0, '../../ssd_keras/bouding_box_utils')
sys.path.insert(0, '../../data_generator_object_detection_2d/')
sys.path.insert(0, '../../data_generator_object_detection_2d/misc_utils')

from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TerminateOnNaN, CSVLogger
from keras import backend as K
from keras.models import load_model
from math import ceil
import numpy as np
from matplotlib import pyplot as plt
from pdb import set_trace as t

from models.keras_ssd7 import build_model
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast


from ssd_input_encoder import SSDInputEncoder
from ssd_output_decoder import decode_detections, decode_detections_fast

from object_detection_2d_data_generator import DataGenerator
from object_detection_2d_misc_utils import apply_inverse_transforms
from data_augmentation_chains.data_augmentation_chain_variable_input_size import DataAugmentationVariableInputSize
from data_augmentation_chains.data_augmentation_chain_constant_input_size import DataAugmentationConstantInputSize
from data_augmentation_chains.data_augmentation_chain_original_ssd import SSDDataAugmentation


###############################################################

img_height = 256 # Height of the input images
img_width = 256 # Width of the input images
img_channels = 3 # Number of color channels of the input images
intensity_mean = 127.5 # Set this to your preference (maybe `None`). The current settings transform the input pixel values to the interval `[-1,1]`.
intensity_range = 127.5 # Set this to your preference (maybe `None`). The current settings transform the input pixel values to the interval `[-1,1]`.
n_classes = 2 # Number of positive classes
scales = None #[0.08, 0.16, 0.32, 0.64, 0.96] # An explicit list of anchor box scaling factors. If this is passed, it will override `min_scale` and `max_scale`.
aspect_ratios = [0.5, 1.0, 2.0] # The list of aspect ratios for the anchor boxes
two_boxes_for_ar1 = False # Whether or not you want to generate two anchor boxes for aspect ratio 1
steps = None # In case you'd like to set the step sizes for the anchor box grids manually; not recommended
offsets = None # In case you'd like to set the offsets for the anchor box grids manually; not recommended
clip_boxes = False # Whether or not to clip the anchor boxes to lie entirely within the image boundaries
variances = [1.0, 1.0, 1.0, 1.0] # The list of variances by which the encoded target coordinates are scaled
normalize_coords = True # Whether or not the model is supposed to use coordinates relative to the image size

###############################################################

# 1: Build the Keras model

K.clear_session() # Clear previous models from memory.

model = build_model(image_size=(img_height, img_width, img_channels),
                    n_classes=n_classes,
                    mode='training',
                    l2_regularization=0.0005,
                    scales=scales,
                    aspect_ratios_global=aspect_ratios,
                    aspect_ratios_per_layer=None,
                    two_boxes_for_ar1=two_boxes_for_ar1,
                    steps=steps,
                    offsets=offsets,
                    clip_boxes=clip_boxes,
                    variances=variances,
                    normalize_coords=normalize_coords,
                    subtract_mean=intensity_mean,
                    divide_by_stddev=intensity_range)

# 2: Optional: Load some weights

model.load_weights('../../ssd7_checkpoints/ssd7_epoch-11_loss-2.3288_val_loss-2.3078.h5', by_name=True)

# 3: Instantiate an Adam optimizer and the SSD loss function and compile the model

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

model.compile(optimizer=adam, loss=ssd_loss.compute_loss)


###############################################################
# TEST #

images_dir_train         = '../../data_folder/train/img/0/'
image_set_filename_train = '../../data_folder_imagesets/imagesets_train.txt'
annotations_dir_train    = '../../data_folder_bboxes_plusminus5/annotations_train/'

images_dir_val         = '../../data_folder/val/img/0/'
image_set_filename_val = '../../data_folder_imagesets/imagesets_val.txt'
annotations_dir_val    = '../../data_folder_bboxes_plusminus5/annotations_val/'

classes = ['background',
           'house']


train_dataset = DataGenerator()
val_dataset = DataGenerator()

train_dataset.parse_xml(images_dirs=[images_dir_train],
                  image_set_filenames=[image_set_filename_train],
                  annotations_dirs=[annotations_dir_train],
                  classes=classes,
                  include_classes='all',
                  exclude_truncated=False,
                  exclude_difficult=False,
                  ret=False)

val_dataset.parse_xml(images_dirs=[images_dir_val],
                  image_set_filenames=[image_set_filename_val],
                  annotations_dirs=[annotations_dir_val],
                  classes=classes,
                  include_classes='all',
                  exclude_truncated=False,
                  exclude_difficult=False,
                  ret=False)

# 1: Set the generator for the predictions.

predict_generator = val_dataset.generate(batch_size=1,
                                         shuffle=True,
                                         transformations=[],
                                         label_encoder=None,
                                         returns={'processed_images',
                                                  'processed_labels',
                                                  'filenames'},
                                         keep_images_without_gt=False)

# 2: Generate samples
for i in range(5):
  batch_images, batch_labels, batch_filenames = next(predict_generator)

  i = 0 # Which batch item to look at

  print("Image:", batch_filenames[i])
  print()
  print("Ground truth boxes:\n")
  print(batch_labels[i])

  # 3: Make a prediction

  y_pred = model.predict(batch_images)

  # 4: Decode the raw prediction `y_pred`

  y_pred_decoded = decode_detections(y_pred,
                                     confidence_thresh=0.5,
                                     iou_threshold=0.45,
                                     top_k=200,
                                     normalize_coords=normalize_coords,
                                     img_height=img_height,
                                     img_width=img_width)

  np.set_printoptions(precision=2, suppress=True, linewidth=90)
  print("Predicted boxes:\n")
  print('   class   conf xmin   ymin   xmax   ymax')
  print(y_pred_decoded[i])

  # 5: Draw the predicted boxes onto the image

  plt.figure(figsize=(20,12))
  plt.imshow(batch_images[i])

  current_axis = plt.gca()

  colors = plt.cm.hsv(np.linspace(0, 1, n_classes+1)).tolist() # Set the colors for the bounding boxes
  classes = ['background', 'house'] # Just so we can print class names onto the image instead of IDs

  # Draw the ground truth boxes in green (omit the label for more clarity)
  for box in batch_labels[i]:
      xmin = box[1]
      ymin = box[2]
      xmax = box[3]
      ymax = box[4]
      label = '{}'.format(classes[int(box[0])])
      current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color='green', fill=False, linewidth=2))  
      #current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor':'green', 'alpha':1.0})

  # Draw the predicted boxes in blue
  for box in y_pred_decoded[i]:
      xmin = box[-4]
      ymin = box[-3]
      xmax = box[-2]
      ymax = box[-1]
      color = colors[int(box[0])]
      label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
      current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color=color, fill=False, linewidth=2))  
      current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor':color, 'alpha':1.0})


  plt.show()