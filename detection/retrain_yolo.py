"""
This is a script that can be used to retrain the YOLOv2 model for your own dataset.
"""
import argparse

import os

import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Lambda, Conv2D
from keras.models import load_model, Model
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping

from yad2k.models.keras_yolo import (preprocess_true_boxes, yolo_body,
									 yolo_eval, yolo_head, yolo_loss)
from yad2k.utils.draw_boxes import draw_boxes

# Args
argparser = argparse.ArgumentParser(
	description="Retrain or 'fine-tune' a pretrained YOLOv2 model for your own data.")

argparser.add_argument(
	'-d',
	'--data_path',
	help="path to numpy data file (.npz) containing np.object array 'boxes' and np.uint8 array 'images'",
	default=os.path.join('..', 'COCO_DATA', 'anotations.txt'))

argparser.add_argument(
	'-a',
	'--anchors_path',
	help='path to anchors file, defaults to yolo_anchors.txt',
	default=os.path.join('model_data', 'yolo_anchors.txt'))

argparser.add_argument(
	'-c',
	'--classes_path',
	help='path to classes file, defaults to pascal_classes.txt',
	default=os.path.join('model_data', 'coco_classes.txt'))

# Default anchor boxes
YOLO_ANCHORS = np.array(
	((0.57273, 0.677385), (1.87446, 2.06253), (3.33843, 5.47434),
	 (7.88282, 3.52778), (9.77052, 9.16828)))

def _main(args):
	data_path = os.path.expanduser(args.data_path)
	classes_path = os.path.expanduser(args.classes_path)
	anchors_path = os.path.expanduser(args.anchors_path)

	class_names = get_classes(classes_path)
	anchors = get_anchors(anchors_path)

	#data = np.load(data_path) # custom data saved as a numpy file.
	#  has 2 arrays: an object array 'boxes' (variable length of boxes in each image)
	#  and an array of images 'images'

	#image_data, boxes = process_data(data['images'], data['boxes'])

	anchors = YOLO_ANCHORS

	#detectors_mask, matching_true_boxes = get_detector_mask(boxes, anchors)

	model_body, model = create_model(anchors, class_names)

	train(
		model,
		class_names,
		anchors,
		data_path
	)

	draw(model_body,
		class_names,
		anchors,
		test_data,
		weights_name='trained_stage_3_best.h5',
		save_all=False)


def get_classes(classes_path):
	'''loads the classes'''
	with open(classes_path) as f:
		class_names = f.readlines()
	class_names = [c.strip() for c in class_names]
	return class_names

def get_anchors(anchors_path):
	'''loads the anchors from a file'''
	if os.path.isfile(anchors_path):
		with open(anchors_path) as f:
			anchors = f.readline()
			anchors = [float(x) for x in anchors.split(',')]
			return np.array(anchors).reshape(-1, 2)
	else:
		Warning("Could not open anchors file, using default.")
		return YOLO_ANCHORS

def process_data(images, boxes=None):
	'''processes the data'''
	#images = [PIL.Image.fromarray(i) for i in images]
	orig_size = [np.array([i.width, i.height]) for i in images]
	#orig_size = np.expand_dims(orig_size, axis=0)

	# Image preprocessing.
	processed_images = [i.resize((416, 416), PIL.Image.BICUBIC) for i in images]
	processed_images = [np.array(image, dtype=np.float) for image in processed_images]
	processed_images = [image/255. for image in processed_images]

	if boxes is not None:
		# Box preprocessing.
		# Original boxes stored as 1D list of class, x_min, y_min, x_max, y_max.
		#boxes = [box.reshape((-1, 4)) for box in boxes]
		# Get extents as y_min, x_min, y_max, x_max, class for comparision with
		# model output.
		#boxes_extents = [box[:, [1, 0, 3, 2]] for box in boxes]
		# Get box parameters as x_center, y_center, box_width, box_height, class.
		boxes_xy = [0.5 * (box[:, 2:4]) + box[:, 0:2] for box in boxes]
		boxes_wh = [box[:, 2:4] for box in boxes]
		boxes_xy = [boxxy / orig_size[i] for i, boxxy in enumerate(boxes_xy)]
		boxes_wh = [boxwh / orig_size[i] for i, boxwh in enumerate(boxes_wh)]
		boxes = [np.concatenate((boxes_xy[i], boxes_wh[i]), axis=1) for i, box in enumerate(boxes)]

		# find the max number of boxes
		max_boxes = 0
		for boxz in boxes:
			if boxz.shape[0] > max_boxes:
				max_boxes = boxz.shape[0]

		# add zero pad for training
		for i, boxz in enumerate(boxes):
			if boxz.shape[0]  < max_boxes:
				zero_padding = np.zeros( (max_boxes-boxz.shape[0], 4), dtype=np.float32)
				boxes[i] = np.vstack((boxz, zero_padding))

		return np.array(processed_images), np.array(boxes)
	else:
		return np.array(processed_images)

def get_detector_mask(boxes, anchors):
	'''
	Precompute detectors_mask and matching_true_boxes for training.
	Detectors mask is 1 for each spatial position in the final conv layer and
	anchor that should be active for the given boxes and 0 otherwise.
	Matching true boxes gives the regression targets for the ground truth box
	that caused a detector to be active or 0 otherwise.
	'''
	detectors_mask = [0 for i in range(len(boxes))]
	matching_true_boxes = [0 for i in range(len(boxes))]
	for i, box in enumerate(boxes):
		detectors_mask[i], matching_true_boxes[i] = preprocess_true_boxes(box, anchors, [416, 416])

	return np.array(detectors_mask), np.array(matching_true_boxes)

def create_model(anchors, class_names, load_pretrained=True, freeze_body=True):
	'''
	returns the body of the model and the model

	# Params:

	load_pretrained: whether or not to load the pretrained model or initialize all weights

	freeze_body: whether or not to freeze all weights except for the last layer's

	# Returns:

	model_body: YOLOv2 with new output layer

	model: YOLOv2 with custom loss Lambda layer

	'''

	detectors_mask_shape = (13, 13, 5, 1)
	matching_boxes_shape = (13, 13, 5, 4)

	# Create model input layers.
	image_input = Input(shape=(416, 416, 3))
	boxes_input = Input(shape=(None, 4))
	detectors_mask_input = Input(shape=detectors_mask_shape)
	matching_boxes_input = Input(shape=matching_boxes_shape)

	# Create model body.
	yolo_model = yolo_body(image_input, len(anchors), len(class_names))
	topless_yolo = Model(yolo_model.input, yolo_model.layers[-2].output)

	if load_pretrained:
		# Save topless yolo:
		topless_yolo_path = os.path.join('model_data', 'yolo_topless.h5')
		if not os.path.exists(topless_yolo_path):
			print("CREATING TOPLESS WEIGHTS FILE")
			yolo_path = os.path.join('model_data', 'yolo.h5')
			model_body = load_model(yolo_path)
			model_body = Model(model_body.inputs, model_body.layers[-2].output)
			model_body.save_weights(topless_yolo_path)
		topless_yolo.load_weights(topless_yolo_path)

	if freeze_body:
		for layer in topless_yolo.layers:
			layer.trainable = False
	final_layer = Conv2D(len(anchors)*(5), (1, 1), activation='linear')(topless_yolo.output)

	model_body = Model(image_input, final_layer)

	# Place model loss on CPU to reduce GPU memory usage.
	with tf.device('/cpu:0'):
		# TODO: Replace Lambda with custom Keras layer for loss.
		model_loss = Lambda(
			yolo_loss,
			output_shape=(1, ),
			name='yolo_loss',
			arguments={'anchors': anchors,
					   'num_classes': len(class_names)})([
						   model_body.output, boxes_input,
						   detectors_mask_input, matching_boxes_input
					   ])

	model = Model(
		[model_body.input, boxes_input, detectors_mask_input,
		 matching_boxes_input], model_loss)

	return model_body, model

def data_generator(base_path, data, batch_size=32):
	i = 0
	while True:
		if i+batch_size >= len(data):
			i = 0
		images = []
		boxes = []
		for j in range(batch_size):
			line = data[i+j]
			parts = line.strip().split(" ")
			file_name = os.path.join(base_path, parts[0])
			image = PIL.Image.open(file_name)
			if np.ndim(np.array(image)) != 3:
				#TODO: find a better method to handle grayscale images
				image = np.array([np.array(image), np.array(image), np.array(image)])
				image = PIL.Image.fromarray(np.transpose(image, (1,2,0)))
			images.append(image)
			dets = []
			for k in range(int(len(parts[1:])/4)):
				dets.append([float(parts[4*k+1]),float(parts[4*k+2]),float(parts[4*k+3]),float(parts[4*k+4])])
			boxes.append(np.array(dets))

		i += batch_size

		image_data, boxes = process_data(images,boxes)

		anchors = YOLO_ANCHORS

		detectors_mask, matching_true_boxes = get_detector_mask(boxes, anchors)

		yield [image_data, boxes, detectors_mask, matching_true_boxes], np.zeros(batch_size)

def train(model, class_names, anchors, data_path, validation_split=0.1):
	'''
	retrain/fine-tune the model

	logs training with tensorboard

	saves training weights in current directory

	best weights according to val_loss is saved as trained_stage_3_best.h5
	'''
	model.compile(
		optimizer='adam', loss={
			'yolo_loss': lambda y_true, y_pred: y_pred
		})  # This is a hack to use the custom loss function in the last layer.


	logging = TensorBoard()
	checkpoint = ModelCheckpoint("trained_stage_3_best.h5", monitor='val_loss',
								 save_weights_only=True, save_best_only=True)
	early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=15, verbose=1, mode='auto')

	with open(data_path) as f:
		data = f.readlines()
	#base path changes for each dataset
	base_path = os.path.join(os.path.dirname(data_path),"train2014")
	if not os.path.exists("trained_stage_1.h5"):
		model.fit_generator(data_generator(base_path, data, batch_size=32), callbacks=[logging], steps_per_epoch=len(data)//32, verbose=1, epochs=5)
		#model.fit([image_data, boxes, detectors_mask, matching_true_boxes],
		#		  np.zeros(len(image_data)),
		#		  validation_split=validation_split,
		#		  batch_size=32,
		#		  epochs=5,
		#		  callbacks=[logging])
		model.save('trained_stage_1.h5')

	model_body, model = create_model(anchors, class_names, load_pretrained=False, freeze_body=False)

	model.load_weights('trained_stage_1.h5')

	model.compile(
		optimizer='adam', loss={
			'yolo_loss': lambda y_true, y_pred: y_pred
		})  # This is a hack to use the custom loss function in the last layer.


	model.fit_generator(data_generator(base_path, data, batch_size=8), callbacks=[logging], steps_per_epoch=len(data)//8, verbose=1, epochs=30)
	#model.fit([image_data, boxes, detectors_mask, matching_true_boxes],
	#		  np.zeros(len(image_data)),
	#		  validation_split=0.1,
	#		  batch_size=8,
	#		  epochs=30,
	#		  callbacks=[logging])

	model.save('trained_stage_2.h5')

	model.fit_generator(data_generator(base_path, data, batch_size=8), callbacks=[logging, checkpoint, early_stopping], steps_per_epoch=len(data)//8, verbose=1, epochs=30)
	#model.fit([image_data, boxes, detectors_mask, matching_true_boxes],
	#		  np.zeros(len(image_data)),
	#		  validation_split=0.1,
	#		  batch_size=8,
	#		  epochs=30,
	#		  callbacks=[logging, checkpoint, early_stopping])

	model.save('trained_stage_3.h5')

def draw(model_body, class_names, anchors, test_data, weights_name='trained_stage_3_best.h5', out_path="output_images", save_all=True):
	'''
	Draw bounding boxes on image data
	'''
	
	print(test_data.shape)
	model_body.load_weights(weights_name)

	# Create output variables for prediction.
	yolo_outputs = yolo_head(model_body.output, anchors, len(class_names))
	input_image_shape = K.placeholder(shape=(2, ))
	boxes, scores = yolo_eval(
		yolo_outputs, input_image_shape, score_threshold=0.07, iou_threshold=0)

	# Run prediction on overfit image.
	sess = K.get_session()  # TODO: Remove dependence on Tensorflow session.

	if  not os.path.exists(out_path):
		os.makedirs(out_path)
	for i in range(len(image_data)):
		out_boxes, out_scores = sess.run(
			[boxes, scores],
			feed_dict={
				model_body.input: image_data[i],
				input_image_shape: [image_data.shape[2], image_data.shape[3]],
				K.learning_phase(): 0
			})
		print('Found {} boxes for image.'.format(len(out_boxes)))
		print(out_boxes)

		# Plot image with predicted boxes.
		image_with_boxes = draw_boxes(image_data[i][0], out_boxes, out_scores)
		# Save the image:
		if save_all or (len(out_boxes) > 0):
			image = PIL.Image.fromarray(image_with_boxes)
			image.save(os.path.join(out_path,str(i)+'.png'))

		# To display (pauses the program):
		# plt.imshow(image_with_boxes, interpolation='nearest')
		# plt.show()



if __name__ == '__main__':
	args = argparser.parse_args()
	_main(args)