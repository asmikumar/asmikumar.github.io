from keras.models import model_from_json
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from PIL import Image
import numpy as np
import flask
from flask import render_template, redirect
from flask import request, url_for, render_template, redirect
import io
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
model = None

UPLOAD_FOLDER = os.path.join(app.root_path ,'static','img')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def load_model():
	# load the pre-trained Keras model (here we are using a model
	# pre-trained on ImageNet and provided by Keras, but you can
	# substitute in your own networks just as easily)
	global model
	#model = ResNet50(weights="imagenet")
	#model = tf.keras.models.load_model('model.h5')
	with open('model/model.json', 'r') as f:
		model = model_from_json(f.read())
	model.load_weights('model/weights.hdf5')

def prepare_image(image, target):
	# if the image mode is not RGB, convert it
	if image.mode != "RGB":
		image = image.convert("RGB")

	# resize the input image and preprocess it
	image = image.resize(target)
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)
	image = imagenet_utils.preprocess_input(image)

	# return the processed image
	return image

@app.route("/", methods=["POST","GET"])
def predict():
	# initialize the data dictionary that will be returned from the view
	#data = {"success": False}
	data = {"success": "Upload — Satellite Image"}
	title = "Predicting Poverty"
	name = "default.png"
	# ensure an image was properly uploaded to our endpoint
	if flask.request.method == "POST":
		if flask.request.files.get("image"):
			
			image1 = flask.request.files["image"]
			# save the image to the upload folder, for display on the webpage.
			image = image1.save(os.path.join(app.config['UPLOAD_FOLDER'], image1.filename))
			
			# read the image in PIL format
			with open(os.path.join(app.config['UPLOAD_FOLDER'], image1.filename), 'rb') as f:
				image = Image.open(io.BytesIO(f.read()))
			
			# preprocess the image and prepare it for classification
			processed_image = prepare_image(image, target=(200, 200))

			'''
			# classify the input image and then initialize the list
			# of predictions to return to the client
			with graph.as_default():
				preds = model.predict(processed_image)
				results = imagenet_utils.decode_predictions(preds)
				data["predictions"] = []
			
			# loop over the results and add them to the list of
			# returned predictions
			for (imagenetID, label, prob) in results[0]:
				r = {"label": label, "probability": float(prob)}
				data["predictions"].append(r)
			'''

			def get_luminosity(category):
				luminosity = None
				if category == 0: # dim
					luminosity = np.random.randint(0, 3)
				elif category == 1: # medium
					luminosity = np.random.randint(3, 35)
				elif category == 2: # bright
					luminosity = np.random.randint(35, 64)
				return luminosity

			def predict_wealth(luminosity):
				slope = 0.07268134920271797
				intercept = -0.3465173128978627
				return np.around(luminosity * slope + intercept, decimals=2)

			index_label_dict = {0: "DIM", 1: "MEDIUM", 2: "BRIGHT"}
			with graph.as_default():
				preds = model.predict(processed_image)
				category = np.argmax(preds)
				label = "Predicted nightlight brightness: {}".format(index_label_dict[category])
				label_wealth = "Predicted wealth index: {}".format(str(predict_wealth(get_luminosity(category))))
				prob = np.max(preds)
				r = {"label": label, "probability": prob, "label_wealth": label_wealth}

				data["predictions"] = []
				data["predictions"].append(r)
			
			# indicate that the request was a success
			data["success"] = "Wealth Predictions"
			title = "predict"
			
			return render_template('index.html', data=data, title = title, name=image1.filename)
	# return the data dictionary as a JSON response
	return render_template('index.html', data = data, title=title, name=name)
# if this is the main thread of execution first load the model and
# then start the server

if __name__ == "__main__":
	print(("* Loading Keras model and Flask starting server..."
		"please wait until server has fully started.(60sec)"))
	load_model()
	global graph
	#graph = tf.get_default_graph()
	graph = tf.compat.v1.get_default_graph()
	app.run(debug=True)