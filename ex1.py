from keras.applications import ResNet50
from keras.applications import InceptionV3
from keras.applications import Xception # TensorFlow ONLY
from keras.applications import VGG16
from keras.applications import VGG19
from keras.applications import imagenet_utils
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


#construct argument parse
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help = "path to the input image")
ap.add_argument("-model", "--model", type=str, default="xception", help="name of pre-trained network to use")
args = vars(ap.parse_args())

#args["image"] = "abc.png"
#args["model"] = "xception"

#define dictionary mapping model to classes inside Keras
MODELS = {"vgg16": VGG16, "vgg19": VGG19, "inception": InceptionV3, "xception": Xception, "resnet": ResNet50}

#ensure valid model name supplied
if args["model"] not in MODELS.keys():
	raise AssertionError("The --model command line should be a key in the 'MODELS' dictionary")

#input shape
inputShape = (224,224)
preprocess = imagenet_utils.preprocess_input

if args["model"] in ("inception", "xception"):
	inputShape = (299,299)
	preprocess = preprocess_input

#pretrained models
print("[INFO] loading {}...".format(args["model"]))
Network = MODELS[args["model"]]
model=Network(weights="imagenet")

#load input images
print("[INFO] loading and pre-processing images...")
image = load_img(args["image"],target_size=inputShape)
image = img_to_array(image)
image = np.expand_dims(image, axis=0)
image = preprocess(image)

#classify the image
print("[INFO] classifying image with '{}'...".format(args["model"]))
preds = model.predict(image)
P = imagenet_utils.decode_predictions(preds)

# loop over the predictions and display the rank-5 predictions +
# probabilities to our terminal
for (i, (imagenetID, label, prob)) in enumerate(P[0]):
	print("{}. {}: {:.2f}%".format(i + 1, label, prob * 100))

#Load Images
from scipy import misc
img2=mpimg.imread(args["image"])
plt.imshow(img2)
plt.show()

plt.hist(img2.ravel(), bins=256, range=(0.0, 1.0), fc='k', ec='k')
plt.show()

(imagenetID, label, prob) = P[0][0]
print(imagenetID, label, prob)

#load the image via OpenCV, draw top predictions, and display on screen
#orig = cv2.imread(args["image"])
#(imagenetID, label, prob) = P[0][0]
#cv2.putText(orig, "Label: {}, {:.2f}%".format(label, prob * 100),
#	(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
#cv2.imshow("Classification", orig)
#cv2.waitKey(0)

