import os
import sys
import json
from ast import literal_eval

models = ["xception", "inception", "vgg", "resnet"]
statusesPath = os.environ["PROJECT_DIR"] + "/statuses"
imagesPath = os.environ["PROJECT_DIR"] + "/dataset"
#bestModel = {model: getBestModel(model) for model in models}

#def getBestModel(model):
#	prefix = "transferlearning_train_ "
#	i = 0
#	minValidationError = 1.
#	bestModel = 0
#	while os.path.isfile(statusesPath + prefix + i + ".status"):
#		with open(statusesPath + prefix + i + ".status", encoding='utf-8') as data_file:
#			data = json.loads(data_file.read())
#			if data["validation_error"] < minValidationError:
#				minValidationError = data["validation_error"]
#				bestModel = i
#		i += 1	
#	return bestModel

def getTestDistribution(test_path):
    trueImages = len([name for name in os.listdir(test_path + '/merger/') if os.path.isfile(test_path + '/merger/' + name)])
    falseImages = len([name for name in os.listdir(test_path + '/noninteracting/') if os.path.isfile(test_path + '/noninteracting/' + name)])
    totalImages = trueImages + falseImages
    print(trueImages, falseImages, totalImages)
    return trueImages/totalImages, falseImages/totalImages, totalImages


def ensemblePredictions():
	perFilePredictions = {}
	for model in models:
#		with open(statusesPath + "/" + model + "/probPreds") as f:
		modelPredictions = (literal_eval(f) for f in open(statusesPath + "/" + model + "/probPreds"))
		for p in modelPredictions:
			for pred in p:
				if pred[0] not in perFilePredictions:
					perFilePredictions[pred[0]] = pred[1]
				else:
					perFilePredictions[pred[0]] = [x + y for x, y in zip(perFilePredictions[pred[0]], pred[1])]

	ensembledPredictions = {}

	for filename, predictions in perFilePredictions.items():
		ensembledPredictions[filename] = [x/len(models) for x in predictions]

	return ensembledPredictions

def applyPosterior(ensembledPredictions):
	weightedPredictions = {}
	for key, value in ensembledPredictions.items():
		weightedPredictions[key] = [value[0]*mergerProportion, value[1]* nonMergerProportion]

	return weightedPredictions

mergerProportion, nonMergerProportion, totalImages = getTestDistribution(imagesPath + "/test")

bestModel = {
	"vgg": 2,
	"inception": 1,
	"xception": 7,
	"resnet": 2
}

ensembledPredictions = ensemblePredictions()
print(sys.argv)
if len(sys.argv) <= 1:
	ensembledPredictions = applyPosterior(ensembledPredictions)

correct = 0
for key, value in ensembledPredictions.items():
	if ("merger" in key and max(value) == value[0]) or  ("noninteracting" in key and max(value) == value[1]):
		correct += 1

if len(sys.argv) > 1  and sys.argv[1] == "withPosterior":
	print("Accuracy of Ensemble Model with Posterior: ", correct/totalImages)

else:
	print("Accuracy of Ensemble Model: ", correct/totalImages)
