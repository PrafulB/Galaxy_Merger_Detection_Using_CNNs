import numpy as np
import time
import json
import os
import sys

from keras.applications.xception import Xception
from keras.preprocessing import image
from keras.models import Model, load_model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.optimizers import SGD


def createGenerator(class_path):

    translation_range=0.05
    zoom_base = (2**(1/2)) * ((1 - translation_range) ** -1)
    zoom_factor=2.
    image_data_generator = image.ImageDataGenerator(rotation_range=360,
                                                    zoom_range=(zoom_base, zoom_base * zoom_factor),
                                                    width_shift_range=translation_range,
                                                    height_shift_range=translation_range,
                                                    vertical_flip=True)

    target_size=(299, 299)
    generator = image_data_generator.flow_from_directory(class_path,
                                                        shuffle=True,
                                                        seed=9,
                                                        batch_size=8,
                                                        target_size=target_size)

    while True:
        from_gen = next(generator)
        images = [from_gen[0]]
        labels = [from_gen[1]]

        cumulative_batch, cumulative_labels = np.vstack(images), np.vstack(labels)
        
        yield cumulative_batch, cumulative_labels

def createTestGenerator(class_path):
    translation_range=0.05
    zoom_base = (2**(1/2)) * ((1 - translation_range) ** -1)
    zoom_factor=2.
    
    image_data_generator = image.ImageDataGenerator(rotation_range=360,
                                                    zoom_range=(zoom_base, zoom_base * zoom_factor),
                                                    width_shift_range=translation_range,
                                                    height_shift_range=translation_range,
                                                    vertical_flip=True)

    target_size=(299, 299)
    generator = image_data_generator.flow_from_directory(class_path,
                                                        shuffle=False,
                                                        seed=9,
                                                        batch_size=1,
                                                        target_size=target_size)
    return generator


def writeStatusToFile(filename, epoch, training_error, training_loss, validation_error, validation_loss):

    info = {
            'epoch': epoch,
            'timestamp': time.time(),
            'training_error': training_error,
            'training_loss': training_loss,
            'validation_error': validation_error,
            'validation_loss': validation_loss
        }

    with open(filename, 'w') as fp:
        json.dump(info, fp)


def train(model, epochs, imagesPath, statusesWritePath, mode, training_type):

    trainingGenerator = createGenerator(imagesPath + '/training')
    validationGenerator = createGenerator(imagesPath + '/validation')

    steps_per_epoch = 3000
    validation_steps = 1500

    for epoch in range(epochs):

        print("Starting {} epoch # {}".format(training_type, epoch))
        
        history = model.fit_generator(trainingGenerator, 
                                      steps_per_epoch = steps_per_epoch,
                                      epochs = 1, 
                                      validation_data = validationGenerator, 
                                      validation_steps = validation_steps, 
                                      use_multiprocessing=True, 
                                      workers=10)
        
        print("Finished {} epoch #{}".format(training_type, epoch))


        training_error, validation_error = (1. - history.history['acc'][0]), (1. - history.history['val_acc'][0])
        training_loss, validation_loss = history.history['loss'][0], history.history['val_loss'][0]

        model.save(checkpointsPath + "/{}_{}_{}.checkpoint".format(mode, training_type, epoch))
        writeStatusToFile(statusesWritePath + "/{}_{}_{}.status".format(mode, training_type, epoch),
                     epoch,
                     training_error,
                     training_loss,
                     validation_error,
                     validation_loss)

    return model

def generateModel(mode, optimizer, lossFn="categorical_crossentropy", metrics=['accuracy']):
    
    if mode == "transferlearning":
        xception=Xception(weights = "imagenet", include_top = False)
    
    else:
        print("Setting up Xception with no learned weights")
        xception=Xception(weights = None, include_top = False)

    output=GlobalAveragePooling2D()(xception.output)
    output=Dense(1024, activation = "relu")(output)
    output=Dense(2, activation = "softmax")(output)
    model=Model(inputs = xception.input, outputs=output)


    # Don't change the weights for Xception at this stage if trying Transfer Learning.
    for layer in xception.layers:
        layer.trainable = False if mode == "transferlearning" else True

    model.compile(optimizer = optimizer, loss = lossFn, metrics = metrics)

    return model


def retrieveModelFromCheckpoint(checkpointsPath, trainFromEpoch, optimizer=SGD(lr=0.5*3e-5, momentum=0.9), lossFn="categorical_crossentropy", metrics=['accuracy'], useFor="train" ):
    try:
        modelCheckpointPath = checkpointsPath+"/train_{}.checkpoint".format(trainFromEpoch)
        model = load_model(modelCheckpointPath)

        if useFor == "train":
            # Train the entire model now.
            for layer in model.layers:
                layer.trainable = True

            model.compile(optimizer = optimizer, loss = lossFn, metrics = metrics)

        return model
    except Exception as e:
        print("Model not found for specified epoch!! Please try again with correct epoch.")
        raise e
        sys.exit(0)


def preTrainModel(imagesPath, checkpointsPath, statusesWritePath, mode, epochs = 30):

    model = generateModel(mode, optimizer="adam")

    model = train(model, epochs, imagesPath, statusesWritePath, mode, training_type='pretrain')

    model.save(checkpointsPath + "/{}_train_0.checkpoint".format(mode))


def trainModel(imagesPath, checkpointsPath, statusesWritePath, mode, trainFromEpoch, epochs=100):

    if mode == "transferlearning":
        model = retrieveModelFromCheckpoint(checkpointsPath, trainFromEpoch, optimizer=SGD(lr=0.5*3e-5, momentum=0.9), useFor="train")

    else:
        model = generateModel(mode, optimizer=SGD(lr=0.5*3e-5, momentum=0.9))

    train(model, epochs, imagesPath, statusesWritePath, mode, training_type='train')

def testModel(imagesPath, checkpointsPath, statusesWritePath, modelEpoch):

    model = retrieveModelFromCheckpoint(checkpointsPath, modelEpoch, useFor="test")
    testingGenerator = createTestGenerator(imagesPath + '/test')

    testingGenerator.reset()

    #getPredictions(model, testingGenerator)

    evaluateModel(model, testingGenerator)


def getPredictions(model, testingGenerator):
    predictions = model.predict_generator(testingGenerator, steps=3998, use_multiprocessing=True, workers=5)
    predictedLabels = np.argmax(predictions, axis=1)
    labels = testingGenerator.class_indices
    labels = dict((v, k) for k,v in labels.items())
    predictedLabels = [labels[i] for i in predictedLabels]

    predctionsByFileName = list(zip(testingGenerator.filenames, predictedLabels))

    print(predctionsByFileName)


def evaluateModel(model, testingGenerator):
    evaluationOnTestData = model.evaluate_generator(generator=testingGenerator)
    print(dict(zip(model.metrics_names, evaluationOnTestData)))
    info = {
            'timestamp': time.time(),
            'test_loss': evaluationOnTestData[0],
            'test_accuracy': evaluationOnTestData[1]
        }

    testStatusFile = statusesWritePath + "/test.status"
    with open(testStatusFile, 'w') as fp:
        json.dump(info, fp)

if __name__ == '__main__':

    imagesPath = os.environ['PROJECT_DIR'] + "/dataset"
    checkpointsPath = os.environ['PROJECT_DIR'] + "/checkpoints"
    statusesWritePath = os.environ['PROJECT_DIR'] + "/statuses"
    print(sys.argv)

    try:
        mode = "transferlearning" if sys.argv[1] == "transferlearning" else "randinit"
    except Exception as e:
        raise e
        print("Mode (argument 1) not set!")
        sys.exit(0)

    startFromEpoch = 0
    try:
        startFromEpoch = sys.argv[2]
        print("Training Model from epoch #{}".format(startFromEpoch))
    except Exception as e:
        raise e
        print("startFromEpoch (argument 2) not set. Training Model from scratch!")
        sys.exit(0)

    try:
        if mode == "transferlearning" and sys.argv[3] != "train" and sys.argv[3] != "test": 
            print("Starting pre-training of Output Layers")
            preTrainModel(imagesPath, checkpointsPath, statusesWritePath, mode)

        if sys.argv[3] != "test":
            print("Starting training of Complete Model")
            trainModel(imagesPath, checkpointsPath, statusesWritePath, mode, startFromEpoch)
    except Exception as e:
        raise e
        print("Run (argument 3) type is not set")
        sys.exit(0)

    print("Running the trained model on test data")
    testModel(imagesPath, checkpointsPath, statusesWritePath, startFromEpoch)
