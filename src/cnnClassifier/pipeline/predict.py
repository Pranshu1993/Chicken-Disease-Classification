import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

class PredictionPipeline:
    def __init__(self,filename):
        self.filename =filename
    
    def predict(self):
        # load model
        model = load_model(os.path.join("artifacts","training", "model.keras"))

        imagename = self.filename
        test_image = image.load_img(imagename, target_size = (224,224))
        test_image = image.img_to_array(test_image)
        print("Before Normalisation: ", test_image)
        test_image = test_image / 255.0
        test_image = np.expand_dims(test_image, axis = 0)
        # result = np.argmax(model.predict(test_image), axis=1)
        result = model.predict(test_image, verbose=0)
        print("The final result: ",result)

        # class_labels = {0: 'Coccidiosis', 1: 'Healthy'}

        if result[0][0] >= 0.5:
            prediction = 'Healthy'
            return [{ "image" : prediction}]
        else:
            prediction = 'Coccidiosis'
            return [{ "image" : prediction}]