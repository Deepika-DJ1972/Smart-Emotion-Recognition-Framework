"""
This file can be used to try a live prediction. 
"""

import keras
import librosa
import numpy as np

from config import EXAMPLES_PATH
from config import MODEL_DIR_PATH


class LivePredictions:
    """
    Main class of the application.
    """

    def __init__(self, file):
        """
        Init method is used to initialize the main parameters.
        """
        self.file=file
        self.path = MODEL_DIR_PATH + 'Emotion_Voice_Detection_Model.h5'
        self.loaded_model = keras.models.load_model(self.path)
    
    def make_predictions(self):
        """
        Method to process the files and create your features.
        """
       
        data, sampling_rate = librosa.load(self.file)
        
        mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=40).T, axis=0)
        #x = np.expand_dims(mfccs, axis=2)
        
        x = np.expand_dims(mfccs, axis=0)
        
        #predictions = self.loaded_model.predict_classes(x)
        #predictions=self.loaded_model.predict(x)
        #predictions=np.round(predictions).astype(int)
        predictions=np.argmax(self.loaded_model.predict(x),axis=1)
        
        #print(type(predictions))
        #print(predictions)
        
        print( "Prediction is", " ", self.convert_class_to_emotion(predictions))

    @staticmethod
    def convert_class_to_emotion(pred):
        """
        Method to convert the predictions (int) into human readable strings.
        """
        
        label_conversion = {'0': 'neutral',
                            '1': 'calm',
                            '2': 'happy',
                            '3': 'sad',
                            '4': 'angry',
                            '5': 'fearful',
                            '6': 'disgust',
                            '7': 'surprised'}

        for key, value in label_conversion.items():
            if int(key) == pred:
                label = value
        return label


if __name__ == '__main__':
    live_prediction = LivePredictions(file="C:\\Users\\steph\\OneDrive\\Desktop\\desktop\\emotion-classification-from-audio-files-master\\examples\\a1.wav")
    live_prediction.loaded_model.summary()
    live_prediction.make_predictions()
    live_prediction = LivePredictions(file="C:\\Users\\steph\\OneDrive\\Desktop\\desktop\\emotion-classification-from-audio-files-master\\examples\\03-01-01-01-01-02-05.wav")
    live_prediction.make_predictions()
    live_prediction = LivePredictions(file="C:\\Users\\steph\\OneDrive\\Desktop\\desktop\\emotion-classification-from-audio-files-master\\examples\\Recording (3).wav")
    live_prediction.make_predictions()
    live_prediction = LivePredictions(file="C:\\Users\\steph\\OneDrive\\Desktop\\desktop\\emotion-classification-from-audio-files-master\\examples\\OAF_calm_happy.wav")
    live_prediction.make_predictions()
    
    