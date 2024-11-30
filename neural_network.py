import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense
from keras.layers import Conv1D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Activation
from keras.models import Sequential
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


from config import SAVE_DIR_PATH
from config import MODEL_DIR_PATH


class TrainModel:

    @staticmethod
    def train_neural_network(X_file_path, y_file_path) -> None:
        X = joblib.load(X_file_path)
        y = joblib.load(y_file_path)

        """
        This function trains the neural network.
        """

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

        x_traincnn = np.expand_dims(X_train, axis=2)
        x_testcnn = np.expand_dims(X_test, axis=2)

        print(x_traincnn.shape, x_testcnn.shape)

        model = Sequential()
        model.add(Conv1D(64, 5, padding='same',
                         input_shape=(40, 1)))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(8))
        model.add(Activation('softmax'))

        print(model.summary())

        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])

        cnn_history = model.fit(x_traincnn, y_train,
                                batch_size=16, epochs=50,
                                validation_data=(x_testcnn, y_test))

        # Loss plotting
        plt.plot(cnn_history.history['loss'])
        plt.plot(cnn_history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig('loss.png')
        plt.close()

        # Accuracy plotting
        plt.plot(cnn_history.history['accuracy'])
        plt.plot(cnn_history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('acc')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig('accuracy.png')

        predictions = model.predict(x_testcnn)
        y_pred = np.argmax(predictions, axis=1)  # Convert probabilities to class labels

        # Convert categorical labels to numerical values
        label_encoder = LabelEncoder()
        y_true = label_encoder.fit_transform(y_test)
        y_pred = label_encoder.transform(y_pred)

        # Compute the confusion matrix
        matrix = confusion_matrix(y_true, y_pred)

        print(classification_report(y_true, y_pred))
        print(matrix)

        model_name = 'Emotion_Voice_Detection_Model.h5'

        # Save model and weights
        if not os.path.isdir(MODEL_DIR_PATH):
            os.makedirs(MODEL_DIR_PATH)
        model_path = os.path.join(MODEL_DIR_PATH, model_name)
        model.save(model_path)
        print('Saved trained model at %s' % model_path)



if __name__ == '__main__':
    print('Training started')
    X_file_path = os.path.join(SAVE_DIR_PATH, "C:\\Users\\steph\\OneDrive\\Desktop\\desktop\\emotion-classification-from-audio-files-master\\joblib_features\\X.joblib")
    y_file_path = os.path.join(SAVE_DIR_PATH, "C:\\Users\\steph\\OneDrive\\Desktop\\desktop\\emotion-classification-from-audio-files-master\\joblib_features\\y.joblib")


print(X_file_path)
print(y_file_path)
NEURAL_NET = TrainModel.train_neural_network(X_file_path, y_file_path)
