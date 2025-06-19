#region imports
from AlgorithmImports import *

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, Dense, Lambda, Flatten, Concatenate
from tensorflow.keras import Model
from tensorflow.keras import metrics
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras import utils
from tensorflow.keras.models import model_from_json
from tensorflow.keras.config import enable_unsafe_deserialization
from tensorflow.keras.saving import register_keras_serializable
from sklearn.preprocessing import StandardScaler
import math
from keras.utils import set_random_seed
#endregion

set_random_seed(0)
enable_unsafe_deserialization()

# Define the variables used to make predictions.
factor_names = ['open', 'high', 'low', 'close', 'volume']  


class Direction:
    """Constants used for labeling price movements."""
    # Labels must be integers because Keras (and most ML Libraries) 
    # only work with numbers.
    
    UP = 0
    DOWN = 1
    STATIONARY = 2


@register_keras_serializable()
def f0(x):
    return tf.split(x, num_or_size_splits=3, axis=1)[0]
    
@register_keras_serializable()
def f1(x):
    return tf.split(x, num_or_size_splits=3, axis=1)[1]

@register_keras_serializable()
def f2(x):
    return tf.split(x, num_or_size_splits=3, axis=1)[2]


class TemporalCNN:
    """Temporal Convolutional Neural Network Model built upon Keras."""
    
    # The name describes the architecture of the Neural Network model.
    # Temporal refers to the fact the layers are separated temporally 
    # into three regions. Convolutional refers to the fact Convolutional
    # layers are used to extract features.

    def __init__(self, model_json, n_tsteps=15):
        # n_tsteps is the number of time steps in time series for one 
        # input/prediction.
        self._n_tsteps = n_tsteps
        self._scaler = StandardScaler()  # Used for Feature Scaling
        
        # Create the model.
        if model_json:
            # Load it from the Object Store.
            self._cnn = model_from_json(model_json)
        else:
            # Create a new one from scratch.
            self._cnn = self._create_model()

        # Compile the model.
        self._cnn.compile(
            optimizer='adam',
            loss=CategoricalCrossentropy(from_logits=True)
        )

    def _create_model(self):
        """Creates the neural network model."""
        inputs = Input(shape=(self._n_tsteps, len(factor_names)))
        
        # Extract features using a Convolutional layers ("CNN").
        feature_extraction = Conv1D(30, 4, activation='relu')(inputs)

        # Split layer into three regions based on time, ("Temporal").
        long_term = Lambda(f0, output_shape=(4, 30))(feature_extraction)
        mid_term = Lambda(f1, output_shape=(4, 30))(feature_extraction)
        short_term = Lambda(f2, output_shape=(4, 30))(feature_extraction)
        
        long_term_conv = Conv1D(1, 1, activation='relu')(long_term)
        mid_term_conv = Conv1D(1, 1, activation='relu')(mid_term)
        short_term_conv = Conv1D(1, 1, activation='relu')(short_term)
        
        # Combine the three layers back into one.
        combined = Concatenate(axis=1)(
            [long_term_conv, mid_term_conv, short_term_conv]
        )
        
        # Flattening is required since our input is a 2D matrix.
        flattened = Flatten()(combined)
        
        # 1 output neuron for each class (Up, Stationary, Down).
        # See the Direction class.
        outputs = Dense(3, activation='softmax')(flattened)
        
        # Specify the input and output layers of the model.
        return Model(inputs=inputs, outputs=outputs)
    
    def _prepare_data(
            self, data, rolling_avg_window_size=5, stationary_threshold=.0001):
        """Prepares the data for a format friendly for our model."""
        # rolling_avg_window_size is the window size for the future mid 
        # prices to average. This average is what the model wants to 
        # predict.
        # stationary_threshold is the maximum change of movement to be 
        # considered stationary for the average mid price stated above.
        df = data[factor_names]
        shift = -(rolling_avg_window_size - 1)
    
        # Define a function to label our data.
        def label_data(row):
            if row['close_avg_change_pct'] > stationary_threshold:
                return Direction.UP
            elif row['close_avg_change_pct'] < -stationary_threshold:
                return Direction.DOWN
            else:
                return Direction.STATIONARY
            
        # Compute the percentage change in the average of the close of 
        # the future 5 time steps at each time step.
        df['close_avg'] = df['close'].rolling(
            window=rolling_avg_window_size
        ).mean().shift(shift) 
        df['close_avg_change_pct'] = \
            (df['close_avg'] - df['close']) / df['close']
         
        # Label data based on direction.
        # axis=1 signifies a row-wise operation (axis=0 is col-wise).
        df['movement_labels'] = df.apply(label_data, axis=1)
        
        # Create lists to store each 2D input matrix and the 
        # corresponding label.
        data = []
        labels = []
        for i in range(len(df)-self._n_tsteps+1+shift):
            label = df['movement_labels'].iloc[i + self._n_tsteps - 1]
            data.append(df[factor_names].iloc[i:i + self._n_tsteps].values)
            labels.append(label)
        data = np.array(data)
        
        # Temporarily reshape the data to 2D since sklearn only works 
        # with 2D data.
        dim1, dim2, dim3 = data.shape
        data = data.reshape(dim1 * dim2, dim3)
        
        # Fit our scaler and transform our data in one method call.
        data = self._scaler.fit_transform(data)
        
        # Return the data to the original shape.
        data = data.reshape(dim1, dim2, dim3)
        
        # Keras needs dummy matrices for classification problems, hence 
        # the need for to_categorical(). num_classes ensures our dummy 
        # matrix has 3 columns, one for each label.
        return data, utils.to_categorical(labels, num_classes=3)

    def train(self, data):
        """Trains the model."""
        data, labels = self._prepare_data(data)
        self._cnn.fit(data, labels, epochs=20)
        return self._cnn.to_json()
        
    def predict(self, input_data):
        """
        Makes a prediction on the direction of the future stock 
        price.
        """
        input_data = self._scaler.transform(
            input_data.fillna(method='ffill').values
        )
        prediction = self._cnn.predict(input_data[np.newaxis, :])[0]
        direction = np.argmax(prediction)
        confidence = prediction[direction]
        return direction, confidence


