#region imports
from AlgorithmImports import *

import pywt
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
#endregion


class SVMWavelet:

    def forecast(self, data):
        """
        Decomposes 1-D array "data" into multiple components using 
        Discrete Wavelet Transform, denoises each component using 
        thresholding, use Support Vector Regression (SVR) to forecast 
        each component, recombine components for aggregate forecast.

        returns: the value of the aggregate forecast 1 time-step into 
        the future
        """
        # Daubechies/Symlets are good choices for denoising.
        w = pywt.Wavelet('sym10') 
        
        threshold = 0.5

        # Decompose the data into wavelet components.
        coeffs = pywt.wavedec(data, w)
        
        # If we want at least 3 levels (components), solve for:
        #   log2(len(data) / wave_length - 1) >= 3
        # In this case, since we wave_length(sym10) == 20, after 
        # solving, we get len(data) >= 152, hence why our RollingWindow 
        # is of length 152 in main.py.

        for i in range(len(coeffs)):
            if i > 0:
                # Don't threshold the approximation coefficients.
                coeffs[i] = pywt.threshold(coeffs[i], threshold*max(coeffs[i]))
            forecasted = self._svm_forecast(coeffs[i])
            coeffs[i] = np.roll(coeffs[i], -1)
            coeffs[i][-1] = forecasted
            
        datarec = pywt.waverec(coeffs, w)
        return datarec[-1]

    def _svm_forecast(self, data, sample_size=10):
        '''
        Paritions `data` and fits an SVM model to this data, then 
        forecasts the value one time-step into the future
        '''
        X, y = self._partition_array(data, size=sample_size)

        gsc = GridSearchCV(
            SVR(), 
            {
                'C': [.05, .1, .5, 1, 5, 10], 
                'epsilon': [0.001, 0.005, 0.01, 0.05, 0.1]
            }, 
            scoring='neg_mean_squared_error'
        )
        
        model = gsc.fit(X, y).best_estimator_
        return model.predict(data[np.newaxis, -sample_size:])[0]
        
    def _partition_array(self, arr, size=None, splits=None):
        '''
        Partitions 1-D array `arr` in a Rolling fashion if `size` is 
        specified, else, divides them into `splits` pieces.

        returns: list of paritioned arrays, list of the values 1 step 
        ahead of each partitioned array
        '''
        arrs = []
        values = []

        if not (bool(size is None) ^ bool(splits is None)):
            raise ValueError('Size XOR Splits should not be None')

        if size:
            arrs = [arr[i:i + size] for i in range(len(arr) - size)]
            values = [arr[i] for i in range(size, len(arr))]

        elif splits:
            size = len(arr) // splits
            if len(arr) % size == 0:
                arrs = [arr[i:i+size] for i in range(size-1, len(arr)-1, size)]
                values = [arr[i] for i in range(2*size - 1, len(arr), size)]
            else:
                arrs = [
                    arr[i:i+size]
                    for i in range(len(arr) % size - 1, len(arr) - 1, size)
                ]
                values = [
                    arr[value].iloc[i]
                    for i in range(len(arr) % size + size - 1, len(arr), size)
                ]
                
        return np.array(arrs), np.array(values)