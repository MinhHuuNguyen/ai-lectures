def print_cnn_docs():
    docs_str = '''
        1. Conv2D(filters, kernel_size, strides, padding, ...) \n
        https://keras.io/api/layers/convolution_layers/convolution2d/ \n
        2. BatchNormalization() \n
        https://keras.io/api/layers/normalization_layers/batch_normalization/ \n
        3. MaxPooling2D(pool_size, strides, padding, ...) \n
        https://keras.io/api/layers/pooling_layers/max_pooling2d/ \n
        4. Flatten() \n
        https://keras.io/api/layers/reshaping_layers/flatten/ \n
        5. Dense(units, activation, ...) \n
        https://keras.io/api/layers/core_layers/dense/ \n
        6. Dropout(rate) \n
        https://keras.io/api/layers/regularization_layers/dropout/
    '''
    print(docs_str)
