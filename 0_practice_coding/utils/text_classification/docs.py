def print_rnn_docs():
    docs_str = '''
        1. Embedding(input_dim, output_dim, ...) \n
        https://keras.io/api/layers/core_layers/embedding/ \n
        2. LSTM(units, ...) \n
        https://keras.io/api/layers/recurrent_layers/lstm/ \n
        3. Bidirectional(layer, ...) \n
        https://keras.io/api/layers/recurrent_layers/bidirectional/ \n
        4. Dense(units, activation, ...) \n
        https://keras.io/api/layers/core_layers/dense/
        5. Dropout(rate) \n
        https://keras.io/api/layers/regularization_layers/dropout/
    '''
    print(docs_str)
