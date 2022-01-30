import tensorflow as tf

def scaledGlorot(shape, dtype=None):
    fan_in, fan_out = shape
    limit = 0.001*tf.math.sqrt(6 / (fan_in + fan_out))
    return tf.random.uniform(shape, minval=-limit, maxval=limit, dtype=dtype)

def initValuePolicyNetwork(stateSpace, actionSpace, hiddenLayers):
    inputs = tf.keras.Input(shape=(stateSpace,))
    for i, size in enumerate(hiddenLayers):
        if i == 0:
            x = tf.keras.layers.Dense(size, kernel_initializer=scaledGlorot, activation='tanh')(inputs)
        else:
            x = tf.keras.layers.Dense(size, kernel_initializer=scaledGlorot, activation='tanh')(x)


    value = tf.keras.layers.Dense(1, kernel_initializer=scaledGlorot, activation = "linear")(x)
    mu = tf.keras.layers.Dense(actionSpace, kernel_initializer=scaledGlorot, activation = "linear")(x)
    sigma = tf.keras.layers.Dense(actionSpace, kernel_initializer=scaledGlorot, activation = "softplus")(x)

    outputs = tf.keras.layers.Concatenate()([value, mu, sigma])
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='valuePolicyNetwork')

    return model


