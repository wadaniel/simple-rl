import tensorflow as tf

def value_policy_loss(mu, sigma, action, tderror):
    return (0.5 * ( (action - mu) / sigma )**2 + 0.5*tf.math.log(2 * np.pi * sigma**2)) * tderror

def value_policy_network(stateSpace, actionSpace, hiddenLayers):
    inputs = tf.keras.Input(shape=(stateSpace,))
    for i, size in enumerate(hiddenLayers):
        if i == 0:
            x = tf.keras.layers.Dense(size, activation='relu')(inputs)
        else:
            x = tf.keras.layers.Dense(size, activation='relu')(x)


    value = tf.keras.layers.Dense(1, activation = "linear")(x)
    mu = tf.keras.layers.Dense(actionSpace, activation = "linear")(x)
    sigma = tf.keras.layers.Dense(actionSpace, activation = "softplus")(x)

    outputs = tf.keras.layers.Concatenate()([value, mu, sigma])
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='value_policy_network')

    return model


