import keras
from keras import layers
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Lambda

def conv_block(
    x,
    filters,
    activation,
    kernel_size=(3, 3),
    strides=1,
    padding="same",
    use_bias=True,
    use_bn=False,
    use_dropout=False,
    drop_value=0.5,
):
    x = layers.Conv1D(
        filters, kernel_size, strides=strides, padding=padding, use_bias=use_bias
    )(x)
    if use_bn:
        x = layers.BatchNormalization()(x)
    x = activation(x)
    if use_dropout:
        x = layers.Dropout(drop_value)(x)
    return x


def get_discriminator_model(in_shape=256, num_classes=3, print_summary=False):
    img_input = layers.Input(shape=(in_shape,))
    class_input = layers.Input(shape=(num_classes,))
    # Redimensionando para trabajar con señales de 256
    # Cambiamos a 32x8 para mantener una relación similar
    x = layers.Reshape((32, 8))(img_input)
    x = conv_block(
        x,
        64,  # Reduciendo número de filtros
        kernel_size=7,  # Kernel más pequeño
        strides=1,
        use_bn=False,
        use_bias=True,
        activation=layers.LeakyReLU(),
        use_dropout=True,
        drop_value=0.5,
    )
    x = conv_block(
        x,
        128,
        kernel_size=7,
        strides=2,
        use_bn=False,
        activation=layers.LeakyReLU(),
        use_bias=True,
        use_dropout=True,
        drop_value=0.5,
    )
    x = conv_block(
        x,
        256,
        kernel_size=7,
        strides=2,
        use_bn=False,
        activation=layers.LeakyReLU(),
        use_bias=True,
        use_dropout=True,
        drop_value=0.5,
    )
    x = conv_block(
        x,
        256,
        kernel_size=7,
        strides=2,
        use_bn=False,
        activation=layers.LeakyReLU(),
        use_bias=True,
        use_dropout=True,
        drop_value=0.5,
    )

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, use_bias=False)(x)
    
    class_ind = Lambda(lambda x: K.argmax(x, axis=-1))(class_input)
    class_embedding = layers.Embedding(num_classes, 128)(class_ind)

    dot_product = Lambda(lambda inputs: tf.reduce_sum(tf.multiply(inputs[0], inputs[1]), axis=1, keepdims=True))([x, class_embedding])

    scalar_function = layers.Dense(1, use_bias=False)(x)

    x = layers.Add()([dot_product, scalar_function])
    
    d_model = keras.models.Model([img_input, class_input], x, name="discriminator")
    if print_summary:
        print(d_model.summary())
    return d_model


def get_derivative_discriminator_model(in_shape=255, num_classes=3, print_summary=False):
    img_input = layers.Input(shape=(in_shape,))
    class_input = layers.Input(shape=(num_classes,))
    x = layers.Dense(256)(img_input)  # Reducido de 512 a 256
    x = layers.LeakyReLU()(x)
    x = layers.Reshape((16, 16))(x)  # Ajustado de (32,16) a (16,16)
    x = conv_block(
        x,
        64,
        kernel_size=5,
        strides=1,
        use_bn=False,
        use_bias=True,
        activation=layers.LeakyReLU(),
        use_dropout=True,
        drop_value=0.5,
    )
    x = conv_block(
        x,
        128,
        kernel_size=5,
        strides=2,
        use_bn=False,
        use_bias=True,
        activation=layers.LeakyReLU(),
        use_dropout=True,
        drop_value=0.5,
    )
    x = conv_block(
        x,
        256,
        kernel_size=5,
        strides=2,
        use_bn=False,
        activation=layers.LeakyReLU(),
        use_bias=True,
        use_dropout=True,
        drop_value=0.5,
    )

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, use_bias=False)(x)
    class_ind = Lambda(lambda x: K.argmax(x, axis=-1))(class_input)
    class_embedding = layers.Embedding(num_classes, 128)(class_ind)

    dot_product = Lambda(lambda inputs: tf.reduce_sum(tf.multiply(inputs[0], inputs[1]), axis=1, keepdims=True))([x, class_embedding])

    scalar_function = layers.Dense(1, use_bias=False)(x)

    x = layers.Add()([dot_product, scalar_function])

    d_model = keras.models.Model([img_input, class_input], x, name="derivative_discriminator")
    if print_summary:
        print(d_model.summary())
    return d_model


def get_second_derivative_discriminator_model(in_shape=254, num_classes=3, print_summary=False):
    img_input = layers.Input(shape=(in_shape,))
    class_input = layers.Input(shape=(num_classes,))
    x = layers.Dense(256)(img_input)  # Reducido de 512 a 256
    x = layers.LeakyReLU()(x)
    x = layers.Reshape((16, 16))(x)  # Ajustado de (32,16) a (16,16)
    x = conv_block(
        x,
        64,
        kernel_size=5,
        strides=1,
        use_bn=False,
        use_bias=True,
        activation=layers.LeakyReLU(),
        use_dropout=True,
        drop_value=0.5,
    )
    x = conv_block(
        x,
        128,
        kernel_size=5,
        strides=2,
        use_bn=False,
        use_bias=True,
        activation=layers.LeakyReLU(),
        use_dropout=True,
        drop_value=0.5,
    )
    x = conv_block(
        x,
        256,
        kernel_size=5,
        strides=2,
        use_bn=False,
        activation=layers.LeakyReLU(),
        use_bias=True,
        use_dropout=True,
        drop_value=0.5,
    )

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, use_bias=False)(x)
    class_ind = Lambda(lambda x: K.argmax(x, axis=-1))(class_input)
    class_embedding = layers.Embedding(num_classes, 128)(class_ind)

    dot_product = Lambda(lambda inputs: tf.reduce_sum(tf.multiply(inputs[0], inputs[1]), axis=1, keepdims=True))([x, class_embedding])
    scalar_function = layers.Dense(1, use_bias=False)(x)

    x = layers.Add()([dot_product, scalar_function])

    d_model = keras.models.Model([img_input, class_input], x, name="derivative2_discriminator")
    if print_summary:
        print(d_model.summary())
    return d_model


def upsample_block(
    x,
    filters,
    activation,
    kernel_size=7,  # Reducido de 18 a 7
    strides=2,
    up_size=2,
    padding="same",
    use_bn=False,
    use_bias=True,
    use_dropout=False,
    drop_value=0.3,
):
    x = layers.UpSampling1D(up_size)(x)
    x = layers.Conv1D(
        filters, kernel_size, strides=strides, padding=padding, use_bias=use_bias
    )(x)

    if use_bn:
        x = layers.BatchNormalization()(x)

    if activation:
        x = activation(x)
    if use_dropout:
        x = layers.Dropout(drop_value)(x)
    return x


def get_generator_model(noise_dim=100, num_classes=3, print_summary=False):
    noise = layers.Input(shape=(noise_dim,))
    class_input = layers.Input(shape=(num_classes,))
    class_embedding = layers.Dense(32, use_bias=False)(class_input)
    combined_input = layers.Concatenate()([noise, class_embedding])
    x = layers.Dense(256, use_bias=False)(combined_input)  # Reducido de 1024 a 256
    x = layers.ReLU()(x)

    x = layers.Reshape((16, 16))(x)  # Ajustado de (32, 32) a (16, 16)
    x = upsample_block(
        x,
        256,  # Reducido de 512 a 256
        layers.ReLU(),
        strides=1,
        use_bias=False,
        use_bn=True,
        padding="same",
        use_dropout=False,
    )
    x = upsample_block(
        x,
        128,
        layers.ReLU(),
        strides=1,
        use_bias=False,
        use_bn=True,
        padding="same",
        use_dropout=False,
    )
    x = upsample_block(
        x,
        64,
        layers.ReLU(),
        strides=1,
        use_bias=False,
        use_bn=True,
        padding="same",
        use_dropout=False,
    )
    x = upsample_block(
        x,
        1,
        layers.Activation('linear'),
        strides=1,
        use_bias=False,
        use_bn=False,
        padding="same",
        use_dropout=False,
    )
    x = layers.Flatten()(x)

    g_model = keras.models.Model([noise, class_input], x, name="generator")
    if print_summary:
        print(g_model.summary())
    return g_model


def get_discriminator_model_mc(in_shape=256, n_classes=3, print_summary=False):
    # Label input
    in_label = layers.Input(shape=(n_classes,))
    # Scale up to image dim with linear activation
    n_nodes = in_shape
    li = layers.Dense(n_nodes)(in_label)
    # Reshape to additional channel
    li = layers.Reshape((in_shape, 1))(li)
    # Image input
    in_image = layers.Input(shape=(256,))  # Cambiado de 1024 a 256
    In = layers.Reshape((in_shape, 1))(in_image)
    # Concat label as a channel
    merge = layers.Concatenate()([In, li])
    # Downsample
    fe = layers.Conv1D(64, 7, strides=2, padding='same')(merge)  # Kernel reducido de 14 a 7
    fe = layers.LeakyReLU(alpha=0.2)(fe)
    fe = layers.SpatialDropout1D(0.5)(fe)

    fe = layers.Conv1D(128, 7, strides=2, padding='same')(fe)
    fe = layers.LeakyReLU(alpha=0.2)(fe)
    fe = layers.SpatialDropout1D(0.5)(fe)

    fe = layers.Conv1D(256, 7, strides=2, padding='same')(fe)
    fe = layers.LeakyReLU(alpha=0.2)(fe)
    fe = layers.SpatialDropout1D(0.5)(fe)

    # Flatten feature map
    fe = layers.Flatten()(fe)
    # Output
    out_layer = layers.Dense(1, activation='sigmoid')(fe)
    # Define model
    model = keras.models.Model([in_image, in_label], out_layer, name="discriminator_mc")
    if print_summary:
        print(model.summary())
    return model


def get_derivative_discriminator_model_mc(in_shape=255, n_classes=3, print_summary=False):
    # Label input
    in_label = layers.Input(shape=(n_classes,))
    # Scale up to image dim with linear activation
    n_nodes = in_shape
    li = layers.Dense(n_nodes)(in_label)
    # Reshape to additional channel
    li = layers.Reshape((in_shape, 1))(li)
    # Image input
    in_image = layers.Input(shape=(in_shape,))
    In = layers.Reshape((in_shape, 1))(in_image)
    # Concat label as a channel
    merge = layers.Concatenate()([In, li])
    # Downsample
    fe = layers.Conv1D(64, 7, strides=2, padding='same')(merge)  # Kernel reducido de 14 a 7
    fe = layers.LeakyReLU(alpha=0.2)(fe)
    fe = layers.SpatialDropout1D(0.5)(fe)

    fe = layers.Conv1D(128, 7, strides=2, padding='same')(fe)
    fe = layers.LeakyReLU(alpha=0.2)(fe)
    fe = layers.SpatialDropout1D(0.5)(fe)

    fe = layers.Conv1D(256, 7, strides=2, padding='same')(fe)
    fe = layers.LeakyReLU(alpha=0.2)(fe)
    fe = layers.SpatialDropout1D(0.5)(fe)

    # Flatten feature map
    fe = layers.Flatten()(fe)
    # Output
    out_layer = layers.Dense(1, activation='sigmoid')(fe)
    # Define model
    model = keras.models.Model([in_image, in_label], out_layer, name="derivative_discriminator_mc")
    if print_summary:
        print(model.summary())
    return model


def get_generator_model_mc(latent_dim=100, n_classes=3, print_summary=False):
    # Image input
    in_lat = layers.Input(shape=(latent_dim,))
    in_label = layers.Input(shape=(n_classes,))

    x = layers.Concatenate()([in_lat, in_label])

    n_nodes = 32 * 256  # Ajustado para generar 256 puntos
    merge = layers.Dense(n_nodes)(x)
    merge = layers.Activation('relu')(merge)

    # Add dimension as there's no 1DTranspose
    merge = layers.Reshape((32, 1, 256))(merge)  # Redimensionado para señales de 256
    
    gen = layers.Conv2DTranspose(128, kernel_size=(7, 1), strides=(2, 1), padding='same')(merge)  # Kernel reducido de 18 a 7
    gen = layers.Activation('relu')(gen)

    gen = layers.Conv2DTranspose(64, kernel_size=(7, 1), strides=(2, 1), padding='same')(gen)
    gen = layers.Activation('relu')(gen)

    gen = layers.Conv2DTranspose(1, kernel_size=(7, 1), strides=(2, 1), padding='same')(gen)
    gen = layers.Activation('linear')(gen)
    
    # Output
    out_layer = layers.Reshape((256,))(gen)  # Cambiado de 1024 a 256
    # Define model
    model = keras.models.Model([in_lat, in_label], [out_layer], name="generator_mc")
    if print_summary:
        print(model.summary())
    return model

