def node_operations(input_layer        = None,
                    n_filters          = 16,
                    n_convs            = 2,
                    kernel_size        = (3, 3, 3),
                    kernel_initializer = 'he_normal',
                    activation         = 'relu'):
    l = Conv3D(n_filters, kernel_size, activation=activation, kernel_initializer=kernel_initializer, padding='same')(input_layer)
    l = BatchNormalization()(l)
    if n_convs > 1:
        for n in range(1, n_convs, 1):
            l = Conv3D(n_filters, kernel_size, activation=activation, kernel_initializer=kernel_initializer, padding='same')(l)
            l = BatchNormalization()(l)
    return l


def unet3dpp(img_height: int         = 64,
             img_width: int          = 64,
             img_depth: int          = 64,
             img_channels: int       = 1,
             kernel_size: tuple      = (3, 3, 3),
             nfilters_input: int     = 32,
             activation: str         = 'relu',
             kernel_initializer: str = 'he_normal',
             activation_output: str  = 'softmax',
             optimizer: str          = 'adam',
             loss: str               = 'binary_crossentropy',
             metrics: list           = ['accuracy'],
             show_summary: bool      = False,
             show_graph: bool        = False):
    n_convs = 2
    # Input
    inputs = Input((img_height, img_width, img_depth, img_channels))
    s = inputs

    # L1
    x00_o = node_operations(input_layer=s,
                            n_filters=1 * nfilters_input,
                            n_convs=n_convs,
                            kernel_size=kernel_size,
                            kernel_initializer=kernel_initializer,
                            activation=activation)

    x10_i = MaxPooling3D()(x00_o)
    x10_o = node_operations(input_layer=x10_i,
                            n_filters=2 * nfilters_input,
                            n_convs=n_convs,
                            kernel_size=kernel_size,
                            kernel_initializer=kernel_initializer,
                            activation=activation)

    x01_i = concatenate([x00_o, Conv3DTranspose(1 * nfilters_input, (2, 2, 2),
                                                strides=(2, 2, 2),
                                                padding='same')(x10_o)])
    x01_o = node_operations(input_layer=x01_i,
                            n_filters=1 * nfilters_input,
                            n_convs=n_convs,
                            kernel_size=kernel_size,
                            kernel_initializer=kernel_initializer,
                            activation=activation)

    # L2
    x20_i = MaxPooling3D()(x10_o)
    x20_o = node_operations(input_layer=x20_i,
                            n_filters=4 * nfilters_input,
                            n_convs=n_convs,
                            kernel_size=kernel_size,
                            kernel_initializer=kernel_initializer,
                            activation=activation)

    x11_i = concatenate([x10_o, Conv3DTranspose(2 * nfilters_input, (2, 2, 2),
                                                strides=(2, 2, 2),
                                                padding='same')(x20_o)])
    x11_o = node_operations(input_layer=x11_i,
                            n_filters=2 * nfilters_input,
                            n_convs=n_convs,
                            kernel_size=kernel_size,
                            kernel_initializer=kernel_initializer,
                            activation=activation)

    x02_i = concatenate([x00_o, x01_o,
                         Conv3DTranspose(1 * nfilters_input, (2, 2, 2),
                                         strides=(2, 2, 2),
                                         padding='same')(x11_o)])
    x02_o = node_operations(input_layer=x02_i,
                            n_filters=1 * nfilters_input,
                            n_convs=n_convs,
                            kernel_size=kernel_size,
                            kernel_initializer=kernel_initializer,
                            activation=activation)

    # L3
    x30_i = MaxPooling3D()(x20_o)
    x30_o = node_operations(input_layer=x30_i,
                            n_filters=8 * nfilters_input,
                            n_convs=n_convs,
                            kernel_size=kernel_size,
                            kernel_initializer=kernel_initializer,
                            activation=activation)

    x21_i = concatenate([x20_o, Conv3DTranspose(4 * nfilters_input, (2, 2, 2),
                                                strides=(2, 2, 2),
                                                padding='same')(x30_o)])
    x21_o = node_operations(input_layer=x21_i,
                            n_filters=2 * nfilters_input,
                            n_convs=n_convs,
                            kernel_size=kernel_size,
                            kernel_initializer=kernel_initializer,
                            activation=activation)

    x12_i = concatenate([x10_o, x11_o,
                         Conv3DTranspose(2 * nfilters_input, (2, 2, 2),
                                         strides=(2, 2, 2),
                                         padding='same')(x21_o)])
    x12_o = node_operations(input_layer=x12_i,
                            n_filters=2 * nfilters_input,
                            n_convs=n_convs,
                            kernel_size=kernel_size,
                            kernel_initializer=kernel_initializer,
                            activation=activation)

    x03_i = concatenate([x00_o, x01_o, x02_o,
                         Conv3DTranspose(1 * nfilters_input, (2, 2, 2),
                                         strides=(2, 2, 2),
                                         padding='same')(x12_o)])
    x03_o = node_operations(input_layer=x03_i,
                            n_filters=1 * nfilters_input,
                            n_convs=n_convs,
                            kernel_size=kernel_size,
                            kernel_initializer=kernel_initializer,
                            activation=activation)

    # L4
    x40_i = MaxPooling3D()(x30_o)
    x40_o = node_operations(input_layer=x40_i,
                            n_filters=16 * nfilters_input,
                            n_convs=n_convs,
                            kernel_size=kernel_size,
                            kernel_initializer=kernel_initializer,
                            activation=activation)

    x31_i = concatenate([x30_o, Conv3DTranspose(8 * nfilters_input, (2, 2, 2),
                                                strides=(2, 2, 2),
                                                padding='same')(x40_o)])
    x31_o = node_operations(input_layer=x31_i,
                            n_filters=8 * nfilters_input,
                            n_convs=n_convs,
                            kernel_size=kernel_size,
                            kernel_initializer=kernel_initializer,
                            activation=activation)

    x22_i = concatenate([x20_o, x21_o,
                         Conv3DTranspose(4 * nfilters_input, (2, 2, 2),
                                         strides=(2, 2, 2),
                                         padding='same')(x31_o)])
    x22_o = node_operations(input_layer=x22_i,
                            n_filters=4 * nfilters_input,
                            n_convs=n_convs,
                            kernel_size=kernel_size,
                            kernel_initializer=kernel_initializer,
                            activation=activation)

    x13_i = concatenate([x10_o, x11_o, x12_o,
                         Conv3DTranspose(2 * nfilters_input, (2, 2, 2),
                                         strides=(2, 2, 2),
                                         padding='same')(x22_o)])
    x13_o = node_operations(input_layer=x13_i,
                            n_filters=2 * nfilters_input,
                            n_convs=n_convs,
                            kernel_size=kernel_size,
                            kernel_initializer=kernel_initializer,
                            activation=activation)

    x04_i = concatenate([x00_o, x01_o, x02_o, x03_o,
                         Conv3DTranspose(1 * nfilters_input, (2, 2, 2),
                                         strides=(2, 2, 2),
                                         padding='same')(x13_o)])
    x04_o = node_operations(input_layer=x04_i,
                            n_filters=1 * nfilters_input,
                            n_convs=n_convs,
                            kernel_size=kernel_size,
                            kernel_initializer=kernel_initializer,
                            activation=activation)

    # Output
    outputs = Conv3D(2, (1, 1, 1), activation=activation_output)(x04_o)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # Show useful information
    if show_summary:
        model.summary()

    if show_graph:
        tf.keras.utils.plot_model(model, show_shapes=True)

    return model
