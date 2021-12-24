def set_vnet_model(img_height:int         = 128,
                   img_width:int          = 128,
                   img_depth:int          = 64,
                   img_channels:int       = 1,
                   kernel_size:tuple      = (5, 5, 5),
                   nfilters_input:int     = 16,
                   activation:str         = 'relu',
                   kernel_initializer:str = 'he_normal',
                   activation_output:str  = 'softmax',
                   optimizer:str          = 'adam',
                   loss:str               = 'binary_crossentropy',
                   metrics:list           = ['accuracy'],
                   show_summary:bool      = False,
                   show_graph:bool        = False):
  # Implementation according to the original article (https://arxiv.org/abs/1606.04797)

  #Input
  inputs = Input((img_height, img_width, img_depth, img_channels))
  s = inputs

  # Downsample - Level 0
  c1 = Conv3D(1*nfilters_input, kernel_size, activation=activation, kernel_initializer=kernel_initializer, padding='same')(s)
  c1 = BatchNormalization()(c1)
  c1 = c1 + s
  c2 = Conv3D(2*nfilters_input, (2, 2, 2), strides=(2, 2, 2))(c1)

  # Downsample - Level 1
  c3 = Conv3D(2*nfilters_input, kernel_size, activation=activation, kernel_initializer=kernel_initializer, padding='same')(c2)
  c3 = BatchNormalization()(c3)
  c3 = Conv3D(2*nfilters_input, kernel_size, activation=activation, kernel_initializer=kernel_initializer, padding='same')(c3)
  c3 = BatchNormalization()(c3)
  c3 = c3 + c2
  c4 = Conv3D(4*nfilters_input, (2, 2, 2), strides=(2, 2, 2))(c3)

  # Downsample - Level 2
  c5 = Conv3D(4*nfilters_input, kernel_size, activation=activation, kernel_initializer=kernel_initializer, padding='same')(c4)
  c5 = BatchNormalization()(c5)
  c5 = Conv3D(4*nfilters_input, kernel_size, activation=activation, kernel_initializer=kernel_initializer, padding='same')(c5)
  c5 = BatchNormalization()(c5)
  c5 = Conv3D(4*nfilters_input, kernel_size, activation=activation, kernel_initializer=kernel_initializer, padding='same')(c5)
  c5 = BatchNormalization()(c5)
  c5 = c5 + c4
  c6 = Conv3D(8*nfilters_input, (2, 2, 2), strides=(2, 2, 2))(c5)

  # Downsample - Level 3
  c7 = Conv3D(8*nfilters_input, kernel_size, activation=activation, kernel_initializer=kernel_initializer, padding='same')(c6)
  c7 = BatchNormalization()(c7)
  c7 = Conv3D(8*nfilters_input, kernel_size, activation=activation, kernel_initializer=kernel_initializer, padding='same')(c7)
  c7 = BatchNormalization()(c7)
  c7 = Conv3D(8*nfilters_input, kernel_size, activation=activation, kernel_initializer=kernel_initializer, padding='same')(c7)
  c7 = BatchNormalization()(c7)
  c7 = c7 + c6
  c8 = Conv3D(16*nfilters_input, (2, 2, 2), strides=(2, 2, 2))(c7)
  
  # Bottleneck - Level 4
  c9 = Conv3D(16*nfilters_input, kernel_size, activation=activation, kernel_initializer=kernel_initializer, padding='same')(c8)
  c9 = BatchNormalization()(c9)
  c9 = Conv3D(16*nfilters_input, kernel_size, activation=activation, kernel_initializer=kernel_initializer, padding='same')(c9)
  c9 = BatchNormalization()(c9)
  c9 = Conv3D(16*nfilters_input, kernel_size, activation=activation, kernel_initializer=kernel_initializer, padding='same')(c9)
  c9 = BatchNormalization()(c9)
  c9 = c9 + c8
  c10 = Conv3DTranspose(16*nfilters_input, (2, 2, 2), strides=(2, 2, 2), padding='same')(c9)
  
  # Upsample - Level 3
  c11 = concatenate([c10, c7])
  c11 = Conv3D(16*nfilters_input, kernel_size, activation=activation, kernel_initializer=kernel_initializer, padding='same')(c11)
  c11 = BatchNormalization()(c11)
  c11 = Conv3D(16*nfilters_input, kernel_size, activation=activation, kernel_initializer=kernel_initializer, padding='same')(c11)
  c11 = BatchNormalization()(c11)
  c11 = Conv3D(16*nfilters_input, kernel_size, activation=activation, kernel_initializer=kernel_initializer, padding='same')(c11)
  c11 = BatchNormalization()(c11)
  c11 = c11 + c10
  c12 = Conv3DTranspose(8*nfilters_input, (2, 2, 2), strides=(2, 2, 2), padding='same')(c11)

  # Upsample - Level 2
  c13 = concatenate([c12, c5])
  c13 = Conv3D(8*nfilters_input, kernel_size, activation=activation, kernel_initializer=kernel_initializer, padding='same')(c13)
  c13 = BatchNormalization()(c13)
  c13 = Conv3D(8*nfilters_input, kernel_size, activation=activation, kernel_initializer=kernel_initializer, padding='same')(c13)
  c13 = BatchNormalization()(c13)
  c13 = Conv3D(8*nfilters_input, kernel_size, activation=activation, kernel_initializer=kernel_initializer, padding='same')(c13)
  c13 = BatchNormalization()(c13)
  c13 = c13 + c12
  c14 = Conv3DTranspose(4*nfilters_input, (2, 2, 2), strides=(2, 2, 2), padding='same')(c13)

  # Upsample - Level 1
  c15 = concatenate([c14, c3])
  c15 = Conv3D(4*nfilters_input, kernel_size, activation=activation, kernel_initializer=kernel_initializer, padding='same')(c15)
  c15 = BatchNormalization()(c15)
  c15 = Conv3D(4*nfilters_input, kernel_size, activation=activation, kernel_initializer=kernel_initializer, padding='same')(c15)
  c15 = BatchNormalization()(c15)
  c15 = c15 + c14
  c16 = Conv3DTranspose(2*nfilters_input, (2, 2, 2), strides=(2, 2, 2), padding='same')(c15)

  # Upsample - Level 0
  c17 = concatenate([c16, c1])
  c17 = Conv3D(2*nfilters_input, kernel_size, activation=activation, kernel_initializer=kernel_initializer, padding='same')(c17)
  c17 = BatchNormalization()(c17)
  c17 = c17 + c16
  
  # Output  
  outputs = Conv3D(2, (1, 1, 1), activation=activation_output)(c17)
    
  model = Model(inputs=[inputs], outputs=[outputs])
  model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

  # Show useful information
  if show_summary:
    model.summary()
  
  if show_graph:
    tf.keras.utils.plot_model(model, show_shapes=True)
  
  return model
