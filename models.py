from keras.layers import (
    Dense,
    Activation,
    Flatten,
    Layer,
    RepeatVector,
    AveragePooling2D,
    Conv2D,
    MaxPooling2D,
    LocallyConnected2D,
    concatenate,
    Concatenate,
    Cropping2D,
    BatchNormalization,
    Input,
    ZeroPadding2D,
    Reshape,
    dot,
    Dot,
    add,
    Lambda,
    Dropout,
    Activation)
from keras.models import Model
from keras import backend as K
from keras.utils import conv_utils
from keras.engine import InputSpec, Layer
from tensorflow import image as tfi
import six
from keras.regularizers import l2
from keras import backend as K
from keras.layers import LeakyReLU
import tensorflow as tf
# from keras.engine import Layer
import keras.backend as K
from keras.callbacks import Callback, EarlyStopping
import numpy as np


# CustomStopper only stops after x epochs
class CustomStopper(EarlyStopping):
    def __init__(self, monitor='val_loss',
                 min_delta=0, patience=0, verbose=0, mode='auto',
                 start_epoch=100):  # add argument for starting epoch
        super(CustomStopper, self).__init__()
        self.start_epoch = start_epoch

    def on_epoch_end(self, epoch, logs=None):
        if epoch > self.start_epoch:
            current = logs.get(self.monitor)
            if current is None:
                warnings.warn(
                    'Early stopping conditioned on metric `%s` '
                    'which is not available. Available metrics are: %s' %
                    (self.monitor, ','.join(list(logs.keys()))),
                    RuntimeWarning)
                return
            if self.monitor_op(current - self.min_delta, self.best):
                self.best = current
                self.wait = 0
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    self.stopped_epoch = epoch
                    self.model.stop_training = True


class MultiGPU_Checkpoint_Callback(Callback):

    def __init__(self, filepath, base_model, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        super(MultiGPU_Checkpoint_Callback, self).__init__()
        self.base_model = base_model
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available,'
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('Epoch %05d: %s improved from %0.5f to '
                                  ' %0.5f, saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.base_model.save_weights(filepath,
                                                         overwrite=True)
                        else:
                            self.base_model.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('Epoch %05d: %s did not improve' %
                                  (epoch + 1, self.monitor))
            else:
                if self.verbose > 0:
                    print('Epoch %05d: saving model to %s' % (epoch + 1,
                                                              filepath))
                if self.save_weights_only:
                    self.base_model.save_weights(filepath, overwrite=True)
                else:
                    self.base_model.save(filepath, overwrite=True)


class ResizeImages(Layer):
    """Resize Images to a specified size

    # Arguments
        output_size: Size of output layer width and height
        data_format: A string,
            one of `channels_last` (default) or `channels_first`.
            The ordering of the dimensions in the inputs.
            `channels_last` corresponds to inputs with shape
            `(batch, height, width, channels)` while `channels_first`
            corresponds to inputs with shape
            `(batch, channels, height, width)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".

    # Input shape
        - If `data_format='channels_last'`:
            4D tensor with shape:
            `(batch_size, rows, cols, channels)`
        - If `data_format='channels_first'`:
            4D tensor with shape:
            `(batch_size, channels, rows, cols)`

    # Output shape
        - If `data_format='channels_last'`:
            4D tensor with shape:
            `(batch_size, pooled_rows, pooled_cols, channels)`
        - If `data_format='channels_first'`:
            4D tensor with shape:
            `(batch_size, channels, pooled_rows, pooled_cols)`
    """
    def __init__(self, output_dim=(1, 1), data_format=None, **kwargs):
        super(ResizeImages, self).__init__(**kwargs)
        data_format = conv_utils.normalize_data_format(data_format)
        self.output_dim = conv_utils.normalize_tuple(output_dim, 2,
                                                     'output_dim')
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.input_spec = InputSpec(ndim=4)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            return (input_shape[0], input_shape[1], self.output_dim[0],
                    self.output_dim[1])
        elif self.data_format == 'channels_last':
            return (input_shape[0], self.output_dim[0], self.output_dim[1],
                    input_shape[3])

    def _resize_fun(self, inputs, data_format):
        try:
            assert K.backend() == 'tensorflow'
            assert self.data_format == 'channels_last'
        except AssertionError:
            print("Only tensorflow backend is supported for the resize layer \
                  and accordingly 'channels_last' ordering")
        output = tfi.resize_images(inputs, self.output_dim, align_corners=True)
        return output

    def call(self, inputs):
        output = self._resize_fun(inputs=inputs, data_format=self.data_format)
        return output

    def get_config(self):
        config = {'output_dim': self.output_dim,
                  'data_format': self.data_format}
        base_config = super(ResizeImages, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class cartridge(Layer):
    '''
    Implemeting a Conv2D with strides=1, and 'valid' padding
    '''

    def __init__(self, img_size=29, **kwargs):
        self.filters = 1
        self.img_size = img_size
        self.k_h, self.k_w = (3, 3)
        super(cartridge, self).__init__(**kwargs)

    def build(self, input_shape):
        self.out_h = self.img_size
        self.out_w = self.img_size
        # allocate vars for kernels
        self.kernels = self.add_weight(name='kernel0',
                                       shape=[6, 1],
                                       initializer='he_normal',
                                       trainable=True)
        super(cartridge, self).build(input_shape)

    def call(self, x):

        kernel = K.repeat_elements(self.kernels, rep=3, axis=0)

        t = []
        for i in range(self.out_h):
            for j in range(self.out_w):
                # take a patch
                # want:
                # X X 0
                # X 0 X
                # 0 X X
                p = []
                xlist = [i+0, i+1, i+0, i+2, i+1, i+2]
                ylist = [j+0, j+0, j+1, j+1, j+2, j+2]
                for ci in range(3):
                    for xyi in range(6):
                        p.append(x[:, xlist[xyi], ylist[xyi], ci])
                # flatten the patch
                p = K.reshape(p, [-1, 18])
                # convolution
                conv = K.dot(p, kernel)
                # gather tensors
                t.append(conv)

        stacked = K.stack(t, axis=1)
        output = K.reshape(stacked, [-1, self.out_h, self.out_w,
                           self.filters])
        return output

    def compute_output_shape(self, input_shape):
        return (None, self.out_h, self.out_w, self.filters)


def fly_eye(img_size, num_predictions, resize, pre_out=0, pre_weight=None):
    if resize == 0:  # means dataset is Cifar10 and has not been resized
        imgs_in = Input(shape=(32, 32, 3))
        imgs = Cropping2D(cropping=((1, 2), (2, 1)))(imgs_in)
    else:
        imgs_in = Input(shape=(resize, resize, 3))
        if resize == 33 and img_size == 29:
            # dataset is flies with max input 29x29
            imgs = Cropping2D(cropping=((2, 2), (2, 2)))(imgs_in)
        elif img_size is not resize:
            imgs = ResizeImages((img_size, img_size))(imgs_in)
        else:
            imgs = imgs_in
    imgs = ZeroPadding2D(padding=((1, 1), (1, 1)))(imgs)

    R1_6 = cartridge()(imgs)

    R1_6 = Activation('relu')(R1_6)

    imgs = Cropping2D(cropping=((1, 1), (1, 1)))(imgs)

    R7 = Conv2D(1, 1, activation='relu', kernel_initializer='he_normal')(imgs)
    R8 = Conv2D(1, 1, activation='relu', kernel_initializer='he_normal')(imgs)

    L1 = Concatenate()([R1_6, R8])
    L1 = LocallyConnected2D(1, (1, 1), activation='relu',
                            kernel_initializer='he_normal')(L1)

    L2 = R1_6
    L2 = LocallyConnected2D(1, (1, 1), activation='relu',
                            kernel_initializer='he_normal')(L2)

    L3 = Concatenate()([R1_6, R7, R8])
    L3 = LocallyConnected2D(1, (1, 1), activation='relu',
                            kernel_initializer='he_normal')(L3)

    L4 = R1_6
    L4 = LocallyConnected2D(1, (1, 1), activation='relu',
                            kernel_initializer='he_normal')(L4)

    L5 = Concatenate()([R1_6, R8])
    L5 = LocallyConnected2D(1, (1, 1), activation='relu',
                            kernel_initializer='he_normal')(L5)

    C2conv_5x5 = ZeroPadding2D(padding=(2, 2))(L5)
    C2conv_5x5 = LocallyConnected2D(1, (5, 5), strides=(1, 1))(C2conv_5x5)
    C2 = Concatenate()([L1, C2conv_5x5])
    C2 = LocallyConnected2D(1, (1, 1), activation='relu',
                            kernel_initializer='he_normal')(C2)

    C3conv_3x3 = ZeroPadding2D(padding=(1, 1))(L5)
    C3conv_3x3 = LocallyConnected2D(1, (3, 3), strides=(1, 1))(C3conv_3x3)
    C3 = Concatenate()([L1, L2, L3, C3conv_3x3])
    C3 = LocallyConnected2D(1, (1, 1), activation='relu',
                            kernel_initializer='he_normal')(C3)

    Mi1conv_3x3 = ZeroPadding2D(padding=(1, 1))(L5)
    Mi1conv_3x3 = LocallyConnected2D(1, (3, 3), strides=(1, 1))(Mi1conv_3x3)
    Mi1 = Concatenate()([R8, L1, L3, C2, C3, Mi1conv_3x3])
    Mi1 = LocallyConnected2D(1, (1, 1), activation='relu',
                             kernel_initializer='he_normal')(Mi1)

    Mi4conv_3x3 = ZeroPadding2D(padding=(1, 1))(L5)
    Mi4conv_3x3 = LocallyConnected2D(1, (3, 3), strides=(1, 1))(Mi4conv_3x3)
    Mi4 = Concatenate()([R8, L2, L3, C2, C3, Mi4conv_3x3])
    Mi4 = LocallyConnected2D(1, (1, 1), activation='relu',
                             kernel_initializer='he_normal')(Mi4)

    Mi9conv_3x3 = Concatenate()([L3, L4])
    Mi9conv_3x3 = ZeroPadding2D(padding=(1, 1))(Mi9conv_3x3)
    Mi9conv_3x3 = LocallyConnected2D(1, (3, 3), strides=(1, 1))(Mi9conv_3x3)
    Mi9 = Concatenate()([R7, R8, L2, Mi9conv_3x3])
    Mi9 = LocallyConnected2D(1, (1, 1), activation='relu',
                             kernel_initializer='he_normal')(Mi9)

    Mi15conv_3x3 = Concatenate()([R8, L5])
    Mi15conv_3x3 = ZeroPadding2D(padding=(1, 1))(Mi15conv_3x3)
    Mi15 = LocallyConnected2D(1, (3, 3), strides=(1, 1))(Mi15conv_3x3)
    Mi15 = LocallyConnected2D(1, (1, 1), activation='relu',
                              kernel_initializer='he_normal')(Mi15)

    Tm20conv_3x3 = ZeroPadding2D(padding=(1, 1))(Mi4)
    Tm20conv_3x3 = LocallyConnected2D(1, (3, 3), strides=(1, 1))(Tm20conv_3x3)
    Tm20 = Concatenate()([R7, R8, L2, C3, Mi1, Tm20conv_3x3])
    Tm20 = LocallyConnected2D(1, (1, 1), activation='relu',
                              kernel_initializer='he_normal')(Tm20)

    Tm1 = Concatenate()([L2, L5, C2, C3, Mi1, Mi4, Mi9])
    Tm1 = LocallyConnected2D(1, (1, 1), activation='relu',
                             kernel_initializer='he_normal')(Tm1)

    Tm2conv_3x3 = ZeroPadding2D(padding=(1, 1))(L4)
    Tm2conv_3x3 = LocallyConnected2D(1, (3, 3), strides=(1, 1))(Tm2conv_3x3)
    Tm2 = Concatenate()([L2, C3, Mi4, Mi9, Tm2conv_3x3])
    Tm2 = LocallyConnected2D(1, (1, 1), activation='relu',
                             kernel_initializer='he_normal')(Tm2)

    Tm3conv_3x3 = Concatenate()([L3, C2, Mi4, Mi9])
    Tm3conv_3x3 = ZeroPadding2D(padding=(1, 1))(Tm3conv_3x3)
    Tm3conv_3x3 = LocallyConnected2D(1, (3, 3), strides=(1, 1))(Tm3conv_3x3)
    Tm3conv_5x5 = Concatenate()([L1, L5, Mi1])
    Tm3conv_5x5 = ZeroPadding2D(padding=(2, 2))(Tm3conv_5x5)
    Tm3conv_5x5 = LocallyConnected2D(1, (5, 5), strides=(1, 1))(Tm3conv_5x5)
    Tm3 = Concatenate()([Tm3conv_5x5, Tm3conv_3x3])
    Tm3 = LocallyConnected2D(1, (1, 1), activation='relu',
                             kernel_initializer='he_normal')(Tm3)

    Tm4conv_3x3 = Concatenate()([L4, Mi4, Mi9])
    Tm4conv_3x3 = ZeroPadding2D(padding=(1, 1))(Tm4conv_3x3)
    Tm4conv_3x3 = LocallyConnected2D(1, (3, 3), strides=(1, 1))(Tm4conv_3x3)
    Tm4conv_5x5 = Concatenate()([L2, C3])
    Tm4conv_5x5 = ZeroPadding2D(padding=(2, 2))(Tm4conv_5x5)
    Tm4conv_5x5 = LocallyConnected2D(1, (5, 5), strides=(1, 1))(Tm4conv_5x5)
    Tm4 = Concatenate()([Tm4conv_5x5, Tm4conv_3x3])
    Tm4 = LocallyConnected2D(1, (1, 1), activation='relu',
                             kernel_initializer='he_normal')(Tm4)

    Tm6conv_3x3 = Concatenate()([Mi1, Mi15])
    Tm6conv_3x3 = ZeroPadding2D(padding=(1, 1))(Tm6conv_3x3)
    Tm6conv_3x3 = LocallyConnected2D(1, (3, 3), strides=(1, 1))(Tm6conv_3x3)
    Tm6conv_5x5 = Concatenate()([L5, Mi9])
    Tm6conv_5x5 = ZeroPadding2D(padding=(2, 2))(Tm6conv_5x5)
    Tm6conv_5x5 = LocallyConnected2D(1, (5, 5), strides=(1, 1))(Tm6conv_5x5)
    Tm6 = Concatenate()([Tm6conv_5x5, Tm6conv_3x3])
    Tm6 = LocallyConnected2D(1, (1, 1), activation='relu',
                             kernel_initializer='he_normal')(Tm6)

    Tm9conv_3x3 = Concatenate()([L4, Mi4])
    Tm9conv_3x3 = ZeroPadding2D(padding=(1, 1))(Tm9conv_3x3)
    Tm9conv_3x3 = LocallyConnected2D(1, (3, 3), strides=(1, 1))(Tm9conv_3x3)
    Tm9 = Concatenate()([L3, C2, C3, Tm9conv_3x3])
    Tm9 = LocallyConnected2D(1, (1, 1), activation='relu',
                             kernel_initializer='he_normal')(Tm9)

    TmY5aconv_3x3 = Concatenate()([L5, Mi4, Mi9])
    TmY5aconv_3x3 = ZeroPadding2D(padding=(1, 1))(TmY5aconv_3x3)
    TmY5a = LocallyConnected2D(1, (3, 3), strides=(1, 1))(TmY5aconv_3x3)
    TmY5a = LocallyConnected2D(1, (1, 1), activation='relu',
                               kernel_initializer='he_normal')(TmY5a)

    T2a = Concatenate()([L2, L5, C2, C3, Mi4, Tm1, Tm2])
    T2a = LocallyConnected2D(1, (1, 1), activation='relu',
                             kernel_initializer='he_normal')(T2a)

    T2conv_3x3 = Concatenate()([Mi1, Tm1, Tm3, Tm4, TmY5a])
    T2conv_3x3 = ZeroPadding2D(padding=(1, 1))(T2conv_3x3)
    T2 = LocallyConnected2D(1, (3, 3), strides=(1, 1))(T2conv_3x3)
    T2 = LocallyConnected2D(1, (1, 1), activation='relu',
                            kernel_initializer='he_normal')(T2)

    T3conv_3x3 = Concatenate()([L2, L4, L5, C2, C3, Mi1, Mi9, Tm1, Tm2, Tm3,
                                Tm6, TmY5a])
    T3conv_3x3 = ZeroPadding2D(padding=(1, 1))(T3conv_3x3)
    T3 = LocallyConnected2D(1, (3, 3), strides=(1, 1))(T3conv_3x3)
    T3 = LocallyConnected2D(1, (1, 1), activation='relu',
                            kernel_initializer='he_normal')(T3)

    # LCwider is a pseudo-type like LC4
    # Takes input from Lob1, Lob2, Lob4
    LCwide = Concatenate()([Tm1, Tm2, Tm3, Tm4, Tm6, Tm9, TmY5a, T2, T2a, T3])
    LCwide = ZeroPadding2D(padding=(2, 2))(LCwide)
    LCwide = LocallyConnected2D(1, (5, 5), strides=(1, 1), activation='relu',
                                kernel_initializer='he_normal')(LCwide)

    # LCnarrow is a pseudo-type like LC17
    # Takes input from Lob2, Lob3, Lob4, Lob5
    LCnarrow = Concatenate()([Tm2, Tm3, Tm4, Tm6, Tm9, Tm20, TmY5a, T2, T2a,
                              T3])
    LCnarrow = ZeroPadding2D(padding=(1, 1))(LCnarrow)
    LCnarrow = LocallyConnected2D(1, (3, 3), strides=(1, 1), activation='relu',
                                  kernel_initializer='he_normal')(LCnarrow)

    brain = Concatenate()([LCnarrow, LCwide])
    x = Flatten()(brain)
    x = Dense(256, activation='relu')(x)
    x = Dense(256, activation='relu')(x)

    if pre_weight is not None and pre_out > 0:
        predictions = Dense(pre_out, activation='softmax')(x)
        model = Model(inputs=imgs_in, outputs=predictions)
        model.load_weights(
            pre_weight)

    predictions = Dense(num_predictions, activation='softmax')(x)

    model = Model(inputs=imgs_in, outputs=predictions)

    return model


def Zeiler_Fergus(input_shape, num_predictions, resize, pre_out=0,
                  pre_weight=None):
    if resize == 0:  # means dataset is Cifar10 and has not been resized
        imgs = Input(shape=(32, 32, 3))
        x = Cropping2D(cropping=((1, 2), (2, 1)))(imgs)
        x = ResizeImages((input_shape, input_shape))(x)
    else:
        imgs = Input(shape=(resize, resize, 3))
        # bottleneck size
        if resize == 33 and input_shape == 29:
            x = Cropping2D(cropping=((2, 2), (2, 2)))(imgs)
        elif resize == 33 and input_shape == 224:
            x = Cropping2D(cropping=((2, 2), (2, 2)))(imgs)
            x = ResizeImages((input_shape, input_shape))(x)
        elif resize == 256 and input_shape == 224:
                x = Cropping2D(cropping=((16, 16), (16, 16)))(imgs)
        elif input_shape is not resize:
            x = ResizeImages((input_shape, input_shape))(imgs)
        else:
            x = imgs

    x = Conv2D(96, (7, 7), strides=(2, 2), padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)

    x = Conv2D(256, (5, 5), strides=(2, 2), padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)

    x = Conv2D(384, (3, 3), strides=(1, 1), padding='same')(x)
    x = Activation('relu')(x)
    x = Conv2D(384, (3, 3), strides=(1, 1), padding='same')(x)
    x = Activation('relu')(x)
    x = Conv2D(256, (3, 3), strides=(1, 1), padding='same')(x)
    x = Activation('relu')(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    x = Flatten()(x)
    x = Dense(4096, activation='relu')(x)
    x = Dense(4096, activation='relu')(x)

    if pre_weight is not None and pre_out > 0:
        predictions = Dense(pre_out, activation='softmax')(x)
        model = Model(inputs=imgs, outputs=predictions)
        model.load_weights(
            pre_weight)

    predictions = Dense(num_predictions, activation='softmax')(x)
    model = Model(inputs=imgs, outputs=predictions)

    return model


def _bn_relu(input):
    """Helper to build a BN -> relu block
    """
    norm = BatchNormalization(axis=CHANNEL_AXIS, momentum=0.6)(input)
    return Activation("relu")(norm)


def _conv_bn_relu(**conv_params):
    """Helper to build a conv -> BN -> relu block
    """
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer",
                                                "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer",
                                                l2(1.e-4))

    def f(input):
        conv = Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer)(input)
        return _bn_relu(conv)

    return f


def _bn_relu_conv(**conv_params):
    """Helper to build a BN -> relu -> conv block.
    This is an improved scheme proposed in
    http://arxiv.org/pdf/1603.05027v2.pdf
    """
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer",
                                                "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer",
                                                l2(1.e-4))

    def f(input):
        activation = _bn_relu(input)
        return Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer)(activation)

    return f


def _shortcut(input, residual):
    """Adds a shortcut between input and residual block and merges them with
    "sum"
    """
    # Expand channels of shortcut to match residual.
    # Stride appropriately to match residual (width, height)
    # Should be int if network architecture is correctly configured.
    input_shape = K.int_shape(input)
    residual_shape = K.int_shape(residual)
    stride_width = int(round(input_shape[ROW_AXIS] / residual_shape[ROW_AXIS]))
    stride_height = int(round(input_shape[COL_AXIS]
                              / residual_shape[COL_AXIS]))
    equal_channels = input_shape[CHANNEL_AXIS] == residual_shape[CHANNEL_AXIS]

    shortcut = input
    # 1 X 1 conv if shape is different. Else identity.
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        shortcut = Conv2D(filters=residual_shape[CHANNEL_AXIS],
                          kernel_size=(1, 1),
                          strides=(stride_width, stride_height),
                          padding="valid",
                          kernel_initializer="he_normal",
                          kernel_regularizer=l2(0.0001))(input)

    return add([shortcut, residual])


def _residual_block(block_function, filters, repetitions,
                    is_first_layer=False):
    """Builds a residual block with repeating bottleneck blocks.
    """
    def f(input):
        for i in range(repetitions):
            init_strides = (1, 1)
            if i == 0 and not is_first_layer:
                init_strides = (2, 2)
            input = block_function(
                filters=filters, init_strides=init_strides,
                is_first_block_of_first_layer=(is_first_layer and i == 0)
                )(input)
        return input

    return f


def basic_block(filters, init_strides=(1, 1),
                is_first_block_of_first_layer=False):
    """Basic 3 X 3 convolution blocks for use on resnets with layers <= 34.
    Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
    """
    def f(input):

        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            conv1 = Conv2D(filters=filters, kernel_size=(3, 3),
                           strides=init_strides,
                           padding="same",
                           kernel_initializer="he_normal",
                           kernel_regularizer=l2(1e-4))(input)
        else:
            conv1 = _bn_relu_conv(filters=filters, kernel_size=(3, 3),
                                  strides=init_strides)(input)

        residual = _bn_relu_conv(filters=filters, kernel_size=(3, 3))(conv1)
        return _shortcut(input, residual)

    return f


def bottleneck(filters, init_strides=(1, 1),
               is_first_block_of_first_layer=False):
    """Bottleneck architecture for > 34 layer resnet.
    Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
    Returns:
        A final conv layer of filters * 4
    """
    def f(input):

        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            conv_1_1 = Conv2D(filters=filters, kernel_size=(1, 1),
                              strides=init_strides,
                              padding="same",
                              kernel_initializer="he_normal",
                              kernel_regularizer=l2(1e-4))(input)
        else:
            conv_1_1 = _bn_relu_conv(filters=filters, kernel_size=(1, 1),
                                     strides=init_strides)(input)

        conv_3_3 = _bn_relu_conv(filters=filters, kernel_size=(3, 3))(conv_1_1)
        residual = _bn_relu_conv(filters=filters * 4, kernel_size=(1, 1)
                                 )(conv_3_3)
        return _shortcut(input, residual)

    return f


def _handle_dim_ordering():
    global ROW_AXIS
    global COL_AXIS
    global CHANNEL_AXIS
    if K.image_dim_ordering() == 'tf':
        ROW_AXIS = 1
        COL_AXIS = 2
        CHANNEL_AXIS = 3
    else:
        CHANNEL_AXIS = 1
        ROW_AXIS = 2
        COL_AXIS = 3


def _get_block(identifier):
    if isinstance(identifier, six.string_types):
        res = globals().get(identifier)
        if not res:
            raise ValueError('Invalid {}'.format(identifier))
        return res
    return identifier


class ResnetBuilder(object):
    @staticmethod
    def build(input_shape, num_outputs, block_fn, repetitions, resize):
        """Builds a custom ResNet like architecture.
        Args:
            input_shape:
                The input shape in the form (nb_channels, nb_rows, nb_cols)
            num_outputs: The number of outputs at final softmax layer
            block_fn:
                The block function to use.
                This is either `basic_block` or `bottleneck`.
                The original paper used basic_block for layers < 50
            repetitions: Number of repetitions of various block units.
                At each block unit, the number of filters are doubled
                and the input size is halved
        Returns:
            The keras `Model`.
        """
        _handle_dim_ordering()
        if len(input_shape) != 3:
            raise Exception("Input shape should be a tuple (nb_channels,\
                             nb_rows, nb_cols)")

        # Load function from str if needed.
        block_fn = _get_block(block_fn)

        # bottleneck size
        if resize == 0:  # means dataset is Cifar10 and has not been resized
            input = Input(shape=(32, 32, 3))
            x = Cropping2D(cropping=((1, 2), (2, 1)))(input)
            x = ResizeImages((input_shape[0], input_shape[1]))(x)
        else:
            input = Input(shape=(resize, resize, 3))
            # bottleneck size
            if resize == 33 and input_shape[0] == 29:
                x = Cropping2D(cropping=((2, 2), (2, 2)))(input)
            elif resize == 33 and input_shape[0] == 224:
                x = Cropping2D(cropping=((2, 2), (2, 2)))(input)
                x = ResizeImages((input_shape[0], input_shape[1]))(x)
            elif resize == 256 and input_shape[0] == 224:
                    x = Cropping2D(cropping=((16, 16), (16, 16)))(input)
            elif input_shape[0] is not resize:
                x = ResizeImages((input_shape[0], input_shape[1]))(input)
            else:
                x = input

        conv1 = _conv_bn_relu(filters=64, kernel_size=(7, 7), strides=(2, 2)
                              )(x)
        pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same"
                             )(conv1)

        block = pool1
        filters = 64
        for i, r in enumerate(repetitions):
            block = _residual_block(block_fn, filters=filters, repetitions=r,
                                    is_first_layer=(i == 0))(block)
            filters *= 2

        # Last activation
        block = _bn_relu(block)

        # Classifier block
        block_shape = K.int_shape(block)
        pool2 = AveragePooling2D(pool_size=(block_shape[ROW_AXIS],
                                 block_shape[COL_AXIS]), strides=(1, 1))(block)
        flatten1 = Flatten()(pool2)
        dense = Dense(units=num_outputs, kernel_initializer="he_normal",
                      activation="softmax")(flatten1)

        model = Model(inputs=input, outputs=dense)
        return model

    @staticmethod
    def build_resnet_18(input_shape, num_outputs, resize):
        return ResnetBuilder.build((input_shape, input_shape, 3), num_outputs,
                                   basic_block, [2, 2, 2, 2], resize)

    @staticmethod
    def build_resnet_34(input_shape, num_outputs, resize):
        return ResnetBuilder.build((input_shape, input_shape, 3), num_outputs,
                                   basic_block, [3, 4, 6, 3], resize)

    @staticmethod
    def build_resnet_50(input_shape, num_outputs, resize):
        return ResnetBuilder.build((input_shape, input_shape, 3), num_outputs,
                                   bottleneck, [3, 4, 6, 3], resize)

    @staticmethod
    def build_resnet_101(input_shape, num_outputs, resize):
        return ResnetBuilder.build((input_shape, input_shape, 3), num_outputs,
                                   bottleneck, [3, 4, 23, 3], resize)

    @staticmethod
    def build_resnet_152(input_shape, num_outputs, resize):
        return ResnetBuilder.build((input_shape, input_shape, 3), num_outputs,
                                   bottleneck, [3, 8, 36, 3], resize)


def reverse_gradient(X, hp_lambda):
    '''Flips the sign of the incoming gradient during training.'''
    try:
        reverse_gradient.num_calls += 1
    except AttributeError:
        reverse_gradient.num_calls = 1

    grad_name = "GradientReversal%d" % reverse_gradient.num_calls

    @tf.RegisterGradient(grad_name)
    def _flip_gradients(op, grad):
        return [tf.negative(grad) * hp_lambda]

    g = K.get_session().graph
    with g.gradient_override_map({'Identity': grad_name}):
        y = tf.identity(X)

    return y


class GradientReversal(Layer):
    '''Flip the sign of gradient during training.'''
    def __init__(self, hp_lambda, **kwargs):
        self.supports_masking = False
        self.hp_lambda = hp_lambda
        super(GradientReversal, self).__init__(**kwargs)

    def build(self, input_shape):
        # Be sure to call this somewhere!
        super(GradientReversal, self).build(input_shape)

    def call(self, x, mask=None):
        return reverse_gradient(x, self.hp_lambda)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {'hp_lambda': self.hp_lambda}
        base_config = super(GradientReversal, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class DAResnetBuilder(object):
    @staticmethod
    def build(input_shape, num_outputs, block_fn, repetitions, resize):
        """Builds a custom ResNet like architecture.
        Args:
            input_shape:
            The input shape in the form (nb_channels, nb_rows, nb_cols)
            num_outputs: The number of outputs at final softmax layer
            block_fn:
            The block function to use. This is either `basic_block` or
            `bottleneck`. The original paper used basic_block for layers < 50
            repetitions: Number of repetitions of various block units.
                At each block unit, the number of filters are doubled
                and the input size is halved
        Returns:
            The keras `Model`.
        """
        _handle_dim_ordering()
        if len(input_shape) != 3:
            raise Exception("Input shape should be a tuple (nb_channels, \
                            nb_rows, nb_cols)")

        # Load function from str if needed.
        block_fn = _get_block(block_fn)

        # bottleneck size
        if resize == 0:  # means dataset is Cifar10 and has not been resized
            input = Input(shape=(32, 32, 3))
            x = Cropping2D(cropping=((1, 2), (2, 1)))(input)
        else:
            input = Input(shape=(resize, resize, 3))
            # bottleneck size
            if resize == 33 and input_shape[0] == 29:
                x = Cropping2D(cropping=((2, 2), (2, 2)))(input)
            elif resize == 33 and input_shape[0] == 224:
                x = Cropping2D(cropping=((2, 2), (2, 2)))(input)
                x = ResizeImages((input_shape[0], input_shape[1]))(x)
            elif resize == 256 and input_shape[0] == 224:
                    x = Cropping2D(cropping=((16, 16), (16, 16)))(input)
            elif input_shape[0] is not resize:
                x = ResizeImages((input_shape[0], input_shape[1]))(input)
            else:
                x = input

        conv1 = _conv_bn_relu(filters=64, kernel_size=(7, 7), strides=(2, 2)
                              )(x)
        pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same"
                             )(conv1)

        block = pool1
        filters = 64
        for i, r in enumerate(repetitions[0]):
            block = _residual_block(block_fn, filters=filters, repetitions=r,
                                    is_first_layer=(i == 0))(block)
            filters *= 2

        # Split off the domain predictor:
        DA_branch = GradientReversal(1)(block)
        DA_branch = Flatten()(DA_branch)
        DA_branch = Dense(256)(DA_branch)
        DA_branch = LeakyReLU()(DA_branch)
        DA_branch = Dropout(0.5)(DA_branch)
        DA_branch = Dense(256)(DA_branch)
        DA_branch = LeakyReLU()(DA_branch)
        DA_branch = Dropout(0.5)(DA_branch)
        Domain = Dense(1, kernel_initializer="he_normal", activation="softmax",
                       name='Domain')(DA_branch)

        for i, r in enumerate(repetitions[1]):
            block = _residual_block(block_fn, filters=filters, repetitions=r,
                                    is_first_layer=(i == 0))(block)
            filters *= 2

        # Last activation
        block = _bn_relu(block)

        # Classifier block
        block_shape = K.int_shape(block)
        pool2 = AveragePooling2D(pool_size=(block_shape[ROW_AXIS],
                                 block_shape[COL_AXIS]), strides=(1, 1))(block)
        flatten1 = Flatten()(pool2)
        ID = Dense(units=num_outputs, kernel_initializer="he_normal",
                   activation="softmax", name='ID')(flatten1)

        model = Model(inputs=input, outputs=[ID, Domain])
        return model

    @staticmethod
    def build_resnet_18(input_shape, num_outputs, resize):
        return DAResnetBuilder.build((input_shape, input_shape, 3),
                                     num_outputs, basic_block,
                                     ([2, 2, 2], [2]), resize)

    @staticmethod
    def build_resnet_34(input_shape, num_outputs, resize):
        return DAResnetBuilder.build((input_shape, input_shape, 3),
                                     num_outputs, basic_block,
                                     ([3, 4, 6], [3]), resize)

    @staticmethod
    def build_resnet_50(input_shape, num_outputs, resize):
        return DAResnetBuilder.build((input_shape, input_shape, 3),
                                     num_outputs, bottleneck,
                                     ([3, 4, 6], [3]), resize)

    @staticmethod
    def build_resnet_101(input_shape, num_outputs, resize):
        return DAResnetBuilder.build((input_shape, input_shape, 3),
                                     num_outputs, bottleneck,
                                     ([3, 4, 23], [3]), resize)

    @staticmethod
    def build_resnet_152(input_shape, num_outputs, resize):
        return DAResnetBuilder.build((input_shape, input_shape, 3),
                                     num_outputs, bottleneck,
                                     ([3, 8, 36], [3]), resize)
