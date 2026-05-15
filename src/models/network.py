"""Neural network definitions for generator, discriminator, and GAN."""

from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, BatchNormalization, LeakyReLU, Concatenate, Dropout, Flatten, Dense, Activation, Multiply, Add
from tensorflow.keras.models import Model
from tensorflow.keras import Sequential
import tensorflow as tf


def _apply_activation(x, func, func_kwargs=None):
    """Apply activation from layer class, layer instance, function, or string."""
    if func_kwargs is None:
        func_kwargs = {}

    if isinstance(func, str):
        return Activation(func, **func_kwargs)(x)

    if isinstance(func, tf.keras.layers.Layer):
        return func(x)

    if isinstance(func, type) and issubclass(func, tf.keras.layers.Layer):
        return func(**func_kwargs)(x)

    if callable(func):
        return Activation(func, **func_kwargs)(x)

    raise TypeError(f'Unsupported activation specification: {func!r}')

def conv_block(input_tensor, output_channels, kernel_size, depth=2,
               func=LeakyReLU, func_kwargs=None, batch_normalization=True,
               use_bias=False, dropout_rate=0.3, dropout_from_layer=1):
    """Apply a sequence of Conv2D layers with optional BatchNorm, activation, and dropout.

    Dropout is only applied from *dropout_from_layer* onwards (0-indexed), so
    early layers are left unregularised.

    Parameters
    ----------
    input_tensor : tf.Tensor
        Input tensor.
    output_channels : int
        Number of filters for each Conv2D layer.
    kernel_size : int or tuple of int
        Convolutional kernel size.
    depth : int, optional
        Number of Conv2D layers in the block. Default is 2.
    func : callable, optional
        Activation function class (e.g. ``LeakyReLU``, ``ReLU``, ``ELU``).
        Default is ``LeakyReLU``.
    func_kwargs : dict or None, optional
        Keyword arguments passed to *func* (e.g. ``{'alpha': 0.1}`` for
        ``LeakyReLU``). Default is ``None`` (empty dict).
    batch_normalization : bool, optional
        Whether to apply BatchNormalisation after each Conv2D. Default is
        ``True``.
    use_bias : bool, optional
        Whether Conv2D layers include a bias term (redundant with BN but
        configurable). Default is ``False``.
    dropout_rate : float, optional
        Dropout probability applied to eligible layers. Default is 0.3.
    dropout_from_layer : int, optional
        0-indexed layer index from which dropout is applied. Default is 1
        (skips the first conv layer).

    Returns
    -------
    tf.Tensor
        Output tensor after all conv layers.
    """
    if func_kwargs is None:
        func_kwargs = {}
    x = input_tensor
    for i in range(depth):
        x = Conv2D(filters=output_channels, kernel_size=kernel_size,
                   activation=None, padding='same', use_bias=use_bias)(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        x = _apply_activation(x, func, func_kwargs)
        if i >= dropout_from_layer:
            x = Dropout(dropout_rate)(x)
    return x


def attention_gate(skip, gating, n_intermediate_filters):
    """Apply an attention gate (Oktay et al. 2018, Attention U-Net).

    Computes a spatial attention map from the skip connection and the gating
    signal (upsampled decoder feature), then weights the skip connection
    element-wise by that map.  This suppresses irrelevant spatial regions
    before the skip connection is concatenated with the decoder output.

    Parameters
    ----------
    skip : tf.Tensor
        Skip-connection tensor from the encoder of shape
        ``(N, H, W, C_skip)``.
    gating : tf.Tensor
        Gating signal from the decoder — same spatial size as *skip* — of
        shape ``(N, H, W, C_gate)``.
    n_intermediate_filters : int
        Number of filters in the intermediate attention projection (typically
        ``C_skip // 2``).

    Returns
    -------
    tf.Tensor
        Skip connection weighted by the learned attention map; same shape as
        *skip*.
    """
    # Project both inputs to a common intermediate space
    theta = Conv2D(n_intermediate_filters, kernel_size=1, padding='same', use_bias=True)(skip)
    phi   = Conv2D(n_intermediate_filters, kernel_size=1, padding='same', use_bias=True)(gating)

    # Add and apply ReLU
    f = Activation('relu')(Add()([theta, phi]))

    # Project to a single-channel attention map and apply sigmoid
    psi = Conv2D(1, kernel_size=1, padding='same', use_bias=True)(f)
    alpha = Activation('sigmoid')(psi)

    # Gate the skip connection elementwise
    attended = Multiply()([skip, alpha])
    return attended


# Define the discriminator (simple CNN)
def get_discriminator(input_shape, depth=2, n_initial_filters=64, filter_size=2,
                      kernel_size=(3, 3), dropout_rate=0.3,
                      func=LeakyReLU, func_kwargs=None, output_activation='sigmoid',
                      block_depth=1, batch_normalization=True, use_bias=False, dropout_from_layer=0):
    """Build a CNN discriminator for the GAN.

    Parameters
    ----------
    input_shape : tuple of int
        Spatial shape of the input tensor ``(H, W, C)``.
    depth : int, optional
        Number of convolutional stages. Default is 2.
    n_initial_filters : int, optional
        Number of filters in the first stage; doubled at each subsequent
        stage. Default is 64.
    filter_size : int, optional
        Multiplicative growth factor for the number of filters per stage.
        Default is 2.
    kernel_size : tuple of int, optional
        Convolutional kernel size used inside each ``conv_block``. Default
        is ``(3, 3)``.
    dropout_rate : float, optional
        Dropout probability. Default is 0.3.
    func : callable, optional
        Activation function class. Default is ``LeakyReLU``.
    func_kwargs : dict or None, optional
        Keyword arguments for *func*. Default is ``None``.
    output_activation : str, optional
        Activation for the final Dense layer. Default is ``'sigmoid'``.
    block_depth : int, optional
        Number of conv layers per stage. Default is 1.
    batch_normalization : bool, optional
        Whether to use BatchNormalisation in each block. Default is ``True``.
    use_bias : bool, optional
        Whether Conv2D layers include a bias term. Default is ``False``.
    dropout_from_layer : int, optional
        0-indexed layer within each block from which dropout is applied.
        Default is 0.

    Returns
    -------
    tf.keras.Model
        Discriminator model mapping an image to a scalar probability.
    """
    if func_kwargs is None:
        func_kwargs = {}
    layers = [Input(shape=input_shape)]
    n_filters = n_initial_filters
    x = layers[0]
    for _ in range(depth):
        x = conv_block(x, n_filters, kernel_size, depth=block_depth,
                       func=func, func_kwargs=func_kwargs, batch_normalization=batch_normalization,
                       use_bias=use_bias, dropout_rate=dropout_rate, dropout_from_layer=dropout_from_layer)
        n_filters *= filter_size
    x = Flatten()(x)
    x = Dense(1, activation=output_activation)(x)
    return Model(inputs=layers[0], outputs=x)

# Define the GAN model
@tf.keras.utils.register_keras_serializable(package='astroUnets')
class GAN(Model):
    """Generative Adversarial Network combining a U-Net generator and a CNN discriminator.

    Attributes
    ----------
    generator : tf.keras.Model
        U-Net generator model.
    discriminator : tf.keras.Model
        CNN discriminator model.
    g_optimizer : tf.keras.optimizers.Optimizer
        Optimiser for the generator.
    d_optimizer : tf.keras.optimizers.Optimizer
        Optimiser for the discriminator.
    adversarial_loss_fn : callable
        Adversarial loss function used by the discriminator and generator's
        adversarial term.
    reconstruction_loss_fn : callable or None
        Supervised reconstruction loss applied to generator outputs against the
        clean target image.
    adversarial_loss_weight : float
        Weight applied to the generator adversarial term.
    reconstruction_loss_weight : float
        Weight applied to the generator reconstruction term.
    label_smoothing : float
        Label smoothing factor applied to real labels in discriminator
        training. Default is 0.1.
    """
    def __init__(
        self,
        generator,
        discriminator,
        g_optimizer,
        d_optimizer,
        adversarial_loss_fn,
        reconstruction_loss_fn=None,
        adversarial_loss_weight=1.0,
        reconstruction_loss_weight=100.0,
        label_smoothing=0.1,
        **kwargs,
    ):
        """Initialise the GAN with its components and hyper-parameters.

        Parameters
        ----------
        generator : tf.keras.Model
            U-Net generator model.
        discriminator : tf.keras.Model
            CNN discriminator model.
        g_optimizer : tf.keras.optimizers.Optimizer
            Optimiser for the generator.
        d_optimizer : tf.keras.optimizers.Optimizer
            Optimiser for the discriminator.
        adversarial_loss_fn : callable
            Adversarial loss function (e.g. ``BinaryCrossentropy``).
        reconstruction_loss_fn : callable or None, optional
            Reconstruction loss for the generator (for example ``MAE`` or
            ``log_cosh``). Default is ``None``.
        adversarial_loss_weight : float, optional
            Weight applied to the generator adversarial term. Default is 1.0.
        reconstruction_loss_weight : float, optional
            Weight applied to the generator reconstruction term. Default is
            100.0.
        label_smoothing : float, optional
            Real-label smoothing coefficient. Default is 0.1.
        """
        super().__init__(**kwargs)
        self.generator = generator
        self.discriminator = discriminator
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.adversarial_loss_fn = adversarial_loss_fn
        self.reconstruction_loss_fn = reconstruction_loss_fn
        self.adversarial_loss_weight = adversarial_loss_weight
        self.reconstruction_loss_weight = reconstruction_loss_weight
        self.label_smoothing = label_smoothing

    def get_config(self):
        """Return a serializable config for Keras checkpointing."""
        config = super().get_config()
        config.update({
            'generator': tf.keras.utils.serialize_keras_object(self.generator),
            'discriminator': tf.keras.utils.serialize_keras_object(self.discriminator),
            'g_optimizer': tf.keras.utils.serialize_keras_object(self.g_optimizer),
            'd_optimizer': tf.keras.utils.serialize_keras_object(self.d_optimizer),
            'adversarial_loss_fn': tf.keras.utils.serialize_keras_object(self.adversarial_loss_fn),
            'reconstruction_loss_fn': tf.keras.utils.serialize_keras_object(self.reconstruction_loss_fn),
            'adversarial_loss_weight': self.adversarial_loss_weight,
            'reconstruction_loss_weight': self.reconstruction_loss_weight,
            'label_smoothing': self.label_smoothing,
        })
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        """Rebuild a GAN from a serialized config."""
        config = dict(config)
        for key in (
            'generator',
            'discriminator',
            'g_optimizer',
            'd_optimizer',
            'adversarial_loss_fn',
            'reconstruction_loss_fn',
        ):
            if config.get(key) is not None:
                config[key] = tf.keras.utils.deserialize_keras_object(
                    config[key],
                    custom_objects=custom_objects,
                )
        return cls(**config)

    def call(self, inputs, training=None, mask=None):
        """Forward pass: delegate to the generator.

        Parameters
        ----------
        inputs : tf.Tensor
            Noisy input image batch.
        training : bool or None, optional
            Passed to the generator's ``call`` method.
        mask : tf.Tensor or None, optional
            Unused Keras mask placeholder kept for API compatibility.

        Returns
        -------
        tf.Tensor
            Reconstructed image batch produced by the generator.
        """
        return self.generator(inputs, training=training)

    def train_step(self, data):
        """Perform one GAN training step (discriminator then generator update).

        Uses soft real-labels (``1 - label_smoothing``) and fake-labels of 0
        for the discriminator.  The generator is trained to fool the
        discriminator by targeting a label of 1.

        Parameters
        ----------
        data : tuple
            ``(x_train, y_train)`` where *x_train* is the noisy image batch
            and *y_train* is the clean reference image batch.

        Returns
        -------
        dict
            ``{'d_loss': discriminator_loss, 'g_loss': generator_loss,``
            ``'g_adv_loss': adversarial_term, 'g_rec_loss': reconstruction_term}``.
        """
        x_train, y_train = data

        batch_size = tf.shape(x_train)[0]

        # Soft labels for label smoothing
        real_labels = tf.ones((batch_size, 1)) * (1.0 - self.label_smoothing)
        fake_labels = tf.zeros((batch_size, 1))

        # Train discriminator (generate fake images inside the tape)
        with tf.GradientTape() as tape:
            fake_images = self.generator(x_train, training=True)
            real_preds = self.discriminator(y_train, training=True)
            fake_preds = self.discriminator(fake_images, training=True)
            d_loss = self.adversarial_loss_fn(real_labels, real_preds) + self.adversarial_loss_fn(fake_labels, fake_preds)

        d_grads = tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(d_grads, self.discriminator.trainable_variables))

        # Train generator — second forward pass is needed so this tape tracks generator gradients
        with tf.GradientTape() as tape:
            fake_images = self.generator(x_train, training=True)
            fake_preds = self.discriminator(fake_images, training=True)
            g_adv_loss = self.adversarial_loss_fn(tf.ones((batch_size, 1)), fake_preds)
            if self.reconstruction_loss_fn is not None:
                g_rec_loss = self.reconstruction_loss_fn(y_train, fake_images)
            else:
                g_rec_loss = tf.constant(0.0, dtype=g_adv_loss.dtype)
            g_loss = (
                self.adversarial_loss_weight * g_adv_loss
                + self.reconstruction_loss_weight * g_rec_loss
            )

        g_grads = tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_grads, self.generator.trainable_variables))

        return {
            "d_loss": d_loss,
            "g_loss": g_loss,
            "g_adv_loss": g_adv_loss,
            "g_rec_loss": g_rec_loss,
        }

def conv_batch_maxpooling(input_tensor, output_channels, kernel_size,
                          pooling_size, func=LeakyReLU, func_kwargs=None,
                          batch_normalization=True, use_bias=False, dropout_rate=0.3,
                          block_depth=2, dropout_from_layer=1):
    """Encoder block: :func:`conv_block` followed by max-pooling.

    Parameters
    ----------
    input_tensor : tf.Tensor
        Input tensor.
    output_channels : int
        Number of filters for each Conv2D layer inside the block.
    kernel_size : int or tuple of int
        Convolutional kernel size.
    pooling_size : int or tuple of int
        Pooling window size for ``MaxPooling2D``.
    func : callable, optional
        Activation function class. Default is ``LeakyReLU``.
    func_kwargs : dict or None, optional
        Keyword arguments for *func*. Default is ``None``.
    batch_normalization : bool, optional
        Whether to apply BatchNormalisation. Default is ``True``.
    use_bias : bool, optional
        Whether Conv2D layers include a bias term. Default is ``False``.
    dropout_rate : float, optional
        Dropout probability. Default is 0.3.
    block_depth : int, optional
        Number of Conv2D layers in the block. Default is 2.
    dropout_from_layer : int, optional
        0-indexed layer from which dropout is applied. Default is 1.

    Returns
    -------
    conv : tf.Tensor
        Output of the conv block — used as the skip connection.
    pool : tf.Tensor
        Output after max-pooling — passed to the next encoder stage.
    """
    conv = conv_block(input_tensor, output_channels, kernel_size, depth=block_depth,
                      func=func, func_kwargs=func_kwargs, batch_normalization=batch_normalization,
                      use_bias=use_bias, dropout_rate=dropout_rate, dropout_from_layer=dropout_from_layer)
    pool = MaxPooling2D(pooling_size, padding='same')(conv)
    return conv, pool

def upsample_and_concat(input_layer, previous_layer, kernel_size, filter_size,
                        func=LeakyReLU, func_kwargs=None, batch_normalization=True, use_bias=False,
                        dropout_rate=0.3, block_depth=2, dropout_from_layer=1, attention=False,
                        kernel_initializer=None):
    """Decoder block: upsample, optionally gate the skip connection, concatenate, then convolve.

    Upsamples via a ``Conv2DTranspose``, optionally applies an attention gate
    to the skip connection, concatenates along the channel axis, and applies a
    ``conv_block``.

    Parameters
    ----------
    input_layer : tf.Tensor
        Input tensor to upsample.
    previous_layer : tf.Tensor
        Skip-connection tensor from the corresponding encoder block.
    kernel_size : int or tuple of int
        Conv kernel size for the ``conv_block``.
    filter_size : int
        Upsampling factor used for ``Conv2DTranspose`` kernel and strides.
    func : callable, optional
        Activation function class. Default is ``LeakyReLU``.
    func_kwargs : dict or None, optional
        Keyword arguments for *func*. Default is ``None``.
    batch_normalization : bool, optional
        Whether to apply BatchNormalisation. Default is ``True``.
    use_bias : bool, optional
        Whether Conv2D layers include a bias term. Default is ``False``.
    dropout_rate : float, optional
        Dropout probability. Default is 0.3.
    block_depth : int, optional
        Number of Conv2D layers in the conv block. Default is 2.
    dropout_from_layer : int, optional
        0-indexed layer from which dropout is applied. Default is 1.
    attention : bool, optional
        If ``True``, apply an attention gate to the skip connection before
        concatenation. Default is ``False``.

    Returns
    -------
    tf.Tensor
        Output tensor after upsampling, (attended) concatenation, and
        ``conv_block``.
    """
    # Number of output channels matches the skip connection depth
    output_channels = previous_layer.shape[-1]

    # Upsample via transposed convolution
    if kernel_initializer is None:
        kernel_initializer = tf.keras.initializers.TruncatedNormal(mean=0, stddev=0.02)
    deconv = Conv2DTranspose(
        filters=output_channels,
        kernel_size=filter_size,
        strides=filter_size,
        padding='same',
        kernel_initializer=kernel_initializer
    )(input_layer)

    # Optionally gate the skip connection using the upsampled decoder signal
    if attention:
        n_intermediate = max(1, output_channels // 2)
        previous_layer = attention_gate(skip=previous_layer, gating=deconv,
                                        n_intermediate_filters=n_intermediate)

    # Concatenate with (attended) skip connection along channel axis
    deconv = Concatenate(axis=-1)([deconv, previous_layer])

    deconv = conv_block(deconv, output_channels, kernel_size, depth=block_depth,
                        func=func, func_kwargs=func_kwargs, batch_normalization=batch_normalization,
                        use_bias=use_bias, dropout_rate=dropout_rate, dropout_from_layer=dropout_from_layer)
    return deconv

def network(input_shape, depth, kernel_size, filter_size, pooling_size, n_of_initial_channels,
            func=LeakyReLU, func_kwargs=None, batch_normalization=True, use_bias=False,
            dropout_rate=0.3, block_depth=2, dropout_from_layer=1, attention=False,
            output_activation=None, kernel_initializer=None):
    """Build a U-Net generator model.

    Parameters
    ----------
    input_shape : tuple of int
        Spatial shape of the input tensor ``(H, W, C)`` — batch dimension
        excluded.
    depth : int
        Number of encoder/decoder stages.
    kernel_size : int or tuple of int
        Convolutional kernel size.
    filter_size : int
        Factor by which the number of channels doubles in each encoder stage
        (and halves in each decoder stage).
    pooling_size : int or tuple of int
        Max-pooling window size.
    n_of_initial_channels : int
        Number of channels in the first encoder stage; multiplied by
        *filter_size* at each subsequent stage.
    func : callable, optional
        Activation function class. Default is ``LeakyReLU``.
    func_kwargs : dict or None, optional
        Keyword arguments for *func*. Default is ``None``.
    batch_normalization : bool, optional
        Whether to apply BatchNormalisation. Default is ``True``.
    use_bias : bool, optional
        Whether Conv2D layers include a bias term. Default is ``False``.
    dropout_rate : float, optional
        Dropout probability. Default is 0.3.
    block_depth : int, optional
        Number of Conv2D layers per encoder/decoder block. Default is 2.
    dropout_from_layer : int, optional
        0-indexed layer within each block from which dropout is applied.
        Default is 1.
    attention : bool, optional
        If ``True``, apply attention gates on all skip connections. Default
        is ``False``.
    output_activation : str or None, optional
        Activation function for the final output Conv2D layer. Default is ``None`` (linear).
    kernel_initializer : None or str or tf.keras.initializers.Initializer, optional
        Kernel initializer for Conv2DTranspose layers. If None, uses TruncatedNormal(mean=0, stddev=0.02).
        Can be a string name (e.g. 'glorot_uniform') or initializer object. Default is ``None``.

    Returns
    -------
    tf.keras.Model
        U-Net model whose input and output shapes both equal *input_shape*
        (single-channel output via a 1×1 Conv2D with linear activation).
    """
    convs = []
    pools = []
    input_tensor = Input(shape=input_shape)
    x = input_tensor
    for i in range(depth):
        conv, pool = conv_batch_maxpooling(x, n_of_initial_channels, kernel_size,
                          pooling_size, func, func_kwargs, batch_normalization, use_bias,
                          dropout_rate, block_depth, dropout_from_layer)
        convs.append(conv)
        pools.append(pool)
        x = pool
        n_of_initial_channels *= filter_size

    x = convs.pop()
    for i in range(depth - 1):
        x = upsample_and_concat(input_layer=x, previous_layer=convs.pop(), kernel_size=kernel_size,
                                filter_size=filter_size, func=func, func_kwargs=func_kwargs,
                                batch_normalization=batch_normalization, use_bias=use_bias,
                                dropout_rate=dropout_rate, block_depth=block_depth,
                                dropout_from_layer=dropout_from_layer, attention=attention,
                                kernel_initializer=kernel_initializer)
        
    output_tensor = Conv2D(1, [1, 1], activation=output_activation)(x)
    model = Model(inputs=input_tensor, outputs=output_tensor)
    return model

