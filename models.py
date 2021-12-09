import jax.experimental.stax as stax
from neural_tangents import stax as nt_stax

#================ inf-width networks =============

def simple_cnn_NT(width_factor=1, num_classes=10):
    print("Using width_factor {}".format(width_factor))
    return nt_stax.serial(
        nt_stax.Conv(out_chan=8*width_factor, filter_shape=(3,3), strides=(2,2), padding='VALID'),
        nt_stax.Relu(),
        nt_stax.Conv(out_chan=16*width_factor, filter_shape=(3,3), strides=(2,2), padding='VALID'),
        nt_stax.Relu(),
        nt_stax.Flatten(),
        nt_stax.Dense(out_dim=num_classes))

def simple_cnn(width_factor=1, num_classes=10):
    return stax.serial(
        stax.Conv(out_chan=8*width_factor, filter_shape=(3,3), strides=(2,2), padding='VALID'),
        stax.Relu,
        stax.Conv(out_chan=16*width_factor, filter_shape=(3,3), strides=(2,2), padding='VALID'),
        stax.Relu,
        stax.Flatten,
        stax.Dense(out_dim=num_classes))


#================ toy network =============

def double_with_shared_params(layer):
    def init_fun(rng, input_shape):
        layer_init_fun, _ = layer
        layer_instance = layer_init_fun(rng, input_shape)
        return layer_instance
    def apply_fun(params, inputs, **kwargs):
        _, layer_apply_fun = layer
        return layer_apply_fun(params, inputs, **kwargs) - layer_apply_fun(params, -inputs, **kwargs)
    return init_fun, apply_fun

def toy_network(k, d):
    # Elementwise multiply by -1 before passing into the same Conv again  
    conv = stax.Conv(out_chan=1, filter_shape=(1, d), strides=(1,1), padding="VALID")
    conv_relu_pair = stax.serial(conv, stax.Relu)

    window_shape = (k,1)
    ones = (1,) * len(window_shape)
    print("ones = {}".format(ones))

    return stax.serial(double_with_shared_params(conv_relu_pair),stax.SumPool(window_shape=window_shape),stax.Flatten)

#================ WRN from neural tangents lib, for linearization =============

# similar to example in https://github.com/google/autol2 but slightly modified to
# replicate original wrn from  https://github.com/szagoruyko/wide-residual-networks/tree/master/pytorch

def _batch_norm_internal(batchnorm, axis=(0, 1, 2)):
    """Layer constructor for a stax.BatchNorm layer with dummy kernel computation.
  Do not use kernels for architectures that include this function."""
    bn = stax.BatchNorm()
    init_fn, apply_fn = bn
    kernel_fn = lambda kernels: kernels
    return init_fn, apply_fn, kernel_fn

def WideResnetBlockNT(channels, strides, channel_mismatch=False):
    Main = nt_stax.serial(_batch_norm_internal(None),
                          nt_stax.Relu(),
                          nt_stax.Conv(channels, (3, 3), strides, padding='SAME'),
                          _batch_norm_internal(None),
                          nt_stax.Relu(),
                          nt_stax.Conv(channels, (3, 3), strides=(1, 1), padding='SAME'))

    Shortcut = nt_stax.Identity if not channel_mismatch else nt_stax.Conv(channels, (1, 1), strides, padding='VALID')

    return nt_stax.serial(nt_stax.FanOut(2),
                       nt_stax.parallel(Main, Shortcut),
                       nt_stax.FanInSum())


def WideResnetGroupNT(num_blocks, channels, strides):
    blocks = []
    blocks += [WideResnetBlockNT(channels, strides, channel_mismatch=True)]
    for _ in range(num_blocks - 1):
        blocks += [WideResnetBlockNT(channels, strides=(1, 1))]
    return nt_stax.serial(*blocks)


def WideResnetNT(num_blocks, k, num_classes=10):
    widths = [int(v * k) for v in (16, 32, 64)]
    return nt_stax.serial(
        nt_stax.Conv(16, (3, 3), strides=(1, 1), padding='SAME'),
        WideResnetGroupNT(num_blocks, widths[0], strides=(1, 1)),
        WideResnetGroupNT(num_blocks, widths[1], strides=(2, 2)),
        WideResnetGroupNT(num_blocks, widths[2], strides=(2, 2)),
        _batch_norm_internal(None),
        nt_stax.Relu(),
        nt_stax.AvgPool((8, 8)),
        nt_stax.Flatten(),
        nt_stax.Dense(num_classes)
    )




#================ WRN =============

# similar to example in https://github.com/google/autol2 but using straight stax instead of neural_tangents.stax
# also slightly modified to replicate original wrn from  https://github.com/szagoruyko/wide-residual-networks/tree/master/pytorch

def WideResnetBlock(channels, strides, channel_mismatch=False):
    Main = stax.serial(stax.BatchNorm(),
                       stax.Relu,
                       stax.Conv(channels, (3, 3), strides, padding="SAME"),
                       stax.BatchNorm(),
                       stax.Relu,
                       stax.Conv(channels, (3, 3), strides=(1, 1), padding="SAME"))

    Shortcut = stax.Identity if not channel_mismatch else stax.Conv(channels, (1, 1), strides, padding='VALID')

    return stax.serial(stax.FanOut(2),
                       stax.parallel(Main, Shortcut),
                       stax.FanInSum)


def WideResnetGroup(num_blocks, channels, strides):
    blocks = []
    blocks += [WideResnetBlock(channels, strides, channel_mismatch=True)]
    for _ in range(num_blocks - 1):
        blocks += [WideResnetBlock(channels, strides=(1, 1))]
    return stax.serial(*blocks)


def WideResnet(num_blocks, k, num_classes=10):
    widths = [int(v * k) for v in (16, 32, 64)]
    return stax.serial(
        stax.Conv(16, (3, 3), strides=(1, 1), padding="SAME"),
        WideResnetGroup(num_blocks, widths[0], strides=(1, 1)),
        WideResnetGroup(num_blocks, widths[1], strides=(2, 2)),
        WideResnetGroup(num_blocks, widths[2], strides=(2, 2)),
        stax.BatchNorm(),
        stax.Relu,
        stax.AvgPool((8, 8)),
        stax.Flatten,
        stax.Dense(num_classes)
    )
