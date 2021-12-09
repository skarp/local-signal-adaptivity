from jax import random
from jax.api import grad, jit
from jax.experimental import optimizers
from jax.experimental.stax import logsoftmax

import neural_tangents as nt
from absl import app,flags
from examples import datasets

import models
from data import *

FLAGS = flags.FLAGS

# Model and NTK flags
NTK = Enum('NTK',['finite','infinite','none'])
Model = Enum('Model', ['cnn','wrn'])
for e in [NTK,Model]:
    e.__str__ = lambda self: self.name
flags.DEFINE_enum_class('ntk', NTK.none, NTK, "which type of NTK to use (none indicates original model)")
flags.DEFINE_enum_class('model', Model.wrn, Model, "which model to use, one of {wrn, cnn}")

# General hyperparameters.
flags.DEFINE_string('loss','ce','loss')
flags.DEFINE_bool('augment',True,'use data augmentation')
flags.DEFINE_bool('cutout',False,'cutout')
flags.DEFINE_integer('batch_size',128,'batch_size')
flags.DEFINE_float('start_lr',0.1,'start_lr')
flags.DEFINE_float('weight_decay',0.0005,'weight_decay')
flags.DEFINE_integer('epochs',200,'epochs')
flags.DEFINE_integer('width_factor',1,'widening factor of cnn or wrn')

flags.DEFINE_bool('evaluate',True,'whether to evaluate the trained model along the way - takes longer')
flags.DEFINE_integer('num_train', -1, 'number of training samples, if less than total')
flags.DEFINE_integer('num_test', -1, 'number of test samples, if less than total')

# Dataset variations (CIFAR-2/4/6, Gaussian noise, and structured noise).
flags.DEFINE_enum_class('dataset',Dataset.cifar10,Dataset,"one of cifar{2,3,6,10}")
flags.DEFINE_integer('image_size', 32, 'size of image to put onto background')
flags.DEFINE_integer('background_size', 32, 'size of background, only settable for background_noise=none or gaussian. no background if this equals image_size')
flags.DEFINE_enum_class('image_noise',ImageNoise.none,ImageNoise,"type of noise to add to image, either none or gaussian")
flags.DEFINE_enum_class('background_noise',BackgroundNoise.none,BackgroundNoise,"type of noise to add to background, either none, gaussian, blocks, or imagenet")
flags.DEFINE_enum_class('block_noise_data',BlockNoiseDataset.cifar6, BlockNoiseDataset, "dataset to use in block noise, if background_noise=blocks. one of cifar{10,6,4}")
flags.DEFINE_float('image_noise_scalar',0.0,'scalar for image noise (std for gaussian)')
flags.DEFINE_float('background_noise_scalar',0.0,'scalar for background noise (std for gaussian, pixel intensity for imagenet and blocks)')
flags.DEFINE_enum_class('placement',Placement.fixed_corner,Placement,"placement of image onto background (fixed_corner, random_corner, or random_loc")

#Inf-width NTK hyperparameter
flags.DEFINE_float('diag_reg', 1e-2, 'diagonal regularization strength for inf-width NTK')

def accuracy(y, y_hat):
    return np.mean(np.argmax(y, axis=1) == np.argmax(y_hat, axis=1))

def main(unused_argv):

    # TODO: param pretty printing

    batch_size = FLAGS.batch_size
    weight_decay = FLAGS.weight_decay
    start_lr = FLAGS.start_lr
    key = random.PRNGKey(0)


    # ============ DATA ==============

    train_images, train_labels, test_images, test_labels = load_data(dataset=FLAGS.dataset,
                                                                     image_size=FLAGS.image_size, background_size=FLAGS.background_size,
                                                                     image_noise=FLAGS.image_noise, background_noise=FLAGS.background_noise,
                                                                     block_noise_data = FLAGS.block_noise_data,
                                                                     image_noise_scalar=FLAGS.image_noise_scalar, background_noise_scalar=FLAGS.background_noise_scalar,
                                                                     placement=FLAGS.placement,
                                                                     num_train=FLAGS.num_train, num_test=FLAGS.num_test)
    print("train_images shape = {}".format(train_images.shape))
    print("test_images shape = {}".format(test_images.shape))
    print("train_labels shape = {}".format(train_labels.shape))
    print("test_labels shape = {}".format(test_labels.shape))
    num_train = train_images.shape[0]


    # ============ TRAIN  =============

    # ---- model ----
    num_classes = train_labels.shape[1]

    if FLAGS.ntk == NTK.finite:
        if FLAGS.model == Model.cnn:
            init_fn, f_orig, _ = models.simple_cnn_NT(width_factor=FLAGS.width_factor)
            _, init_params = init_fn(key, (-1, FLAGS.background_size, FLAGS.background_size, 1))
        if FLAGS.model == Model.wrn:
            init_fn, f_orig, _ = models.WideResnetNT(num_blocks=1, k=FLAGS.width_factor, num_classes=num_classes)
            _, init_params = init_fn(key, (-1, FLAGS.background_size, FLAGS.background_size, 3))
        f = nt.linearize(f_orig, init_params)
    elif FLAGS.ntk == NTK.infinite:
        if FLAGS.model == Model.cnn:
            _, _, kernel_fn = models.simple_cnn_NT(width_factor=FLAGS.width_factor)
    else:
        if FLAGS.model == Model.cnn:
            init_fn, f = models.simple_cnn(width_factor=FLAGS.width_factor)
            _, init_params = init_fn(key, (-1, FLAGS.background_size, FLAGS.background_size, 1))
        if FLAGS.model == Model.wrn:
            init_fn, f = models.WideResnet(num_blocks=1, k=FLAGS.width_factor, num_classes=num_classes)
            _, init_params = init_fn(key, (-1, FLAGS.background_size, FLAGS.background_size, 3))


    if FLAGS.ntk == NTK.infinite:
        predict_fn = nt.predict.gradient_descent_mse_ensemble(nt.batch(kernel_fn,50),  # TODO: other batch size?
                                                              train_images,
                                                              train_labels,
                                                              diag_reg=FLAGS.diag_reg,
                                                              diag_reg_absolute_scale=False)

        y_train_ntk = predict_fn(x_test=train_images, get='ntk')
        print("train acc = {}".format(np.mean(np.argmax(train_labels,axis=-1) == np.argmax(y_train_ntk, axis=-1))))

        y_test_ntk = predict_fn(x_test=test_images, get='ntk')
        print("test acc = {}".format(np.mean(np.argmax(test_labels,axis=-1) == np.argmax(y_test_ntk, axis=-1))))

        return

    #if using cnn/wrn or finite-width ntk, continue to training loop:

    # ---- optimizer ----
    steps_per_epoch = num_train // batch_size
    epochs_factor = int(FLAGS.epochs/200)

    if FLAGS.ntk == NTK.finite:
        learning_rate_fn = optimizers.piecewise_constant(steps_per_epoch * epochs_factor * np.array([140,170]),
                                                        start_lr*np.array([1.,0.2,0.2**2]))
        opt_init, opt_apply, get_params = optimizers.adam(learning_rate_fn)
    else:
        learning_rate_fn = optimizers.piecewise_constant(steps_per_epoch * epochs_factor * np.array([80,100,120]),
                                                         start_lr*np.array([1.,0.2,0.2**2,0.2**3]))
        opt_init, opt_apply, get_params = optimizers.momentum(learning_rate_fn, 0.9)

    opt_apply = jit(opt_apply)
    state = opt_init(init_params)

    # ---- loss ----

    if FLAGS.loss=='mse':
        loss = lambda fx, y: ((fx - y) ** 2).sum(axis=1)
    elif FLAGS.loss=='ce':
        loss = lambda fx, y: -np.mean(logsoftmax(fx) * y, axis=1)


    if weight_decay > 0.0:
        weight_penalty = lambda params: weight_decay * optimizers.l2_norm(params)**2
        batch_loss =  lambda params, x, y: np.mean(loss(f(params,x), y)) + weight_penalty(params)
    else:
        batch_loss = lambda params, x, y: np.mean(loss(f(params,x), y))

    grad_loss = jit(grad(batch_loss))


    # ---- main loop ----

    for epoch in range(FLAGS.epochs):
        print("Starting epoch {} with {} steps".format(epoch, steps_per_epoch))
        for i, (x,y) in enumerate(datasets.minibatch(train_images, train_labels,batch_size,1)):

            params = get_params(state)


            gl = grad_loss(params, x, y)
            state = opt_apply(epoch * steps_per_epoch + i, gl, state)

            # compute metrics
            if i % steps_per_epoch == 0:
                print('{}\t{:.4f}'.format(epoch, batch_loss(params, x, y)))

                if (epoch < 10*epochs_factor and epoch % epochs_factor == 0) or (epoch % (10*epochs_factor) == 0):
                    if FLAGS.evaluate:    
                        print("Printing metrics at epoch {}".format(epoch))


                        train_output = []
                        for i in range(0,len(train_images),100):
                            x = train_images[i:i+100]
                            train_output.append(f(params, x))
                        train_output = np.concatenate(train_output)
                        print('\ttrain: loss {}\tacc {}'.format(np.mean(loss(train_output, train_labels)),accuracy(train_output, train_labels)))


                        test_output = []
                        print("len(test_images) = {}".format(len(test_images)))
                        for i in range(0,len(test_images),100):
                            x = test_images[i:i+100]
                            test_output.append(f(params, x))
                        test_output = np.concatenate(test_output)

                        print('\ttest: loss {}\tacc {}'.format(np.mean(loss(test_output, test_labels)),accuracy(test_output, test_labels)))

if __name__ == '__main__':
    app.run(main)

