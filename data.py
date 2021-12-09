import jax.numpy as np
import numpy as onp
import tensorflow_datasets as tfds
import tensorflow as tf
from enum import Enum


def _one_hot(x, k, dtype=np.float32):
    """Create a one-hot encoding of x of size k."""
    return onp.array(x[:, None] == onp.arange(k), dtype)

CIFAR_ANIMAL_IDS =  [2,3,4,5,6,7]
CIFAR_VEHICLE_IDS = [0,1,8,9]

Dataset = Enum('Dataset', ['cifar10','cifar6','cifar4','cifar2', 'mnist'])
ImageNoise = Enum('ImageNoise',['none','gaussian'])
BackgroundNoise = Enum('BackgroundNoise',['none','gaussian','blocks','imagenet'])
BlockNoiseDataset = Enum('BlockNoiseDataset',['cifar10','cifar6','cifar4'])
Placement = Enum('Placement', ['fixed_corner','random_corner','random_loc'])
for e in [Dataset,ImageNoise,BackgroundNoise,BlockNoiseDataset,Placement]:
    e.__str__ = lambda self: self.name


def load_data(dataset=Dataset.cifar10,
              image_size=32, background_size=32,
              image_noise=ImageNoise.none, background_noise=BackgroundNoise.none,
              block_noise_data = BlockNoiseDataset.cifar6,
              image_noise_scalar=1.0, background_noise_scalar=1.0,
              placement=Placement.fixed_corner, num_train=-1, num_test=-1):

    if dataset == Dataset.mnist:
        train_images, train_ys, test_images, test_ys =mnist(num_train, num_test)  # TODO: not passing image size for mnist
    else:
        c10_train_xs, c10_train_ys, c10_test_xs, c10_test_ys = cifar_with_size(image_size)  # TODO: not passing num examples for cifar
        if dataset == Dataset.cifar10:
            train_images, train_ys, test_images, test_ys = c10_train_xs, c10_train_ys, c10_test_xs, c10_test_ys
        elif dataset == Dataset.cifar2:
            train_ys = onp.vectorize(lambda label: 1 if label in CIFAR_ANIMAL_IDS else 0)(c10_train_ys)
            test_ys = onp.vectorize(lambda label: 1 if label in CIFAR_ANIMAL_IDS else 0)(c10_test_ys)
            train_images, test_images = c10_train_xs, c10_test_xs
        elif dataset == Dataset.cifar4:
            train_images, train_ys, test_images, test_ys = cifar_with_only(CIFAR_VEHICLE_IDS,  c10_train_xs, c10_train_ys, c10_test_xs, c10_test_ys)
        elif dataset == Dataset.cifar6:
            train_images, train_ys, test_images, test_ys = cifar_with_only(CIFAR_ANIMAL_IDS,  c10_train_xs, c10_train_ys, c10_test_xs, c10_test_ys)

    if image_noise==ImageNoise.gaussian:
        train_images += onp.random.normal(scale=image_noise_scalar,size=train_images.shape)
        test_images += onp.random.normal(scale=image_noise_scalar,size=test_images.shape)

    if image_size == background_size:
        train_xs, test_xs = train_images, test_images
    else:
        if background_noise == BackgroundNoise.none:
            background_train = onp.zeros((train_images.shape[0], background_size, background_size, train_images.shape[-1]))
            background_test = onp.zeros((test_images.shape[0], background_size, background_size, train_images.shape[-1]))
        elif background_noise == BackgroundNoise.gaussian:
            background_train = onp.random.normal(scale=background_noise_scalar,size=(train_images.shape[0], background_size, background_size, train_images.shape[-1]))
            background_test = onp.random.normal(scale=background_noise_scalar,size=(test_images.shape[0], background_size, background_size, train_images.shape[-1]))
        elif background_noise == BackgroundNoise.blocks:
            c10_train_xs, _, c10_test_xs, _ = cifar_with_size(background_size//2)
            if block_noise_data == BlockNoiseDataset.cifar4:
                background_images,_,_,_ = cifar_with_only(CIFAR_VEHICLE_IDS,  c10_train_xs, c10_train_ys, c10_test_xs, c10_test_ys)
            elif block_noise_data == BlockNoiseDataset.cifar6:
                background_images,_,_,_ = cifar_with_only(CIFAR_ANIMAL_IDS,  c10_train_xs, c10_train_ys, c10_test_xs, c10_test_ys)
            background_train = blocks(background_images, train_images.shape[0])
            background_test = blocks(background_images, test_images.shape[0])
        elif background_noise == BackgroundNoise.imagenet:
            background_train = load_imagenet_plants(train_images.shape[0])
            background_test  = load_imagenet_plants(test_images.shape[0])

        background_train *= background_noise_scalar
        background_test *= background_noise_scalar

        train_xs, test_xs = place_images(placement, train_images, background_train), place_images(placement, test_images, background_test)

    k = max(train_ys) + 1
    print('There are {:} categories'.format(k))
    train_ys, test_ys = _one_hot(train_ys, k),  _one_hot(test_ys, k)

    return train_xs, train_ys, test_xs, test_ys


def mnist(num_train=-1, num_test=-1, size=28):
    ds_train, ds_test = tfds.load('mnist', split=['train','test'],data_dir='data/')
    if size != 28:
        ds_train = ds_train.map(lambda ex: {'image':tf.image.resize(ex['image'], [size,size])/255.,
                                            'label':ex['label']})
        ds_test  = ds_test.map(lambda ex: {'image':tf.image.resize(ex['image'], [size,size])/255.,
                                           'label':ex['label']})

    ds_train = ds_train.batch(60000).as_numpy_iterator()
    ds_train = next(ds_train)

    ds_test = ds_test.batch(10000).as_numpy_iterator()
    ds_test = next(ds_test)

    train_xs, train_ys, test_xs, test_ys = (ds_train["image"], ds_train["label"], ds_test["image"], ds_test["label"])

    if size == 28:
        train_xs = train_xs / 255.
        test_xs = test_xs / 255.

    num_train = num_train if num_train > 0 else 60000
    num_test = num_test if num_test > 0 else 10000
    keep_idx_train = onp.random.choice(train_xs.shape[0],size=num_train)
    keep_idx_test = onp.random.choice(test_xs.shape[0],size=num_test)

    return train_xs[keep_idx_train], train_ys[keep_idx_train], test_xs[keep_idx_test], test_ys[keep_idx_test]

def cifar_with_size(size):
    ds_train, ds_test = tfds.load('cifar10', split=['train','test'],data_dir='data/')
    if size != 32:
        ds_train = ds_train.map(lambda ex: {'id': ex['id'],
                                            'image':tf.image.resize(ex['image'], [size,size])/255.,
                                            'label':ex['label']})
        ds_test  = ds_test.map(lambda ex: {'id': ex['id'],
                                           'image':tf.image.resize(ex['image'], [size,size])/255.,
                                           'label':ex['label']})

    ds_train = ds_train.batch(50000).as_numpy_iterator()
    ds_train = next(ds_train)

    ds_test = ds_test.batch(10000).as_numpy_iterator()
    ds_test = next(ds_test)

    train_xs, train_ys, test_xs, test_ys = (ds_train["image"], ds_train["label"], ds_test["image"], ds_test["label"])
    if size == 32:
        # otherwise this had to be done above during resizing, per tf.image api
        train_xs = train_xs / 255.
        test_xs = test_xs / 255.

    return train_xs, train_ys, test_xs, test_ys

def cifar_with_only(class_ids, train_xs, train_ys, test_xs, test_ys):
    is_allowed_train = onp.equal(train_ys.reshape(-1,1),onp.array(class_ids).reshape(1,-1)).any(axis=-1)
    is_allowed_test = onp.equal(test_ys.reshape(-1,1),onp.array(class_ids).reshape(1,-1)).any(axis=-1)
    train_xs, train_ys = train_xs[is_allowed_train], train_ys[is_allowed_train]
    test_xs, test_ys = test_xs[is_allowed_test], test_ys[is_allowed_test]
    train_ys = onp.vectorize(lambda label: class_ids.index(label))(train_ys)
    test_ys = onp.vectorize(lambda label: class_ids.index(label))(test_ys)
    return train_xs, train_ys, test_xs, test_ys

def blocks(dataset, num_examples):
    num_images = dataset.shape[0]
    out = onp.concatenate([onp.concatenate([dataset[onp.random.choice(num_images, size=num_examples)]
                                            for i in range(2)],axis=1)
                           for j in range(2)],axis=2)
    return out


def crop(images, size):
    N = images.shape[0]
    a = size
    b = images.shape[1]
    right  = onp.random.random_integers(low=0,high=b-a,size=(N))
    top  = onp.random.random_integers(low=0,high=b-a,size=(N))
    r = right[:,None,None] + onp.arange(a)[None,:]
    t = top[:,None,None] + onp.arange(a)[:,None]
    return images[onp.arange(N)[:,None,None],t,r]

def load_imagenet_plants(num_examples,size=32):
    imagenet_plants = onp.load('data/imagenet_plants.npz')['arr_0']
    num_images = imagenet_plants.shape[0]
    if size == 32:
        out = imagenet_plants[onp.random.choice(num_images, size=num_examples)]
    elif size < 32:
        out = imagenet_plants[onp.random.choice(num_images, size=num_examples)]
        out = crop(out, size)
    elif size > 32:
        num_reps = onp.ceil(size/32).astype(int)
        out =onp.zeros((num_examples,32*num_reps, 32*num_reps, 3))
        for i in range(num_reps):
            for j in range(num_reps):
                out[:,i*32:(i+1)*32,j*32:(j+1)*32] = imagenet_plants[onp.random.choice(num_images, size=num_examples)]
        if size % 32 != 0:
            out = crop(out, size)
    return out

def place_images(placement, images, backgrounds):
    N = images.shape[0]
    a = images.shape[1]
    b = backgrounds.shape[1]
    if placement == Placement.fixed_corner:
        right = a*onp.ones((N),dtype=int)
        top = a*onp.ones((N),dtype=int)
    elif placement == Placement.random_corner:
        right = onp.random.choice([0,a],size=(N))
        top = onp.random.choice([0,a],size=(N))
    elif placement == Placement.random_loc:
        right  = onp.random.random_integers(low=0,high=b-a,size=(N))
        top  = onp.random.random_integers(low=0,high=b-a,size=(N))
    r = right[:,None,None] + onp.arange(a)[None,:]
    t = top[:,None,None] + onp.arange(a)[:,None]
    out = backgrounds.copy()
    out[onp.arange(N)[:,None,None],t,r] = images
    return out
