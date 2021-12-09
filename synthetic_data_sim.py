from jax import random
from jax.api import grad, jit
from jax.experimental import optimizers
import jax.numpy as np

import pandas as pd
import numpy as onp

from examples import datasets

from absl import app
from absl import flags

import models
import pickle

FLAGS = flags.FLAGS



# General hyperparameters.
flags.DEFINE_integer('batch_size',2000,'batch_size')
flags.DEFINE_float('lr_w',0.1/1000,'lr of w')
flags.DEFINE_integer('epochs',10000,'epochs')
flags.DEFINE_integer('num_test',2000,'number of test examples')

# Dataset variations (Gaussian noise).
flags.DEFINE_integer('d',10,'dimension')
flags.DEFINE_integer('k',100,'num rows')
flags.DEFINE_float('noise_scalar',1.,'multiplies noise sigma')
flags.DEFINE_string('results_dir','','results_dir')
flags.DEFINE_integer('eval_freq',1,'eval frequancy in steps')

flags.DEFINE_boolean('evaluate',True,'whether to evaluate')

def _one_hot(x, k, dtype=np.float32):
    """Create a one-hot encoding of x of size k."""
    return onp.array(x[:, None] == onp.arange(k), dtype)

def get_toy_dataset(n, k, sigma, w_star):
    w_star = w_star.reshape(-1,)
    d = w_star.shape[0]
    ys = onp.random.choice([-1,1],size=(n,1))
    signal_indices = onp.random.choice(range(k),size=(n,))
    xs = onp.random.normal(loc=0,scale=sigma,size=(n,k,d))
    for i in range(n):
        to_add = ys[i,0]*w_star
        xs[i,signal_indices[i],:] = to_add  # +=
    xs = onp.expand_dims(xs, axis=-1)
    return np.array(xs), np.array(ys)

def load_toy_data(n_train, n_test, k, sigma, w_star):
    train_xs, train_ys = get_toy_dataset(n_train, k, sigma, w_star)
    test_xs, test_ys = get_toy_dataset(n_test, k, sigma, w_star)
    return train_xs, train_ys, test_xs, test_ys

def accuracy(fx, y):
    return np.mean(y * fx >= 0)


def main(unused_argv):
    print("eval_freq", FLAGS.eval_freq)
    exp_name = "lr_w_{}_d_{}_k_{}".format(FLAGS.lr_w, FLAGS.d, FLAGS.k)

    key = random.PRNGKey(0)

    sigma = FLAGS.noise_scalar*np.log(FLAGS.k)/np.sqrt(FLAGS.k)
    w_star = np.ones((FLAGS.d,1))/np.sqrt(FLAGS.d)

    num_train = FLAGS.batch_size


    init_fn, f = models.toy_network(k=FLAGS.k, d=FLAGS.d)
    _, init_params = init_fn(key, (-1, FLAGS.k, FLAGS.d, 1))
    init_w_radius = 1e-6
    init_params[0][0] = (init_w_radius*init_params[0][0][0]/onp.linalg.norm(init_params[0][0][0]),
                         init_params[0][0][1])
    # ---- optimizer ----
    steps_per_epoch = num_train // FLAGS.batch_size
    opt_init, opt_apply, get_params = optimizers.sgd(FLAGS.lr_w)
    opt_apply = jit(opt_apply)
    state = opt_init(init_params)

    # ---- loss ----
    loss = lambda fx, y: np.log(1 + np.exp(-y * fx))
    batch_loss = lambda params, x, y: np.mean(loss(f(params,x), y))
    grad_loss = jit(grad(batch_loss))

    # ---- main loop ----

    info = dict()
    info['d'] = FLAGS.d
    info['k'] = FLAGS.k
    info['batch_size'] = FLAGS.batch_size
    info['lr_w'] = FLAGS.lr_w
    info['noise_scalar'] = FLAGS.noise_scalar
    results = pd.DataFrame(columns=list(info.keys())+['c','o','delta_w_perp_inner_prod','b','grad_b','grad_w_norm','grad_c','loss','accuracy'])

    step = 0
    w_star_perp_old = None
    for epoch in range(FLAGS.epochs):

        train_images, train_labels, test_images, test_labels = load_toy_data(n_train=num_train,
                                                                             n_test=FLAGS.num_test,
                                                                             k=FLAGS.k,
                                                                             sigma=sigma,
                                                                             w_star=w_star)
        for i, (x, y) in enumerate(datasets.minibatch(train_images, train_labels, FLAGS.batch_size, 1)):
            params = get_params(state)
            grads = grad_loss(params, x, y)

            if step % FLAGS.eval_freq == 0:
                row = info.copy()
                row['step'] =step
                epoch += 1

                if FLAGS.evaluate:

                    test_output = []
                    for i, (x, y) in enumerate(datasets.minibatch(test_images, test_labels, FLAGS.num_test, 1)):
                        test_output.append(f(params, x))
                    test_output = np.concatenate(test_output)

                    w  = onp.asarray(params[0][0][0]).reshape(FLAGS.d,1)
                    c = (w.T@w_star).item()

                    w_star_perp_unnorm = w - c*w_star
                    o = onp.linalg.norm(w_star_perp_unnorm)
                    w_star_perp = w_star_perp_unnorm/o
                    if w_star_perp_old is None:
                        delta_w_perp = onp.array(0)
                    else:
                        delta_w_perp = w_star_perp_old.T@w_star_perp
                    w_star_perp_old = w_star_perp
                    row['c']= c
                    row['o']= o
                    row['delta_w_perp_inner_prod'] = delta_w_perp.item()
                    row['b']=params[0][0][1].item()
                    row['grad_b']=grads[0][0][1].item()
                    row['grad_w_norm'] = onp.linalg.norm(grads[0][0][0])
                    row['grad_c'] = (onp.asarray(grads[0][0][0]).reshape(FLAGS.d,1).T@w_star).item()
                    row['loss'] = np.mean(loss(test_output, test_labels)).item()
                    row['accuracy']= accuracy(test_output, test_labels).item()
                    results = results.append(row,ignore_index=True)
                    results.to_pickle(FLAGS.results_dir+'/' + exp_name+'.pkl')
            grads[0][0] = (grads[0][0][0], grads[0][0][1]/FLAGS.k)
            state = opt_apply(epoch * steps_per_epoch + i, grads, state)
            step += 1

        del train_images
        del train_labels
        del test_images
        del test_labels

    pickle.dump(params, open(FLAGS.results_dir+'/' + exp_name+'_final_model.pkl', "wb"))


if __name__ == '__main__':
    app.run(main)

