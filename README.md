# Code for [Local Signal Adaptivity: Provable Feature Learning in Neural Networks Beyond Kernels](https://papers.nips.cc/paper/2021/hash/d064bf1ad039ff366564f352226e7640-Abstract.html)
by Stefani Karp, Ezra Winston, Yuanzhi Li, Aarti Singh

NeurIPS 2021



### Setup
Python requirements are given in requirements.txt.

To install jax for gpu usage, see instructions at [github.com/google/jax](https://github.com/google/jax).


### Contents
The file ```train.py``` runs the CIFAR-variant experiments, and the file ```synthetic_data_sim.py``` runs the synthetic-data experiments with the model used in the theory.

Examples of both types of runs are given in ```main_paper_experiments.sh```.

Calling ```train.py``` allows for specification of the image dataset, noise type, image placement, etc.

Examples of various data variants are given in ```data_examples.ipynb```.
