#!/bin/bash -l

#figure 3
result_dir="toy_results"
python -u synthetic_data_sim.py --d 10 --k 1000 --results_dir ${result_dir} \
                                   --batch_size 500 --num_test 5000 --lr_w  0.1  \
                                   --epochs 10000 --eval_freq 100


#figure 4 (CIFAR-10 noise=ImageNet)
placement="random_loc"
background_noise="imagenet"
dataset="cifar10"
for scalar in 0.0 0.25 0.5 0.75 1.0
do
  python -u train.py --dataset ${dataset} --noaugment \
                        --background_noise ${background_noise} --placement ${placement} \
                        --background_noise_scalar ${scalar} \
                        --image_size 16 --background_size 32 \
                        --ntk finite --width_factor 10 \
                        --batch_size 50 --start_lr 0.001 --weight_decay 0.0

  python -u train.py --dataset ${dataset} --noaugment \
                        --background_noise ${background_noise} --placement ${placement} \
                        --background_noise_scalar ${scalar} \
                        --image_size 16 --background_size 32 \
                        --width_factor 10

done


#figure 4 (CIFAR-Vehicles)
placement="random_corner"
background_noise="blocks"
dataset="cifar2"
for scalar in 0.0 0.25 0.5 0.75 1.0
do
  python -u train.py --dataset ${dataset} --noaugment \
                        --background_noise ${background_noise} --placement ${placement} \
                        --background_noise_scalar ${scalar} \
                        --image_size 16 --background_size 32 \
                        --ntk finite --width_factor 10 \
                        --batch_size 50 --start_lr 0.001 --weight_decay 0.0

  python -u train.py --dataset ${dataset} --noaugment \
                        --background_noise ${background_noise} --placement ${placement} \
                        --background_noise_scalar ${scalar} \
                        --image_size 16 --background_size 32 \
                        --width_factor 10


#figure 4 (CIFAR-2 noise=ImageNet)
placement="random_loc"
background_noise="imagenet"
dataset="cifar2"
for scalar in 0.0 0.25 0.5 0.75 1.0
do
  python -u train.py --dataset ${dataset} --noaugment \
                        --background_noise ${background_noise} --placement ${placement} \
                        --background_noise_scalar ${scalar} \
                        --image_size 16 --background_size 32 \
                        --ntk finite --width_factor 10 \
                        --batch_size 50 --start_lr 0.001 --weight_decay 0.0

  python -u train.py --dataset ${dataset} --noaugment \
                        --background_noise ${background_noise} --placement ${placement} \
                        --background_noise_scalar ${scalar} \
                        --image_size 16 --background_size 32 \
                        --width_factor 10
done



#figure 4 (CIFAR-2 noise=Gaussian)
placement="random_loc"
background_noise="gaussian"
dataset="cifar2"
for scalar in 0.0 0.5 1.0 1.5 2.0
do
  python -u train.py --dataset ${dataset} --noaugment \
                        --background_noise ${background_noise} --placement ${placement} \
                        --background_noise_scalar ${scalar} \
                        --image_size 16 --background_size 32 \
                        --ntk finite --width_factor 10 \
                        --batch_size 50 --start_lr 0.001 --weight_decay 0.0

  python -u train.py --dataset ${dataset} --noaugment \
                        --background_noise ${background_noise} --placement ${placement} \
                        --background_noise_scalar ${scalar} \
                        --image_size 16 --background_size 32 \
                        --width_factor 10
done


#figure 6 (MNIST-20k)
image_noise="none"
background_noise="gaussian"
placement="random_loc"
image_size=28
background_size=42
diag_reg=1e-5
num_train=20000
num_test=10000

for noise_scalar in 0.0 0.5 1.0 1.5 2.0
do
  python -u train.py --model cnn --dataset mnist --noaugment \
   --placement ${placement} --image_noise ${image_noise} --background_noise ${background_noise} --image_size 28 \
   --background_size ${background_size} --background_noise_scalar ${noise_scalar} \
   --num_train ${num_train} --num_test ${num_test} \
   --ntk infinite --diag_reg ${diag_reg} \
    > mnist_im${image_size}_bg${background_size}_bg-noise_${background_noise}_${noise_scalar}_${placement}_num-train-${num_train}_diag-reg-${diag_reg}_inf_ntk.out

  python -u train.py --model cnn --dataset mnist --noaugment \
   --placement ${placement} --image_noise ${image_noise} --background_noise ${background_noise} --image_size 28 \
   --background_size ${background_size} --background_noise_scalar ${noise_scalar} \
   --num_train ${num_train} --num_test ${num_test} \
   --ntk finite --batch_size 50 --start_lr 0.001 --weight_decay 0.0 \
    > mnist_im${image_size}_bg${background_size}_bg-noise_${background_noise}_${noise_scalar}_${placement}_num-train-${num_train}_finite_ntk.out

  python -u train.py --model cnn --dataset mnist --noaugment \
   --placement ${placement} --image_noise ${image_noise} --background_noise ${background_noise} --image_size 28 \
   --background_size ${background_size} --background_noise_scalar ${noise_scalar} \
   --num_train ${num_train} --num_test ${num_test} \
    > mnist_im${image_size}_bg${background_size}_bg-noise_${background_noise}_${noise_scalar}_${placement}_num-train-${num_train}_finite_cnn.out
done


#figure 6 (MNIST)
image_noise="none"
background_noise="gaussian"
placement="random_loc"
image_size=28
background_size=42

for noise_scalar in 0.0 0.5 1.0 1.5 2.0
do
  for width_factor in 50 100
  do
    python -u train.py --model cnn --dataset mnist --noaugment \
   --placement ${placement} --image_noise ${image_noise} --background_noise ${background_noise} --image_size 28 \
   --background_size ${background_size} --background_noise_scalar ${noise_scalar} \
   --ntk finite --width_factor ${width_factor} --batch_size 50 --start_lr 0.001 --weight_decay 0.0 \
    > mnist_im${image_size}_bg${background_size}_bg-noise_${background_noise}_${noise_scalar}_${placement}_width-factor-${width_factor}_finite_ntk.out
  done

  python -u train.py --model cnn --dataset mnist --noaugment \
   --placement ${placement} --image_noise ${image_noise} --background_noise ${background_noise} --image_size 28 \
   --background_size ${background_size} --background_noise_scalar ${noise_scalar} \
    > mnist_im${image_size}_bg${background_size}_bg-noise_${background_noise}_${noise_scalar}_${placement}_finite_cnn.out
done
