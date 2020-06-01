# DAUA
The implementation of Domain Agnostic Learning for Unbiased Authentication method by Keras
 
## Dependencies
The code runs with Python and requires Tensorflow of version 1.2.1 or higher and Keras of version 2.0 or higher. Please `pip install` the following packages:
- `numpy`
- `tensorflow` 
- `keras`
- `pandas`
 
## CMNIST data

Download `colored_mnist.h5` in https://drive.google.com/drive/folders/1EH9GM9TTsfcWxYV5QtwCtyTtJWrcP3h5

And put `colored_mnist.h5` in the `data/` folder

Run the following command in shell:

```shell
python train_CMNIST.py
```
 
See `train_CMNIST.py` for details. 

## CelebA data

Download `celeba_img_align_5p_size64.h5` in https://drive.google.com/drive/folders/1EH9GM9TTsfcWxYV5QtwCtyTtJWrcP3h5

And put `celeba_img_align_5p_size64.h5` in the `data/` folder

Run the following command in shell:

```shell
python train_CelebA.py
```
 
See `train_CelebA.py` for details. 

## Mobile data

Download `device_transfer.h5` in https://drive.google.com/drive/folders/1EH9GM9TTsfcWxYV5QtwCtyTtJWrcP3h5

And put `device_transfer.h5` in the `data/` folder

Run the following command in shell:

```shell
python train_mobile.py
```
 

See `train_mobile.py` for details. 
