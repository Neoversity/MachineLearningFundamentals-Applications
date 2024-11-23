# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 00:52:38 2024

@author: anton
"""

# import torch
# print(torch.cuda.is_available())  # Має повернути True
# print(torch.version.cuda)        # Має повернути 11.8
# print(torch.cuda.get_device_name(0))  # Має повернути 'NVIDIA GeForce RTX 3080 Ti'

import tensorflow as tf
import torch
print("TensorFlow version:", tf.__version__)
print("Built with CUDA support:", tf.test.is_built_with_cuda())
print("GPU Available:", tf.config.list_physical_devices('GPU'))


print("GPU доступний:", torch.cuda.is_available())
print("Назва GPU:", torch.cuda.get_device_name(0))


