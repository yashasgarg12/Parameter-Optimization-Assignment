import numpy as np


def gen_random_params():
    kernel_list = ['linear', 'poly', 'rbf', 'sigmoid']
    gamma = np.random.uniform(0.0, 1.0)
    C = np.random.uniform(0.0, 1.0)
    kernel = np.random.choice(kernel_list)

    return C, kernel, gamma


#Path: main.py