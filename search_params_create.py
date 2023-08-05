#!/usr/bin/env python3

import os
import argparse
import math
from pathlib import Path

import k_diffusion as K

import random
import numpy as np


def main():
    writer = K.utils.CSVLogger(f'search_params.csv',
                               ['ssd_mean', 'ssd_std', 'sigma_min', 'sigma_max', 'lr', 'weight_decay'])

    lr = 2e-4
    weight_decay = 1e-4
    locs = np.linspace(-5.0, -0.5, num=10)
    scales = np.linspace(2.0, 0.5, num=10)
    for loc in locs:
        for scale in scales:
            sigma_min = np.exp(loc - 5 * scale)
            sigma_max = np.exp(loc + 5 * scale)
            writer.write(loc, scale, sigma_min, sigma_max, lr, weight_decay)


if __name__ == '__main__':
    main()
    # compare_ds()
