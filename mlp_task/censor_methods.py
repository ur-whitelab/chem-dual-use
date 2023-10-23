import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import random
from dataclasses import dataclass

def noising(
        data, xnoiselevel, ynoiselevel, threshold=0, noise_region='above', plot=True
    ):
    x, y = data
    x = x.numpy()
    y = y.numpy()
    x_random = np.random.normal(size=x.shape)
    y_random = np.random.normal(size=y.shape)
    if noise_region == 'above':
        xnoise = xnoiselevel * x_random * np.select([y>threshold],[1], 0).reshape(-1,1)
        ynoise = ynoiselevel * y_random * np.select([y>threshold],[1], 0)
    elif noise_region == 'below':
        xnoise = xnoiselevel * x_random * np.select([y<threshold],[1], 0).reshape(-1,1)
        ynoise = ynoiselevel * y_random * np.select([y<threshold],[1], 0)
    else:
          raise Exception("'targetregion' argument can only accept 'below' or 'above'")
    xn = x + np.float32(xnoise)
    yn = y + np.float32(ynoise)
    if plot:
        if noise_region == 'above':
            mask = y > threshold
            title_suffix = 'y > threshold'
        else:
            mask = y < threshold
            title_suffix = 'y < threshold'
        # to be certain only desired region is noised
        if ynoiselevel:
            # separate non-noised vs noised region by color
            plt.scatter(y[mask], yn[mask], c='C1')
            plt.scatter(y[~mask], yn[~mask], c='C0')
            plt.title(f'Y noise at level {ynoiselevel}')
            plt.tight_layout()
            plt.savefig(f'sanitycheck_yn{xnoiselevel}.png',dpi=300)
            plt.show()

        if xnoiselevel:
            plt.scatter(x[mask, 0], xn[mask, 0], c='C1')
            plt.scatter(x[~mask, 0], xn[~mask, 0], c='C0')
            plt.title(f'X noise at level {xnoiselevel}')
            plt.ylim(-8,8)
            plt.xlim(-2,2)
            plt.ylabel('noised x')
            plt.xlabel('true x')
            plt.tight_layout()
            plt.savefig(f'sanitycheck_xn{xnoiselevel}.png',dpi=300)
            plt.show()


            plt.tight_layout()
            # # Plot a histogram for gaussian noise
            # plt.hist(x_random[:,0], bins=30)
            # plt.title('Distribution of Random Normal Noise (X)')
            # plt.xlabel('Value')
            # plt.ylabel('Frequency')
            # plt.show()

    return torch.tensor(xn), torch.tensor(yn)

def omit_sensitive_data_probabilistic(
        data, threshold, sensitive_region, omit_probability=1, plot=False
    ):
    """
    Filters and returns data by omitting a fraction of data in the sensitive region.

    Args:
    data (tuple): Input data containing features and labels.
    threshold (float): Value determining the sensitive region.
    sensitive_region (str): 'above' or 'below' to specify the sensitive region.
    omit_probability (float): Chance of omitting a data point in the sensitive region.
    plot (boolean): Prints a plot of filtered y vs. raw y for sanity check.

    Returns:
    tuple: Filtered features and labels as tensors.

    """
    if sensitive_region not in ['above','below']:
        raise ValueError("sensitive_region must be either 'above' or 'below'")
    if not 0 <= omit_probability <= 1:
        raise ValueError("omit_probability must be in the range [0,1]")

    x, y = data
    filtered_x = []
    filtered_y = []
    for i, label in enumerate(y):
        is_in_sensitive_region = (
            (sensitive_region == 'above' and label.item() > threshold) or
            (sensitive_region == 'below' and label.item() < threshold)
        )
        if is_in_sensitive_region and torch.rand(1).item() <= omit_probability:
            # exclude data point if it's in sensitive region and within omission prob
            continue

        filtered_x.append(x[i])
        filtered_y.append(label)

    # Convert back to tensors, preserving original data types and shapes
    filtered_x = torch.stack(filtered_x)
    filtered_y = torch.tensor(filtered_y, dtype=y.dtype)

    if plot:
        plt.figure()
        plt.plot(y.numpy(), color='blue', label='Original Y Values')
        plt.plot(filtered_y.numpy(), color='red', label='Filtered Y Values')
        plt.axhline(y=threshold, color='grey', linestyle='--', label='Threshold')
        plt.xlabel('Index') # filter_y indices doesn't refer to the original y...
        plt.ylabel('Y Value')
        plt.title('Sensitive Data Omission')
        plt.legend()
        plt.show()

    return filtered_x, filtered_y

# deterministic omission
def omit_sensitive_data(data, threshold, sensitive_region, omit_fraction=1, plot=False):
    """
    Filters and returns data by omitting a fraction of data in the sensitive region.

    Args:
    data (tuple): Input data containing features and labels.
    threshold (float): Value determining the sensitive region.
    sensitive_region (str): 'above' or 'below' to specify the sensitive region.
    omit_fraction (float): Fraction of data points to be omitted in the sensitive region.
    plot (boolean): Prints a plot of filtered y vs. raw y for sanity check.

    Returns:
    tuple: Filtered features and labels as tensors.
    """
    
    if sensitive_region not in ['above', 'below']:
        raise ValueError("sensitive_region must be either 'above' or 'below'")
    if not 0 <= omit_fraction <= 1:
        raise ValueError("omit_fraction must be in the range [0,1]")

    x, y = data
    sensitive_data = []
    non_sensitive_data = []

    for i, label in enumerate(y):
        if (sensitive_region == 'above' and label.item() > threshold) or \
           (sensitive_region == 'below' and label.item() < threshold):
            sensitive_data.append((x[i], label))
        else:
            non_sensitive_data.append((x[i], label))

    omit_size = int(omit_fraction * len(sensitive_data))
    kept_sensitive_data = random.sample(sensitive_data, len(sensitive_data) - omit_size)

    final_data = kept_sensitive_data + non_sensitive_data

    # Convert to tensor form
    filtered_x = torch.stack([data_point[0] for data_point in final_data])
    filtered_y = torch.tensor([data_point[1] for data_point in final_data], dtype=y.dtype)
    
    if plot:
        plt.figure()
        plt.plot(y.numpy(), color='blue', label='Original Y Values')
        plt.scatter(range(len(filtered_y.numpy())), filtered_y.numpy(), color='red', label='Filtered Y Values', s=10)
        plt.axhline(y=threshold, color='grey', linestyle='--', label='Threshold')
        plt.xlabel('Index') # Note: filtered_y indices may not align with original y...
        plt.ylabel('Y Value')
        plt.title('Sensitive Data Omission')
        plt.legend()
        plt.show()
    
    return filtered_x, filtered_y

# def filterdata_by_label(data, threshold, omitregion):
#     if omitregion not in ['above', 'below']:
#         raise ValueError("omitregion must be either 'above' or 'below'")
#     x,y = data
#     filtered_x = []
#     filtered_y = []

#     for i, label in enumerate(y):
#         if omitregion == 'above':
#             if label < threshold:
#                 filtered_x.append(x[i])
#                 filtered_y.append(label)
#         else:
#             if label > threshold:
#                 filtered_x.append(x[i])
#                 filtered_y.append(label)

#     if len(filtered_x) == 1:
#         filtered_x = filtered_x[0]
#     else:
#         x_shape = filtered_x[0].shape
#         filtered_x = torch.stack(filtered_x).reshape(len(filtered_x), *x_shape)

#     if len(filtered_y) == 1:
#         filtered_y = filtered_y[0]
#     else:
#         filtered_y = torch.tensor(filtered_y)

#     return filtered_x, filtered_y