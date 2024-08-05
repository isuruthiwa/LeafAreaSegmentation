# helper_functions.py

import numpy as np
from matplotlib import pyplot as plt


def show_mask(image, mask, score, output_dir, file_name, random_color=False):
    plt.imshow(image)
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    plt.gca().imshow(mask_image)

    plt.title(f"Mask, Score: {score:.3f}", fontsize=18)

    # Save the plot as an image
    plt.savefig(f'{output_dir}/{file_name}', format='png', dpi=300)

    plt.show()
