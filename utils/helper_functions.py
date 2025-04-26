# helper_functions.py
import matplotlib
import numpy as np
from matplotlib import pyplot as plt, patches
matplotlib.use('agg')

def show_mask(image, plot, bounding_box, mask, score, output_dir, file_name, random_color=False):
    plt.imshow(image)
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    plt.gca().imshow(mask_image)

    rect = patches.Rectangle((bounding_box[0], bounding_box[1]),
                             (bounding_box[2]-bounding_box[0]),
                             (bounding_box[3]-bounding_box[1]),
                             linewidth=2,
                             edgecolor='r',
                             facecolor='none')

    # Add the rectangle to the plot
    plt.gca().add_patch(rect)

    # plt.title(f"Segmented Mask, Score: {score:.3f}", fontsize=18)

    # Save the plot as an image
    plt.savefig(f'{output_dir}/{file_name}', format='png', dpi=300, bbox_inches='tight')

    if plot:
        plt.show()

    plt.close()

def crop_image_from_bounding_box(mask, bounding_box):
    # Crop the image using the bounding box

    # Crop the mask using NumPy slicing
    cropped_mask = mask[int(bounding_box[1]):int(bounding_box[3]), int(bounding_box[0]):int(bounding_box[2])]

    # show_plot(cropped_mask)
    return cropped_mask


def show_plot(image, title):
    # Display the cropped mask
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.show()
