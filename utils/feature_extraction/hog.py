import numpy as np
import scipy.io as sio
from tqdm import tqdm
from skimage import draw

def _normalize_block(block, eps=1e-7):
    block = block / np.sqrt(np.sum(block[:] ** 2, axis=(1, 2, 3)) + eps ** 2).reshape(block.shape[0], 1, 1, 1)
    return block


def _gradient(images):
    row_gradient = np.zeros_like(images, dtype=float)
    col_gradient = np.zeros_like(images, dtype=float)

    row_gradient[:, :, :, 1:-1] = -images[:, :, :, :-2] + images[:, :, :, 2:]
    col_gradient[:, :, 1:-1, :] = -images[:, :, :-2, :] + images[:, :, 2:, :]

    return row_gradient, col_gradient


def _build_histogram(magnitude, orientation, num_orientations = 9, pixels_per_cell=(4, 4)):
    assert pixels_per_cell[0] % 2 == 0 and pixels_per_cell[1] % 2 == 0

    n_images, n_row, n_col = magnitude.shape
    n_row_per_cell, n_col_per_cell = pixels_per_cell
    n_row -= n_row % n_row_per_cell
    n_col -= n_col % n_col_per_cell
    n_cells_row = n_row // n_row_per_cell
    n_cells_col = n_col // n_col_per_cell
    deg_per_orientation = 180.0 / num_orientations

    orientation_bins = np.zeros((n_images, n_cells_row, n_cells_col, num_orientations), dtype=float)
    tmp_bins = np.zeros((n_images, n_row, n_col, num_orientations), dtype=float)

    for orientation_id in tqdm(range(num_orientations)):
        orientation_start = deg_per_orientation * orientation_id
        orientation_end = deg_per_orientation * (orientation_id + 1)
        flag_1 = orientation_start <= orientation
        flag_2 = orientation < orientation_end
        flag = np.logical_and(flag_1, flag_2)
        tmp_bins[:, :, :, orientation_id] = np.where(flag, magnitude, 0)

    for cell_id_in_row in tqdm(range(n_cells_row)):
        row_range_start = cell_id_in_row * n_row_per_cell
        row_range_end = (cell_id_in_row + 1) * n_row_per_cell

        for cell_id_in_col in range(n_cells_col):
            col_range_start = cell_id_in_col * n_col_per_cell
            col_range_end = (cell_id_in_col + 1) * n_col_per_cell

            cell = np.sum(tmp_bins[:, row_range_start:row_range_end, col_range_start:col_range_end, :], axis=(1,2))
            orientation_bins[:, cell_id_in_row, cell_id_in_col, :] = cell / (n_row_per_cell * n_col_per_cell)

    return orientation_bins

def hog(images, num_orientations=9, pixels_per_cell=(4, 4), cells_per_block=(1, 1), flatten=True):
    assert images.ndim == 4

    print(f"==> Begin feature extraction using HOG")
    print(f"==> Parameters: num_orientations = {num_orientations}, pixels_per_cell = {pixels_per_cell}, cells_per_block = {cells_per_block}, flatten = {flatten}")

    # Step 1: Gradient computation
    # compute row and column gradients
    print(f"==> Begin Step 1: computing gradients")
    row_gradient, col_gradient = _gradient(images)

    # compute the magnitude of gradients
    magnitude = np.hypot(row_gradient, col_gradient)

    # select the channel with maximal gradient magitude
    max_channel = magnitude.argmax(axis=1)
    image_id, row_id, col_id = np.meshgrid(np.arange(images.shape[0]),
                                           np.arange(images.shape[2]),
                                           np.arange(images.shape[3]),
                                           indexing='ij',
                                           sparse=True)
    row_gradient = row_gradient[image_id, max_channel, row_id, col_id]
    col_gradient = col_gradient[image_id, max_channel, row_id, col_id]
    magnitude = magnitude[image_id, max_channel, row_id, col_id]

    # Step 2: Orientation binning
    print(f"==> Begin Step 2: building histogram of gradients")
    orientation = np.rad2deg(np.arctan2(row_gradient, col_gradient)) % 180
    orientation_bins = _build_histogram(magnitude=magnitude, orientation=orientation, num_orientations=num_orientations,
                                        pixels_per_cell=pixels_per_cell)

    # Step 3 and 4: Descriptor blocks and Block normalization
    print(f"==> Begin Step 3 & 4: constructing blocks and normalize them")
    n_images = images.shape[0]
    n_cells_row = images.shape[2] // pixels_per_cell[0]
    n_cells_col = images.shape[3] // pixels_per_cell[1]
    n_blocks_row = (n_cells_row - cells_per_block[0]) + 1
    n_blocks_col = (n_cells_col - cells_per_block[1]) + 1
    normalized_blocks = np.zeros(
        (n_images, n_blocks_row, n_blocks_col, cells_per_block[0], cells_per_block[1], num_orientations), dtype=float)

    for row in tqdm(range(n_blocks_row)):
        for col in range(n_blocks_col):
            block = orientation_bins[:, row: row + cells_per_block[0], col: col + cells_per_block[1], :]
            normalized_blocks[:, row, col, :] = _normalize_block(block)

    if flatten:
        normalized_blocks = normalized_blocks.reshape(normalized_blocks.shape[0], -1)

    print(f"==> End HOG feature extraction")

    return normalized_blocks

# Here the methods of visualizing hog is based on skimage
# The hog feature extraction method is implemented by myself
def visualize_hog(image, num_orientations=9, pixels_per_cell=(4, 4)):
    images = image.reshape(1, *image.shape) / 255

    row_gradient, col_gradient = _gradient(images)

    # compute the magnitude of gradients
    magnitude = np.hypot(row_gradient, col_gradient)

    # select the channel with maximal gradient magitude
    max_channel = magnitude.argmax(axis=1)
    image_id, row_id, col_id = np.meshgrid(np.arange(images.shape[0]),
                                           np.arange(images.shape[2]),
                                           np.arange(images.shape[3]),
                                           indexing='ij',
                                           sparse=True)
    row_gradient = row_gradient[image_id, max_channel, row_id, col_id]
    col_gradient = col_gradient[image_id, max_channel, row_id, col_id]
    magnitude = magnitude[image_id, max_channel, row_id, col_id]

    # Step 2: Orientation binning
    print(f"==> Begin Step 2: building histogram of gradients")
    orientation = np.rad2deg(np.arctan2(row_gradient, col_gradient)) % 180
    orientation_bins = _build_histogram(magnitude=magnitude, orientation=orientation, num_orientations=num_orientations,
                                        pixels_per_cell=pixels_per_cell)

    n_cells_row = images.shape[2] // pixels_per_cell[0]
    n_cells_col = images.shape[3] // pixels_per_cell[1]

    radius = min(pixels_per_cell[0], pixels_per_cell[1]) // 2 - 1
    orientations_arr = np.arange(num_orientations)
    # set dr_arr, dc_arr to correspond to midpoints of orientation bins
    orientation_bin_midpoints = (
            np.pi * (orientations_arr + .5) / num_orientations)
    dr_arr = radius * np.sin(orientation_bin_midpoints)
    dc_arr = radius * np.cos(orientation_bin_midpoints)
    hog_image = np.zeros((images.shape[2], images.shape[3]), dtype=float)
    for r in range(n_cells_row):
        for c in range(n_cells_col):
            for o, dr, dc in zip(orientations_arr, dr_arr, dc_arr):
                centre = tuple([r * pixels_per_cell[0] + pixels_per_cell[0] // 2,
                                c * pixels_per_cell[1] + pixels_per_cell[1] // 2])
                rr, cc = draw.line(int(centre[0] - dc),
                                   int(centre[1] + dr),
                                   int(centre[0] + dc),
                                   int(centre[1] - dr))
                hog_image[rr, cc] += orientation_bins[0, r, c, o]

    return hog_image

def visualize(image, hog_image):
    import matplotlib.pyplot as plt
    from skimage import exposure

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 4), sharex=True, sharey=True)

    ax1.axis('off')
    ax1.imshow(image.transpose(1, 2, 0) / 255)
    ax1.set_title('Original Image')

    # Rescale histogram for better display
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

    ax2.axis('off')
    ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    ax2.set_title('Visualization of HOG')
    plt.savefig('display1.png')


if __name__ == '__main__':
    # parameters given
    path = '../../data/train_32x32.mat'
    images = sio.loadmat(path)['X'].astype(float)
    images = images.transpose(3, 2, 0, 1)
    labels = sio.loadmat(path)['y']
    # images = images[:10]
    pixels_per_cell = (2, 2)
    cells_per_block = (1, 1)
    num_orientations = 9

    # feature = hog(images, num_orientations=num_orientations, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block, flatten=True)

    index = 2

    print(f"Index: {index}, label: {labels[index]}")
    image = images[index]
    hog_image = visualize_hog(image, num_orientations=num_orientations, pixels_per_cell=pixels_per_cell)
    visualize(image, hog_image)