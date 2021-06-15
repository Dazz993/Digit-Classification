import numpy as np
import scipy.io as sio
from tqdm import tqdm

def _normalize_block(block, eps=1e-7):
    block = block / np.sqrt(np.sum(block ** 2) + eps ** 2)
    return block


def _gradient(images):
    row_gradient = np.zeros_like(images, dtype=float)
    col_gradient = np.zeros_like(images, dtype=float)

    row_gradient[:, :, :, 1:-1] = -images[:, :, :, :-2] + images[:, :, :, 2:]
    col_gradient[:, :, 1:-1, :] = -images[:, :, :-2, :] + images[:, :, 2:, :]

    return row_gradient, col_gradient


def _build_histogram_1(magnitude, orientation, num_orientations = 9, pixels_per_cell=(4, 4)):
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

    for orientation_id in range(num_orientations):
        orientation_start = deg_per_orientation * orientation_id
        orientation_end = deg_per_orientation * (orientation_id + 1)
        flag_1 = orientation_start <= orientation
        flag_2 = orientation < orientation_end
        flag = np.logical_and(flag_1, flag_2)
        tmp_bins[:, :, :, orientation_id] = np.where(flag, magnitude, 0)

    for cell_id_in_row in range(n_cells_row):
        row_range_start = cell_id_in_row * n_row_per_cell
        row_range_end = (cell_id_in_row + 1) * n_row_per_cell

        for cell_id_in_col in range(n_cells_col):
            col_range_start = cell_id_in_col * n_col_per_cell
            col_range_end = (cell_id_in_col + 1) * n_col_per_cell

            cell = np.sum(tmp_bins[:, row_range_start:row_range_end, col_range_start:col_range_end, :], axis=(1,2))
            orientation_bins[:, cell_id_in_row, cell_id_in_col, :] = cell / (n_row_per_cell * n_col_per_cell)

    return orientation_bins

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

    for image_id in tqdm(range(n_images)):
        for cell_id_in_row in range(n_cells_row):
            row_range_start = cell_id_in_row * n_row_per_cell
            row_range_end = (cell_id_in_row + 1) * n_row_per_cell

            for cell_id_in_col in range(n_cells_col):
                col_range_start = cell_id_in_col * n_col_per_cell
                col_range_end = (cell_id_in_col + 1) * n_col_per_cell

                for row in range(row_range_start, row_range_end):
                    for col in range(col_range_start, col_range_end):

                        for orientation_id in range(num_orientations):
                            orientation_start = deg_per_orientation * orientation_id
                            orientation_end = deg_per_orientation * (orientation_id + 1)
                            if orientation_start <= orientation[image_id, row, col] < orientation_end:
                                orientation_bins[image_id, cell_id_in_row, cell_id_in_col, orientation_id] += magnitude[image_id, row, col]

                orientation_bins[image_id, cell_id_in_row, cell_id_in_col, :] /= (n_row_per_cell * n_col_per_cell)

    return orientation_bins

if __name__ == '__main__':
    # parameters given
    path = '../../data/test_32x32.mat'
    images = sio.loadmat(path)['X'].astype(float)
    images = images.transpose(3, 2, 0, 1)
    images = images[:10]
    pixels_per_cell = (2, 2)
    cells_per_block = (1, 1)
    num_orientations = 9

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
    orientation_bins = _build_histogram(magnitude=magnitude, orientation=orientation, num_orientations=num_orientations, pixels_per_cell=pixels_per_cell)
    orientation_bins_1 = _build_histogram_1(magnitude=magnitude, orientation=orientation, num_orientations=num_orientations,
                                        pixels_per_cell=pixels_per_cell)

    print((np.max(abs(orientation_bins - orientation_bins_1))))

    # Step 3 and 4: Descriptor blocks and Block normalization
    print(f"==> Begin Step 3 & 4: constructing blocks and normalize them")
    n_images = images.shape[0]
    n_cells_row = images.shape[2] // pixels_per_cell[0]
    n_cells_col = images.shape[3] // pixels_per_cell[1]
    n_blocks_row = (n_cells_row - cells_per_block[0]) + 1
    n_blocks_col = (n_cells_col - cells_per_block[1]) + 1
    normalized_blocks = np.zeros((n_images, n_blocks_row, n_blocks_col, cells_per_block[0], cells_per_block[1], num_orientations), dtype=float)

    for image_id in range(n_images):
        for row in range(n_blocks_row):
            for col in range(n_blocks_col):
                block = orientation_bins[image_id, row : row + cells_per_block[0], col : col + cells_per_block[1], :]
                normalized_blocks[image_id, row, col, :] = _normalize_block(block)