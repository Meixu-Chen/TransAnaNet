import numpy as np
import scipy.ndimage as ndimage


def transform_matrix_offset_center_3d(matrix, x, y, z):
    offset_matrix = np.array([[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]])
    return ndimage.interpolation.affine_transform(matrix, offset_matrix)


def random_shift(img_numpy, max_percentage=0.2):
    dim1, dim2, dim3 = img_numpy.shape
    m1, m2, m3 = int(dim1 * max_percentage / 2), int(dim1 * max_percentage / 2), int(dim1 * max_percentage / 2)
    d1 = np.random.randint(-m1, m1)
    d2 = np.random.randint(-m2, m2)
    d3 = np.random.randint(-m3, m3)
    return transform_matrix_offset_center_3d(img_numpy, d1, d2, d3)


class RandomShift(object):
    def __init__(self, max_percentage=0.2):
        self.max_percentage = max_percentage

    def __call__(self, img_numpy, label=None):
        
        dim1, dim2, dim3, dim4 = img_numpy.shape
        m1, m2, m3 = 0, int(dim2 * self.max_percentage / 2), int(dim3 * self.max_percentage / 2)
        d1 = 0
        d2 = np.random.randint(-m2, m2)
        d3 = np.random.randint(-m3, m3)

        for i in range(dim4):
            img_numpy[:,:,:,i] = transform_matrix_offset_center_3d(img_numpy[:,:,:,i], d1, d2, d3)

        if label.any() != None:
            dim1, dim2, dim3, dim4 = label.shape
            for i in range(dim4):
                label[:,:,:,i] = transform_matrix_offset_center_3d(label[:,:,:,i], d1, d2, d3)
        return img_numpy, label
