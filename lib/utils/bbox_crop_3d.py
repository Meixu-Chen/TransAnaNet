import numpy as np

class bbox_Mask3DCrop(self, Image_NPArray, Mask_NPArray, Pad=True):
    # Will calculate the 3D coordinates of the minimum bounding box that would fit the Mask for the associated image
    # Will use the center of the Mask to create the desired 3D bounding box size
    # Will crop the Image and Mask arrays to fit the associated 3D bounding box size
    # INPUT COMMENTS:
        # Image_NPArray and Mask_NPArray:
            # These should both be Numpy arrays.
            # The Mask should have already been aligned to the Image prior to going into this function,
            # and the Mask and Image arrays should have the same shape.
            # If either is off, the cropping will most likely be off on the Image, and the function will return incorrect results.
        # z_bb, y_bb, and x_bb (these are included in the sets class, and can be called upon via self):
            # These should be integers (no decimals allowed).
            # These will be used as a symmetric expansion in the specified direction (x, y, z) when creating the desired bounding box size.
            # The shape of the returned array will be the Integer * 2 + 1.
            # For example, if z_bb = 10, then the shape for x will be 21
            # (10 padded on the left, 1 from the center, and 10 padded on the right, which when summed together give 21)
        # Pad
            # If Pad==True, then the image and mask will be padded by z_bb, y_bb, and x_bb by 0s
            # This will prevent the calculated ROI from having dimensions going outside of the numpy array
    # OUTPUT COMMENTS:
        # Image_NPArray:
            # Cropped from the center of the mask to contain the specified voxels in each direction (by z_bb, y_bb, and x_bb above)
            # See description of anticipated shape under input comments above.
        # Mask_NPArray:
            # Cropped from the center of the mask to contain the specified voxels in each direction (by z_bb, y_bb, and x_bb above)
            # See description of anticipated shape under input comments above.
    z_bb = self.z_bb
    y_bb = self.y_bb
    x_bb = self.x_bb
    if Pad == True:
        MaskPad = np.pad(Mask_NPArray, ((z_bb, z_bb), (y_bb, y_bb), (x_bb, x_bb)), 'constant')
        ImagePad = np.pad(Image_NPArray, ((z_bb, z_bb), (y_bb, y_bb), (x_bb, x_bb)), 'constant')
    else:
        MaskPad = Mask_NPArray
        ImagePad = Image_NPArray
    # Bounding box coordinates are created from the center of the Mask
    x = np.any(MaskPad, axis=(1, 2))
    y = np.any(MaskPad, axis=(0, 2))
    z = np.any(MaskPad, axis=(0, 1))
    zmin, zmax = np.where(x)[0][[0, -1]]
    ymin, ymax = np.where(y)[0][[0, -1]]
    xmin, xmax = np.where(z)[0][[0, -1]]
    z_center = (zmin + zmax) / 2
    y_center = (ymin + ymax) / 2
    x_center = (xmin + xmax) / 2
    z_bb_min = int(z_center - z_bb)
    z_bb_max = int(z_center + z_bb) + 1
    y_bb_min = int(y_center - y_bb)
    y_bb_max = int(y_center + y_bb) + 1
    x_bb_min = int(x_center - x_bb)
    x_bb_max = int(x_center + x_bb) + 1
    assert MaskPad.shape == ImagePad.shape, 'MaskPad shape is not equal to ImagePad shape in the bbox_Mask3DCrop function.'
    assert z_bb_max >= zmax, print('The calculated z_bb_max (' + str(
        z_bb_max) + ') does not cover the entire ROI, with the zmax boundary being ' + str(zmax) +
                                   '. The difference in size is: ' + str(abs(z_bb_max - zmax)) + '.')
    assert z_bb_min <= zmin, print('The calculated z_bb_min (' + str(
        z_bb_min) + ') does not cover the entire ROI, with the zmin boundary being ' + str(zmin) +
                                   '. The difference in size is: ' + str(abs(z_bb_min - zmin)) + '.')
    assert y_bb_max >= ymax, print('The calculated y_bb_max (' + str(
        y_bb_max) + ') does not cover the entire ROI, with the ymax boundary being ' + str(ymax) +
                                   '. The difference in size is: ' + str(abs(y_bb_max - ymax)) + '.')
    assert y_bb_min <= ymin, print('The calculated y_bb_min (' + str(
        y_bb_min) + ') does not cover the entire ROI, with the ymin boundary being ' + str(ymin) +
                                   '. The difference in size is: ' + str(abs(y_bb_min - ymin)) + '.')
    assert x_bb_max >= xmax, print('The calculated x_bb_max (' + str(
        x_bb_max) + ') does not cover the entire ROI, with the xmax boundary being ' + str(xmax) +
                                   '. The difference in size is: ' + str(abs(x_bb_max - xmax)) + '.')
    assert x_bb_min <= xmin, print('The calculated x_bb_min (' + str(
        x_bb_min) + ') does not cover the entire ROI, with the xmin boundary being ' + str(xmin) +
                                   '. The difference in size is: ' + str(abs(x_bb_min - xmin)) + '.')
    Mask_NPArray = MaskPad[z_bb_min:z_bb_max, y_bb_min:y_bb_max, x_bb_min:x_bb_max]
    Image_NPArray = ImagePad[z_bb_min:z_bb_max, y_bb_min:y_bb_max, x_bb_min:x_bb_max]
    return Image_NPArray, Mask_NPArray