import numpy as np
from PIL import Image
from numba import jit
from tqdm import tqdm
from abc import abstractmethod, abstractstaticmethod
from os.path import basename
from typing import List
import functools


def NI_decor(fn):
    def wrap_fn(self, *args, **kwargs):
        try:
            return fn(self, *args, **kwargs)
        except NotImplementedError as e:
            print(e)
    return wrap_fn


class SeamImage:
    def __init__(self, img_path: str, vis_seams: bool = True):
        """ SeamImage initialization.

        Parameters:
            img_path (str): image local path
            vis_seams (bool): if true, another version of the original image shall be store, and removed seams should be marked on it
        """
        #################
        # Do not change #
        #################
        self.path = img_path

        self.gs_weights = np.array([[0.299, 0.587, 0.114]]).T

        self.rgb = self.load_image(img_path)
        self.resized_rgb = self.rgb.copy()

        self.vis_seams = vis_seams
        if vis_seams:
            self.seams_rgb = self.rgb.copy()

        self.h, self.w = self.rgb.shape[:2]

        try:
            self.gs = self.rgb_to_grayscale(self.rgb)
            self.resized_gs = self.gs.copy()
            self.cumm_mask = np.ones_like(self.gs, dtype=bool)
        except NotImplementedError as e:
            print(e)

        try:
            self.E = self.calc_gradient_magnitude()
        except NotImplementedError as e:
            print(e)
        ################

        # additional attributes you might find useful
        self.seam_history = []
        self.seam_balance = 0

        # This might serve you to keep tracking original pixel indices
        self.idx_map_h, self.idx_map_v = np.meshgrid(
            range(self.w), range(self.h))

    @NI_decor
    def rgb_to_grayscale(self, np_img):
        """ Converts a np RGB image into grayscale (using self.gs_weights).
        Parameters
            np_img : ndarray (float32) of shape (h, w, 3) 
        Returns:
            grayscale image (float32) of shape (h, w, 1)

        Guidelines & hints:
            Use NumpyPy vectorized matrix multiplication for high performance.
            To prevent outlier values in the boundaries, we recommend to pad them with 0.5
        """

        return np.dot(np_img, self.gs_weights)

    @NI_decor
    def calc_gradient_magnitude(self):
        """ Calculate gradient magnitude of a grayscale image

        Returns:
            A gradient magnitude image (float32) of shape (h, w)

        Guidelines & hints:
            - In order to calculate a gradient of a pixel, only its neighborhood is required.
            - keep in mind that values must be in range [0,1]
            - np.gradient or other off-the-shelf tools are NOT allowed, however feel free to compare yourself to them
        """
        # Using padding to maintain original dimensions
        padded = np.pad(self.resized_gs.squeeze(), pad_width=1, mode='edge')
            
        # Forward difference in x direction: I(x+1,y)−I(x,y)
        dx = padded[1:-1, 2:] - padded[1:-1, 1:-1]

        # Forward difference in y direction: I(x,y+1)−I(x,y)
        dy = padded[2:, 1:-1] - padded[1:-1, 1:-1]

        # Compute gradient magnitude as sqrt(dx^2 + dy^2)
        magnitude = np.sqrt(np.square(dx) + np.square(dy))

        # Normalize to range [0,1]
        max_val = np.max(magnitude)
        if max_val > 0:
            magnitude = magnitude / max_val

        return magnitude

    def update_ref_mat(self):
        for i, s in enumerate(self.seam_history[-1]):
            self.idx_map[i, s:] = np.roll(self.idx_map[i, s:], -1)

    def reinit(self):
        """
        Re-initiates instance and resets all variables.
        """
        self.__init__(img_path=self.path)

    @staticmethod
    def load_image(img_path, format='RGB'):
        return np.asarray(Image.open(img_path).convert(format)).astype('float32') / 255.0

    def paint_seams(self):
        for s in self.seam_history:
            for i, s_i in enumerate(s):
                self.cumm_mask[self.idx_map_v[i, s_i],
                               self.idx_map_h[i, s_i]] = False
        cumm_mask_rgb = np.stack([self.cumm_mask.squeeze()] * 3, axis=2)
        self.seams_rgb = np.where(cumm_mask_rgb, self.seams_rgb, [1, 0, 0])

    def seams_removal(self, num_remove: int):
        """ Iterates num_remove times and removes num_remove vertical seams

        Parameters:
            num_remove (int): number of vertical seams to be removed

        Guidelines & hints:
        As taught, the energy is calculated from top to bottom.
        You might find the function np.roll useful.

        This step can be divided into a couple of steps:
            i) init/update matrices (E, mask) where:
                - E is the gradient magnitude matrix
                - mask is a boolean matrix for removed seams
            iii) find the best seam to remove and store it
            iv) index update: when a seam is removed, index mapping should be updated in order to keep track indices for next iterations
            v) seam removal: create the carved image with the chosen seam (and update seam visualization if desired)
            Note: the flow described below is a recommendation. You may implement seams_removal as you wish, but it needs to support:
            - removing seams a couple of times (call the function more than once)
            - visualize the original image with removed seams marked in red (for comparison)
        """
        for _ in tqdm(range(num_remove)):
            self.E = self.calc_gradient_magnitude()
            self.mask = np.ones_like(self.E, dtype=bool)

            seam = self.find_minimal_seam()
            self.seam_history.append(seam)
            if self.vis_seams:
                self.update_ref_mat()
            self.remove_seam(seam)
            if self.vis_seams:
                self.paint_seams()

    @NI_decor
    def find_minimal_seam(self) -> List[int]:
        """
        Finds the seam with the minimal energy.
        Returns:
            The found seam, represented as a list of indexes
        """
        # h, w = self.h, self.w
        # E = self.gs.copy()
        # M = np.full((h + 2, w + 2), np.inf, dtype=float)
        # M[1:-1, 1:-1] = E  
        # C = np.zeros_like(M, dtype=int)
        # path = []
        
        # for i in range(2, h + 1):
        #     for j in range(1, w + 1):
        #         min_val = min(M[i - 1, j - 1], M[i - 1, j], M[i - 1, j + 1])
        #         M[i, j] = E[i - 1, j - 1] + min_val
                
        #         if min_val == M[i - 1, j - 1]:
        #             C[i, j] = j - 1
        #         elif min_val == M[i - 1, j]:
        #             C[i, j] = j
        #         else:
        #             C[i, j] = j + 1
            
        # C = C[1:-1, 1:-1] -1
        # M = M[1:-1, 1:-1]

        # min_idx = np.argmin(M[-1])

        # for i in range(h - 1, -1, -1):
        #     path.append(min_idx)
        #     min_idx = C[i, min_idx]

        # return path[::-1]
    

    @NI_decor
    def remove_seam(self, seam: List[int]):
        """ Removes a seam from self.rgb (you may create a resized version, like self.resized_rgb)

        Guidelines & hints:
        In order to apply the removal, you might want to extend the seam mask to support 3 channels (rgb) using:
        3d_mak = np.stack([1d_mask] * 3, axis=2)
        ...and then use it to create a resized version.
        
        :arg seam: The seam to remove
        """
        # Decrease width since we're removing a seam
        self.w -= 1
        
        # Create a mask to identify pixels to keep (True) and remove (False)
        self.mask = np.ones((self.h, self.w + 1), dtype=bool)
        
        # Vectorized approach to mark seam pixels for removal - much faster than looping
        self.mask[np.arange(self.h), seam] = False
        
        # Create 3D mask for RGB image
        mask_3D = np.stack([self.mask] * 3, axis=2)
        
        # Apply masks to resize the images
        self.resized_gs = self.resized_gs[self.mask].reshape(self.h, self.w, 1)
        self.resized_rgb = self.resized_rgb[mask_3D].reshape(self.h, self.w, 3)
        
        # Update energy matrix if necessary for the next iteration
        if hasattr(self, 'E') and self.E is not None:
            self.E = self.calc_gradient_magnitude()


    @NI_decor
    def rotate_mats(self, clockwise: bool):
        """
        Rotates the matrices either clockwise or counter-clockwise.
        """
        # For np.rot90, k=1 means 90 degrees counter-clockwise, k=-1 means 90 degrees clockwise
    # For np.rot90, k=1 means 90 degrees counter-clockwise, k=-1 means 90 degrees clockwise
        rotation = -1 if clockwise else 1
        
        # List of matrices to rotate with their axes parameters
        matrices_to_rotate = [
            ('gs', (0, 1)),
            ('rgb', (0, 1)),
            ('resized_rgb', (0, 1)),
            ('resized_gs', (0, 1)),
            ('cumm_mask', (0, 1)),
            ('E', (0, 1)),
            ('idx_map_h', None),
            ('idx_map_v', None),
            ('seams_rgb', (0, 1))
        ]
        
        # Rotate each matrix if it exists
        for attr_name, axes in matrices_to_rotate:
            if hasattr(self, attr_name) and getattr(self, attr_name) is not None:
                attr = getattr(self, attr_name)
                if axes is not None:
                    setattr(self, attr_name, np.rot90(attr, k=rotation, axes=axes))
                else:
                    setattr(self, attr_name, np.rot90(attr, k=rotation))
        
        # Swap height and width
        self.h, self.w = self.w, self.h
        
        # Swap horizontal and vertical index maps
        if hasattr(self, 'idx_map_h') and hasattr(self, 'idx_map_v'):
            self.idx_map_h, self.idx_map_v = self.idx_map_v, self.idx_map_h



    @NI_decor
    def seams_removal_vertical(self, num_remove: int):
        """ A wrapper for removing num_remove horizontal seams (just a recommendation)

        Parameters:
            num_remove (int): umber of vertical seam to be removed
        """

        self.idx_map = self.idx_map_h
        self.seams_removal(num_remove)
        self.seam_history.clear()

    @NI_decor
    def seams_removal_horizontal(self, num_remove: int):
        """ Removes num_remove horizontal seams by rotating the image, removing vertical seams, and restoring the original rotation.

        Parameters:
            num_remove (int): number of horizontal seam to be removed
        """
        self.rotate_mats(True)
        self.seams_removal_vertical(num_remove)
        self.rotate_mats(False)
        

    """
    BONUS SECTION
    """

    @NI_decor
    def seams_addition(self, num_add: int):
        """ BONUS: adds num_add seams to the image

            Parameters:
                num_add (int): number of horizontal seam to be removed

            Guidelines & hints:
            - This method should be similar to removal
            - You may use the wrapper functions below (to support both vertical and horizontal addition of seams)
            - Visualization: paint the added seams in green (0,255,0)

        """
        raise NotImplementedError(
            "TODO (Bonus): Implement SeamImage.seams_addition")

    @NI_decor
    def seams_addition_horizontal(self, num_add: int):
        """ A wrapper for removing num_add horizontal seams (just a recommendation)

        Parameters:
            num_add (int): number of horizontal seam to be added

        Guidelines & hints:
            You may find np.rot90 function useful

        """
        raise NotImplementedError(
            "TODO (Bonus): Implement SeamImage.seams_addition_horizontal")

    @NI_decor
    def seams_addition_vertical(self, num_add: int):
        """ A wrapper for removing num_add vertical seams (just a recommendation)

        Parameters:
            num_add (int): number of vertical seam to be added
        """

        raise NotImplementedError(
            "TODO (Bonus): Implement SeamImage.seams_addition_vertical")


class GreedySeamImage(SeamImage):
    """Implementation of the Seam Carving algorithm using a greedy approach"""
    @NI_decor
    def find_minimal_seam(self) -> List[int]:
        """
        Finds the minimal seam by using a greedy algorithm.

        Guidelines & hints:
        The first pixel of the seam should be the pixel with the lowest cost.
        Every row chooses the next pixel based on which neighbor has the lowest cost.
        """
        seam = np.zeros(self.h, dtype=int)
        seam[0] = np.argmin(self.E[0])

        for r in range(1,self.h):
            start_idx = max(0, seam[r-1] - 1)
            finish_idx = min(self.w, seam[r - 1] + 2)

            local_min = np.argmin(self.E[r, start_idx:finish_idx])

            seam[r] = start_idx + local_min
        return seam




class DPSeamImage(SeamImage):
    """
    Implementation of the Seam Carving algorithm using dynamic programming (DP).
    """

    def __init__(self, *args, **kwargs):
        """ DPSeamImage initialization.
        """
        super().__init__(*args, **kwargs)
        try:
            self.M = self.calc_M()
        except NotImplementedError as e:
            print(e)

    @NI_decor
    def find_minimal_seam(self) -> List[int]:
        """
        Finds the minimal seam by using dynamic programming.

        Guidelines & hints:
        As taught, the energy is calculated from top to bottom.
        You might find the function np.roll useful.

        This step can be divided into a couple of steps:
            i) init/update matrices (M, backtracking matrix) where:
                - M is the cost matrix
                - backtracking matrix is an idx matrix used to track the minimum seam from bottom up
            ii) fill in the backtrack matrix corresponding to M
            iii) seam backtracking: calculates the actual indices of the seam
        """
        raise NotImplementedError(
            "TODO: implement DPSeamImage.find_minimal_seam")

    @NI_decor
    def calc_M(self):
        """ Calculates the matrix M discussed in lecture (with forward-looking cost)

        Returns:
            An energy matrix M (float32) of shape (h, w)

        Guidelines & hints:
            As taught, the energy is calculated from top to bottom.
            You might find the function 'np.roll' useful.
        """
        raise NotImplementedError("TODO: Implement DPSeamImage.calc_M")

    def init_mats(self):
        self.M = self.calc_M()
        self.backtrack_mat = np.zeros_like(self.M, dtype=int)

    @staticmethod
    @jit(nopython=True)
    def calc_bt_mat(M, E, GS, backtrack_mat):
        """ Fills the BT back-tracking index matrix. This function is static in order to support Numba. To use it, uncomment the decorator above.

        Recommended parameters (member of the class, to be filled):
            M: np.ndarray (float32) of shape (h,w)
            E: np.ndarray (float32) of shape (h,w)
            GS: np.ndarray (float32) of shape (h,w)
            backtrack_mat: np.ndarray (int32) of shape (h,w): to be filled here

        Guidelines & hints:
            np.ndarray is a reference type. Changing it here may affect it on the outside.
        """
        raise NotImplementedError("TODO: Implement DPSeamImage.calc_bt_mat")
        h, w = M.shape


def scale_to_shape(orig_shape: np.ndarray, scale_factors: list):
    """ Converts scale into shape

    Parameters:
        orig_shape (np.ndarray): original shape [y,x]
        scale_factors (list): scale factors for y,x respectively

    Returns
        the new shape
    """
    raise NotImplementedError("TODO: Implement scale_to_shape")


def resize_seam_carving(seam_img: SeamImage, shapes: np.ndarray):
    """ Resizes an image using Seam Carving algorithm

    Parameters:
        seam_img (SeamImage) The SeamImage instance to resize
        shapes (np.ndarray): desired shape (y,x)

    Returns
        the resized rgb image
    """
    raise NotImplementedError("TODO: Implement resize_seam_carving")


def bilinear(image, new_shape):
    """
    Resizes an image to new shape using bilinear interpolation method
    :param image: The original image
    :param new_shape: a (height, width) tuple which is the new shape
    :returns: the image resized to new_shape
    """
    in_height, in_width, _ = image.shape
    out_height, out_width = new_shape
    new_image = np.zeros(new_shape)
    ### Your code here###

    def get_scaled_param(org, size_in, size_out):
        scaled_org = (org * size_in) / size_out
        scaled_org = min(scaled_org, size_in - 1)
        return scaled_org
    scaled_x_grid = [get_scaled_param(x, in_width, out_width)
                     for x in range(out_width)]
    scaled_y_grid = [get_scaled_param(
        y, in_height, out_height) for y in range(out_height)]
    x1s = np.array(scaled_x_grid, dtype=int)
    y1s = np.array(scaled_y_grid, dtype=int)
    x2s = np.array(scaled_x_grid, dtype=int) + 1
    x2s[x2s > in_width - 1] = in_width - 1
    y2s = np.array(scaled_y_grid, dtype=int) + 1
    y2s[y2s > in_height - 1] = in_height - 1
    dx = np.reshape(scaled_x_grid - x1s, (out_width, 1))
    dy = np.reshape(scaled_y_grid - y1s, (out_height, 1))
    c1 = np.reshape(image[y1s][:, x1s] * dx + (1 - dx) *
                    image[y1s][:, x2s], (out_width, out_height, 3))
    c2 = np.reshape(image[y2s][:, x1s] * dx + (1 - dx) *
                    image[y2s][:, x2s], (out_width, out_height, 3))
    new_image = np.reshape(c1 * dy + (1 - dy) * c2,
                           (out_height, out_width, 3)).astype(int)
    return new_image
