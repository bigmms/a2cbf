import numpy as np
import sys
import cv2
#from scipy.misc import imresize
#from BF import bilateral_filter_own

class State():
    def __init__(self, size, n_actions):
        self.size = size
        self.image = np.zeros(size, dtype=np.float32)
        self.n_actions = n_actions
        self.filter_diameter = 15
        self.sigma_s = 30
        # self.sigma_r = 10


    def reset(self, img):
        self.image = img
        self.sigma_r_map = (np.ones((self.size[0], self.size[2], self.size[3])) * 10).astype(np.float32)

    def step(self, act):
        actions = np.arange(-10, 11, 2, dtype=np.float32) / 10
        # actions = np.array([2, 1, 0.5, 0.2, 0.1, 0, -0.1, -0.2, -0.5, -1, -2]).astype(np.float32)
        pre_sigma_r_map = np.copy(self.sigma_r_map)
        sigma_r_map = np.copy(act)
        for i in range(self.n_actions):
            sigma_r_map = np.where(sigma_r_map == i, actions[i], sigma_r_map)
        self.sigma_r_map = (sigma_r_map + self.sigma_r_map).astype(np.float32)

        img_255 = np.copy(self.image) * 255
        for p in range(len(self.image)):
            for j in np.unique(self.sigma_r_map[p]):
                I = cv2.bilateralFilter(img_255[p, 0, :, :], self.filter_diameter, j, self.sigma_s)
                self.image[p, 0, :, :] = np.where(self.sigma_r_map[p] == j, I, img_255[p, 0, :, :])
        self.image = np.divide(self.image, 255, dtype=np.float32)
        # self.sigma_r_map = np.where(self.sigma_r_map < 0, pre_sigma_r_map, self.sigma_r_map)
        # self.sigma_r_map = np.where(self.sigma_r_map > 1, pre_sigma_r_map, self.sigma_r_map)