import os
import numpy as np
import cv2
from glob import glob


class MiniBatchLoader(object):

    def __init__(self, noise_path, gt_path, crop_size):
        # load data paths
        self.noise_path_infos = glob(noise_path + '*.png')
        self.gt_path_infos = glob(gt_path + '*.png')
        self.noise_path_infos.sort()
        self.gt_path_infos.sort()
        self.crop_size = crop_size

    def count_paths(self):
        return len(self.noise_path_infos)

    def load_training_data(self, indices):
        return self.load_data([self.noise_path_infos, self.gt_path_infos], indices)

    def load_data(self, path_infos, indices):
        mini_batch_size = len(indices)
        in_channels = 1
        xs = np.zeros((mini_batch_size, in_channels, self.crop_size, self.crop_size)).astype(np.float32)
        xs2 = np.zeros((mini_batch_size, in_channels, self.crop_size, self.crop_size)).astype(np.float32)

        for i, index in enumerate(indices):
            h, w = 0, 0
            noise_path = path_infos[0][index]
            gt_path = path_infos[1][index]

            img_noise = cv2.imread(noise_path, 0)
            img_gt = cv2.imread(gt_path, 0)

            if img_noise is None or img_gt is None:
                raise RuntimeError("invalid image: {i}".format(i=img_noise))
            if len(img_noise.shape) != 3 or len(img_gt.shape) != 3:
                pass

            try:
                h, w = img_noise.shape
            except ValueError:
                print(index)

            if np.random.rand() > 0.5:
                img_noise = np.fliplr(img_noise)
                img_gt = np.fliplr(img_gt)

            if np.random.rand() > 0.5:
                angle = 10 * np.random.rand()
                if np.random.rand() > 0.5:
                    angle *= -1
                M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
                img_noise = cv2.warpAffine(img_noise, M, (w, h))
                img_gt = cv2.warpAffine(img_gt, M, (w, h))

            rand_range_h = h - self.crop_size
            rand_range_w = w - self.crop_size
            x_offset = np.random.randint(rand_range_w)
            y_offset = np.random.randint(rand_range_h)

            img_noise = img_noise[y_offset:y_offset + self.crop_size, x_offset:x_offset + self.crop_size]
            img_gt = img_gt[y_offset:y_offset + self.crop_size, x_offset:x_offset + self.crop_size]

            img_noise = (img_noise / 255).astype(np.float32)
            img_gt = (img_gt / 255).astype(np.float32)

            xs2[i, 0, :, :] = img_noise.astype(np.float32)
            xs[i, 0, :, :] = img_gt.astype(np.float32)
        return xs, xs2

    def load_testing_data(self):
        imgs, gts = [], []
        for i in range(self.count_paths()):
            img = cv2.imread(self.noise_path_infos[i])
            img = (img / 255.0).astype(np.float32)
            img = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0)
            imgs.append(img)

            gt = cv2.imread(self.gt_path_infos[i])
            gt = (gt / 255.0).astype(np.float32)
            # gt = np.expand_dims(gt, axis=0)
            gts.append(gt)

        return gts, imgs

    def random_load_data(self, batch_size, img_size):
        total_data_size = self.count_paths()
        indexes = np.random.choice(total_data_size, batch_size)
        imgs = np.zeros((batch_size, 1, img_size, img_size)).astype(np.float32)
        gts = np.zeros((batch_size, 1, img_size, img_size)).astype(np.float32)
        for i in range(batch_size):
            img = cv2.imread(self.noise_path_infos[indexes[i]], 0)
            gt = cv2.imread(self.gt_path_infos[indexes[i]], 0)
            imgs[i, 0, :, :] = img
            gts[i, 0, :, :] = gt
        imgs = (imgs / 255).astype(np.float32)
        gts = (gts / 255).astype(np.float32)

        return gts, imgs