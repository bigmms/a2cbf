import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sewar.full_ref import psnr

def compute_reward(gt, previous_image, current_image):
    reward = np.square(gt - previous_image) * 255 - np.square(gt - current_image) * 255
    return reward.astype(np.float32)

def psnr_testing(GT, img, print_out=False):
    xs2 = np.zeros((GT.shape[0])).astype(np.float32)
    for i, (gt, img) in enumerate(zip(GT, img)):
        xs2[i] = psnr(gt[0], img[0], MAX=1.0)
    if not print_out:
        return xs2
    else:
        avg_psnr = sum(xs2) / len(xs2)
        return avg_psnr


class PICTURE():
    def __init__(self, version, TRAIN_BATCH_SIZE, EPISODE_LEN):
        self.version = version
        self.TRAIN_BATCH_SIZE = TRAIN_BATCH_SIZE
        self.EPISODE_LEN = EPISODE_LEN

    def draw_pic(self, img, e, step, isGT=False):
        for p in range(img.shape[0]):
            path = './valid/episode_%04d/' % e
            os.makedirs(path, exist_ok=True)
            I = img[p, 0, :, :] * 255
            if isGT:
                cv2.imwrite(path + '/GT.jpg', I)
            else:
                if type(step) == str:
                    cv2.imwrite(path + '/step_%s.jpg' % step, I)
                else:
                    cv2.imwrite(path + '/step_%02d.jpg' % step, I)

    def draw_psnr(self, data, e, batch, name):
        steps, psnrs = data
        # for i in range(TRAIN_BATCH_SIZE):
        #     plt.plot(steps, psnrs)
        #     plt.ylabel(name)
        #     plt.xlabel('step')
        #     plt.savefig(path + "/batch_%04d.jpg" % batch)
        #     plt.clf()
        for p in range(self.TRAIN_BATCH_SIZE):
            path = './pic_%s/episode_%04d/batch_%04d//patch_%04d/' % (self.version, e, batch, p)
            os.makedirs(path, exist_ok=True)
            BP = []
            for j in range((self.EPISODE_LEN + 1)):
                BP.append(psnrs[j][p])
            plt.plot(steps, BP, color='r')
            plt.ylabel(name)
            plt.xlabel('step')
            plt.savefig(path + "/patch_%04d_psnr.jpg" % p)
            plt.clf()

    def draw_rewardself(self, data, e, name):
        plt.plot(data[0], data[1], color='r')
        plt.ylabel(name)
        plt.xlabel('state')
        path = './pic_%s/episode_%04d/' % (self.version, e)
        # os.makedirs('./%s_graph_%s/' % (name, VERSION), exist_ok=True)
        plt.savefig(path + "/episode_%04d.jpg" % e)
        plt.clf()

    def action_map_3(self, img, step, ch):  # 3 channel
        path = './action_map/'
        os.makedirs(path, exist_ok=True)
        b, w, h = img.shape
        colors = [(148, 126, 148), (166, 3, 228), (240, 6, 155), (92, 165, 91), (131, 195, 68),
                  (93, 159, 207), (170, 32, 153), (143, 52, 75), (220, 169, 151), (118, 150, 146),
                  (0, 0, 255), (159, 121, 123), (117, 24, 237), (121, 93, 211), (212, 211, 24),
                  (57, 232, 238), (66, 235, 74), (184, 146, 44), (0, 125, 255)]
        for b in range(img.shape[0]):
            seg_img = np.zeros((w, h, 3))
            for c in range(len(colors)):
                seg_img[:, :, 0] += ((img[0, :, :] == c) * (colors[c][0]))
                seg_img[:, :, 1] += ((img[0, :, :] == c) * (colors[c][1]))
                seg_img[:, :, 2] += ((img[0, :, :] == c) * (colors[c][2]))
            cv2.imwrite(path + 'step_%02d_%d.jpg' % (step, ch), seg_img)

    def action_map_1(self, img, e, step):  # 1 channel
        path = './valid/episode_%04d/' % e
        os.makedirs(path, exist_ok=True)
        b, w, h = img.shape
        colors = [(148, 126, 148), (166, 3, 228), (240, 6, 155), (92, 165, 91), (131, 195, 68),
                  (93, 159, 207), (170, 32, 153), (143, 52, 75), (220, 169, 151), (118, 150, 146),
                  (0, 0, 255), (159, 121, 123), (117, 24, 237), (121, 93, 211), (212, 211, 24),
                  (57, 232, 238), (66, 235, 74), (184, 146, 44), (0, 125, 255)]

        for b in range(img.shape[0]):
            seg_img = np.zeros((w, h, 3))
            for c in range(len(colors)):
                seg_img[:, :, 0] += ((img[0, :, :] == c) * (colors[c][0]))
                seg_img[:, :, 1] += ((img[0, :, :] == c) * (colors[c][1]))
                seg_img[:, :, 2] += ((img[0, :, :] == c) * (colors[c][2]))
            cv2.imwrite(path + 'step_%02d.jpg' % step, seg_img)