import os
import matplotlib as mpl
import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
if os.environ.get('DISPLAY', '') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
from utility_file.mini_batch_loader import *
from utility_file.MyFCN import *
import sys
import time
# from utility_file.State_Adjust_Pixel import State
from utility_file.State_BF import State
from sewar.full_ref import psnr, ssim, msssim
from utility_file.pixelwise_a3c import *
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="A2CBF TEST")
    parser.add_argument("--LEARNING_RATE", default=0.00001, type=int, help="learning rate")
    parser.add_argument("--TRAIN_BATCH_SIZE", default=64, type=int, help="batch size")
    parser.add_argument("--TEST_BATCH_SIZE", default=1, type=int, help="testing batch size")
    parser.add_argument("--GAMMA", default=0.95, type=int, help="discount factor")
    parser.add_argument("--GPU_ID", default=0, type=int, help="GPU")
    parser.add_argument("--N_ACTIONS", default=11, type=int, help="kinds of action")
    parser.add_argument("--CROP_SIZE", default=256, type=int, help="size of img")
    parser.add_argument("--EPISODE_LEN", default=40, type=int, help="step")
    parser.add_argument("--TESTING_DATA_PATH", default='./dataset/testsets/', type=str, help="path of val img")
    parser.add_argument("--TESTING_GT_PATH", default='./dataset/testsets_gt/', type=str, help="path of val gt")
    parser.add_argument("--LOAD_MODEL_PATH", default="./my_model/model.npz", type=str, help="path to load model")
    parser.add_argument("--SAVE_DIR", default="./result/", type=str, help="path to save result")

    return parser.parse_args()

# LOAD_MODEL_PATH = './my_model_v1/denoise_myfcn_107/model.npz'

def test(loader, agent, opt):
    os.makedirs(opt.SAVE_DIR, exist_ok=True)
    sum_psnr, sum_ssim, sum_msssim = 0, 0, 0
    test_data_size = loader.count_paths()
    gts, imgs = loader.load_testing_data()
    c = 1
    for raw_hr, raw_lr in zip(gts, imgs):
        print(c)
        w, h = raw_lr.shape[2], raw_lr.shape[3]
        output = np.zeros((w, h, raw_lr.shape[1]))
        tmp_img = [[], [], []]
        for channel in range(raw_lr.shape[1]):
            current_state = State((opt.TEST_BATCH_SIZE, 1, w, h), opt.N_ACTIONS)
            current_state.reset(raw_lr[:, channel: (channel + 1), :, :])
            # draw_pic(raw_hr, i, 'GT')
            for t in range(opt.EPISODE_LEN):
                action = agent.act(current_state.image)
                current_state.step(action)
                # action_map(action, t, channel)
                tmp_img[channel].append(current_state.image[0])
            output[:, :, channel] = current_state.image[0][0]
        psnr_value = compute_psnr(raw_hr, output)
        ssim_value = compute_ssim(raw_hr, output)
        msssim_value = compute_msssim(raw_hr, output)
        print(psnr_value, ssim_value, msssim_value)
        cv2.imwrite('%s/%04d.png' % (opt.SAVE_DIR, c), output * 255)
        sum_psnr += psnr_value
        sum_ssim += ssim_value
        sum_msssim += msssim_value
        agent.stop_episode()
        c += 1
    print("avg psnr: ", sum_psnr / test_data_size, ", avg ssim: ", sum_ssim / test_data_size, ", avg msssim: ", sum_msssim/ test_data_size)
    sys.stdout.flush()

def compute_psnr(GT, img, max_value=1.0):
    
    return psnr(GT, img, MAX=max_value)

def compute_ssim(GT, img, max_value=1.0):
    
    return ssim(GT, img, MAX=max_value)[0]

def compute_msssim(GT, img, max_value=1.0):
    
    return msssim(GT, img, MAX=max_value).real

def main(opt):
    mini_batch_loader = MiniBatchLoader(
        opt.TESTING_DATA_PATH,
        opt.TESTING_GT_PATH,
        opt.CROP_SIZE)

    my_gpu = chainer.cuda.get_device_from_id(opt.GPU_ID).use()
    # chainer.cuda.get_device_from_id(opt.GPU_ID).use()

    model = MyFcn(opt.N_ACTIONS, opt.TRAIN_BATCH_SIZE, opt.CROP_SIZE, my_gpu)
    optimizer = chainer.optimizers.Adam(alpha=opt.LEARNING_RATE)
    optimizer.setup(model)
    agent = PixelWiseA3C(my_gpu, model, optimizer, opt.EPISODE_LEN, opt.GAMMA)
    chainer.serializers.load_npz(opt.LOAD_MODEL_PATH, agent.model)
    agent.act_deterministically = True
    agent.model.to_gpu(device=my_gpu)
    test(mini_batch_loader, agent, opt)

def dle_file():
    paths = glob('./img_output02/fivek/*.png')
    paths.sort()
    c = 3751
    for p in paths:
        os.rename(p, './img_output02/fivek/%04d_PACBF.png' % c)
        c += 1

if __name__ == '__main__':
    opt = parse_args()
    main(opt)
