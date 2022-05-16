import os
import sys
import matplotlib as mpl
if os.environ.get('DISPLAY', '') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
from utility_file.mini_batch_loader import *
from utility_file.State_BF import State
from utility_file.MyFCN import *
from utility_file.pixelwise_a3c import *
from utility_file.utility import PICTURE, compute_reward, psnr_testing
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="A2CBF TRAIN")
    parser.add_argument("--LEARNING_RATE", default=0.00001, type=int, help="learning rate")
    parser.add_argument("--GAMMA", default=0.95, type=int, help="discount factor")
    parser.add_argument("--GPU_ID", default=0, type=int, help="GPU")
    parser.add_argument("--N_EPISODES", default=100, type=int, help="episodes")
    parser.add_argument("--N_ACTIONS", default=11, type=int, help="kinds of action")
    parser.add_argument("--CROP_SIZE", default=256, type=int, help="size of img")
    parser.add_argument("--TRAIN_BATCH_SIZE", default=1, type=int, help="batch size")
    parser.add_argument("--EPISODE_LEN", default=10, type=int, help="every EPISODE_LEN time updata model one time (step)")
    parser.add_argument("--TRAINING_NOISE_PATH", default='./dataset/trainsets/', type=str, help="path of training img")
    parser.add_argument("--TRAINING_GT_PATH", default='./dataset/trainsets_gt/', type=str, help="path of training gt")
    parser.add_argument("--SAVE_PATH", default="./my_model/a2cbf_", type=str, help="path to save model")
    parser.add_argument("--VALIDATION_NOISE_PATH", default='./dataset/valsets/', type=str, help="path of val img")
    parser.add_argument("--VALIDATION_GT_PATH", default='./dataset/valsets_gt/', type=str, help="path of val gt")
    return parser.parse_args()

def main(opt):
    my_gpu = chainer.backends.cuda.get_device_from_id(opt.GPU_ID)
    ''' model setup '''

    model = MyFcn(opt.N_ACTIONS, opt.TRAIN_BATCH_SIZE, opt.CROP_SIZE, my_gpu)
    optimizer = chainer.optimizers.Adam(alpha=opt.LEARNING_RATE)
    optimizer.setup(model)
    agent = PixelWiseA3C(my_gpu, model, optimizer, opt.EPISODE_LEN, opt.GAMMA)
    agent.model.to_gpu(device=my_gpu)
    # load pre-train model
    # chainer.serializers.load_npz('./my_model_v1/denoise_myfcn_168/optimizer.npz', optimizer)
    # chainer.serializers.load_npz('./my_model_v1/denoise_myfcn_168/model.npz', agent.model)
    ''' training data setup '''
    mini_batch_loader = MiniBatchLoader(
        opt.TRAINING_NOISE_PATH,
        opt.TRAINING_GT_PATH,
        opt.CROP_SIZE)
    train_data_size = mini_batch_loader.count_paths()
    indices = np.random.permutation(train_data_size)
    total_batch = train_data_size // opt.TRAIN_BATCH_SIZE
    ''' env setup '''
    current_state = State((opt.TRAIN_BATCH_SIZE, 1, opt.CROP_SIZE, opt.CROP_SIZE), opt.N_ACTIONS)

    for episode in range(1, opt.N_EPISODES + 1):
        print("episode %d" % episode)
        for i in range(total_batch):
            sys.stdout.flush()
            r = indices[i * opt.TRAIN_BATCH_SIZE:(i + 1) * opt.TRAIN_BATCH_SIZE]
            raw_gt, raw_noise = mini_batch_loader.load_training_data(r)
            current_state.reset(raw_noise)
            reward = np.zeros(raw_gt.shape, raw_gt.dtype)
            ori_psnr = psnr_testing(raw_gt, current_state.image, print_out=True)
            for t in range(opt.EPISODE_LEN):
                previous_image = current_state.image.copy()
                action = agent.act_and_train(current_state.image, reward)
                current_state.step(action)
                reward = compute_reward(raw_gt, previous_image, current_state.image)

            agent.stop_episode_and_train(current_state.image, reward, True)
            final_psnr = psnr_testing(raw_gt, current_state.image, print_out=True)
            print("[episode: %d / %d] [batch: %d / %d] [ori_psnr: %f] [final_psnr: %f]"
                  % (episode, opt.N_EPISODES, (i + 1), total_batch, ori_psnr, final_psnr))
            sys.stdout.flush()
            optimizer.alpha = opt.LEARNING_RATE * ((1 - episode / opt.N_EPISODES) ** 0.9)
            # if (i + 1) % 2 == 0:
        validation(episode, agent, opt)
        agent.save(opt.SAVE_PATH + "%04d" % episode)


def validation(episode, agent, opt):
    VAL_CROP_SIZE, VAL_BATCH_SIZE = 256, 1

    ''' vaild data setup '''
    mini_batch_loader = MiniBatchLoader(
        opt.VALIDATION_NOISE_PATH,
        opt.VALIDATION_GT_PATH,
        VAL_CROP_SIZE)

    '''env setup'''
    val_current_state = State((opt.TRAIN_BATCH_SIZE, 1, VAL_CROP_SIZE, VAL_CROP_SIZE), opt.N_ACTIONS)

    picture = PICTURE(VAL_BATCH_SIZE, opt.EPISODE_LEN)

    for i in range(VAL_BATCH_SIZE):
        steps, psnr_per_step = [], []
        val_gt, val_noise = mini_batch_loader.random_load_data(VAL_BATCH_SIZE, VAL_CROP_SIZE)
        val_current_state.reset(val_noise)

        picture.draw_pic(val_gt, episode, 0, isGT=True)
        picture.draw_pic(val_noise, episode, 0)

        # psnr_per_step.append(psnr_testing(val_gt, val_current_state.image))
        for t in range(opt.EPISODE_LEN):
            action = agent.act(val_current_state.image)
            val_current_state.step(action)
            # psnr_per_step.append(psnr_testing(val_gt, val_current_state.image))

            picture.action_map_1(action, episode, t)


        picture.draw_pic(val_current_state.image, episode, 'final')

if __name__ == '__main__':
    opt = parse_args()
    main(opt)