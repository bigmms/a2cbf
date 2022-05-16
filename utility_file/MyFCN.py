import chainer
import chainer.links as L
import chainer.functions as F
from chainer import Chain
from chainer.links.caffe import CaffeFunction
import chainerrl
from chainerrl.agents import a3c
from transformer import get_encoder_decoder
from transformer.utils import subsequent_mask
import numpy as np
import cupy

class DilatedConvBlock(chainer.Chain):

    def __init__(self, d_factor, weight, bias, channel=64):
        super(DilatedConvBlock, self).__init__(
            diconv=L.DilatedConvolution2D(in_channels=channel, out_channels=64, ksize=3, stride=1, pad=d_factor,
                                          dilate=d_factor, nobias=False, initialW=weight, initial_bias=bias))
        self.train = True

    def __call__(self, x):
        h = F.relu(self.diconv(x))
        return h

class Transformer(Chain):
    """
        This class shows how the transformer could be used.
        The copy transformer used in our example consists of an encoder/decoder
        stack with a stack size of two. We use greedy decoding during test time,
        to get the predictions of the transformer.

        :param vocab_size: vocab_size determines the number of classes we want to distinguish.
        Since we only want to copy numbers, the vocab_size is the same for encoder and decoder.
        :param max_len: determines the maximum sequence length, since we have no end of sequence token.
        :param start_symbol: determines the begin of sequence token.
        :param transformer_size: determines the number of hidden units to be used in the transformer
    """

    def __init__(self, vocab_size, transformer_size=512):
        super().__init__()
        self.vocab_size = vocab_size
        self.transformer_size = transformer_size

        self.model = get_encoder_decoder(
            vocab_size,
            vocab_size,
            N=2,
            model_size=transformer_size,
        )
        self.mask = subsequent_mask(self.transformer_size)
        self.classifier = L.Linear(transformer_size, vocab_size)


    def __call__(self, x):
        result = self.model(x, self.mask)
        # result = self.model(x, t, None, self.mask)
        return self.classifier(result, n_batch_axes=2)

class MyFcn(chainer.Chain, a3c.A3CModel):

    def __init__(self, n_actions, batch, size, my_gpu):
        w = chainer.initializers.HeNormal()
        net = CaffeFunction('./utility_file/initial_weight/zhang_cvpr17_denoise_15_gray.caffemodel').to_gpu(my_gpu)
        super(MyFcn, self).__init__(
            conv1=L.Convolution2D(1, 64, 3, stride=1, pad=1, nobias=False, initialW=net.layer1.W.data, initial_bias=net.layer1.b.data),
            diconv2=DilatedConvBlock(2, net.layer3.W.data, net.layer3.b.data),
            diconv3=DilatedConvBlock(3, net.layer6.W.data, net.layer6.b.data),
            diconv4=DilatedConvBlock(4, net.layer9.W.data, net.layer9.b.data),
            diconv5_pi=DilatedConvBlock(3, net.layer12.W.data, net.layer12.b.data),
            diconv6_pi=DilatedConvBlock(2, net.layer15.W.data, net.layer15.b.data),
            conv7_pi=chainerrl.policies.SoftmaxPolicy(L.Convolution2D(64, n_actions, 3, stride=1, pad=1, nobias=False, initialW=w)),
            diconv5_V=DilatedConvBlock(3, net.layer12.W.data, net.layer12.b.data),
            diconv6_V=DilatedConvBlock(2, net.layer15.W.data, net.layer15.b.data),
            conv7_V=L.Convolution2D(64, 1, 3, stride=1, pad=1, nobias=False, initialW=net.layer18.W.data, initial_bias=net.layer18.b.data,),
            transformer=Transformer(vocab_size=(size * size), transformer_size=1024)
        )
        self.train = True
        self.batch = batch

    # def extract_patch(self, img_in):
    #     for i in


    def pi_and_v(self, x):
        # self.transformer(np.reshape(x, (self.batch, -1)))
        x1 = F.relu(self.conv1(x))  # H * W * 64
        x2 = self.diconv2(x1)  # H/2 * W/2 * 64
        x3 = self.diconv3(x2)  # H/4 * W/4 * 64
        x4 = self.diconv4(x3)  # H/8 * W/8 * 64
        # hidden_feature = self.transformer(F.reshape(x4, (self.batch, -1)))
        h_V = self.diconv5_V(x4)
        h_V = self.diconv6_V(h_V)
        vout = F.sigmoid(self.conv7_V(h_V))

        h_pi = self.diconv5_pi(x4)
        h_pi = self.diconv6_pi(h_pi)
        pout = self.conv7_pi(h_pi)
        return pout, vout