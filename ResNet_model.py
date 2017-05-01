import sys
import os
import numpy as np
import PIL.Image as Image
import glob
import pickle as pkl
from lasagne.layers import Conv2DLayer as ConvLayer
#from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers import ElemwiseSumLayer
from lasagne.layers import InputLayer
from lasagne.layers import DenseLayer
from lasagne.layers import GlobalPoolLayer
from lasagne.layers import PadLayer
from lasagne.layers import ExpressionLayer
from lasagne.layers import NonlinearityLayer
from lasagne.nonlinearities import softmax, rectify
from lasagne.layers import batch_norm
from lasagne.layers import Pool2DLayer as PoolLayer
import time
import theano
import theano.tensor as T
import lasagne

def residual_block(l,increase_dim=False, projection=False):
        input_num_filters = l.output_shape[1]
        if increase_dim:
            first_stride = (2,2)
            out_num_filters = input_num_filters*2
        else:
            first_stride = (1,1)
            out_num_filters = input_num_filters

        stack_1 = batch_norm(ConvLayer(l, num_filters=out_num_filters, filter_size=(3,3), stride=first_stride, nonlinearity=rectify, pad='same'))
        stack_2 = batch_norm(ConvLayer(stack_1, num_filters=out_num_filters, filter_size=(3,3), stride=(1,1), nonlinearity=None, pad='same'))
        
        # add shortcut connections
        if increase_dim :
            if projection:
                # projection shortcut, as option B in paper
                projection = batch_norm(ConvLayer(l, num_filters=out_num_filters, filter_size=(1,1), stride=(2,2), nonlinearity=None, pad='same', b=None))
                block = NonlinearityLayer(ElemwiseSumLayer([stack_2, projection]),nonlinearity=rectify)
            else:
                # identity shortcut, as option A in paper
                identity = ExpressionLayer(l, lambda X: X[:, :, ::2, ::2], lambda s: (s[0], s[1], s[2]//2, s[3]//2))
                padding = PadLayer(identity, [out_num_filters//4,0,0], batch_ndim=1)
                block = NonlinearityLayer(ElemwiseSumLayer([stack_2, padding]),nonlinearity=rectify)
        else:
            block = NonlinearityLayer(ElemwiseSumLayer([stack_2, l]),nonlinearity=rectify)
        
        return block

def build_ResNet(input_var=None):
    
    l_in = InputLayer(shape=(None, 3, 64, 64), input_var=input_var)

    # first layer, output is 32 x 64 x 64
    l = batch_norm(ConvLayer(l_in, num_filters=32, filter_size=(3,3), stride=(1,1), nonlinearity=rectify, pad='same'))
    
    
    # first stack of residual blocks, output is 32 x 64 x 64
    for _ in range(5):
        l = residual_block(l)

    # second stack of residual blocks, output is 64 x 32 x 32
    l = residual_block(l, increase_dim=True)
    for _ in range(1,5):
        l = residual_block(l)

    # stack of simple blocks, output is 128 x 32 x 32
    for _ in range(1,5):
        l = batch_norm(ConvLayer(l, num_filters=128, filter_size=(3,3), stride=(1,1), nonlinearity=rectify, pad='same'))
    
    # stack of simple blocks, output is 32 x 32 x 64 
    for _ in range(1,5):
        l = batch_norm(ConvLayer(l, num_filters=64, filter_size=(3,3), stride=(1,1), nonlinearity=rectify, pad='same'))
     
 # stack of simple blocks, output is 32 x 32 x 32   
    for _ in range(1,5):
        l = batch_norm(ConvLayer(l, num_filters=32, filter_size=(3,3), stride=(1,1), nonlinearity=rectify, pad='same'))
        
#     last layer simple convolution, output is 3 x 32 x 32   
    l = ConvLayer(l, num_filters=3, filter_size=(3,3), stride=1, nonlinearity=None, pad='same')
        
        
    return l
