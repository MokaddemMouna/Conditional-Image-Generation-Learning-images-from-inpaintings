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

theano.config.floatX = 'float32'

def generate_and_show_sample(fn, nb, seed=1993):

    it = Iterator(img_path="val2014")
    choice = range(len(it))
    
    if seed > 0:
        np.random.seed(seed)
        np.random.shuffle(choice)

    choice = choice[:nb]

#     try:
    xs, ys, cs = zip(*[it[i] for i in choice])
    loss, preds = fn(xs, ys)
    
    show_sample(xs,ys,preds,nb)

#     except:
#         print "Oups!"
        
        
def show_sample(xs, ys, preds, nb):
    

    for i in range(nb):
	#print nb
	#print i
        img_true = np.copy(xs[i])
        center = (int(np.floor(img_true.shape[0] / 2.)), int(np.floor(img_true.shape[1] / 2.)))

        img_true[center[0] - 16:center[0] + 16, center[1] - 16:center[1] + 16, :] = ys[i]

#         plt.subplot(2, nb, i+1)
#         plt.imshow(img_true)
        img_true = (img_true*255).astype('uint8')
        Image.fromarray(img_true).show()

        img_pred = np.copy(xs[i])
        img_pred[center[0] - 16:center[0] + 16, center[1] - 16:center[1] + 16, :] = preds[i]
#         plt.subplot(2, nb, nb+i+1)
#         plt.imshow(img_pred)
        img_pred = (img_pred*255).astype('uint8')
        Image.fromarray(img_pred).show()
        
        print "Saving images..."
        Image.fromarray(img_true).save("/u/mokaddem/IFT6266/test_images/img_true"+str(i)+".png","PNG")
        Image.fromarray(img_pred).save("/u/mokaddem/IFT6266/test_images/img_pred"+str(i)+".png","PNG")

        

#     plt.show()
