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

def x_y_cap(data):

        xs, ys, caps = zip(*[x for x in data if x is not None])
        return np.array(xs), np.array(ys), caps

def generate_and_show_sample(fn, nb=1):

    it = Iterator(img_path="val2014",load_caption=True)
    
    x,y,caps = x_y_cap(it)
    
    all_caps = [item for sublist in caps for item in sublist]
    
    g_batch_idx = np.random.choice(len(all_caps), nb, replace=False)
    random_caps = [all_caps[idx] for idx in g_batch_idx]
    g_capts_batch = captions_to_embedded_matrix(embedding_model, random_caps)
    
    choice = range(len(it))
    
    np.random.seed(1993)
    np.random.shuffle(choice)

    choice = choice[:nb]

#     try:
    xs, ys, cs = zip(*[it[i] for i in choice])
    for i in range(nb):
        caption = np.random.choice(g_capts_batch, size=(1, 300), replace=False)
        noise = np.random.uniform(-1., 1., size=(1, 100))
        preds = fn(noise,caption)
        show_sample(xs[i],ys[i],preds,i,nb)

#     except:
#         print "Oups!"
               

def show_sample(xs, ys, preds,seed, nb=1):
    

    img_true = np.copy(xs)
    center = (int(np.floor(img_true.shape[0] / 2.)), int(np.floor(img_true.shape[1] / 2.)))

    img_true[center[0] - 16:center[0] + 16, center[1] - 16:center[1] + 16, :] = ys

#         plt.subplot(2, nb, i+1)
#         plt.imshow(img_true)
    img_true = (img_true*255).astype('uint8')
#     Image.fromarray(img_true).show()

    img_pred = np.copy(xs)
    img_pred[center[0] - 16:center[0] + 16, center[1] - 16:center[1] + 16, :] = preds
#         plt.subplot(2, nb, nb+i+1)
#         plt.imshow(img_pred)
    img_pred = (img_pred*255).astype('uint8')
#     Image.fromarray(img_pred).show()


    Image.fromarray(img_true).save("/home/mouna/Documents/Project/test_images/img_true"+str(seed)+".jpg","JPEG")
    Image.fromarray(img_pred).save("/home/mouna/Documents/Project/test_images/img_pred"+str(seed)+".jpg","JPEG")

#     plt.show()

def init_google_word2vec_model():
    """
    Copy locally the Google model and load it
    Returns the model
    """

    src_dir = '/home/mouna/Documents/Project/'

    model_name = 'GoogleNews-vectors-negative300.bin.gz'


    print 'Loading Word2Vec model...'
    model = gensim.models.KeyedVectors.load_word2vec_format(src_dir + model_name, binary=True)
    print 'Loading completed.'

    return model

def captions_to_embedded_matrix(embedding_model, captions_dict):
    """
    Converts captions into embedded matrix using a gensim KeyedVector model
    Returns a matrix of size (batch_size, embedded vector size)
    """

    # create empty matrix that will store the real valued
    embedded_matrix = np.empty(shape=(len(captions_dict), 300), dtype=theano.config.floatX)


    for i, capt in enumerate(captions_dict):

        # combine all sentences for a given image into one string, lowercase and then split
        words = " ".join(capt).lower().split()

        # filter the words in the caption based on vocab of model
        filtrd_words = filter(lambda x: x in embedding_model.vocab, words)

        # get embedding vector
        vector = embedding_model[filtrd_words]

        # average over all words and store in matrix
        embedded_matrix[i] = np.average(vector, axis=0)

    if embedded_matrix.ndim == 1:
        embedded_matrix = np.expand_dims(embedded_matrix, axis=0)

    return embedded_matrix
