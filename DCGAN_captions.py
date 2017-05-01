def build_generator(input_var=None, embedding_var=None):
    

    noise_layer = InputLayer(shape=(None, 100),
                                        input_var=input_var)
    
    embedding_layer = InputLayer(shape=(None, 300), input_var=embedding_var)
    
    network = ConcatLayer([noise_layer, embedding_layer], axis=1)
    

    
    network = BatchNormLayer(DenseLayer(network,4*4*64*8, nonlinearity=rectify))
    
    
    network = ReshapeLayer(network, shape=(-1, 64*8, 4, 4))
    
    
    network = TransposedConv2DLayer(network, num_filters=64*4, filter_size=(5, 5),
                                                   stride=(2, 2), crop=2, nonlinearity= rectify,output_size = 8)

    network = BatchNormLayer(network)
    
    network = TransposedConv2DLayer(network, num_filters=64*2,
              filter_size=(5, 5), stride=(2, 2), crop=2, nonlinearity= rectify,output_size = 16)

    network = BatchNormLayer(network)
    network = TransposedConv2DLayer(network, num_filters=64,
              filter_size=(5, 5), stride=(2, 2), crop=2, nonlinearity= rectify,output_size = 32)

    network = TransposedConv2DLayer(network, num_filters=3,
                                                   filter_size=(5, 5), stride=(2, 2), crop=2,
                                                   nonlinearity=sigmoid,output_size = 32)
    
#     print lasagne.layers.get_output_shape(network, input_shapes=None)
    
    return network


def build_discriminator(input_var=None):
    

    lrelu = LeakyRectify(0.2)

    network = InputLayer(shape=(None, 3, 32, 32),
                                        input_var=input_var)

    network = Conv2DLayer(network, num_filters=128/4, filter_size=(5, 5),
                                         stride=2, pad=2, nonlinearity=lrelu)
    network = BatchNormLayer(network)

    network = Conv2DLayer(network, num_filters=128/2, filter_size=(5, 5),
                                         stride=2, pad=2, nonlinearity=lrelu)
    network = BatchNormLayer(network)

    network = Conv2DLayer(network, num_filters=128, filter_size=(5, 5),
                                         stride=2, pad=2, nonlinearity=lrelu)
    network = BatchNormLayer(network)


    network = FlattenLayer(network)
    network = DenseLayer(network, 1, nonlinearity=lasagne.nonlinearities.sigmoid)
    
#     print lasagne.layers.get_output_shape(network, input_shapes=None)

    return network
    
    def load_dataset(batch_size=128):

    train_iter = Iterator(nb_sub=1280,batch_size=batch_size, img_path = 'train2014', extract_center=True,load_caption=True)
    val_iter = Iterator(nb_sub=1280, batch_size=batch_size, img_path = 'val2014',extract_center=True,load_caption=True)

    return train_iter, val_iter


# ############################## Main program ################################
# Everything else will be handled in our main program now. We could pull out
# more functions to better separate the code, but it wouldn't make it any
# easier to read.

def train(num_epochs=2, TRAIN_STEPS_GEN = 10, TRAIN_STEPS_DISCR = 15,
          lr=0.0002, example=19, save_freq=100,
          batch_size=128, verbose_freq=100,
          model_file="/home/mouna/Documents/Project/test.npz",
          reload=False,
          **kwargs):

    # Load the dataset
    print "Loading data..."
    train_iter, val_iter = load_dataset(batch_size)
    
    
    x,y,capts = x_y_cap(train_iter)
    
    all_caps = [item for sublist in capts for item in sublist]
    

    #some monitoring stuff
    val_loss = []
    train_loss = []

    # Prepare Theano variables for inputs and targets
    noise_var = T.matrix('noise',dtype=theano.config.floatX)
    input = T.tensor4('real_img',dtype=theano.config.floatX)
    inpt_embd = T.matrix('inpt_embedding',dtype=theano.config.floatX)
#     target = T.tensor4('targets',dtype='float32')

    input_var = input.transpose((0, 3, 1, 2))
    

    generator = build_generator(noise_var,inpt_embd)
    discriminator = build_discriminator(input_var)
    
    sample = lasagne.layers.get_output(generator)
    
    # Create expression for passing real data through the discriminator
    real_out = lasagne.layers.get_output(discriminator)
    # Create expression for passing fake data through the discriminator
    fake_out = lasagne.layers.get_output(discriminator,
            sample)
    
    # Create loss expressions
    generator_loss = - T.mean(T.log(fake_out))
    discriminator_loss = - T.mean(T.log(real_out)) - T.mean(T.log(1 - fake_out))
    
    # Create update expressions for training
    generator_params = lasagne.layers.get_all_params(generator, trainable=True)
    
    discriminator_params = lasagne.layers.get_all_params(discriminator, trainable=True)
    
    generator_updates = lasagne.updates.adam(generator_loss, generator_params, learning_rate=0.0005, beta1=0.6)
    
    discriminator_updates = lasagne.updates.adam(discriminator_loss, discriminator_params, learning_rate=0.001, beta1=0.9)
    
    print "Computing the functions..."


    train_generator_fn = None
    train_discriminator_fn = None
    
    train_generator_fn = theano.function([noise_var,inpt_embd], generator_loss,
                                              updates=generator_updates, allow_input_downcast=True)

    train_discriminator_fn = theano.function([input,noise_var,inpt_embd], discriminator_loss,
                                              updates=discriminator_updates, allow_input_downcast=True)

    # Compile a second function computing the validation loss and accuracy:
    generate_sample_fn = theano.function([noise_var,inpt_embd], sample.transpose((0, 2, 3, 1)), allow_input_downcast=True)
    
    t = time.time()
    embedding_model = init_google_word2vec_model()
    print 'Embedding model was loaded in %s secs' % np.round(time.time() - t, 0)

    # Finally, launch the training loop.
    print "Starting training..."
    # We iterate over epochs:
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        print epoch
        train_batches = 0
        d_train_step = 0
        disc_loss = 0
        gen_loss = 0
        start_time = time.time()
        for i, batch in enumerate(train_iter):
            t_batch = time.time()
            d_train_step += 1
            inputs, targets, caps = batch
            # generate batch of uniform samples
            rdm_d = np.random.uniform(-1., 1., size=(len(inputs), 100))
            # generate embeddings for the batch
            print "mouna"
            d_capts_batch = captions_to_embedded_matrix(embedding_model, caps)
            print "mouna"
            disc_loss = train_discriminator_fn(targets,rdm_d,d_capts_batch)
            print '- train discr batch %s, loss %s in %s sec' % (num_batch, np.round(disc_loss, 4),
                                                                         np.round(time.time() - t_batch, 2))
            if d_train_step > = TRAIN_STEPS_DISCR:
                # reset discriminator step counter
                d_train_step = 0
                # train the generator for given number of steps
                for _ in xrange(TRAIN_STEPS_GEN):
                    rdm_g = np.random.uniform(-1., 1., size=(batch_size, 100))
                    g_batch_idx = np.random.choice(len(all_caps), batch_size, replace=False)
                    random_caps = [all_caps[idx] for idx in g_batch_idx]
                    g_capts_batch = captions_to_embedded_matrix(embedding_model, random_caps)
                    gen_loss = train_generator_fn(rdm_g,g_capts_batch)
                    print '- train gen step %s, loss %s' % (_ + 1, np.round(gen_loss, 4))
            train_batches += 1


            
#         print "batch {} of epoch {} of {} took {:.3f}s".format(i, epoch + 1, num_epochs, time.time() - start_time)
#         print "  training generator loss", gen_loss
#         print "  training discriminator loss" , disc_loss
            
    
    # Generate some samples 
        if epoch == 1:
            generate_and_show_sample(generate_sample_fn, nb=example)


if __name__ == '__main__':
    train()
        
