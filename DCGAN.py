def build_generator(input_var=None):
    

    network = InputLayer(shape=(None, 100),
                                        input_var=input_var)

    network = ReshapeLayer(network, (-1, 100, 1, 1))
    network = BatchNormLayer(network)
    network = TransposedConv2DLayer(network, num_filters=1024, filter_size=(4, 4),
                                                   stride=(1, 1), nonlinearity= rectify)

    network = BatchNormLayer(network)
    network = TransposedConv2DLayer(network, num_filters=512,
              filter_size=(5, 5), stride=(2, 2), crop=2, nonlinearity= rectify,output_size = 8)

    network = BatchNormLayer(network)
    network = TransposedConv2DLayer(network, num_filters=256,
              filter_size=(5, 5), stride=(2, 2), crop=2, nonlinearity= rectify,output_size = 16)

    network = TransposedConv2DLayer(network, num_filters=3,
                                                   filter_size=(5, 5), stride=(2, 2), crop=2,
                                                   nonlinearity=sigmoid,output_size = 32)
    
    return network


def build_discriminator(input_var=None):
    

    lrelu = LeakyRectify(0.2)

    network = InputLayer(shape=(None, 3, 32, 32),
                                        input_var=input_var)

    network = Conv2DLayer(network, num_filters=1024/4, filter_size=(5, 5),
                                         stride=2, pad=2, nonlinearity=lrelu)
    network = BatchNormLayer(network)

    network = Conv2DLayer(network, num_filters=1024/2, filter_size=(5, 5),
                                         stride=2, pad=2, nonlinearity=lrelu)
    network = BatchNormLayer(network)

    network = Conv2DLayer(network, num_filters=1024, filter_size=(5, 5),
                                         stride=2, pad=2, nonlinearity=lrelu)
    network = BatchNormLayer(network)


    network = FlattenLayer(network)
    network = DenseLayer(network, 1, nonlinearity=lasagne.nonlinearities.sigmoid)

    return network
    
    def load_dataset(batch_size=128):

    train_iter = Iterator(nb_sub=1280,batch_size=batch_size, img_path = 'train2014', extract_center=True)
    val_iter = Iterator(nb_sub=1280, batch_size=batch_size, img_path = 'val2014',extract_center=True)

    return train_iter, val_iter


# ############################## Main program ################################
# Everything else will be handled in our main program now. We could pull out
# more functions to better separate the code, but it wouldn't make it any
# easier to read.

def train(num_epochs=2,
          lr=0.0002, example=19, save_freq=100,
          batch_size=128, verbose_freq=100,
          model_file="/home/mouna/Documents/Project/test.npz",
          reload=False,
          **kwargs):

    # Load the dataset
    print "Loading data..."
    train_iter, val_iter = load_dataset(batch_size)

    #some monitoring stuff
    val_loss = []
    train_loss = []

    # Prepare Theano variables for inputs and targets
    noise_var = T.matrix('noise',dtype='float32')
    input = T.tensor4('real_img',dtype='float32')
#     target = T.tensor4('targets',dtype='float32')

    input_var = input.transpose((0, 3, 1, 2))

    generator = build_generator(noise_var)
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
    
    generator_updates = lasagne.updates.adam(generator_loss, generator_params, learning_rate=lr, beta1=0.5)
    
    discriminator_updates = lasagne.updates.adam(discriminator_loss, discriminator_params, learning_rate=lr, beta1=0.5)
    
    print "Computing the functions..."


    train_generator_fn = None
    train_discriminator_fn = None
    
    train_generator_fn = theano.function([noise_var], generator_loss,
                                              updates=generator_updates, allow_input_downcast=True)

    train_discriminator_fn = theano.function([noise_var,input], discriminator_loss,
                                              updates=discriminator_updates, allow_input_downcast=True)

    # Compile a second function computing the validation loss and accuracy:
    generate_sample_fn = theano.function([noise_var], sample.transpose((0, 2, 3, 1)), allow_input_downcast=True)
    
    # Reloading
    if reload:
        options = pkl.load(open(model_file+'.pkl'))
        kwargs = options


    # Finally, launch the training loop.
    print "Starting training..."
    # We iterate over epochs:
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        print epoch
        train_batches = 0
        start_time = time.time()
        for i, batch in enumerate(train_iter):
            inputs, targets, caps = batch
            noise = np.random.normal(size=(len(inputs), 100))
            disc_loss = train_discriminator_fn(noise, targets)
            gen_loss = train_generator_fn(noise)
            train_batches += 1


            
            print "batch {} of epoch {} of {} took {:.3f}s".format(i, epoch + 1, num_epochs, time.time() - start_time)
            print "  training generator loss", gen_loss
            print "  training discriminator loss" , disc_loss
            
    
    # Generate some samples 
        if epoch == 1:
            generate_and_show_sample(generate_sample_fn, nb=example)


if __name__ == '__main__':
    train()
        
