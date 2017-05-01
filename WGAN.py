def build_generator(input_var=None):
    

    network = InputLayer(shape=(None, 100),
                                        input_var=input_var)

    network = ReshapeLayer(network, (-1, 100, 1, 1))
    network = BatchNormLayer(network)
    network = TransposedConv2DLayer(network, num_filters=128, filter_size=(4, 4),
                                                   stride=(1, 1), nonlinearity= rectify)

    network = BatchNormLayer(network)
    network = TransposedConv2DLayer(network, num_filters=128/2,
              filter_size=(5, 5), stride=(2, 2), crop=2, nonlinearity= rectify,output_size = 8)

    network = BatchNormLayer(network)
    network = TransposedConv2DLayer(network, num_filters=128/4,
              filter_size=(5, 5), stride=(2, 2), crop=2, nonlinearity= rectify,output_size = 16)

    network = TransposedConv2DLayer(network, num_filters=3,
                                                   filter_size=(5, 5), stride=(2, 2), crop=2,
                                                   nonlinearity=tanh,output_size = 32)
    
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

    return network
    
    def load_dataset(batch_size=128):

    train_iter = Iterator(nb_sub=1280,batch_size=batch_size, img_path = 'train2014', extract_center=True)
    val_iter = Iterator(nb_sub=1280, batch_size=batch_size, img_path = 'val2014',extract_center=True)

    return train_iter, val_iter


# ############################## Main program ################################
# Everything else will be handled in our main program now. We could pull out
# more functions to better separate the code, but it wouldn't make it any
# easier to read.

def train(num_epochs=1, epochsize=5, clip=0.01,
          lr=0.00005, example=19, save_freq=100,
          batch_size=64, verbose_freq=100,
          model_file="/home/mouna/Documents/Project/test.npz",
          reload=False,
          **kwargs):

    # Load the dataset
    print "Loading data..."
    train_iter, val_iter = load_dataset(batch_size)
    
    def x_y_cap(data):
    
        xs, ys, caps = zip(*[x for x in data if x is not None])
        return np.array(xs), np.array(ys), caps
    
    x,y,caps = x_y_cap(train_iter)
    
    

    #some monitoring stuff
    val_loss = []
    train_loss = []

    # Prepare Theano variables for inputs and targets
    noise_var = T.matrix('noise',dtype='float32')
    input = T.tensor4('real_img',dtype='float32')
#     target = T.tensor4('targets',dtype='float32')

    input_var = input.transpose((0, 3, 1, 2))

    generator = build_generator(noise_var)
    critic = build_discriminator(input_var)
    
    sample = lasagne.layers.get_output(generator)
    
    # Create expression for passing real data through the discriminator
    real_out = lasagne.layers.get_output(critic)
    # Create expression for passing fake data through the discriminator
    fake_out = lasagne.layers.get_output(critic,
            sample)
    
    # Create score expressions to be maximized (i.e., negative losses)
    generator_score = fake_out.mean()
    critic_score = real_out.mean() - fake_out.mean()
    
    # Create update expressions for training
    generator_params = lasagne.layers.get_all_params(generator, trainable=True)
    
    critic_params = lasagne.layers.get_all_params(critic, trainable=True)
    
    generator_updates = lasagne.updates.rmsprop(-generator_score, generator_params, learning_rate=lr)
    
    critic_updates = lasagne.updates.rmsprop(-critic_score, critic_params, learning_rate=lr)
    
    # Clip critic parameters in a limited range around zero (except biases)
    for param in lasagne.layers.get_all_params(critic, trainable=True,regularizable=True):
        critic_updates[param] = T.clip(critic_updates[param], -clip, clip)
        
    # Instantiate a symbolic noise generator to use for training
    from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
    srng = RandomStreams(seed=np.random.randint(2147462579, size=6))
    noise = srng.uniform((batch_size, 100))
    
    print "Computing the functions..."
    
    generator_train_fn = None
    critic_train_fn = None
    
    # Compile functions performing a training step on a mini-batch (according
    # to the updates dictionary) and returning the corresponding score:
    generator_train_fn = theano.function([], generator_score, givens={noise_var: noise},
                                         updates=generator_updates, allow_input_downcast=True)
    critic_train_fn = theano.function([input], critic_score, givens={noise_var: noise},
                                      updates=critic_updates, allow_input_downcast=True)

    # Compile another function generating some data
    gen_fn = theano.function([noise_var],lasagne.layers.get_output(generator,
                             deterministic=True).transpose((0, 2, 3, 1)), allow_input_downcast=True)
    
    

    # Finally, launch the training loop.
    print "Starting training..."
    # We iterate over epochs:
    generator_updates = 0
    for epoch in range(num_epochs):
        start_time = time.time()

        # In each epoch, we do `epochsize` generator updates. Usually, the
        # critic is updated 5 times before every generator update. For the
        # first 25 generator updates and every 500 generator updates, the
        # critic is updated 100 times instead, following the authors' code.
        critic_scores = []
        generator_scores = []
        for _ in range(epochsize):
            if (generator_updates < 25) or (generator_updates % 500 == 0):
                critic_runs = 100
            else:
                critic_runs = 5
            for _ in range(critic_runs):
                n = np.random.randint(low=0, high=10)
                targets = y[n]
#                 batch = next(train_iter)
#                 inputs, targets, caps = batch
                critic_scores.append(critic_train_fn(targets))
            generator_scores.append(generator_train_fn())
            generator_updates += 1

            
            # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(epoch + 1, num_epochs, time.time() - start_time))
        print("  generator score:\t\t{}".format(np.mean(generator_scores)))
        print("  Wasserstein distance:\t\t{}".format(np.mean(critic_scores)))
            
    
    # Generate some samples 
        if epoch == 0:
            generate_and_show_sample(gen_fn, nb=example)


if __name__ == '__main__':
    train()
        
