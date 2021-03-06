from __future__ import print_function
import tensorflow as tf
import numpy as np
import time
import os
import errno

class RegressionEvaluator:
    """Trains and evaluates a classifier on some datasets passed as argument.

    Uses a number of training tricks, namely, early stopping, keeps the model 
    that achieves the best validation performance, reduces the step size 
    after the validation performance fails to increases for some number of 
    epochs.

    """

    def __init__(self, train_dataset, val_dataset, in_d, cell_size, model_path, 
            training_epochs_max=200, time_minutes_max=180.0, 
            stop_patience=20, rate_patience=7, batch_patience=np.inf, 
            save_patience=2, rate_mult=0.5, batch_mult=2, 
            optimizer_type='adam', sgd_momentum=0.99,
            learning_rate_init=1e-3, learning_rate_min=1e-6, batch_size_init=32, 
            display_step=1, output_to_terminal=False):

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.in_d = list(in_d)
        self.cell_size = cell_size
        self.training_epochs = training_epochs_max
        self.time_minutes_max = time_minutes_max
        self.display_step = display_step
        self.stop_patience = stop_patience
        self.rate_patience = rate_patience
        self.batch_patience = batch_patience
        self.save_patience = save_patience
        self.rate_mult = rate_mult
        self.batch_mult = batch_mult
        self.learning_rate_init = learning_rate_init
        self.learning_rate_min = learning_rate_min
        self.batch_size_init = batch_size_init
        self.optimizer_type = optimizer_type
        self.output_to_terminal = output_to_terminal
        self.sgd_momentum = sgd_momentum
        self.model_path = model_path

        
        if not os.path.exists(os.path.dirname(model_path)):
            try:
                os.makedirs(os.path.dirname(model_path))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

    def eval_model(self, b):
        tf.reset_default_graph()

        input_ph = tf.placeholder("float", [None] + self.in_d)
        label_ph = tf.placeholder("float", [None, self.cell_size, self.cell_size, 3])
        
        learning_rate = tf.placeholder("float")
        learning_rate_val = self.learning_rate_init
        batch_size = self.batch_size_init
        sgd_momentum = self.sgd_momentum

        # compilation 
        train_feed = {}
        eval_feed = {}
        output = b.compile(input_ph, train_feed, eval_feed)
        saver = tf.train.Saver()
        file = open("log.txt","w")
        # Define loss and optimizer
        object_mask = label_ph[...,0]
        x = label_ph[...,1]
        y = label_ph[...,2]
        mask_indices = tf.where(object_mask > 0.0)
        x_hat = tf.gather_nd(x , mask_indices)
        y_hat = tf.gather_nd(y , mask_indices)

        boundary = self.cell_size * self.cell_size
        output_object_mask = tf.reshape(output[:,:boundary],[-1,self.cell_size,self.cell_size])
        output_x = tf.reshape(output[:,boundary:boundary*2],[-1,self.cell_size,self.cell_size])
        output_y = tf.reshape(output[:,boundary*2:],[-1,self.cell_size,self.cell_size])
        output_x_hat = tf.gather_nd(output_x , mask_indices)
        output_y_hat = tf.gather_nd(output_y , mask_indices)

        mask_loss = tf.reduce_sum(tf.square(output_object_mask - object_mask))
        x_loss = tf.reduce_sum(tf.square(output_x_hat - x_hat))
        y_loss = tf.reduce_sum(tf.square(output_y_hat - y_hat))
        loss = mask_loss + x_loss + y_loss
        # chooses the optimizer. (this can be put in a function).
        if self.optimizer_type == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        elif self.optimizer_type == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(
                learning_rate=learning_rate)
        elif self.optimizer_type == 'sgd_mom':
            optimizer = tf.train.MomentumOptimizer(
                learning_rate=learning_rate, momentum=sgd_momentum)
        else:
            raise ValueError("Unknown optimizer.")
        optimizer = optimizer.minimize(loss)

        # Initializing the variables
        init = tf.global_variables_initializer()

        # Launch the graph
        with tf.Session(
                #config=tf.ConfigProto(
                #    allow_soft_placement=True
                #)
            ) as sess:
            sess.run(init)

            # for early stopping
            best_vacc = - np.inf
            best_vacc_saved = - np.inf
            stop_counter = self.stop_patience
            rate_counter = self.rate_patience
            batch_counter = self.batch_patience
            save_counter = self.save_patience
            time_start = time.time()

            min_loss = np.inf
            min_loss_saved = np.inf

            train_num_examples = self.train_dataset.get_num_examples()
            val_num_examples = self.val_dataset.get_num_examples()
            # Training cycle
            for epoch in xrange(self.training_epochs):
                avg_loss = 0.
                total_batch = int(train_num_examples / batch_size)
                # Loop over all batches
                for i in xrange(total_batch):
                    batch_x, batch_y = self.train_dataset.next_batch(batch_size)
                    #print((batch_x.shape, batch_y.shape))
                    #import ipdb; ipdb.set_trace()
                    # Run optimization op (backprop) and cost op (to get loss value)
                    train_feed.update({input_ph: batch_x, 
                                       label_ph: batch_y, 
                                       learning_rate: learning_rate_val})

                    _, l = sess.run([optimizer,loss], feed_dict=train_feed)

                    #_, c = sess.run([optimizer, loss], feed_dict=train_feed)
                    # Compute average loss
                    #avg_cost += c / total_batch
                    avg_loss += l / total_batch
                # early stopping
                batch_x_val, batch_y_val = self.val_dataset.next_batch(12)
                eval_feed.update({input_ph: batch_x_val, 
                                  label_ph: batch_y_val})
                l_val = sess.run(loss, feed_dict=eval_feed)
                #print (avg_loss)
                #print (l_val)
                #vacc = compute_accuracy(self.val_dataset, eval_feed, batch_size)

                # Display logs per epoch step
                if self.output_to_terminal and epoch % self.display_step == 0:
                    print("Time:", "%7.1f" % (time.time() - time_start),
                          "Epoch:", '%04d' % (epoch+1),
                          "cost=", "{:.9f}".format(avg_loss),
                          "val_loss=", "{:.9f}".format(l_val),
                          "learn_rate=", '%.3e' % learning_rate_val)
                if l_val < min_loss:

                    min_loss = l_val
                    stop_counter = self.stop_patience
                    rate_counter = self.rate_patience
                    batch_counter = self.batch_patience
                    save_counter = self.save_patience
                else:
                    stop_counter -= 1
                    rate_counter -= 1
                    batch_counter -= 1
                    if stop_counter == 0:
                        break   
                    if rate_counter == 0:
                        learning_rate_val *= self.rate_mult
                        rate_counter = self.rate_patience

                        if learning_rate_val < self.learning_rate_min:
                            learning_rate_val = self.learning_rate_min

                    if batch_counter == 0:
                        batch_size *= self.batch_mult
                        batch_counter = self.batch_patience

                    if l_val < min_loss_saved:
                        save_counter -= 1

                        if save_counter == 0:
                            file.write("%f\n" %l_val)
                            print("min loss recorded")

                            save_counter = self.save_patience
                            min_loss_saved = l_val

                time_now = time.time()
                if (time_now - time_start) / 60.0 > self.time_minutes_max:
                    break

                '''
                if best_vacc < vacc: 
                    best_vacc = vacc
                    # reinitialize all the counters.
                    stop_counter = self.stop_patience
                    rate_counter = self.rate_patience
                    batch_counter = self.batch_patience
                    save_counter = self.save_patience
                else:
                    stop_counter -= 1
                    rate_counter -= 1
                    batch_counter -= 1
                    if stop_counter == 0:
                        break   

                    if rate_counter == 0:
                        learning_rate_val *= self.rate_mult
                        rate_counter = self.rate_patience

                        if learning_rate_val < self.learning_rate_min:
                            learning_rate_val = self.learning_rate_min

                    if batch_counter == 0:
                        batch_size *= self.batch_mult
                        batch_counter = self.batch_patience

                    if best_vacc_saved < vacc:
                        save_counter -= 1

                        if save_counter == 0:
                            save_path = saver.save(sess, self.model_path)
                            print("Model saved in file: %s" % save_path)

                            save_counter = self.save_patience
                            best_vacc_saved = vacc

                # at the end of the epoch, if spent more time than budget, exit.
                time_now = time.time()
                if (time_now - time_start) / 60.0 > self.time_minutes_max:
                    break
                
            # if the model saved has better performance than the current model,
            # load it.
            if best_vacc_saved > vacc:
                saver.restore(sess, self.model_path)
                print("Model restored from file: %s" % save_path)

            print("Optimization Finished!")

            vacc = compute_accuracy(self.val_dataset, eval_feed, batch_size)
            print("Validation accuracy: %f" % vacc)
            if self.test_dataset != None:
                tacc = compute_accuracy(self.test_dataset, eval_feed, batch_size)
                print("Test accuracy: %f" % tacc)
        '''
        return -l_val



