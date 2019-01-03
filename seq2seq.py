import numpy as np
import tensorflow as tf
import copy
import os
import policy_generator as pg
import matplotlib.pyplot as plt


class model():

    def __init__(self,config):
        self.config = config
        self.wallet_change = []
        self.wallet = None



    def build_graph(self,feed_previous = False):

        from tensorflow.contrib import rnn
        from tensorflow.python.ops import variable_scope
        from tensorflow.python.framework import dtypes


        print("Building graph")

        tf.reset_default_graph()

        global_step = tf.Variable(
                      initial_value=0,
                      name="global_step",
                      trainable=False,
                      collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])

        weights = {
            'out': tf.get_variable('Weights_out', \
                                   shape = [self.config.hidden_dim, self.config.output_dim], \
                                   dtype = tf.float32, \
                                   initializer = tf.truncated_normal_initializer()),
        }
        print("    - Weights init")

        biases = {
            'out': tf.get_variable('Biases_out', \
                                   shape = [self.config.output_dim], \
                                   dtype = tf.float32, \
                                   initializer = tf.constant_initializer(0.)),
        }
        print("    - Biases init")
        with tf.variable_scope('Seq2seq'):
            # Encoder: inputs
            enc_inp = [
                tf.placeholder(tf.float32, shape=(None, self.config.input_dim), name="inp_{}".format(t))
                   for t in range(self.config.input_seq_len)
            ]

            # Decoder: target outputs
            target_seq = [
                tf.placeholder(tf.float32, shape=(None, self.config.output_dim), name="y".format(t))
                  for t in range(self.config.output_seq_len)
            ]

            # Give a "GO" token to the decoder.
            # If dec_inp are fed into decoder as inputs, this is 'guided' training; otherwise only the
            # first element will be fed as decoder input which is then 'un-guided'
            dec_inp = [ tf.zeros_like(target_seq[0], dtype=tf.float32, name="GO") ] + target_seq[:-1]

            with tf.variable_scope('LSTMCell'):
                cells = []
                for i in range(self.config.num_stacked_layers):
                    with tf.variable_scope('RNN_{}'.format(i)):
                        cells.append(tf.contrib.rnn.LSTMCell(self.config.hidden_dim))
                cell = tf.contrib.rnn.MultiRNNCell(cells)

            def _rnn_decoder(decoder_inputs,
                            initial_state,
                            cell,
                            loop_function=None,
                            scope=None):

                with variable_scope.variable_scope(scope or "rnn_decoder"):
                    state = initial_state
                    outputs = []
                    prev = None
                    for i, inp in enumerate(decoder_inputs):
                        if loop_function is not None and prev is not None:
                            with variable_scope.variable_scope("loop_function", reuse=True):
                                inp = loop_function(prev, i)
                        if i > 0:
                            variable_scope.get_variable_scope().reuse_variables()
                        output, state = cell(inp, state)
                        outputs.append(output)
                        if loop_function is not None:
                            prev = output
                return outputs, state

            def _basic_rnn_seq2seq(encoder_inputs,
                                  decoder_inputs,
                                  cell,
                                  feed_previous,
                                  dtype=dtypes.float32,
                                  scope=None):

                with variable_scope.variable_scope(scope or "basic_rnn_seq2seq"):
                    enc_cell = copy.deepcopy(cell)
                    _, enc_state = rnn.static_rnn(enc_cell, encoder_inputs, dtype=dtype)
                    if feed_previous:
                        return _rnn_decoder(decoder_inputs, enc_state, cell, _loop_function)
                    else:
                        return _rnn_decoder(decoder_inputs, enc_state, cell)

            def _loop_function(prev, _):

                return tf.matmul(prev, weights['out']) + biases['out']

            dec_outputs, dec_memory = _basic_rnn_seq2seq(
                enc_inp,
                dec_inp,
                cell,
                feed_previous = feed_previous
            )

            reshaped_outputs = [tf.matmul(i, weights['out']) + biases['out'] for i in dec_outputs]

        # Training loss and optimizer
        with tf.variable_scope('Loss'):
            # L2 loss
            output_loss = 0
            for _y, _Y in zip(reshaped_outputs, target_seq):
                output_loss += tf.reduce_mean(tf.pow(_y - _Y, 2)) #MSE

            # L2 regularization for weights and biases
            reg_loss = 0
            for tf_var in tf.trainable_variables():
                if 'Biases_' in tf_var.name or 'Weights_' in tf_var.name:
                    reg_loss += tf.reduce_mean(tf.nn.l2_loss(tf_var))

            loss = output_loss + self.config.lambda_l2_reg * reg_loss

        with tf.variable_scope('Optimizer'):
            optimizer = tf.contrib.layers.optimize_loss(
                    loss=loss,
                    learning_rate=self.config.learning_rate,
                    global_step=global_step,
                    optimizer='Adam',
                    clip_gradients=self.config.GRADIENT_CLIPPING)

        saver = tf.train.Saver

        return dict(
            enc_inp = enc_inp,
            target_seq = target_seq,
            train_op = optimizer,
            loss=loss,
            saver = saver,
            reshaped_outputs = reshaped_outputs,
            )



    def rescale_data(self,data,scaling_factor,scaling_bias):
        return data*scaling_factor + scaling_bias



    def mse(self,data,new_data):
        return np.mean((data-new_data)**2)



    def mape(self,data,new_data):
        return np.mean(np.abs((data - new_data) / data)) * 100



    def train(self,X,Y):


        train_losses = []
#         val_losses = []

        rnn_model = self.build_graph(feed_previous=False)

        saver = tf.train.Saver()
        print("Saver init")
        init = tf.global_variables_initializer()
        print("Started and initialised")


        with tf.Session() as sess:
            sess.run(init)


            for e in range(self.config.epochs):
                print(e+1,'/',self.config.epochs)
                for i in range(0,X.shape[0],self.config.batch_size):

                    # Try to get batch size, otherwise use remaining data
                    try:

                        batch_X_train = np.reshape(X[i:i+self.config.batch_size],(self.config.batch_size,len(X[0]),1))
                        batch_Y_train = np.reshape(Y[i:i+self.config.batch_size],(self.config.batch_size,len(Y[0])))

                    except:
                        batch_X_train = np.reshape(X[i:],(X[i:].shape[0],len(X[0]),1))
                        batch_Y_train = np.reshape(Y[i:],(Y[i:].shape[0],len(Y[0])))

                    feed_dict = {rnn_model['enc_inp'][t]: batch_X_train[:,t].reshape(-1,self.config.input_dim) for t in range(self.config.input_seq_len)} #input
                    feed_dict.update({rnn_model['target_seq'][t]: batch_Y_train[:,t].reshape(-1,self.config.output_dim) for t in range(self.config.output_seq_len)}) #target output
                    _, loss_t = sess.run([rnn_model['train_op'], rnn_model['loss']], feed_dict)
#                     print(loss_t)
                    train_losses = np.append(train_losses,loss_t)

        # #             batch_val_input,batch_val_output = get_val_data()
        # #             # Validation set
        # #             feed_dict = {rnn_model['enc_inp'][t]: batch_val_input[t].reshape(1,1) for t in range(input_seq_len)}
        # #             feed_dict.update({rnn_model['target_seq'][t]: np.zeros([1, output_dim]) for t in range(output_seq_len)})
        # #             final_preds = sess.run(rnn_model['reshaped_outputs'], feed_dict)

        # #             val_losses = np.append(val_losses,mse(final_preds,batch_val_output[0]))


            temp_saver = rnn_model['saver']()
            save_path = temp_saver.save(sess, os.path.join(self.config.save_path, self.config.save_name))

            plt.plot(train_losses)
#             print("Model saved at: ", self.config,save_path,self.config.save_name)




    def backtest(self,X,Y,wallet):
        print("Backtesting...")

        self.wallet = wallet

        policy = pg.policy_generator(wallet)
        self.wallet_change = [wallet]
        predictions = np.array([])


        rnn_model = self.build_graph(feed_previous=True)
        init = tf.global_variables_initializer()
        with tf.Session() as sess:

            sess.run(init)
            saver = rnn_model['saver']().restore(sess, os.path.join(self.config.save_path, self.config.save_name))


            for i in range(X.shape[0]):
                backtest_X = np.reshape(X[i],(len(X[i]),1))
                backtest_Y = np.reshape(Y[i],(len(Y[i],)))


                feed_dict = {rnn_model['enc_inp'][t]: backtest_X[t].reshape(-1,self.config.input_dim) for t in range(self.config.input_seq_len)}
                feed_dict.update({rnn_model['target_seq'][t]: np.zeros([1, self.config.output_dim]) for t in range(self.config.output_seq_len)})
                forecast = np.asarray(sess.run(rnn_model['reshaped_outputs'], feed_dict))


                policy.generate(self.rescale_data(forecast,self.config.scaling_factor,self.config.scaling_bias))

                policy.execute_policy(true_data = self.rescale_data(backtest_Y,self.config.scaling_factor,self.config.scaling_bias))
                self.wallet_change = np.append(self.wallet_change,policy.wallet)

        self.wallet = policy.wallet
        print("Complete")
