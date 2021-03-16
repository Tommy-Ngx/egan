'''GAIN function.
Date: 2020/02/28
Reference: J. Yoon, J. Jordon, M. van der Schaar, "GAIN: Missing Data 
           Imputation using Generative Adversarial Nets," ICML, 2018.
Paper Link: http://proceedings.mlr.press/v80/yoon18a/yoon18a.pdf
Contact: jsyoon0823@gmail.com
'''

# Necessary packages
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import tensorflow.compat.v1 as tf1
tf1.disable_v2_behavior() 
import copy
import numpy as np
from tqdm import tqdm

from utils import normalization, renormalization, rounding
from utils import xavier_init
from utils import binary_sampler, uniform_sampler, sample_batch_index, sample_batch_binary


def egain (miss_data_x, gain_parameters):
#def Egain(miss_data_x, gain_parameters):
    '''Impute missing values in data_x
  
  Args:
    - miss_data_x: missing data
    - gain_parameters: GAIN network parameters:
      - batch_size: Batch size
      - alpha: Hyperparameter
      - iterations: Iterations
      
  Returns:
    - imputed_data: imputed data
  '''
    # Define mask matrix
    m = 1 - np.isnan(miss_data_x)

    # System parameters
    batch_size = gain_parameters['batch_size']
    # hint_rate = gain_parameters['hint_rate']
    alpha = gain_parameters['alpha']
    iterations = gain_parameters['iterations']

    loss_type = ['trickLogD', 'minimax', 'ls']
    nloss = 3
    beta = 1.0
    ncandi = 1#1#3
    nbest = 1#1#3
    nD = 1  # # of discrim updates for each gen update
    # Other parameters
    no, dim = miss_data_x.shape
    # Hidden state dimensions
    h_dim = int(dim)

    # Normalization
    norm_data, norm_parameters = normalization(miss_data_x)
    norm_data_x = np.nan_to_num(norm_data, 0)

    ## GAIN architecture
    #tf.reset_default_graph()
    tf.compat.v1.get_default_graph()
    # Input placeholders
    # Data vector
    X = tf1.placeholder(tf.float32, shape=[None, dim])
    # Mask vector
    M = tf1.placeholder(tf.float32, shape=[None, dim])
    # B vector
    B = tf1.placeholder(tf.float32, shape=[None, dim])

    # Discriminator variables
    D_W1 = tf.Variable(xavier_init([dim * 2, h_dim]))  # Data + Hint as inputs
    D_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

    D_W2 = tf.Variable(xavier_init([h_dim, h_dim]))
    D_b2 = tf.Variable(tf.zeros(shape=[h_dim]))

    D_W3 = tf.Variable(xavier_init([h_dim, dim]))
    D_b3 = tf.Variable(tf.zeros(shape=[dim]))  # Multi-variate outputs

    theta_D = [D_W1, D_W2, D_W3, D_b1, D_b2, D_b3]

    # Generator variables
    # Data + Mask as inputs (Random noise is in missing components)
    G_W1 = tf.Variable(xavier_init([dim * 2, h_dim]))
    G_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

    G_W2 = tf.Variable(xavier_init([h_dim, h_dim]))
    G_b2 = tf.Variable(tf.zeros(shape=[h_dim]))

    G_W3 = tf.Variable(xavier_init([h_dim, dim]))
    G_b3 = tf.Variable(tf.zeros(shape=[dim]))

    theta_G = [G_W1, G_W2, G_W3, G_b1, G_b2, G_b3]

    ## GAIN functions
    # Generator
    def generator(x, m):
        # Concatenate Mask and Data
        inputs = tf.concat(values=[x, m], axis=1)
        G_h1 = tf.nn.relu(tf.matmul(inputs, G_W1) + G_b1)
        G_h2 = tf.nn.relu(tf.matmul(G_h1, G_W2) + G_b2)
        # MinMax normalized output
        G_prob = tf.nn.sigmoid(tf.matmul(G_h2, G_W3) + G_b3)
        return G_prob

    # Discriminator
    def discriminator(x, h):
        # Concatenate Data and Hint
        inputs = tf.concat(values=[x, h], axis=1)
        D_h1 = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)
        D_h2 = tf.nn.relu(tf.matmul(D_h1, D_W2) + D_b2)
        D_logit = tf.matmul(D_h2, D_W3) + D_b3
        D_prob = tf.nn.sigmoid(D_logit)
        return D_prob

    ## GAIN structure
    # Hint vector
    H = B * M + 0.5 * (1 - B) # 0.5 => 0.1
    # Generator
    G_sample = generator(X, M)
    D_prob_g = discriminator(X * M + G_sample * (1 - M), H)

    # Combine with observed data
    fake_X = tf1.placeholder(tf.float32, shape=[None, dim])
    # Hint vector
    Hat_X = X * M + fake_X * (1 - M)

    # D loss
    D_prob = discriminator(Hat_X, H)
    D_loss_temp = -tf.reduce_mean((M * tf1.log(D_prob + 1e-8) + (1 - M) * tf1.log(1. - D_prob + 1e-8)))
    D_loss = D_loss_temp
    D_solver = tf1.train.AdamOptimizer(learning_rate=0.002, beta1=0.5, beta2=0.99).minimize(D_loss, var_list=theta_D)

    # G loss
    G_loss_logD = -tf.reduce_mean((1 - M) * tf1.log(D_prob_g + 1e-8))
    G_loss_minimax = tf.reduce_mean((1 - M) * tf1.log(1. - D_prob_g + 1e-8))
    G_loss_ls = tf1.reduce_mean((1-M)*tf1.square(D_prob_g - 1))

    MSE_loss = tf.reduce_mean((M * X - M * G_sample) ** 2) / tf.reduce_mean(M)

    G_loss_logD_all = G_loss_logD + alpha * MSE_loss
    G_loss_minimax_all = G_loss_minimax + alpha * MSE_loss
    G_loss_ls_all = G_loss_ls + alpha * MSE_loss

    G_solver_logD = tf1.train.AdamOptimizer(learning_rate=0.002, beta1=0.5, beta2=0.99).minimize(G_loss_logD_all, var_list=theta_G)
    G_solver_minimax = tf1.train.AdamOptimizer(learning_rate=0.002, beta1=0.5, beta2=0.99).minimize(G_loss_minimax_all, var_list=theta_G)
    G_solver_ls = tf1.train.AdamOptimizer(learning_rate=0.002, beta1=0.5, beta2=0.99).minimize(G_loss_ls_all, var_list=theta_G)

    # Fitness function
    Fq_score = tf.reduce_mean((1 - M) * D_prob)
    Fd_score = - tf1.log(tf.reduce_sum(tf.square(tf.gradients(D_loss_temp, theta_D[0])))
                        + tf.reduce_sum(tf.square(tf.gradients(D_loss_temp, theta_D[1])))
                        + tf.reduce_sum(tf.square(tf.gradients(D_loss_temp, theta_D[2])))
                        + tf.reduce_sum(tf.square(tf.gradients(D_loss_temp, theta_D[3])))
                        + tf.reduce_sum(tf.square(tf.gradients(D_loss_temp, theta_D[4])))
                        + tf.reduce_sum(tf.square(tf.gradients(D_loss_temp, theta_D[5])))
                        )

    ## Iterations

    sess = tf1.Session()
    # Start Iterations

    gen_new_params = []
    fitness_best = np.zeros(nbest)
    fitness_candi = np.zeros(ncandi)
    # for it in tqdm(range(iterations)):
    for it in tqdm(range(iterations)):
        # Train candidates G
        if it == 0:
            for can_i in range(0, ncandi):
                sess.run(tf1.global_variables_initializer())
                batch_idx = sample_batch_index(no, batch_size)
                X_mb = norm_data_x[batch_idx, :]
                M_mb = m[batch_idx, :]
                Z_mb = uniform_sampler(0.0, 0.01, batch_size, dim)
                X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb
                B_mb = sample_batch_binary(dim, batch_size)
                gen_samples = sess.run([G_sample], feed_dict={X: X_mb, M: M_mb})[0]
                fq_score, fd_score = sess.run([Fq_score, Fd_score],
                                              feed_dict={X: X_mb, M: M_mb, fake_X: gen_samples, B: B_mb})
                fitness = fq_score + beta * fd_score
                fitness_best[can_i] = fitness
                params = []
                for param in theta_G:
                    params.append(sess.run(param))
                gen_new_params.append(params)
            gen_best_params = copy.deepcopy(gen_new_params)
        else:
            # generate new candidate
            gen_old_params = copy.deepcopy(gen_new_params)
            # print(gen_old_params[0][-1])
            # print(it)
            for can_i in range(ncandi):
                for type_i in range(nloss):
                    batch_idx = sample_batch_index(no, batch_size)
                    X_mb = norm_data_x[batch_idx, :]
                    M_mb = m[batch_idx, :]
                    Z_mb = uniform_sampler(0.0, 0.01, batch_size, dim) # update 1.0 ==> 0.01
                    X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb
                    B_mb = sample_batch_binary(dim, batch_size)
                    # Load and update weights
                    for i in range(len(theta_G)):
                        theta_G[i].load(gen_old_params[can_i][i], sess)
                    loss = loss_type[type_i]
                    if loss == 'trickLogD':
                        sess.run([G_solver_minimax], feed_dict={X: X_mb, M: M_mb, B: B_mb})
                    elif loss == 'minimax':
                        sess.run([G_solver_logD], feed_dict={X: X_mb, M: M_mb, B: B_mb})
                    elif loss == 'ls':
                        sess.run([G_solver_ls], feed_dict={X: X_mb, M: M_mb, B: B_mb})

                    # calculate fitness score
                    gen_samples = sess.run([G_sample], feed_dict={X: X_mb, M: M_mb})[0]
                    fq_score, fd_score = sess.run([Fq_score, Fd_score],
                                                  feed_dict={X: X_mb, M: M_mb, fake_X: gen_samples, B: B_mb})
                    fitness = fq_score + beta * fd_score
                    # print(fitness)
                    gap = fitness_best - fitness
                    if min(gap) < 0:
                        idx_replace = np.argmin(gap)
                        params = []
                        for param in theta_G:
                            params.append(sess.run(param))
                        gen_best_params[idx_replace] = params
                        fitness_best[idx_replace] = fitness

                    if can_i * nloss + type_i < ncandi:
                        idx = can_i * nloss + type_i
                        params = []
                        for param in theta_G:
                            params.append(sess.run(param))
                        gen_new_params[idx] = params
                        fitness_candi[idx] = fitness
                    else:
                        gap = fitness_candi - fitness
                        if min(gap) < 0:
                            idx_replace = np.argmin(gap)
                            params = []
                            for param in theta_G:
                                params.append(sess.run(param))
                            gen_new_params[idx_replace] = params
                            fitness_candi[idx_replace] = fitness
        # Train D
        for i in range(nD):
            batch_idx = sample_batch_index(no, batch_size)
            X_mb = norm_data_x[batch_idx, :]
            M_mb = m[batch_idx, :]
            Z_mb = uniform_sampler(0.0, 0.01, batch_size, dim) # 1.0 ==> 0.01
            X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb
            B_mb = sample_batch_binary(dim, batch_size)
            # impute data for each candidat
            for can_i in range(ncandi):
                for w in range(len(theta_G)):
                    theta_G[w].load(gen_new_params[can_i][w], sess)
                if can_i == ncandi - 1:
                    gen_samples_cani = sess.run([G_sample],
                                                feed_dict={X: X_mb[can_i * batch_size // ncandi:],
                                                           M: M_mb[can_i * batch_size // ncandi:]})[0]
                else:
                    gen_samples_cani = sess.run([G_sample],
                                            feed_dict={X: X_mb[can_i*batch_size//ncandi:(can_i + 1) * batch_size // ncandi],
                                                       M: M_mb[can_i*batch_size//ncandi:(can_i + 1) * batch_size // ncandi]})[0]
                # print(gen_samples_cani.shape)
                if can_i == 0:
                    gen_samples = gen_samples_cani
                else:
                    gen_samples = np.append(gen_samples, gen_samples_cani, axis=0)
            sess.run([D_solver], feed_dict={X: X_mb, M: M_mb, fake_X: gen_samples, B: B_mb})

    ## Return imputed data

    idx = np.argmax(fitness_best)
    # print(idx)
    for i in range(len(theta_G)):
        theta_G[i].load(gen_best_params[idx][i], sess)

    Z_mb = uniform_sampler(0.0, 0.01, no, dim)
    M_mb = m
    X_mb = norm_data_x
    X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb

    imputed_data = sess.run([G_sample], feed_dict={X: X_mb, M: M_mb})[0]
    sess.close()
    imputed_data = m * norm_data_x + (1 - m) * imputed_data

    # Renormalization
    imputed_data = renormalization(imputed_data, norm_parameters)

    # Rounding
    imputed_data = rounding(imputed_data, miss_data_x)

    return imputed_data
