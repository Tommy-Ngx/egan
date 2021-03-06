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

import numpy as np
from tqdm import tqdm

from utils import normalization, renormalization, rounding
from utils import xavier_init
from utils import binary_sampler, uniform_sampler, sample_batch_index


def gain (data_x, gain_parameters):
  '''Impute missing values in data_x
  
  Args:
    - data_x: original data with missing values
    - gain_parameters: GAIN network parameters:
      - batch_size: Batch size
      - hint_rate: Hint rate
      - alpha: Hyperparameter
      - iterations: Iterations
      
  Returns:
    - imputed_data: imputed data
  '''
  # Define mask matrix
  data_m = 1-np.isnan(data_x)
  
  # System parameters
  batch_size = gain_parameters['batch_size']
  hint_rate = gain_parameters['hint_rate']
  alpha = gain_parameters['alpha']
  iterations = gain_parameters['iterations']
  
  # Other parameters
  no, dim = data_x.shape
  
  # Hidden state dimensions
  h_dim = int(dim)
  
  # Normalization
  norm_data, norm_parameters = normalization(data_x)
  norm_data_x = np.nan_to_num(norm_data, 0)
  
  ## GAIN architecture   
  # Input placeholders
  # Data vector
  X = tf1.placeholder(tf.float32, shape = [None, dim])
  # Mask vector 
  M = tf1.placeholder(tf.float32, shape = [None, dim])
  # Hint vector
  H = tf1.placeholder(tf.float32, shape = [None, dim])
  

  # Discriminator variables
  D_W1 = tf.Variable(xavier_init([dim*2, h_dim])) # Data + Hint as inputs
  D_b1 = tf.Variable(tf.zeros(shape = [h_dim]))
  
  D_W2 = tf.Variable(xavier_init([h_dim, h_dim]))
  D_b2 = tf.Variable(tf.zeros(shape = [h_dim]))
  
  D_W3 = tf.Variable(xavier_init([h_dim, dim]))
  D_b3 = tf.Variable(tf.zeros(shape = [dim]))  # Multi-variate outputs
  
  theta_D = [D_W1, D_W2, D_W3, D_b1, D_b2, D_b3]
  

  #Generator variables
  # Data + Mask as inputs (Random noise is in missing components)
  G_W1 = tf.Variable(xavier_init([dim*2, h_dim]))  
  G_b1 = tf.Variable(tf.zeros(shape = [h_dim]))
  
  G_W2 = tf.Variable(xavier_init([h_dim, h_dim]))
  G_b2 = tf.Variable(tf.zeros(shape = [h_dim]))
  
  G_W3 = tf.Variable(xavier_init([h_dim, dim]))
  G_b3 = tf.Variable(tf.zeros(shape = [dim]))
  
  theta_G = [G_W1, G_W2, G_W3, G_b1, G_b2, G_b3]
  
  ## GAIN functions
  # Generator
  def generator(x,m):
    # Concatenate Mask and Data
    inputs = tf.concat(values = [x, m], axis = 1) 
    G_h1 = tf.nn.relu(tf.matmul(inputs, G_W1) + G_b1) # lrelu 
    G_h2 = tf.nn.relu(tf.matmul(G_h1, G_W2) + G_b2)   # lrelu 
    # MinMax normalized output
    G_prob = tf.nn.sigmoid(tf.matmul(G_h2, G_W3) + G_b3) 
    return G_prob
      
  # Discriminator
  def discriminator(x, h):
    # Concatenate Data and Hint
    inputs = tf.concat(values = [x, h], axis = 1) 
    D_h1 = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)   # lrelu 
    D_h2 = tf.nn.relu(tf.matmul(D_h1, D_W2) + D_b2)     # lrelu 
    D_logit = tf.matmul(D_h2, D_W3) + D_b3  
    D_prob = tf.nn.sigmoid(D_logit)
    return D_prob
  


  ## GAIN structure HeG  MinG  LsG 3 generation
  # Generator
  #G_sample = generator(X, M)

  G_sample_z    = generator(X, M)
  G_sample_HeG  = generator(X, M)
  G_sample_MinG = generator(X, M)
  G_sample_LsG  = generator(X, M)
 
  # Combine with observed data

  Hat_X_z    = X * M + G_sample_z    * (1-M)
  Hat_X_HeG  = X * M + G_sample_HeG  * (1-M)
  Hat_X_MinG = X * M + G_sample_MinG * (1-M)
  Hat_X_LsG  = X * M + G_sample_LsG  * (1-M)
  
  # Discriminator
  #D_prob = discriminator(Hat_X, H)

  D_prob_z   = discriminator(Hat_X_z   , H)
  D_prob_HeG = discriminator(Hat_X_HeG , H)
  D_prob_MinG= discriminator(Hat_X_MinG, H)
  D_prob_LsG = discriminator(Hat_X_LsG , H)


  
  ## GAIN loss
  #D_loss_temp = -tf.reduce_mean(M * tf1.log(D_prob + 1e-8) \
  #                              + (1-M) * tf1.log(1. - D_prob + 1e-8))

  D_loss_temp_z   = -tf.reduce_mean(M * tf1.log(D_prob_z + 1e-8) \
                                + (1-M) * tf1.log(1. - D_prob_z + 1e-8))  
  D_loss_temp_HeG = -tf.reduce_mean(M * tf1.log(D_prob_HeG + 1e-8) \
                                + (1-M) * tf1.log(1. - D_prob_HeG + 1e-8)) 
  D_loss_temp_MinG = -tf.reduce_mean(M * tf1.log(D_prob_MinG + 1e-8) \
                                + (1-M) * tf1.log(1. - D_prob_MinG + 1e-8)) 
  D_loss_temp_LsG = -tf.reduce_mean(M * tf1.log(D_prob_LsG + 1e-8) \
                                + (1-M) * tf1.log(1. - D_prob_LsG + 1e-8)) 


  #G_loss_temp = -tf.reduce_mean((1-M) * tf1.log(D_prob + 1e-8))
  G_loss_temp_HeG  = -tf.reduce_mean((1-M) * tf1.log(D_prob_HeG + 1e-8))
  G_loss_temp_MinG = -tf.reduce_mean((1-M) * tf1.log(D_prob_MinG + 1e-8))
  G_loss_temp_LsG  = -tf.reduce_mean((1-M) * tf1.log(D_prob_LsG + 1e-8))

  
  #MSE_loss = \
  #tf.reduce_mean((M * X - M * G_sample)**2) / tf.reduce_mean(M)

  MSE_loss_z  = \
  tf.reduce_mean((M * X - M * G_sample_z)**2) / tf.reduce_mean(M)  
  MSE_loss_HeG  = \
  tf.reduce_mean((M * X - M * G_sample_HeG)**2) / tf.reduce_mean(M)
  MSE_loss_MinG = \
  tf.reduce_mean((M * X - M * G_sample_MinG)**2) / tf.reduce_mean(M)
  MSE_loss_LsG = \
  tf.reduce_mean((M * X - M * G_sample_LsG)**2) / tf.reduce_mean(M)


  #D_loss = D_loss_temp
  D_loss_z    = D_loss_temp_z
  D_loss_HeG  = D_loss_temp_HeG
  D_loss_MinG = D_loss_temp_MinG
  D_loss_LsG  = D_loss_temp_LsG


  #G_loss = G_loss_temp + alpha * MSE_loss
  G_loss_HeG  = G_loss_temp_HeG  + alpha * MSE_loss_HeG
  G_loss_MinG = G_loss_temp_MinG + alpha * MSE_loss_MinG
  G_loss_LsG  = G_loss_temp_LsG  + alpha * MSE_loss_LsG
  
  
  ## GAIN solver
  #D_solver = tf1.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
  D_solver_z    = tf1.train.AdamOptimizer().minimize(D_loss_z, var_list=theta_D)
  D_solver_HeG  = tf1.train.AdamOptimizer().minimize(D_loss_HeG, var_list=theta_D)
  D_solver_MinG = tf1.train.AdamOptimizer().minimize(D_loss_MinG, var_list=theta_D)
  D_solver_LsG  = tf1.train.AdamOptimizer().minimize(D_loss_LsG, var_list=theta_D)

  #G_solver = tf1.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)
  G_solver_HeG  = tf1.train.AdamOptimizer().minimize(G_loss_HeG, var_list=theta_G)
  G_solver_MinG = tf1.train.AdamOptimizer().minimize(G_loss_MinG, var_list=theta_G)
  G_solver_LsG  = tf1.train.AdamOptimizer().minimize(G_loss_LsG , var_list=theta_G)
  
  ## Iterations
  sess = tf1.Session()
  sess.run(tf1.global_variables_initializer())
   
  mutNum=3
  loss_type=['heuristic','minimax','ls']

  # Start Iterations (training)
  for it in tqdm(range(iterations)):    
      
    # Sample batch
    batch_idx = sample_batch_index(no, batch_size)
    X_mb = norm_data_x[batch_idx, :]  
    M_mb = data_m[batch_idx, :]  
    # Sample random vectors  
    Z_mb = uniform_sampler(0, 0.01, batch_size, dim) 
    # Sample hint vectors
    H_mb_temp = binary_sampler(hint_rate, batch_size, dim)
    H_mb = M_mb * H_mb_temp
    

    # Combine random vectors with observed vectors
    X_mb = M_mb * X_mb + (1-M_mb) * Z_mb 
      
    if it==0:
      _, D_loss_curr_z = sess.run([D_solver_z, D_loss_temp_z], 
                              feed_dict = {M: M_mb, X: X_mb, H: H_mb})
    else:
      for type_i in range(mutNum):
        if loss_type[type_i]=='heuristic':

    #_, D_loss_curr = sess.run([D_solver, D_loss_temp], 
    #                          feed_dict = {M: M_mb, X: X_mb, H: H_mb})
          _, D_loss_curr_HeG = sess.run([D_solver_HeG, D_loss_temp_HeG], 
                              feed_dict = {M: M_mb, X: X_mb, H: H_mb})

        elif loss_type[type_i]=='minimax':
          _, D_loss_curr_MinG = sess.run([D_solver_MinG, D_loss_temp_MinG], 
                              feed_dict = {M: M_mb, X: X_mb, H: H_mb})

        elif loss_type[type_i]=='ls':
          _, D_loss_curr_LsG = sess.run([D_solver_LsG, D_loss_temp_LsG], 
                              feed_dict = {M: M_mb, X: X_mb, H: H_mb})

    
    #_, G_loss_curr, MSE_loss_curr = \
    #sess.run([G_solver, G_loss_temp, MSE_loss],
    #         feed_dict = {X: X_mb, M: M_mb, H: H_mb})

    _, G_loss_curr_HeG, MSE_loss_curr = \
    sess.run([G_solver_HeG, G_loss_temp_HeG, MSE_loss_HeG],
             feed_dict = {X: X_mb, M: M_mb, H: H_mb})



  ## Return imputed data      
  Z_mb = uniform_sampler(0, 0.01, no, dim) 
  M_mb = data_m
  X_mb = norm_data_x          
  X_mb = M_mb * X_mb + (1-M_mb) * Z_mb 
      
  imputed_data = sess.run([G_sample_HeG], feed_dict = {X: X_mb, M: M_mb})[0]
  
  imputed_data = data_m * norm_data_x + (1-data_m) * imputed_data
  
  # Renormalization
  imputed_data = renormalization(imputed_data, norm_parameters)  
  
  # Rounding
  imputed_data = rounding(imputed_data, data_x)  
          
  return imputed_data