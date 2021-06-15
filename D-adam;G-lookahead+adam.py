import tensorflow as tf
import random
import numpy as np
import matplotlib.pyplot as plt
import Scripts
from collections import defaultdict
from Lookahead import Lookahead

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/")
# 55000个图片，每个图片大小28*28
x_train = mnist.train.images[:55000,:]

##random image
# randomNum = random.randint(0,55000)# produce a random number in range of 0 to 55000
# image = x_train[randomNum].reshape([28,28])
# plt.imshow(image, cmap=plt.get_cmap('gray_r'))
# plt.show()

#Our first weight matrix (or filter) will be of size 5x5 and will have a output depth of 8.
def conv2d(x, W):
  return tf.nn.conv2d(input=x, filter=W, strides=[1, 1, 1, 1], padding='SAME')#strides: the first and last elements must be 1. the second is
                                                                              #stride in horizontal direction. the third is stride in vertical direction.

def avg_pool_2x2(x):
  return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')# ksize=[1, height, width, 1]
#
def discriminator(x_image, reuse=False):
    with tf.variable_scope('discriminator') as scope:
        if (reuse):
            tf.get_variable_scope().reuse_variables()
        #First Conv and Pool Layers
        W_conv1 = tf.get_variable('d_wconv1', [5, 5, 1, 8], initializer=tf.truncated_normal_initializer(stddev=0.02))
        b_conv1 = tf.get_variable('d_bconv1', [8], initializer=tf.constant_initializer(0))
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  # 若conv2d(x_image, W_conv1) + b_conv1 大于0，则h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)，否则h_conv1 = 0
        h_pool1 = avg_pool_2x2(h_conv1) #pooling窗口大小是2*2

        #Second Conv and Pool Layers
        W_conv2 = tf.get_variable('d_wconv2', [5, 5, 8, 16], initializer=tf.truncated_normal_initializer(stddev=0.02))
        b_conv2 = tf.get_variable('d_bconv2', [16], initializer=tf.constant_initializer(0))
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = avg_pool_2x2(h_conv2)

        #First Fully Connected Layer
        W_fc1 = tf.get_variable('d_wfc1', [7 * 7 * 16, 32], initializer=tf.truncated_normal_initializer(stddev=0.02))
        b_fc1 = tf.get_variable('d_bfc1', [32], initializer=tf.constant_initializer(0))
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*16])  # tf.reshape(tensor, shape, name=None)将tensor变换为参数shape的形式。 —1代表的用原矩阵的总元素的个数除以另一个确定的维度，就能得到-1维度的值
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1) # tf.matmul(a,b)将矩阵a和矩阵b相乘

        #Second Fully Connected Layer
        W_fc2 = tf.get_variable('d_wfc2', [32, 1], initializer=tf.truncated_normal_initializer(stddev=0.02))
        b_fc2 = tf.get_variable('d_bfc2', [1], initializer=tf.constant_initializer(0))

        #Final Layer
        y_conv=(tf.matmul(h_fc1, W_fc2) + b_fc2)
    return y_conv


def generator(z, batch_size, z_dim, reuse=False):
    with tf.variable_scope('generator') as scope:
        if (reuse):
            tf.get_variable_scope().reuse_variables()
        g_dim = 64  # Number of filters of first layer of generator
        c_dim = 1  # Color dimension of output (MNIST is grayscale, so c_dim = 1 for us)
        s = 28  # Output size of the image
        s2, s4, s8, s16 = int(s / 2), int(s / 4), int(s / 8), int(
            s / 16)  # We want to slowly upscale the image, so these values will help
        # make that change gradual.

        h0 = tf.reshape(z, [batch_size, s16 + 1, s16 + 1, 25])  # z is tensor
        h0 = tf.nn.relu(h0)
        # Dimensions of h0 = batch_size x 2 x 2 x 25

        # First DeConv Layer
        output1_shape = [batch_size, s8, s8, g_dim * 4]
        W_conv1 = tf.get_variable('g_wconv1', [5, 5, output1_shape[-1], int(h0.get_shape()[-1])],
                                  # output1_shape[-1]=g_dim*4,   int(h0.get_shape()[-1])=25
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        b_conv1 = tf.get_variable('g_bconv1', [output1_shape[-1]], initializer=tf.constant_initializer(.1))
        H_conv1 = tf.nn.conv2d_transpose(h0, W_conv1, output_shape=output1_shape,
                                         strides=[1, 2, 2, 1], padding='SAME') + b_conv1
        H_conv1 = tf.contrib.layers.batch_norm(inputs=H_conv1, center=True, scale=True, is_training=True, scope="g_bn1")
        H_conv1 = tf.nn.relu(H_conv1)
        # Dimensions of H_conv1 = batch_size x 3 x 3 x 256

        # Second DeConv Layer
        output2_shape = [batch_size, s4 - 1, s4 - 1, g_dim * 2]
        W_conv2 = tf.get_variable('g_wconv2', [5, 5, output2_shape[-1], int(H_conv1.get_shape()[-1])],
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        b_conv2 = tf.get_variable('g_bconv2', [output2_shape[-1]], initializer=tf.constant_initializer(.1))
        H_conv2 = tf.nn.conv2d_transpose(H_conv1, W_conv2, output_shape=output2_shape,
                                         strides=[1, 2, 2, 1], padding='SAME') + b_conv2
        ###H_conv1:value,is usually a tensor, W_conv2:卷积核，它要求是一个Tensor，具有[filter_height, filter_width, out_channels, in_channels]这样的shape，
        ###具体含义是[卷积核的高度，卷积核的宽度，卷积核个数，图像通道数.

        H_conv2 = tf.contrib.layers.batch_norm(inputs=H_conv2, center=True, scale=True, is_training=True, scope="g_bn2")
        ###Batch Normalization通过减少内部协变量加速神经网络的训练.

        H_conv2 = tf.nn.relu(H_conv2)
        # Dimensions of H_conv2 = batch_size x 6 x 6 x 128

        # Third DeConv Layer
        output3_shape = [batch_size, s2 - 2, s2 - 2, g_dim * 1]
        W_conv3 = tf.get_variable('g_wconv3', [5, 5, output3_shape[-1], int(H_conv2.get_shape()[-1])],
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        b_conv3 = tf.get_variable('g_bconv3', [output3_shape[-1]], initializer=tf.constant_initializer(.1))
        H_conv3 = tf.nn.conv2d_transpose(H_conv2, W_conv3, output_shape=output3_shape,
                                         strides=[1, 2, 2, 1], padding='SAME') + b_conv3
        H_conv3 = tf.contrib.layers.batch_norm(inputs=H_conv3, center=True, scale=True, is_training=True, scope="g_bn3")
        H_conv3 = tf.nn.relu(H_conv3)
        # Dimensions of H_conv3 = batch_size x 12 x 12 x 64

        # Fourth DeConv Layer
        output4_shape = [batch_size, s, s, c_dim]
        W_conv4 = tf.get_variable('g_wconv4', [5, 5, output4_shape[-1], int(H_conv3.get_shape()[-1])],
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        b_conv4 = tf.get_variable('g_bconv4', [output4_shape[-1]], initializer=tf.constant_initializer(.1))
        H_conv4 = tf.nn.conv2d_transpose(H_conv3, W_conv4, output_shape=output4_shape,
                                         strides=[1, 2, 2, 1], padding='VALID') + b_conv4
        H_conv4 = tf.nn.tanh(H_conv4)
        # Dimensions of H_conv4 = batch_size x 28 x 28 x 1

    return H_conv4

#placeholder
sess = tf.Session()
z_dimensions = 100
z_test_placeholder = tf.placeholder(tf.float32, [None, z_dimensions])
 #create a variable (sample_image) that holds the output of the generator,
 #and also initialize the random noise vector that we’ll use as input.
sample_image = generator(z_test_placeholder, 1, z_dimensions)
label_z = discriminator(sample_image, reuse = False)
test_z = np.random.uniform(-1, 1, [1,z_dimensions])
# initialize all the variables, feed our test_z into the placeholder
sess.run(tf.global_variables_initializer())
temp = (sess.run(sample_image, feed_dict={z_test_placeholder: test_z}))

#my_i = temp.squeeze()   #numpy.squeeze(a,axis = None)   axis的取值可为None 或 int 或 tuple of ints, 可选。若axis为空，则删除所有单维度的条目；
#plt.imshow(my_i, cmap='gray_r')
#plt.show()

batch_size = 16
tf.reset_default_graph() #Since we changed our batch size (from 1 to 16), we need to reset our Tensorflow graph

sess = tf.Session()
x_placeholder = tf.placeholder("float", shape = [None,28,28,1]) #Placeholder for input images to the discriminator
z_placeholder = tf.placeholder(tf.float32, [None, z_dimensions]) #Placeholder for input noise vectors to the generator

# the inputs and outputs of our networks
Dx = discriminator(x_placeholder) #Dx will hold discriminator prediction probabilities for the real MNIST images
Gz = generator(z_placeholder, batch_size, z_dimensions) #Gz holds the generated images
Dg = discriminator(Gz, reuse=True) #Dg will hold discriminator prediction probabilities for generated images
g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = Dg, labels = tf.ones_like(Dg))) # ensure forward compatibility: function needs to have logits and labels args explicitly used

#loss function
d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = Dx, labels = tf.ones_like(Dx)))
d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = Dg, labels = tf.zeros_like(Dg)))
d_loss = d_loss_real + d_loss_fake

#we need to create 2 lists, one with the discriminator’s weights and one with the generator’s weights.
tvars = tf.trainable_variables()
d_vars = [var for var in tvars if 'd_' in var.name]
g_vars = [var for var in tvars if 'g_' in var.name]

#we specify our two optimizers
adam = tf.train.AdamOptimizer()
#trainerD = adam.minimize(d_loss, var_list=d_vars)

trainerD = adam.minimize(d_loss, var_list=d_vars)
trainerG = adam.minimize(g_loss, var_list=g_vars)

d_sum_loss = 0
g_sum_loss = 0
d_sum_fakeloss = 0
d_sum_realloss = 0

#opt = adam.minimize(g_loss, var_list=g_vars)
#trainerG = [opt,g_loss,Gz]

list_g_loss = []
list_d_loss = []
list_format = []

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    #g_vars = [var for var in tvars if 'g_' in var.name]

    #lookahea = Lookahead(model_vars=g_vars, k=5, alpha=0.5)
    #trainerG += lookahea.get_ops()


    iterations = 500
    for i in range(iterations):
        z_batch = np.random.uniform(-1, 1, size=[batch_size, z_dimensions])
        real_image_batch = mnist.train.next_batch(batch_size)
        real_image_batch = np.reshape(real_image_batch[0],[batch_size,28,28,1])
        #for k in range(5):
        _, gLoss = sess.run([trainerG, g_loss], feed_dict={z_placeholder: z_batch})  # Update the generator
        _,dLoss,dLoss_fake,dLoss_real = sess.run([trainerD, d_loss, d_loss_fake, d_loss_real],feed_dict={z_placeholder:z_batch,x_placeholder:real_image_batch}) #Update the discriminator


        d_sum_loss += dLoss
        d_sum_fakeloss += dLoss_fake
        d_sum_realloss += dLoss_real

        g_sum_loss += gLoss


        if i % 100 == 0:
            d_sum_loss = d_sum_loss / 100
            g_sum_loss = g_sum_loss / 100
            d_sum_fakeloss = d_sum_fakeloss / 100
            d_sum_realloss = d_sum_realloss / 100
            list_g_loss.append(g_sum_loss)
            list_d_loss.append(d_sum_loss)
            list_format.append(format(i))
            print('Iter: {}'.format(i))
            print('G loss: {:.4f}'.format(gLoss))
            print('D loss: {:.4f}'.format(dLoss))
            print('D Fake loss: {:.4f}'.format(d_sum_fakeloss))
            print('D Real loss: {:.4f}'.format(d_sum_realloss))
            d_sum_loss = 0
            g_sum_loss = 0
            d_sum_fakeloss = 0
            d_sum_realloss = 0

    print("g_loss=", list_g_loss)
    plt.plot(list_format, list_g_loss)
    plt.plot(list_format, list_d_loss)
    plt.show()






    # sample_image = generator(z_placeholder, 1, z_dimensions, reuse=True)
    # z_batch = np.random.uniform(-1, 1, size=[1, z_dimensions])
    # temp = sess.run(sample_image, feed_dict={z_placeholder: z_batch})
    # lable_z = discriminator(temp, reuse=True)
    # my_i = temp.squeeze()
    # print("lable_z=",label_z)
    # plt.imshow(my_i, cmap='gray_r')
    # plt.show()
# #update G and D
# sess.run(tf.global_variables_initializer())
# iterations = 30000
# for i in range(iterations):
#     z_batch = np.random.uniform(-1, 1, size=[batch_size, z_dimensions])
#     real_image_batch = mnist.train.next_batch(batch_size)
#     real_image_batch = np.reshape(real_image_batch[0],[batch_size,28,28,1])
#     _,dLoss = sess.run([trainerD, d_loss],feed_dict={z_placeholder:z_batch,x_placeholder:real_image_batch}) #Update the discriminator
#
#
#     _,gLoss = sess.run([train_op,g_loss],feed_dict={z_placeholder:z_batch}) #Update the generator
#
#
#     if i % 1000 == 0:
#         print('Iter: {}'.format(i))
#         print('G loss: {:.4f}'.format(gLoss))
#         print('D loss: {:.4f}'.format(dLoss))

#a sample image looks like after training.
# sample_image = generator(z_placeholder, 1, z_dimensions, reuse=True)
# z_batch = np.random.uniform(-1, 1, size=[1, z_dimensions])
# temp = sess.run(sample_image, feed_dict={z_placeholder: z_batch})
# my_i = temp.squeeze()
# plt.imshow(my_i, cmap='gray_r')
# plt.show()










