#-*- coding: utf-8 -*-
from __future__ import division
import os
import time
import tensorflow as tf
import numpy as np
import math

import matplotlib.pyplot as plt

from ops import *
from utils import *
from lookahead_11 import LookaheadOptimizer
from glob import glob
import os
import imageio
#import inception_score

#from datetime import datetime
#import matplotlib.pyplot as plt

y_lr = []

class GAN(object):

    def __init__(self, sess, epoch, batch_size, dataset_name, checkpoint_dir,
                 result_dir, log_dir,learningRateD,learningRateG):
        self.sess = sess
        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
        self.result_dir = result_dir
        self.log_dir = log_dir


        self.epoch = epoch
        self.batch_size = batch_size
        self.model_name = "GAN_old"     # name for checkpoint
        self.learningRateD = 0.001
        self.learningRateG = 0.001


        if dataset_name == 'mnist' or dataset_name == 'fashion-mnist': # fix
            # parameters
            self.input_height = 28
            self.input_width = 28
            self.output_height = 28
            self.output_width = 28

            self.z_dim = 62         # dimension of noise-vector
            self.c_dim = 1

            # train
            self.learning_rate = 0.0002
            self.beta1 = 0.5

            # test
            self.sample_num = 64  # number of generated images to be saved

            # load mnist
            self.data_X, self.data_y = load_mnist(self.dataset_name)

            # get number of batches for a single epoch
            self.num_batches = len(self.data_X) // self.batch_size # 700 = 70000 / 100

        elif dataset_name == 'cifar10':
            # parameters
            self.input_height = 32
            self.input_width = 32
            self.output_height = 32
            self.output_width = 32

            self.z_dim = 100         # dimension of noise-vector
            self.c_dim = 3  # color dimension
            self.p = 5

            # train
            #self.learning_rate = 0.0002 # 1e-3, 1e-4
            self.learningRateD = 1e-3
            self.learningRateG = 1e-4
            self.beta1 = 0.5

            # test
            self.sample_num = 64  # number of generated images to be saved

            # load cifar10
            self.data_X =[]
            self.data_y =[]
            data_X1, data_y1 = load_cifar10()
            label =[argmax(one_hot)for one_hot in data_y1]
            print("label",label)

            for i in range (len(data_X1)):
                if label[i] == 4:
                    self.data_X.append(data_X1[i])
                    self.data_y.append(data_y1[i])

            # print("data_X", self.data_X.shape)
            # print("data_Y", self.data_y.shape)

            #validatin images

            '''
            # revice image data // M*N*3 // RGB float32 : value must set between 0. with 1.
            vMin = np.amin(self.data_X[0])
            vMax = np.amax(self.data_X[0])
            img_arr = self.data_X[0].reshape(32*32*3,1) # flatten
            for i, v in enumerate(img_arr):
                img_arr[i] = (v-vMin)/(vMax-vMin)
            img_arr = img_arr.reshape(32,32,3) # M*N*3
            # matplot display
            plt.subplot(1,1,1),plt.imshow(img_arr, interpolation='nearest')
            plt.title("pred.:{}".format(np.argmax(self.data_y[0]),fontsize=10))
            plt.axis("off")
            imgName = "{}.png".format(datetime.now())
            imgName = imgName.replace(":","_")
           ` #plt.savefig(os.path.join(".\\pic_result",imgName))
            plt.savefig(imgName)
            plt.show()`            
            '''

            # get number of batches for a single epoch
            #print(len(self.data_X),len(self.data_y))
            #self.num_batches = self.data_X.get_shape()[0] // self.batch_size
            self.num_batches = len(self.data_X) // self.batch_size
            print("num_batches",self.num_batches)
        else:
            raise NotImplementedError

    def discriminator(self, x, is_training=True, reuse=False):
        # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
        # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
        with tf.variable_scope("discriminator", reuse=reuse):

            if self.dataset_name == 'cifar10':
                print("D:x",x.get_shape()) # 32, 32, 3 = 3072
                net = lrelu(conv2d(x, 64, 5, 5, 2, 2, name='d_conv1'+'_'+self.dataset_name))
                print("D:net1",net.get_shape())
                net = lrelu(bn(conv2d(net, 128, 5, 5, 2, 2, name='d_conv2'+'_'+self.dataset_name), is_training=is_training, scope='d_bn2'))
                print("D:net2",net.get_shape())
                net = lrelu(bn(conv2d(net, 256, 5, 5, 2, 2, name='d_conv3'+'_'+self.dataset_name), is_training=is_training, scope='d_bn3'))
                print("D:net3",net.get_shape())
                net = lrelu(bn(conv2d(net, 512, 5, 5, 2, 2, name='d_conv4'+'_'+self.dataset_name), is_training=is_training, scope='d_bn4'))
                print("D:net4",net.get_shape())
                net = tf.reshape(net, [self.batch_size, -1])#-1代表这一维根据计算得出
                print("D:net5",net.get_shape())
                out_logit = linear(net, 1, scope='d_fc5'+'_'+self.dataset_name)
                print("D:out_logit",out_logit.get_shape())
                out = tf.nn.sigmoid(out_logit)
                print("D:out",out.get_shape())
                print("------------------------")

            else: # mnist / fashion mnist
                #print(x.get_shape())
                net = lrelu(conv2d(x, 64, 4, 4, 2, 2, name='d_conv1'+'_'+self.dataset_name))
                net = lrelu(bn(conv2d(net, 128, 4, 4, 2, 2, name='d_conv2'+'_'+self.dataset_name), is_training=is_training, scope='d_bn2'))
                net = tf.reshape(net, [self.batch_size, -1])
                net = lrelu(bn(linear(net, 1024, scope='d_fc3'+'_'+self.dataset_name), is_training=is_training, scope='d_bn3'))
                out_logit = linear(net, 1, scope='d_fc4'+'_'+self.dataset_name)
                out = tf.nn.sigmoid(out_logit)

            return out, out_logit, net

    def generator(self, z, is_training=True, reuse=False):
        # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
        # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
        with tf.variable_scope("generator", reuse=reuse):

            if self.dataset_name == 'cifar10':
                h_size = 32
                h_size_2 = 16
                h_size_4 = 8
                h_size_8 = 4
                h_size_16 = 2

                print("G:z",z.get_shape())
                net = linear(z, 512*h_size_16*h_size_16, scope='g_fc1'+'_'+self.dataset_name)
                print("G:net_linear",net.get_shape())
                net = tf.nn.relu(
                    bn(tf.reshape(net, [self.batch_size, h_size_16, h_size_16, 512]),is_training=is_training, scope='g_bn1')
                    )
                print("G:tf.nn.relu",net.get_shape())
                net = tf.nn.relu(
                    bn(deconv2d(net, [self.batch_size, h_size_8, h_size_8, 256], 5, 5, 2, 2, name='g_dc2'+'_'+self.dataset_name),is_training=is_training, scope='g_bn2')
                    )
                print("G:dconv1",net.get_shape())
                net = tf.nn.relu(
                    bn(deconv2d(net, [self.batch_size, h_size_4, h_size_4, 128], 5, 5, 2, 2, name='g_dc3'+'_'+self.dataset_name),is_training=is_training, scope='g_bn3')
                    )
                print("G:dconv2",net.get_shape())
                net = tf.nn.relu(
                    bn(deconv2d(net, [self.batch_size, h_size_2, h_size_2, 64], 5, 5, 2, 2, name='g_dc4'+'_'+self.dataset_name),is_training=is_training, scope='g_bn4')
                    )
                print("G:dconv3",net.get_shape())
                out = tf.nn.tanh(
                    deconv2d(net, [self.batch_size, self.output_height, self.output_width, self.c_dim], 5, 5, 2, 2, name='g_dc5'+'_'+self.dataset_name)
                    )
                print("G:dconv4 out",out.get_shape())
                #print("------------------------")

            else: # mnist / fashon mnist
                h_size = 28
                h_size_2 = 14
                h_size_4 = 7

                net = tf.nn.relu(bn(linear(z, 1024, scope='g_fc1'+'_'+self.dataset_name), is_training=is_training, scope='g_bn1'))
                net = tf.nn.relu(bn(linear(net, 128 * h_size_4 * h_size_4, scope='g_fc2'+'_'+self.dataset_name), is_training=is_training, scope='g_bn2'))
                net = tf.reshape(net, [self.batch_size, h_size_4, h_size_4, 128]) #  8 8  128
                net = tf.nn.relu(
                    bn(deconv2d(net, [self.batch_size, h_size_2, h_size_2, 64], 4, 4, 2, 2, name='g_dc3'+'_'+self.dataset_name), is_training=is_training,scope='g_bn3')
                    )

                out = tf.nn.sigmoid(deconv2d(net, [self.batch_size, self.output_height, self.output_width, self.c_dim], 4, 4, 2, 2, name='g_dc4'+'_'+self.dataset_name))

            return out

    def build_model(self):

        image_dims = [self.input_height, self.input_width, self.c_dim]
        bs = self.batch_size # 100

        """ Graph Input """
        # images
        self.inputs = tf.placeholder(tf.float32, [bs] + image_dims, name='real_images')
        #!!!!!!!!!!!!!!!!!!!



        self.global_step = tf.placeholder(tf.int32)
        #self.decay_steps = tf.placeholder(tf.int32)
        self.step_epoch = tf.placeholder(tf.float32, [1,1])





        # noises
        self.z = tf.placeholder(tf.float32, [bs, self.z_dim], name='z')
        print("self.z",self.z.shape)
        self.learningRateD = 1e-3

        """ Loss Function """

        # output of D for real images
        D_real, D_real_logits, _ = self.discriminator(self.inputs, is_training=True, reuse=False)

        # output of D for fake images
        G = self.generator(self.z, is_training=True, reuse=False)
        D_fake, D_fake_logits, _ = self.discriminator(G, is_training=True, reuse=True)

        # get loss for discriminator
        d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits, labels=tf.ones_like(D_real)))
        d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.zeros_like(D_fake)))

        self.d_loss = d_loss_real + d_loss_fake

        # get loss for generator
        self.g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.ones_like(D_fake)))

        """ Training """
        # divide trainable variables into a group for D and a group for G
        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'd_' in var.name]
        g_vars = [var for var in t_vars if 'g_' in var.name]

        # optimizers
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):

            #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

                # 重启余弦衰减

            # self.learning_rate = tf.train.cosine_decay_restarts(learning_rate=0.001, global_step=self.global_step,
            #                                                     first_decay_steps=tf.ceil(10/(tf.exp(self.global_step/20, name=None))))
            #
            # self.learning_rate = tf.train.cosine_decay_restarts(learning_rate=0.001, global_step=self.global_step,
            #                                                     first_decay_steps = 10)
            # self.d_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1).minimize(self.d_loss, var_list=d_vars)
            #########################  Restart learning rate
            self.learning_rate = tf.train.cosine_decay_restarts(learning_rate=0.001, global_step=self.global_step,
                                                                first_decay_steps=40, t_mul=1.0, m_mul=0.8,
                                                                alpha=0)
            self.d_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1).minimize(self.d_loss,
                                                                                                 var_list=d_vars)
            ##################################################

            ############################# Stabel learning rate
            # self.d_optim = tf.train.AdamOptimizer(0.001, beta1=self.beta1).minimize(self.d_loss, var_list=d_vars)

            ##################################################
            self.g_optim = tf.train.AdamOptimizer(0.001, beta1=self.beta1).minimize(self.g_loss, var_list=g_vars)

        """" Testing """
        self.fake_images = self.generator(self.z, is_training=False, reuse=True)

        """ Summary """
        d_loss_real_sum = tf.summary.scalar("d_loss_real", d_loss_real)
        d_loss_fake_sum = tf.summary.scalar("d_loss_fake", d_loss_fake)
        d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
        g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)


        # final summary operations
        self.g_sum = tf.summary.merge([d_loss_fake_sum, g_loss_sum])
        self.d_sum = tf.summary.merge([d_loss_real_sum, d_loss_sum])

    def train(self):

        # initialize all variables
        tf.global_variables_initializer().run()

        # graph inputs for visualize training results
        self.sample_z = np.random.uniform(-1, 1, size=(self.batch_size , self.z_dim)) # 100, 62
        self.test_images = self.data_X[0:self.batch_size]

        # saver to save model
        self.saver = tf.train.Saver()

        # summary writer
        self.writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)

        # restore check-point if it exits
        #could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        # if could_load:
        #     start_epoch = (int)(checkpoint_counter / self.num_batches)
        #     start_batch_id = checkpoint_counter - start_epoch * self.num_batches
        #     counter = checkpoint_counter
        #     print(" [*] Load SUCCESS")
        #     print(" [!] START_EPOCH is ",start_epoch," START_BATCH_ID is ", start_batch_id)
        # else:
        start_epoch = 0
        start_batch_id = 0
        counter = 1
            #print(" [!] Load failed...")
        # t_vars = tf.trainable_variables()
        # d_vars = [var for var in t_vars if 'd_' in var.name]
        # g_vars = [var for var in t_vars if 'g_' in var.name]
        # self.d_optim = tf.train.AdamOptimizer(self.learningRateD, beta1=self.beta1).minimize(self.d_loss,var_list=d_vars)
        # self.g_optim = tf.train.AdamOptimizer(self.learningRateG, beta1=self.beta1).minimize(self.g_loss,var_list=g_vars)

        # loop for epoch
        start_time = time.time()
        i = 0
        y_lr = []
        y__lr =  []
        D_loss = []
        G_loss = []

        fid = []

        for epoch in range(start_epoch, self.epoch):
            # if epoch%5 ==0:
            #     tf.global_variables_initializer().run()
            # get batch data
            for idx in range(start_batch_id, self.num_batches):
                i += 1
                batch_images = self.data_X[idx*self.batch_size:(idx+1)*self.batch_size]
                batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)
                ##FID



                # update D network

                ###################################### Restart learning rate
                _, summary_str, d_loss,lr= self.sess.run([self.d_optim, self.d_sum, self.d_loss, self.learning_rate],
                                                             feed_dict={self.inputs: batch_images, self.z: batch_z, self.global_step:epoch})
                ###################################################################

                ####################################  Stabel learning rate
                # _, summary_str, d_loss = self.sess.run([self.d_optim, self.d_sum, self.d_loss],
                #         feed_dict={self.inputs: batch_images, self.z: batch_z})

                ###################################################################
                # _, summary_str, d_loss, lr = self.sess.run([self.d_optim, self.d_sum, self.d_loss, self.learning_rate],
                #                                            feed_dict={self.inputs: batch_images, self.z: batch_z,
                #                                                       self.global_step: epoch * 50})
                self.writer.add_summary(summary_str, counter)
                D_loss.append(d_loss)



                # update G network
                #self.sess.run([self.g_optim], feed_dict={self.inputs: batch_images, self.z: batch_z})
                # update G twice to make sure that d_loss does not go to zero



                _, summary_str, g_loss = self.sess.run([self.g_optim, self.g_sum, self.g_loss],
                                                          feed_dict={self.inputs: batch_images, self.z: batch_z})
                G_loss.append(g_loss)
                self.writer.add_summary(summary_str, counter)
                # y_lr.append(lr)
                # display training status
                counter += 1
            print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" % (epoch, idx, self.num_batches, time.time() - start_time, d_loss, g_loss))


            # y__lr.append(y_lr[len(y_lr)-1])
            # print("~~~~~~~~~~~~~~~~~~~~~lr =", y__lr)




                #save training results for every 300 steps
                # if np.mod(counter, 300) == 0:
                #     samples = self.sess.run(self.fake_images,
                #                             feed_dict={self.z: self.sample_z, self.inputs: self.test_images})
                #     tot_num_samples = min(self.sample_num, self.batch_size)  # 64
                #     manifold_h = int(np.floor(np.sqrt(tot_num_samples)))  # 8
                #     manifold_w = int(np.floor(np.sqrt(tot_num_samples)))  # 8
                #     save_images(samples[:manifold_h * manifold_w, :, :, :], [manifold_h, manifold_w],
                #                 '.\\' + self.result_dir + '\\' + self.model_name + '_train_{:02d}_{:04d}.png'.format(
                #                     epoch, idx))

            # After an epoch, start_batch_id is set to zero
            # non-zero value is only for the first epoch after loading pre-trained model
            start_batch_id = 0

            # save model
            #print(self.checkpoint_dir)
            self.save(self.checkpoint_dir, counter)

            # show temporal results

            if epoch > 299:
                self.visualize_results(epoch)
                print("D_loss=",D_loss)
                print("G_loss=",G_loss)
                # axis_D = list(range(len(D_loss) - 1))
                # mean_D, variance_D = tf.nn.moments(D_loss, axes_D=0)
                # axis_G = list(range(len(G_loss) - 1))
                # mean_G, variance_G = tf.nn.moments(G_loss, axes_G=0)
                # print("mean_D=",mean_D)
                # print("variance_D=", variance_D)
                # print("mean_G=", mean_G)
                # print("variance_G=", variance_G)


        # save model for final step
        self.save(self.checkpoint_dir, counter)
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # x_lr = range(self.epoch)
        # plt.plot(x_lr, y_lr, 'r-', linewidth=2)
        # plt.title('cosine_decay')
        # plt.show()

        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    def visualize_results(self, epoch):
        tot_num_samples = min(self.sample_num, self.batch_size) # 64, 100
        image_frame_dim = int(np.floor(np.sqrt(tot_num_samples))) # 8

        """ random condition, random noise """

        z_sample = np.random.uniform(-1, 1, size=(self.batch_size, self.z_dim)) # 100, 100

        samples = self.sess.run(self.fake_images, feed_dict={self.z: z_sample})
        #
        # save_matplot_img(samples[:image_frame_dim * image_frame_dim, :, :, :], [1, 1],
        #             'D:/test11111111111111/test11111111111111/results_single' + '/' + self.model_name + '_epoch%03d' % epoch + '_test_all_classes.png')
        if epoch>298:
            for i in range(50):
                image_path = 'D:/test11111111111111/test11111111111111/images_Ablation\s_class4' + '/' + self.model_name + 'number%03d' %i
                plt.imshow(samples[i], interpolation='nearest')
                plt.axis("off")
                plt.savefig(image_path)

        # filenames = glob(os.path.join('./target_ship', '*.*'))
        # file_name = imageio.imread(filenames)
        # really_images = [file_name for filename in filenames]
        # really_images = np.array(really_images)
        # samples = samples[:50]
        # inception_score.calculate_FID(samples)

    @property
    def model_dir(self):
        return "{}_{}_{}_{}".format(
            self.dataset_name, self.batch_size,
            self.output_height, self.output_width)

    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        dd = os.path.join(checkpoint_dir, self.model_name+'.model')
        #print("saving to...", dd)

        self.saver.save(self.sess,os.path.join(checkpoint_dir, self.model_name+'.model'), global_step=step)

    def load(self, checkpoint_dir):
        import re
        #print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
            #print(" [*] Success to read [{}], counter [{}]".format(ckpt_name,counter))
            return True, counter
        else:
            #print(" [*] Failed to find a checkpoint")
            return False, 0