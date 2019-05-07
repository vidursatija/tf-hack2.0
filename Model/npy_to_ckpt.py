# convert npy to checkpoint and test if works
import time
import numpy as np
import tensorflow as tf
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()

VGG_MEAN = [103.939, 116.779, 123.68]

class Vgg16(tf.Module):
    def __init__(self, vgg16_npy_path):
        super(Vgg16, self).__init__()
        print("Class VGG16")
        self.data_dict = np.load(vgg16_npy_path, encoding='latin1').item()
        print("npy file loaded")
        self.var_dict = dict()
        self.define_conv_layer("conv1_1") # , ksize=3, in_filters=3, out_filters=64)
        self.define_conv_layer("conv1_2") # , ksize=3, in_filters=64, out_filters=64)

        self.define_conv_layer("conv2_1") # , ksize=3, in_filters=64, out_filters=128)
        self.define_conv_layer("conv2_2") # , ksize=3, in_filters=128, out_filters=128)

        self.define_conv_layer("conv3_1") # , ksize=3, in_filters=128, out_filters=256)
        self.define_conv_layer("conv3_2") # , ksize=3, in_filters=256, out_filters=256)
        self.define_conv_layer("conv3_3") # , ksize=3, in_filters=256, out_filters=256)

        self.define_conv_layer("conv4_1") # , ksize=3, in_filters=256, out_filters=512)
        self.define_conv_layer("conv4_2") # , ksize=3, in_filters=512, out_filters=512)
        self.define_conv_layer("conv4_3") # , ksize=3, in_filters=512, out_filters=512)

        self.define_conv_layer("conv5_1") # , ksize=3, in_filters=512, out_filters=512)
        self.define_conv_layer("conv5_2") # , ksize=3, in_filters=512, out_filters=512)
        self.define_conv_layer("conv5_3") # , ksize=3, in_filters=512, out_filters=512)

        self.define_fc_layer("fc6") # , in_size=7*7*512, out_size=4096)
        self.define_fc_layer("fc7") # , in_size=4096, out_size=4096)
        self.define_fc_layer("fc8") # , in_size=4096, out_size=1000)

    @tf.function(input_signature=[tf.TensorSpec([None, 224, 224, 3], tf.float32)])
    def __call__(self, bgr):
        """
        load variable from npy to build the VGG

        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        """

#         rgb_scaled = rgb * 255.0

#         # Convert RGB to BGR
#         red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)
#         bgr = tf.concat(axis=3, values=[
#             blue - VGG_MEAN[0],
#             green - VGG_MEAN[1],
#             red - VGG_MEAN[2],
#         ])

        conv1_1 = self.conv_layer(bgr, "conv1_1", ksize=3, in_filters=3, out_filters=64)
        conv1_2 = self.conv_layer(conv1_1, "conv1_2", ksize=3, in_filters=64, out_filters=64)
        pool1 = self.max_pool(conv1_2, 'pool1')

        conv2_1 = self.conv_layer(pool1, "conv2_1", ksize=3, in_filters=64, out_filters=128)
        conv2_2 = self.conv_layer(conv2_1, "conv2_2", ksize=3, in_filters=128, out_filters=128)
        pool2 = self.max_pool(conv2_2, 'pool2')

        conv3_1 = self.conv_layer(pool2, "conv3_1", ksize=3, in_filters=128, out_filters=256)
        conv3_2 = self.conv_layer(conv3_1, "conv3_2", ksize=3, in_filters=256, out_filters=256)
        conv3_3 = self.conv_layer(conv3_2, "conv3_3", ksize=3, in_filters=256, out_filters=256)
        pool3 = self.max_pool(conv3_3, 'pool3')

        conv4_1 = self.conv_layer(pool3, "conv4_1", ksize=3, in_filters=256, out_filters=512)
        conv4_2 = self.conv_layer(conv4_1, "conv4_2", ksize=3, in_filters=512, out_filters=512)
        conv4_3 = self.conv_layer(conv4_2, "conv4_3", ksize=3, in_filters=512, out_filters=512)
        pool4 = self.max_pool(conv4_3, 'pool4')

        conv5_1 = self.conv_layer(pool4, "conv5_1", ksize=3, in_filters=512, out_filters=512)
        conv5_2 = self.conv_layer(conv5_1, "conv5_2", ksize=3, in_filters=512, out_filters=512)
        conv5_3 = self.conv_layer(conv5_2, "conv5_3", ksize=3, in_filters=512, out_filters=512)
        pool5 = self.max_pool(conv5_3, 'pool5')

        fc6 = self.fc_layer(pool5, "fc6", in_size=7*7*512, out_size=4096)
        relu6 = tf.nn.relu(fc6)

        fc7 = self.fc_layer(relu6, "fc7", in_size=4096, out_size=4096)
        relu7 = tf.nn.relu(fc7)

        fc8 = self.fc_layer(relu7, "fc8", in_size=4096, out_size=1000)
        fc8 = tf.nn.softmax(fc8, axis=-1)
        
        return fc8

    @tf.function
    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    @tf.function
    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def define_conv_layer(self, name): # , ksize, in_filters, out_filters):
        # with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        self.var_dict[name+"_filt"] = self.get_conv_filter(name) # , ksize, in_filters, out_filters)
        self.var_dict[name+"_conv_biases"] = self.get_bias(name) # , out_filters)

    def define_fc_layer(self, name): # , in_size, out_size):
        # with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        self.var_dict[name+"_weights"] = self.get_fc_weight(name) # , in_size, out_size)
        self.var_dict[name+"_biases"] = self.get_bias(name) # , out_size)
    
    @tf.function
    def conv_layer(self, bottom, name, ksize, in_filters, out_filters):
        # with tf.variable_scope(name):
        conv = tf.nn.conv2d(bottom, self.var_dict[name+"_filt"], [1, 1, 1, 1], padding='SAME')

        bias = tf.nn.bias_add(conv, self.var_dict[name+"_conv_biases"])

        relu = tf.nn.relu(bias)
        return relu

    @tf.function
    def fc_layer(self, bottom, name, in_size, out_size):
        # with tf.variable_scope(name):
        x = tf.reshape(bottom, [-1, in_size])

        fc = tf.nn.bias_add(tf.matmul(x, self.var_dict[name+"_weights"]), self.var_dict[name+"_biases"])

        return fc

    def get_conv_filter(self, name):
        return tf.Variable(tf.constant(self.data_dict[name][0]), name=name+"_filter")

    def get_bias(self, name):
        return tf.Variable(tf.constant(self.data_dict[name][1]), name=name+"_biases")

    def get_fc_weight(self, name):
        return tf.Variable(tf.constant(self.data_dict[name][0]), name=name+"_weights")

# with tf.Session(config=tf.ConfigProto(gpu_options=(tf.GPUOptions(per_process_gpu_memory_fraction=0.7)))) as sess:
v = Vgg16("vgg16.npy")
var_list = [v for k, v in v.var_dict.items()]
checkpoint = tf.train.Checkpoint(model=v)
checkpoint.save("vgg16-tf1-13-1")
        