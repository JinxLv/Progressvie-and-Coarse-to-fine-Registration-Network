from re import T
from keras import layers
import tensorflow as tf
import tflearn
from keras.layers import Conv3D, Activation, UpSampling3D,BatchNormalization,Conv3DTranspose,Add,Concatenate,Dropout
from tflearn.initializations import normal
from .spatial_transformer import Dense3DSpatialTransformer
from .utils import Network, ReLU, LeakyReLU,Softmax,Sigmoid,channel_attention
from tensorflow.contrib.layers import instance_norm
import keras.backend as K
import keras
from .layers import VecInt

#from keras.layers.core import Lambda

def convolve(opName, inputLayer, outputChannel, kernelSize, stride, stddev=1e-2, reuse=False, weights_init='uniform_scaling'):
    return tflearn.layers.conv_3d(inputLayer, outputChannel, kernelSize, strides=stride,
                                  padding='same', activation='linear', bias=True, scope=opName, reuse=reuse, weights_init=weights_init)

def leakyReLU(inputLayer,opName, alpha = 0.1):
    return LeakyReLU(inputLayer,alpha,opName+'_leakilyrectified')

def inLeakyReLU(inputLayer,opName,alpha = 0.1):
    IN = instance_norm(inputLayer, scope=opName+'_IN')
    return LeakyReLU(IN,alpha,opName+'_leakilyrectified')

def convInLeakyReLU(opName, inputLayer, outputChannel, kernelSize, stride, alpha=0.1, stddev=1e-2, reuse=False):
    conv = convolve(opName, inputLayer,outputChannel, kernelSize, stride, stddev, reuse)
    conv_In = instance_norm(conv, scope=opName+'_IN')
    return LeakyReLU(conv_In,alpha,opName+'_leakilyrectified')

def convolveReLU(opName, inputLayer, outputChannel, kernelSize, stride, stddev=1e-2, reuse=False):
    return ReLU(convolve(opName, inputLayer,
                         outputChannel,
                         kernelSize, stride, stddev=stddev, reuse=reuse),
                opName+'_rectified')

def convolveSoftmax(opName, inputLayer, outputChannel, kernelSize, stride, stddev=1e-2, reuse=False):
    return Softmax(convolve(opName, inputLayer,
                         outputChannel,
                         kernelSize, stride, stddev=stddev, reuse=reuse),
                opName+'_softmax')

def convolveSigmoid(opName, inputLayer, outputChannel, kernelSize, stride, stddev=1e-2, reuse=False):
    return Sigmoid(convolve(opName, inputLayer,
                         outputChannel,
                         kernelSize, stride, stddev=stddev, reuse=reuse),
                opName+'_sigmoid')

def convolveLeakyReLU(opName, inputLayer, outputChannel, kernelSize, stride, alpha=0.1, stddev=1e-2, reuse=False):
    return LeakyReLU(convolve(opName, inputLayer,
                              outputChannel,
                              kernelSize, stride, stddev, reuse),
                     alpha, opName+'_leakilyrectified')

def upconvolve(opName, inputLayer, outputChannel, kernelSize, stride, targetShape, stddev=1e-2, reuse=False, weights_init='uniform_scaling'):
    return tflearn.layers.conv.conv_3d_transpose(inputLayer, outputChannel, kernelSize, targetShape, strides=stride,
                                                 padding='same', activation='linear', bias=False, scope=opName, reuse=reuse, weights_init=weights_init)


def upconvolveLeakyReLU(opName, inputLayer, outputChannel, kernelSize, stride, targetShape, alpha=0.1, stddev=1e-2, reuse=False):
    return LeakyReLU(upconvolve(opName, inputLayer,
                                outputChannel,
                                kernelSize, stride,
                                targetShape, stddev, reuse),
                     alpha, opName+'_rectified')

def upconvolveInLeakyReLU(opName, inputLayer, outputChannel, kernelSize, stride, targetShape, alpha=0.1, stddev=1e-2, reuse=False):
    up_in = instance_norm(upconvolve(opName, inputLayer,outputChannel,kernelSize, stride,targetShape, stddev, reuse), scope=opName+'_IN')
    return LeakyReLU(up_in,alpha, opName+'_rectified')


class PCNet(Network):
    def __init__(self, name, flow_multiplier=1., channels=16, **kwargs):
        super().__init__(name, **kwargs)
        self.flow_multiplier = flow_multiplier
        self.channels = channels
        self.step = 7
        self.reconstruction = Dense3DSpatialTransformer()
        
    def build(self, img1,img2):
        '''
            img1, img2, flow : tensor of shape [batch, X, Y, Z, C]
        '''
        dims = 3
        c = self.channels
        def resblock(inputLayer,opName,channel):
            residual = inputLayer
            conv1 = inLeakyReLU(inputLayer,opName)
            conv1_1_name  = opName[:opName.find('_')]+'_1'+opName[opName.find('_'):]
            print(conv1_1_name)
            conv1_1 = convolve(conv1_1_name,conv1, channel,   3, 1)
            add1 = Add()([conv1_1, residual])
            conv1_1 = inLeakyReLU(add1,conv1_1_name)
            return conv1_1
        
        #Encoding fixed image
        conv0_fixed = convInLeakyReLU('conv0_fixed',   img1, c,   3, 1)

        conv1_fixed = convolve('conv1_fixed',   conv0_fixed, 2*c,   3, 2) 
        conv1_1_fixed = resblock(conv1_fixed,'conv1_fixed',2*c)
        
        conv2_fixed = convolve('conv2_fixed',   conv1_1_fixed,      4*c,   3, 2) 
        conv2_1_fixed = resblock(conv2_fixed,'conv2_fixed',4*c) 

        conv3_fixed = convolve('conv3_fixed',   conv2_1_fixed,      8*c,   3, 2)
        conv3_1_fixed = resblock(conv3_fixed,'conv3_fixed',8*c)

        #Encoding moving image
        conv0_float = convInLeakyReLU('conv0_float',   img2, c,   3, 1)

        conv1_float = convolve('conv1_float',   conv0_float,2*c,   3, 2)
        conv1_1_float = resblock(conv1_float,'conv1_float',  2*c)

        conv2_float = convolve('conv2_float',   conv1_1_float,      4*c,   3, 2)
        conv2_1_float = resblock(conv2_float,'conv2_float',  4*c )

        conv3_float = convolve('conv3_float',   conv2_1_float,      8*c,   3, 2)
        conv3_1_float = resblock(conv3_float,'conv3_float',   8*c) 

        shape0 = conv0_fixed.shape.as_list()
        shape1 = conv1_fixed.shape.as_list()
        shape2 = conv2_fixed.shape.as_list()
        shape3 = conv3_fixed.shape.as_list()

        concat_bottleNeck = tf.concat([conv3_1_fixed,conv3_1_float],4,'concat_bottleNeck')
        concat_bottleNeck = convInLeakyReLU('conv_bottleNeck_contact_1', concat_bottleNeck,   8*c,  3, 1)
        concat_bottleNeck = convInLeakyReLU('conv_bottleNeck_contact_2', concat_bottleNeck,  8*c,  3, 1)

        predict_cache = []

        #   warping scale 2
        pred3 = convolve('pred3', concat_bottleNeck, dims, 3, 1)
        predict_cache.append(pred3)
        
        deconv2 = upconvolveInLeakyReLU('deconv2', concat_bottleNeck, shape2[4], 4, 2, shape2[1:4])
        warping_field_2 = self.DFI_module(predict_cache)
        conv2_1_float = self.reconstruction([conv2_1_float,warping_field_2])
        concat2 = self.NFF_module(conv2_1_fixed,conv2_1_float, deconv2,2)

        #   warping scale 1
        pred2 = convolve('pred2', concat2, dims, 3, 1)#20*20*20
        predict_cache.append(pred2)
        
        deconv1 = upconvolveInLeakyReLU('deconv1', concat2, shape1[4], 4, 2, shape1[1:4])
        warping_field_1 = self.DFI_module(predict_cache)
        conv1_1_float = self.reconstruction([conv1_1_float,warping_field_1])
        concat1 = self.NFF_module(conv1_1_fixed,conv1_1_float, deconv1,1)
        
        #   warping scale 0
        pred1 = convolve('pred1', concat1, dims, 3, 1)#80*80*80 
        predict_cache.append(pred1)
        
        deconv0 = upconvolveInLeakyReLU('deconv0', concat1, shape0[4], 4, 2, shape0[1:4])
        warping_field_0 = self.DFI_module(predict_cache)
        conv0_float = self.reconstruction([conv0_float,warping_field_0])
        concat0 = self.NFF_module(conv0_fixed,conv0_float, deconv0,0)

        concat0 = convolve('concat_conv', concat0, c, 3, 1)
        pred0 = convolve('pred0', concat0, dims, 3, 1) 
        
        #final layer's deformation field
        pred0 = VecInt(method='ss',name = 'VTN_flow_int',int_steps= self.step)(pred0)
        progress_0 = self.reconstruction([warping_field_0,pred0])+pred0
        
        return {'flow': progress_0}


    def DFI_module(self,prediction_list):
        list_num = len(prediction_list)
        level = 5 - list_num
        print(len(prediction_list))
        
        prediction_cache = []
        channel = 16
        for i,prediction in enumerate(prediction_list):
            prediction_cache.append(UpSampling3D(size = (2**(list_num-i),2**(list_num-i),2**(list_num-i)))(prediction))

        concatenate_0 = tf.concat(prediction_cache,4,'att_concat_{}_0'.format(level))
        conv_1 = convolve('att_conv_{}_1'.format(level),concatenate_0,channel*list_num,3,1)
        conv_2 = convolve('att_conv_{}_2'.format(level),conv_1,channel*list_num,3,1)
        
        for i,prediction in enumerate(prediction_cache):
            weight_map = convolveSigmoid('att_weight_{}_{}'.format(level,i),conv_2,3,3,1)
            prediction_cache[i] = tf.multiply(prediction,weight_map)
            if i==0:
                progress_field = prediction_cache[i]
            else:
                progress_field = progress_field+prediction_cache[i]
            print(progress_field)
        
        progress_field = VecInt(method='ss',name = 'VTN_flow_int',int_steps= self.step)(progress_field)
        return progress_field

    def NFF_module(self,fixed_fm,float_fm,decon_fm,level):
        concatenate_fm = tf.concat([float_fm,fixed_fm,decon_fm],4,'att_fusion')
        
        channel = fixed_fm.shape.as_list()[4]
        conv_1 = convolve('att_fusion_conv_{}_1'.format(level),concatenate_fm,channel,1,1)
        conv_2 = convolve('att_fusion_conv_{}_2'.format(level),conv_1,channel,3,1)
        weight_map = convolveSoftmax('att_fusion_conv__{}_3'.format(level),conv_2,3,3,1)
        
        concatenate_1 = tf.concat([tf.multiply(float_fm,tf.expand_dims(weight_map[...,0],-1)),tf.multiply(fixed_fm,tf.expand_dims(weight_map[...,1],-1)),\
            tf.multiply(decon_fm,tf.expand_dims(weight_map[...,2],-1))],4,'att_fusion_concat_{}_1'.format(level))
        channel_wise = channel_attention(concatenate_1,'channel_wise_att_%d'%level)#,'channel_wise_att_%d'%level)
        return concatenate_1*channel_wise


class VTN(Network):
    def __init__(self, name, flow_multiplier=1., channels=16, **kwargs):
        super().__init__(name, **kwargs)
        self.flow_multiplier = flow_multiplier
        self.channels = channels

    def build(self, img1, img2):
        '''
            img1, img2, flow : tensor of shape [batch, X, Y, Z, C]
        '''
        concatImgs = tf.concat([img1, img2], 4, 'concatImgs')

        dims = 3
        c = self.channels
        conv1 = convolveLeakyReLU(
            'conv1',   concatImgs, c,   3, 2)  # 64 * 64 * 64
        conv2 = convolveLeakyReLU(
            'conv2',   conv1,      c*2,   3, 2)  # 32 * 32 * 32
        conv3 = convolveLeakyReLU('conv3',   conv2,      c*4,   3, 2)
        conv3_1 = convolveLeakyReLU('conv3_1', conv3,      c*4,   3, 1)
        conv4 = convolveLeakyReLU(
            'conv4',   conv3_1,    c*8,  3, 2)  # 16 * 16 * 16
        conv4_1 = convolveLeakyReLU('conv4_1', conv4,      c*8,  3, 1)
        conv5 = convolveLeakyReLU(
            'conv5',   conv4_1,    c*16,  3, 2)  # 8 * 8 * 8
        conv5_1 = convolveLeakyReLU('conv5_1', conv5,      c*16,  3, 1)
        conv6 = convolveLeakyReLU(
            'conv6',   conv5_1,    c*32,  3, 2)  # 4 * 4 * 4
        conv6_1 = convolveLeakyReLU('conv6_1', conv6,      c*32,  3, 1)
        # 16 * 32 = 512 channels

        shape0 = concatImgs.shape.as_list()
        shape1 = conv1.shape.as_list()
        shape2 = conv2.shape.as_list()
        shape3 = conv3.shape.as_list()
        shape4 = conv4.shape.as_list()
        shape5 = conv5.shape.as_list()
        shape6 = conv6.shape.as_list()

        pred6 = convolve('pred6', conv6_1, dims, 3, 1)
        upsamp6to5 = upconvolve('upsamp6to5', pred6, dims, 4, 2, shape5[1:4])
        deconv5 = upconvolveLeakyReLU(
            'deconv5', conv6_1, shape5[4], 4, 2, shape5[1:4])
        concat5 = tf.concat([conv5_1, deconv5, upsamp6to5], 4, 'concat5')

        pred5 = convolve('pred5', concat5, dims, 3, 1)
        upsamp5to4 = upconvolve('upsamp5to4', pred5, dims, 4, 2, shape4[1:4])
        deconv4 = upconvolveLeakyReLU(
            'deconv4', concat5, shape4[4], 4, 2, shape4[1:4])
        concat4 = tf.concat([conv4_1, deconv4, upsamp5to4],
                            4, 'concat4')  # channel = 512+256+2

        pred4 = convolve('pred4', concat4, dims, 3, 1)
        upsamp4to3 = upconvolve('upsamp4to3', pred4, dims, 4, 2, shape3[1:4])
        deconv3 = upconvolveLeakyReLU(
            'deconv3', concat4, shape3[4], 4, 2, shape3[1:4])
        concat3 = tf.concat([conv3_1, deconv3, upsamp4to3],
                            4, 'concat3')  # channel = 256+128+2

        pred3 = convolve('pred3', concat3, dims, 3, 1)
        upsamp3to2 = upconvolve('upsamp3to2', pred3, dims, 4, 2, shape2[1:4])
        deconv2 = upconvolveLeakyReLU(
            'deconv2', concat3, shape2[4], 4, 2, shape2[1:4])
        concat2 = tf.concat([conv2, deconv2, upsamp3to2],
                            4, 'concat2')  # channel = 128+64+2

        pred2 = convolve('pred2', concat2, dims, 3, 1)
        upsamp2to1 = upconvolve('upsamp2to1', pred2, dims, 4, 2, shape1[1:4])
        deconv1 = upconvolveLeakyReLU(
            'deconv1', concat2, shape1[4], 4, 2, shape1[1:4])
        concat1 = tf.concat([conv1, deconv1, upsamp2to1], 4, 'concat1')
        pred0 = upconvolve('upsamp1to0', concat1, dims, 4, 2, shape0[1:4])

        return {'flow': pred0 * 20 * self.flow_multiplier}

class VoxelMorph(Network):
    def __init__(self, name, flow_multiplier=1., channels=16, **kwargs):
        super().__init__(name, **kwargs)
        self.flow_multiplier = flow_multiplier
        self.encoders = [m * channels for m in [1, 2, 2, 2]]
        self.decoders = [m * channels for m in [2, 2, 2, 2, 2, 1, 1]] + [3]

    def build(self, img1, img2):
        '''
            img1, img2, flow : tensor of shape [batch, X, Y, Z, C]
        '''
        concatImgs = tf.concat([img1, img2], 4, 'concatImgs')

        conv1 = convolveLeakyReLU(
            'conv1',   concatImgs, self.encoders[0],     3, 2)  # 64 * 64 * 64
        conv2 = convolveLeakyReLU(
            'conv2',   conv1,      self.encoders[1],   3, 2)  # 32 * 32 * 32
        conv3 = convolveLeakyReLU(
            'conv3',   conv2,      self.encoders[2],   3, 2)  # 16 * 16 * 16
        conv4 = convolveLeakyReLU(
            'conv4',   conv3,      self.encoders[3],   3, 2)  # 8 * 8 * 8
 
        net = convolveLeakyReLU('decode4', conv4, self.decoders[0], 3, 1)
        net = tf.concat([UpSampling3D()(net), conv3], axis=-1)
        net = convolveLeakyReLU('decode3',   net, self.decoders[1], 3, 1)
        net = tf.concat([UpSampling3D()(net), conv2], axis=-1)
        net = convolveLeakyReLU('decode2',   net, self.decoders[2], 3, 1)
        net = tf.concat([UpSampling3D()(net), conv1], axis=-1)
        net = convolveLeakyReLU('decode1',   net, self.decoders[3], 3, 1)
        net = convolveLeakyReLU('decode1_1', net, self.decoders[4], 3, 1)
        net = tf.concat([UpSampling3D()(net), concatImgs], axis=-1)
        net = convolveLeakyReLU('decode0',   net, self.decoders[5], 3, 1)
        if len(self.decoders) == 8:
            net = convolveLeakyReLU('decode0_1', net, self.decoders[6], 3, 1)
        net = convolve(
            'flow', net, self.decoders[-1], 3, 1, weights_init=normal(stddev=1e-5))
        return {
            'flow': net * self.flow_multiplier
        }

def affine_flow(W, b, len1, len2, len3):
    b = tf.reshape(b, [-1, 1, 1, 1, 3])
    xr = tf.range(-(len1 - 1) / 2.0, len1 / 2.0, 1.0, tf.float32)
    xr = tf.reshape(xr, [1, -1, 1, 1, 1])
    yr = tf.range(-(len2 - 1) / 2.0, len2 / 2.0, 1.0, tf.float32)
    yr = tf.reshape(yr, [1, 1, -1, 1, 1])
    zr = tf.range(-(len3 - 1) / 2.0, len3 / 2.0, 1.0, tf.float32)
    zr = tf.reshape(zr, [1, 1, 1, -1, 1])
    wx = W[:, :, 0]
    wx = tf.reshape(wx, [-1, 1, 1, 1, 3])
    wy = W[:, :, 1]
    wy = tf.reshape(wy, [-1, 1, 1, 1, 3])
    wz = W[:, :, 2]
    wz = tf.reshape(wz, [-1, 1, 1, 1, 3])
    return (xr * wx + yr * wy) + (zr * wz + b)

def det3x3(M):
    M = [[M[:, i, j] for j in range(3)] for i in range(3)]
    return tf.add_n([
                M[0][0] * M[1][1] * M[2][2],
                M[0][1] * M[1][2] * M[2][0],
                M[0][2] * M[1][0] * M[2][1]
            ]) - tf.add_n([
                M[0][0] * M[1][2] * M[2][1],
                M[0][1] * M[1][0] * M[2][2],
                M[0][2] * M[1][1] * M[2][0]
            ])


class VTNAffineStem(Network):
    def __init__(self, name, flow_multiplier=1., **kwargs):
        super().__init__(name, **kwargs)
        self.flow_multiplier = flow_multiplier

    def build(self, img1, img2):
        
            #img1, img2, flow : tensor of shape [batch, X, Y, Z, C]
        
        concatImgs = tf.concat([img1, img2], 4, 'coloncatImgs')

        dims = 3
        conv1 = convolveLeakyReLU(
            'conv1',   concatImgs, 16,   3, 2)  # 64 * 64 * 64
        conv2 = convolveLeakyReLU(
            'conv2',   conv1,      32,   3, 2)  # 32 * 32 * 32
        conv3 = convolveLeakyReLU('conv3',   conv2,      64,   3, 2)
        conv3_1 = convolveLeakyReLU(
            'conv3_1', conv3,      64,   3, 1)
        conv4 = convolveLeakyReLU(
            'conv4',   conv3_1,    128,  3, 2)  # 16 * 16 * 16
        conv4_1 = convolveLeakyReLU(
            'conv4_1', conv4,      128,  3, 1)
        conv5 = convolveLeakyReLU(
            'conv5',   conv4_1,    256,  3, 2)  # 8 * 8 * 8
        conv5_1 = convolveLeakyReLU(
            'conv5_1', conv5,      256,  3, 1)
        conv6 = convolveLeakyReLU(
            'conv6',   conv5_1,    512,  3, 2)  # 4 * 4 * 4
        conv6_1 = convolveLeakyReLU(
            'conv6_1', conv6,      512,  3, 1)
        ks = conv6_1.shape.as_list()[1:4]
        conv7_W = tflearn.layers.conv_3d(
            conv6_1, 9, ks, strides=1, padding='valid', activation='linear', bias=False, scope='conv7_W')
        conv7_b = tflearn.layers.conv_3d(
            conv6_1, 3, ks, strides=1, padding='valid', activation='linear', bias=False, scope='conv7_b')

        I = [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]]
        W = tf.reshape(conv7_W, [-1, 3, 3]) * self.flow_multiplier
        b = tf.reshape(conv7_b, [-1, 3]) * self.flow_multiplier

        A = W + I 
        # the flow is displacement(x) = place(x) - x = (Ax + b) - x
        # the model learns W = A - I.

        sx, sy, sz = img1.shape.as_list()[1:4]
        flow = affine_flow(W, b, sx, sy, sz)
        #print(flow.shape.as_list(),'&'*30)
        # determinant should be close to 1
        det = det3x3(A)
        det_loss = tf.nn.l2_loss(det - 1.0)
        # should be close to being orthogonal
        # C=A'A, a positive semi-definite matrix
        # should be close to I. For this, we require C
        # has eigen values close to 1 by minimizing
        # k1+1/k1+k2+1/k2+k3+1/k3.
        # to prevent NaN, minimize
        # k1+eps + (1+eps)^2/(k1+eps) + ...
        eps = 1e-5
        epsI = [[[eps * elem for elem in row] for row in Mat] for Mat in I]
        C = tf.matmul(A, A, True) + epsI

        def elem_sym_polys_of_eigen_values(M):
            M = [[M[:, i, j] for j in range(3)] for i in range(3)]
            sigma1 = tf.add_n([M[0][0], M[1][1], M[2][2]])
            sigma2 = tf.add_n([
                M[0][0] * M[1][1],
                M[1][1] * M[2][2],
                M[2][2] * M[0][0]
            ]) - tf.add_n([
                M[0][1] * M[1][0],
                M[1][2] * M[2][1],
                M[2][0] * M[0][2]
            ])
            sigma3 = tf.add_n([
                M[0][0] * M[1][1] * M[2][2],
                M[0][1] * M[1][2] * M[2][0],
                M[0][2] * M[1][0] * M[2][1]
            ]) - tf.add_n([
                M[0][0] * M[1][2] * M[2][1],
                M[0][1] * M[1][0] * M[2][2],
                M[0][2] * M[1][1] * M[2][0]
            ])
            return sigma1, sigma2, sigma3
        s1, s2, s3 = elem_sym_polys_of_eigen_values(C)
        ortho_loss = s1 + (1 + eps) * (1 + eps) * s2 / s3 - 3 * 2 * (1 + eps)
        ortho_loss = tf.reduce_sum(ortho_loss)

        return {
            'flow': flow,
            'W': W,
            'b': b,
            'det_loss': det_loss,
            'ortho_loss': ortho_loss,
            'det_A':det
        }
