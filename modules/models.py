from re import S
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.applications import MobileNetV2, ResNet50
from tensorflow.keras.layers import Input, Conv2D, ReLU, LeakyReLU
from modules.anchor2 import decode_tf, prior_box_tf


def _regularizer(weights_decay):
    """l2 regularizer"""
    return tf.keras.regularizers.l2(weights_decay)


def _kernel_init(scale=1.0, seed=None):
    """He normal initializer"""
    return tf.keras.initializers.he_normal()


class BatchNormalization(tf.keras.layers.BatchNormalization):
    """Make trainable=False freeze BN for real (the og version is sad).
       ref: https://github.com/zzh8829/yolov3-tf2
    """
    def __init__(self, axis=-1, momentum=0.9, epsilon=1e-5, center=True,
                 scale=True, name=None, **kwargs):
        super(BatchNormalization, self).__init__(
            axis=axis, momentum=momentum, epsilon=epsilon, center=center,
            scale=scale, name=name, **kwargs)

    def call(self, x, training=False):
        if training is None:
            training = tf.constant(False)
        training = tf.logical_and(training, self.trainable)

        return super().call(x, training)

class Backbone():
    
    def __init__(self, backbone_type='ResNet50', use_pretrain=True, input_shape=(640,640,3)):
        weights = None
        if use_pretrain:
            weights = 'imagenet'
            
        if backbone_type == 'ResNet50':
            self.extractor = ResNet50(
                input_shape=input_shape, include_top=False, weights=weights)
            self.pick_layer1 = 80  # [size/8, size/8, 512] -> [80, 80, 512]
            self.pick_layer2 = 142  # [size/16, size/16, 1024] -> [40, 40, 1024]
            self.pick_layer3 = 174  # [size/32, size/32, 2048] -> [20, 20, 2048]
            self.preprocess = tf.keras.applications.resnet.preprocess_input
        elif backbone_type == 'MobileNetV2':
            self.extractor = MobileNetV2(
                input_shape=input_shape, include_top=False, weights=weights)
            self.pick_layer1 = 54  # [80, 80, 32]
            self.pick_layer2 = 116  # [40, 40, 96]
            self.pick_layer3 = 143  # [20, 20, 160]
            self.preprocess = tf.keras.applications.mobilenet_v2.preprocess_input

        self.backbone_type = backbone_type
        self.model = None
    
    def build_backbone(self):
        
        if self.model is None:
            self.model = Model(self.extractor.input,
                        (self.extractor.layers[self.pick_layer1].output,
                        self.extractor.layers[self.pick_layer2].output,
                        self.extractor.layers[self.pick_layer3].output),
                        name=self.backbone_type + '_extrator')
        
        
        return self.model
    

class ConvUnit(tf.keras.layers.Layer):
    """Conv + BN + Act"""
    def __init__(self, f, k, s, wd, act=None, name='ConvBN', **kwargs):
        super(ConvUnit, self).__init__(name=name, **kwargs)
        self.conv = Conv2D(filters=f, kernel_size=k, strides=s, padding='same',
                           kernel_initializer=_kernel_init(),
                           kernel_regularizer=_regularizer(wd),
                           use_bias=False, name='conv')
        self.bn = BatchNormalization(name='bn')

        if act is None:
            self.act_fn = tf.identity
        elif act == 'relu':
            self.act_fn = ReLU()
        elif act == 'lrelu':
            self.act_fn = LeakyReLU(0.1)
        else:
            raise NotImplementedError(
                'Activation function type {} is not recognized.'.format(act))

    def call(self, x):
        return self.act_fn(self.bn(self.conv(x)))


class FPN(tf.keras.layers.Layer):
    """Feature Pyramid Network"""
    def __init__(self, out_ch, wd, name='FPN', **kwargs):
        super(FPN, self).__init__(name=name, **kwargs)
        act = 'relu'
        if (out_ch <= 64):
            act = 'lrelu'

        self.output1 = ConvUnit(f=out_ch, k=1, s=1, wd=wd, act=act)
        self.output2 = ConvUnit(f=out_ch, k=1, s=1, wd=wd, act=act)
        self.output3 = ConvUnit(f=out_ch, k=1, s=1, wd=wd, act=act)
        self.merge1 = ConvUnit(f=out_ch, k=3, s=1, wd=wd, act=act)
        self.merge2 = ConvUnit(f=out_ch, k=3, s=1, wd=wd, act=act)

    def call(self, x):
        output1 = self.output1(x[0])  # [80, 80, out_ch]
        output2 = self.output2(x[1])  # [40, 40, out_ch]
        output3 = self.output3(x[2])  # [20, 20, out_ch]

        up_h, up_w = tf.shape(output2)[1], tf.shape(output2)[2]
        up3 = tf.image.resize(output3, [up_h, up_w], method='nearest')
        output2 = output2 + up3
        output2 = self.merge2(output2)

        up_h, up_w = tf.shape(output1)[1], tf.shape(output1)[2]
        up2 = tf.image.resize(output2, [up_h, up_w], method='nearest')
        output1 = output1 + up2
        output1 = self.merge1(output1)

        return output1, output2, output3


class SSH(tf.keras.layers.Layer):
    """Single Stage Headless Layer"""
    def __init__(self, out_ch, wd, name='SSH', **kwargs):
        super(SSH, self).__init__(name=name, **kwargs)
        assert out_ch % 4 == 0
        act = 'relu'
        if (out_ch <= 64):
            act = 'lrelu'

        self.conv_3x3 = ConvUnit(f=out_ch // 2, k=3, s=1, wd=wd, act=None)

        self.conv_5x5_1 = ConvUnit(f=out_ch // 4, k=3, s=1, wd=wd, act=act)
        self.conv_5x5_2 = ConvUnit(f=out_ch // 4, k=3, s=1, wd=wd, act=None)

        self.conv_7x7_2 = ConvUnit(f=out_ch // 4, k=3, s=1, wd=wd, act=act)
        self.conv_7x7_3 = ConvUnit(f=out_ch // 4, k=3, s=1, wd=wd, act=None)

        self.relu = ReLU()

    def call(self, x):
        conv_3x3 = self.conv_3x3(x)

        conv_5x5_1 = self.conv_5x5_1(x)
        conv_5x5 = self.conv_5x5_2(conv_5x5_1)

        conv_7x7_2 = self.conv_7x7_2(conv_5x5_1)
        conv_7x7 = self.conv_7x7_3(conv_7x7_2)

        output = tf.concat([conv_3x3, conv_5x5, conv_7x7], axis=3)
        output = self.relu(output)

        return output


class BboxHead(tf.keras.layers.Layer):
    """Bbox Head Layer"""
    def __init__(self, num_anchor, wd, name='BboxHead', pointNum=2, **kwargs):
        super(BboxHead, self).__init__(name=name, **kwargs)
        self.length = pointNum * 2
        self.num_anchor = num_anchor
        self.conv = Conv2D(filters=num_anchor * self.length, kernel_size=1, strides=1)

    def call(self, x):
        h, w = tf.shape(x)[1], tf.shape(x)[2]
        x = self.conv(x)

        return tf.reshape(x, [-1, h * w * self.num_anchor, self.length])

class ParamHead(tf.keras.layers.Layer):
    """Param Head Layer"""
    def __init__(self, num_anchor, wd, name='ParamHead', paramNum=62, **kwargs):
        super(ParamHead, self).__init__(name=name, **kwargs)
        self.length = paramNum
        self.num_anchor = num_anchor
        self.conv = Conv2D(filters=num_anchor * self.length, kernel_size=1, strides=1)
        
    def call(self, x):
        h, w = tf.shape(x)[1], tf.shape(x)[2]
        x = self.conv(x)
        return tf.reshape(x, [-1, h * w * self.num_anchor, self.length])  
    
class LandmarkHead(tf.keras.layers.Layer):
    """Landmark Head Layer"""
    def __init__(self, num_anchor, wd, name='LandmarkHead', pointNum=68, **kwargs):
        super(LandmarkHead, self).__init__(name=name, **kwargs)
        self.length = pointNum * 2
        self.num_anchor = num_anchor
        self.conv = Conv2D(filters=num_anchor * self.length, kernel_size=1, strides=1)

    def call(self, x):
        h, w = tf.shape(x)[1], tf.shape(x)[2]
        x = self.conv(x)

        return tf.reshape(x, [-1, h * w * self.num_anchor, self.length])


class ClassHead(tf.keras.layers.Layer):
    """Class Head Layer"""
    def __init__(self, num_anchor, wd, name='ClassHead', classNum=2, **kwargs):
        super(ClassHead, self).__init__(name=name, **kwargs)
        self.length = classNum
        self.num_anchor = num_anchor
        self.conv = Conv2D(filters=num_anchor * self.length, kernel_size=1, strides=1)

    def call(self, x):
        h, w = tf.shape(x)[1], tf.shape(x)[2]
        x = self.conv(x)

        return tf.reshape(x, [-1, h * w * self.num_anchor, self.length])


class RetinaFaceModel(tf.keras.Model):
    """Retina Face Model"""
    
    def __init__(self, cfg, iou_th=0.4, score_th=0.02, name='RetinaFaceModel', **kwargs):
        super(RetinaFaceModel, self).__init__(name=name, **kwargs)
        
        self.cfg = cfg
        self.iou_th = iou_th
        self.score_th = score_th
        self.input_size = cfg['input_size']# if training else None # 450
        wd = cfg['weights_decay'] # 5e-4
        out_ch = cfg['out_channel'] # 256
        num_anchor = len(cfg['min_sizes'][0]) # 2
        backbone_type = cfg['backbone_type'] # ResNet50
        
        backbone = Backbone(backbone_type=backbone_type, use_pretrain=True, input_shape=(self.input_size,self.input_size,3))
        self.preprocess = backbone.preprocess
        
        self.backbone = backbone.build_backbone()
        
        self.fpn = FPN(out_ch=out_ch, wd=wd)
        
        self.ssh = [SSH(out_ch=out_ch, wd=wd, name=f'SSH_{i}') for i in range(3)]
        
        self.box_heads = [BboxHead(num_anchor, wd=wd, name=f'BboxHead_{i}') for i in range(3)]
        
        self.param_heads = [ParamHead(num_anchor, wd=wd, name=f'ParamHead_{i}') for i in range(3)]
        
        self.landm_heads = [LandmarkHead(num_anchor, wd=wd, name=f'LandmarkHead_{i}') for i in range(3)]
        
        self.class_head = [ClassHead(num_anchor, wd=wd, name=f'ClassHead_{i}') for i in range(3)]
        
    def call(self, inputs, training=False):
        
        x = self.preprocess(inputs)
        x = self.backbone(x)
        
        feature1, feature2, feature3 = self.fpn(x)
        
        ssh1 = self.ssh[0](feature1)
        ssh2 = self.ssh[1](feature2)
        ssh3 = self.ssh[2](feature3)
        
        bbox_regressions = tf.concat(
            [self.box_heads[0](ssh1), self.box_heads[1](ssh2), self.box_heads[2](ssh3)], axis=1
        )
        
        param_regressions = tf.concat(
            [self.param_heads[0](ssh1), self.param_heads[1](ssh2), self.param_heads[2](ssh3)], axis=1
        )
        
        landm_regressions = tf.concat(
            [self.landm_heads[0](ssh1), self.landm_heads[1](ssh2), self.landm_heads[2](ssh3)], axis=1
        )
        
        classifications = tf.concat(
            [self.class_head[0](ssh1), self.class_head[1](ssh2), self.class_head[2](ssh3)], axis=1
        )
        
        classifications = tf.keras.layers.Softmax(axis=-1)(classifications)

        if training:
            out = (bbox_regressions, landm_regressions, param_regressions, classifications)
            
        else:
            # only for batch size 1
            preds = tf.concat(  # [bboxes, landms, landms_valid, conf]
                [bbox_regressions[0], 
                 landm_regressions[0], 
                 param_regressions[0],
                 tf.ones_like(classifications[0, :, 0][..., tf.newaxis]),
                 classifications[0, :, 1][..., tf.newaxis]], axis=1)
            
            priors = prior_box_tf((tf.shape(inputs)[1], tf.shape(inputs)[2]),
                                self.cfg['min_sizes'],  self.cfg['steps'], self.cfg['clip'])
            decode_preds = decode_tf(preds, priors, self.cfg['variances'])

            selected_indices = tf.image.non_max_suppression(
                boxes=decode_preds[:, :4],
                scores=decode_preds[:, -1],
                max_output_size=tf.shape(decode_preds)[0],
                iou_threshold=self.iou_th,
                score_threshold=self.score_th)

            out = tf.gather(decode_preds, selected_indices)
            
        return out