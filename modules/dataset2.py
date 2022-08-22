import tensorflow as tf
import tensorflow_datasets as tfds
from modules.anchor2 import encode_tf

def DecodeParams(param):
    if len(param) != 62:
        raise('wrong param length')
    
    r_dim, t_dim, shape_dim, exp_dim = 9, 3, 40, 10 #62
        
    rotation = tf.reshape(param[:r_dim], (3, 3))
    translation = tf.reshape(param[r_dim:r_dim+t_dim], (t_dim, 1))
    
    shape = tf.reshape(param[r_dim+t_dim:r_dim+t_dim+shape_dim], (shape_dim, 1))
    exp = tf.reshape(param[r_dim+t_dim+shape_dim:], (exp_dim, 1))
    
    return rotation, translation, shape, exp

def reconstruct_landmark(bfm, param, img_size=450):
    R, t, shape, exp = DecodeParams(param)
    
    face = (bfm.u_base + bfm.w_shp_base @ shape + bfm.w_exp_base @ exp)
    face = tf.reshape(face, [-1, 68, 3])
    face = tf.transpose(face, perm=[0, 2, 1]) #128,3,68
            
    vertex = R @ face + t
    
    offset_y = tf.constant([0, img_size + 1, 0], dtype=tf.float64, shape=[3,1])
    offset_y = tf.tile(offset_y, [1, 68])
                
    y_inv = tf.constant([1.0, -1.0, 1.0], dtype=tf.float64, shape=[3,1])
    y_inv = tf.tile(y_inv, [1, 68])
                
    vertex = tf.multiply(vertex, y_inv)
    vertex = offset_y + vertex

    return vertex

def get_facebox2d(landmark_2d, ratio=0.1):
    
    x_max = tf.reduce_max(landmark_2d[:, 0, :])
    x_min = tf.reduce_min(landmark_2d[:, 0, :])
    y_max = tf.reduce_max(landmark_2d[:, 1, :])
    y_min = tf.reduce_min(landmark_2d[:, 1, :])
    
    x, y, w, h = x_min, y_min, x_max - x_min, y_max - y_min
    
    new_w, new_h = w * (1+ratio), h * (1+ratio)
    shift_x, shift_y = (new_w - w) / 2, (new_h - h) / 2
    new_x, new_y = (x - shift_x), (y - shift_y)
    
    facebox = [new_x, new_y, new_x+new_w, new_y+new_h]
    facebox = tf.reshape(facebox, [-1, 4])
    return facebox

def _parse_tfds(bfm, img_dim, priors, match_thresh, 
                ignore_thresh, variances, numFace=1):
    
    def parse_tfds(dataset):
        
        labels = tf.TensorArray(tf.float32, size=0, dynamic_size=True) #, dynamic_size=True, clear_after_read=False
        image = dataset['image']

        for n in tf.range(numFace):
            l = 0
            label = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
            param = dataset['param']
            
            lmk_3d = reconstruct_landmark(bfm, param, img_dim)
            lmk_2d = tf.cast(lmk_3d[:, :2], tf.float32)
            
            facebox = tf.squeeze(get_facebox2d(lmk_2d))
            
            # stack and normalize facebox [0, 1]
            for i in range(facebox.shape[0]):
                label = label.write(l, facebox[i] / img_dim)
                l += 1
            
            # stack and normalize lmk [0, 1]
            lmk_2d = tf.transpose(tf.squeeze(lmk_2d))
            for i in range(lmk_2d.shape[0]):
                for j in range(lmk_2d.shape[1]):
                    label = label.write(l, lmk_2d[i][j] / img_dim)
                    l +=  1
            
            param = tf.cast(param, tf.float32)
            for p in param:
                label = label.write(l, p)
                l += 1

            # valid
            label = label.write(l, tf.constant(1., dtype=tf.float32))
            l += 1
                
            labels = labels.write(n, label.stack())
                
        labels = labels.stack()

        image, labels = _transform_data(img_dim, priors, match_thresh, 
                                        ignore_thresh, variances)(image, labels)
    
        return image, labels
    
    return parse_tfds


def _transform_data(img_dim, priors, match_thresh, ignore_thresh, variances,
                    using_crop=False, using_resize=False, using_flip=False, 
                    using_distort=True, using_encoding=True):
    def transform_data(img, labels):
        img = tf.cast(img, tf.float32)

        if using_crop:
            # randomly crop
            img, labels = _crop(img, labels)

            # padding to square
            img = _pad_to_square(img)

        if using_resize:
            # resize
            img, labels = _resize(img, labels, img_dim)

        # randomly left-right flip
        if using_flip:
            img, labels = _flip(img, labels)

        # distort
        if using_distort:
            img = _distort(img)

        # encode labels to feature targets
        if using_encoding:
            labels = encode_tf(labels=labels, priors=priors,
                               match_thresh=match_thresh,
                               ignore_thresh=ignore_thresh,
                               variances=variances)

        return img, labels
    return transform_data

def load_tfds_dataset(bfm, tfds_name, batch_size, img_dim,
                          using_bin=True, using_flip=True, using_distort=True,
                          using_encoding=True, priors=None, match_thresh=0.45,
                          ignore_thresh=0.3, variances=[0.1, 0.2],
                          shuffle=False, buffer_size=10240):
    """load dataset from tfrecord"""
    if not using_encoding:
        assert batch_size == 1  # dynamic data len when using_encoding
    else:
        assert priors is not None

    split = 0.1
    raw_dataset = tfds.load(tfds_name, 
                            data_dir=tfds_name,
                            split='train[:{}%]'.format(int(split*100)))
    # test_dataset = tfds.load(tfds_name, 
    #                          data_dir=tfds_name,
    #                          split='train[{}%:]'.format(int(split*100)))
    train_data_num = raw_dataset.cardinality().numpy()
    print("Load training data: {}".format(train_data_num))
    
    raw_dataset = raw_dataset.repeat()
    if shuffle:
        raw_dataset = raw_dataset.shuffle(buffer_size=buffer_size)
    dataset = raw_dataset.map(
        _parse_tfds(bfm, img_dim, priors, match_thresh, ignore_thresh, variances),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset


###############################################################################
#   Data Augmentation                                                         #
###############################################################################
def _flip(img, labels):
    flip_case = tf.random.uniform([], 0, 2, dtype=tf.int32)

    def flip_func():
        flip_img = tf.image.flip_left_right(img)
        flip_labels = tf.stack([1 - labels[:, 2],  labels[:, 1],
                                1 - labels[:, 0],  labels[:, 3],
                                1 - labels[:, 6],  labels[:, 7],
                                1 - labels[:, 4],  labels[:, 5],
                                1 - labels[:, 8],  labels[:, 9],
                                1 - labels[:, 12], labels[:, 13],
                                1 - labels[:, 10], labels[:, 11],
                                labels[:, 14]], axis=1)

        return flip_img, flip_labels

    img, labels = tf.case([(tf.equal(flip_case, 0), flip_func)],
                          default=lambda: (img, labels))

    return img, labels


def _crop(img, labels, max_loop=250):
    shape = tf.shape(img)

    def matrix_iof(a, b):
        """
        return iof of a and b, numpy version for data augenmentation
        """
        lt = tf.math.maximum(a[:, tf.newaxis, :2], b[:, :2])
        rb = tf.math.minimum(a[:, tf.newaxis, 2:], b[:, 2:])

        area_i = tf.math.reduce_prod(rb - lt, axis=2) * \
            tf.cast(tf.reduce_all(lt < rb, axis=2), tf.float32)
        area_a = tf.math.reduce_prod(a[:, 2:] - a[:, :2], axis=1)
        return area_i / tf.math.maximum(area_a[:, tf.newaxis], 1)

    def crop_loop_body(i, img, labels):
        valid_crop = tf.constant(1, tf.int32)

        pre_scale = tf.constant([0.3, 0.45, 0.6, 0.8, 1.0], dtype=tf.float32)
        scale = pre_scale[tf.random.uniform([], 0, 5, dtype=tf.int32)]
        short_side = tf.cast(tf.minimum(shape[0], shape[1]), tf.float32)
        h = w = tf.cast(scale * short_side, tf.int32)
        h_offset = tf.random.uniform([], 0, shape[0] - h + 1, dtype=tf.int32)
        w_offset = tf.random.uniform([], 0, shape[1] - w + 1, dtype=tf.int32)
        roi = tf.stack([w_offset, h_offset, w_offset + w, h_offset + h])
        roi = tf.cast(roi, tf.float32)

        value = matrix_iof(labels[:, :4], roi[tf.newaxis])
        valid_crop = tf.cond(tf.math.reduce_any(value >= 1),
                             lambda: valid_crop, lambda: 0)

        centers = (labels[:, :2] + labels[:, 2:4]) / 2
        mask_a = tf.reduce_all(
            tf.math.logical_and(roi[:2] < centers, centers < roi[2:]),
            axis=1)
        labels_t = tf.boolean_mask(labels, mask_a)
        valid_crop = tf.cond(tf.reduce_any(mask_a),
                             lambda: valid_crop, lambda: 0)

        img_t = img[h_offset:h_offset + h, w_offset:w_offset + w, :]
        h_offset = tf.cast(h_offset, tf.float32)
        w_offset = tf.cast(w_offset, tf.float32)
        labels_t = tf.stack(
            [labels_t[:, 0] - w_offset,  labels_t[:, 1] - h_offset,
             labels_t[:, 2] - w_offset,  labels_t[:, 3] - h_offset,
             labels_t[:, 4] - w_offset,  labels_t[:, 5] - h_offset,
             labels_t[:, 6] - w_offset,  labels_t[:, 7] - h_offset,
             labels_t[:, 8] - w_offset,  labels_t[:, 9] - h_offset,
             labels_t[:, 10] - w_offset, labels_t[:, 11] - h_offset,
             labels_t[:, 12] - w_offset, labels_t[:, 13] - h_offset,
             labels_t[:, 14]], axis=1)

        return tf.cond(valid_crop == 1,
                       lambda: (max_loop, img_t, labels_t),
                       lambda: (i + 1, img, labels))

    _, img, labels = tf.while_loop(
        lambda i, img, labels: tf.less(i, max_loop),
        crop_loop_body,
        [tf.constant(-1), img, labels],
        shape_invariants=[tf.TensorShape([]),
                          tf.TensorShape([None, None, 3]),
                          tf.TensorShape([None, 15])])

    return img, labels


def _pad_to_square(img):
    height = tf.shape(img)[0]
    width = tf.shape(img)[1]

    def pad_h():
        img_pad_h = tf.ones([width - height, width, 3]) * \
            tf.reduce_mean(img, axis=[0, 1], keepdims=True)
        return tf.concat([img, img_pad_h], axis=0)

    def pad_w():
        img_pad_w = tf.ones([height, height - width, 3]) * \
            tf.reduce_mean(img, axis=[0, 1], keepdims=True)
        return tf.concat([img, img_pad_w], axis=1)

    img = tf.case([(tf.greater(height, width), pad_w),
                   (tf.less(height, width), pad_h)], default=lambda: img)

    return img


def _resize(img, labels, img_dim):
    w_f = tf.cast(tf.shape(img)[1], tf.float32)
    h_f = tf.cast(tf.shape(img)[0], tf.float32)
    # locs = tf.stack([labels[:, 0] / w_f,  labels[:, 1] / h_f,
    #                  labels[:, 2] / w_f,  labels[:, 3] / h_f,
    #                  labels[:, 4] / w_f,  labels[:, 5] / h_f,
    #                  labels[:, 6] / w_f,  labels[:, 7] / h_f,
    #                  labels[:, 8] / w_f,  labels[:, 9] / h_f,
    #                  labels[:, 10] / w_f, labels[:, 11] / h_f,
    #                  labels[:, 12] / w_f, labels[:, 13] / h_f], axis=1)
    lmkNum = 68
    faceBoxNum = 2
    labels_remain = labels[:, (lmkNum+faceBoxNum)*2:]
    locs_ = []
    for i in range(lmkNum+faceBoxNum):
        locs_.append(labels[:, i*2] / w_f)
        locs_.append(labels[:, i*2+1] / h_f)
    labels = tf.stack(locs_, axis=1)
    labels = tf.clip_by_value(labels, 0, 1)
    labels = tf.concat([labels, labels_remain], axis=1)

    resize_case = tf.random.uniform([], 0, 5, dtype=tf.int32)

    def resize(method):
        def _resize():
            return tf.image.resize(
                img, [img_dim, img_dim], method=method, antialias=True)
        return _resize

    img = tf.case([(tf.equal(resize_case, 0), resize('bicubic')),
                   (tf.equal(resize_case, 1), resize('area')),
                   (tf.equal(resize_case, 2), resize('nearest')),
                   (tf.equal(resize_case, 3), resize('lanczos3'))],
                  default=resize('bilinear'))

    return img, labels


def _distort(img):
    img = tf.image.random_brightness(img, 0.4)
    img = tf.image.random_contrast(img, 0.5, 1.5)
    img = tf.image.random_saturation(img, 0.5, 1.5)
    img = tf.image.random_hue(img, 0.1)

    return img
