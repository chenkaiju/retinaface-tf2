from absl import app, flags
from absl.flags import FLAGS
import tensorflow as tf
import os
import cv2
import numpy as np

from modules.models import RetinaFaceModel
from modules.dataset2 import unpack_label
from modules.anchor2 import prior_box
from modules.utils import set_memory_growth, load_yaml
from modules.utils2 import draw_landmarks

FILE_DIR = os.path.dirname(os.path.realpath(__file__))


flags.DEFINE_string('input_path', "./photo/faces.jpg",
                    'input image path')
flags.DEFINE_string('cfg_path', './configs/retinaface_res50.yaml',
                    'config file path')
flags.DEFINE_string('gpu', '0', 
                    'which gpu to use')
flags.DEFINE_float('iou_th', 0.4, 'iou threshold for nms')
flags.DEFINE_float('score_th', 0.5, 'score threshold for nms')

def main(_):
    # init
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
    set_memory_growth()
    
    cfg = load_yaml(FLAGS.cfg_path)

    img_dim = cfg['input_size']
    
    # define prior box
    priors = prior_box((img_dim, img_dim),
                       cfg['min_sizes'], cfg['steps'], cfg['clip'])
    
    # define network
    model = RetinaFaceModel(cfg, training=False, 
                            iou_th=FLAGS.iou_th, score_th=FLAGS.score_th)
    
    # load checkpoint
    checkpoint_dir = "./checkpoints/" + cfg['sub_name']
    checkpoint = tf.train.Checkpoint(model=model)
    if tf.train.latest_checkpoint(checkpoint_dir):
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
        print("[*] load ckpt from {}.".format(
            tf.train.latest_checkpoint(checkpoint_dir)))
    else:
        print("[*] Cannot find ckpt from {}.".format(checkpoint_dir))
        exit()
    
    resultDir = os.path.join(FILE_DIR, "result")
    
    input_fn = FLAGS.input_path.split("/")[-1]
    image = cv2.imread(FLAGS.input_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (img_dim, img_dim))

    image = tf.convert_to_tensor(image)

    output = model(tf.cast(image[tf.newaxis, ...], tf.float32))
    
    image = tf.clip_by_value(image, 0, 255)
    image = tf.cast(image, tf.uint8)
    
    faceBox, landmarks, _, _, _ = unpack_label(output, priors)

    if not os.path.exists(resultDir):
        os.mkdir(resultDir)
    
    outputPath = os.path.join(resultDir, input_fn)
    
    draw_landmarks(image.numpy(), landmarks.numpy(), faceBox.numpy(), 
                    img_dim, outputPath)
    
    print("Done drawing {}".format(outputPath))
            

    return

if __name__ == '__main__':
    app.run(main)
