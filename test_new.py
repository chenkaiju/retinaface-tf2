from absl import app, flags
from absl.flags import FLAGS
import tensorflow as tf
import os

from modules.models import RetinaFaceModel
from modules.dataset2 import unpack_label
from modules.anchor2 import prior_box
from modules.utils import set_memory_growth, load_yaml
from modules.utils2 import load_dataset, draw_landmarks

FILE_DIR = os.path.dirname(os.path.realpath(__file__))

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
    batch_size = cfg['batch_size']
    numBatchToTake = 5
    
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
    
    (_, _), (val_dataset, val_data_num) = load_dataset(cfg, priors, 
                                                       load_train=False, load_valid=True)
    
    for i, (inputs, labels) in enumerate(val_dataset.take(numBatchToTake)):
        
        
        for j, (image, label) in enumerate(zip(inputs, labels)):
            
            output = model(image[tf.newaxis, ...])
            
            image = tf.clip_by_value(image, 0, 255)
            image = tf.cast(image, tf.uint8)
            
            faceBox, landmarks, _, _, _ = unpack_label(output, priors)
            
            if not os.path.exists(resultDir):
                os.mkdir(resultDir)
            
            outputPath = os.path.join(resultDir, "{}_{}.jpg".format(i, j))
            
            draw_landmarks(image.numpy(), landmarks.numpy(), faceBox.numpy(), 
                           img_dim, outputPath)
            
            print("Done drawing {}".format(outputPath))
            

    return

if __name__ == '__main__':
    app.run(main)
