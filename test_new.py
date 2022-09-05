from absl import app, flags
from absl.flags import FLAGS
import tensorflow as tf
import os

from modules.models import RetinaFaceModel
from modules.dataset2 import unpack_label
from modules.anchor2 import prior_box, decode_tf
from modules.utils import set_memory_growth, load_yaml
from modules.utils2 import load_dataset, draw_result, post_process_pred


FILE_DIR = os.path.dirname(os.path.realpath(__file__))

flags.DEFINE_string('cfg_path', './configs/retinaface_res50.yaml',
                    'config file path') # retinaface_res50 retinaface_mbv2
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
    numBatchToTake = 5
    
    # define prior box
    priors = prior_box((img_dim, img_dim),
                       cfg['min_sizes'], cfg['steps'], cfg['clip'])
    
    # define network
    model = RetinaFaceModel(cfg, iou_th=FLAGS.iou_th, score_th=FLAGS.score_th)
    
    # load checkpoint
    checkpoint_dir = "./checkpoints/" + cfg['sub_name']
    checkpoint = tf.train.Checkpoint(model=model)
    if tf.train.latest_checkpoint(checkpoint_dir):
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()
        print("[*] load ckpt from {}.".format(
            tf.train.latest_checkpoint(checkpoint_dir)))
    else:
        print("[*] Cannot find ckpt from {}.".format(checkpoint_dir))
        exit()
    
    resultDir = os.path.join(FILE_DIR, "result")
    
    (_, _), (val_dataset, val_data_num) = load_dataset(cfg, priors, 
                                                       load_train=False, load_valid=True)
    
    if not os.path.exists(resultDir):
        os.mkdir(resultDir)
    
    for i, (inputs, labels) in enumerate(val_dataset.take(numBatchToTake)):
  
            output = model(inputs, training=False)

            bbox_regressions = output[0]
            landm_regressions = output[1]
            param_regressions = output[2]
            classifications = output[3]
            
            # [bboxes, landms, params, conf]
            preds = tf.concat(  
                    [bbox_regressions, 
                     landm_regressions, 
                     param_regressions,
                     classifications[:, :, 1][..., tf.newaxis]], axis=-1)
                            
            img = tf.cast(inputs, tf.uint8)
            res = []
            for idx in range(tf.shape(preds)[0].numpy()):

                post_out_pred = post_process_pred(preds[idx], priors, cfg['variances'], FLAGS.iou_th, FLAGS.score_th)
                face_box, landmarks, _, _, _ = unpack_label(post_out_pred, priors)
                
                post_out_gt = post_process_pred(labels[idx], priors, cfg['variances'], FLAGS.iou_th, FLAGS.score_th)                
                face_box_gt, landmarks_gt, _, _, _ = unpack_label(post_out_gt, priors)

                outputPath = os.path.join(resultDir, "{}_{}.jpg".format(i, idx))

                org_img = img[idx].numpy()
                org_img = draw_result(org_img, landmarks.numpy(), face_box.numpy(), outputPath, color=(255,255,0))
                org_img = draw_result(org_img, landmarks_gt.numpy(), face_box_gt.numpy(), outputPath, color=(0,255,0))
            
                print("Done drawing {}".format(outputPath))
            
    return

if __name__ == '__main__':
    app.run(main)
