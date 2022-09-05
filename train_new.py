from absl import app, flags, logging
from absl.flags import FLAGS
import os
import tensorflow as tf

from modules.utils import (set_memory_growth, load_yaml, ProgressBar)
from modules.utils2 import load_dataset, draw_result, post_process_pred
from modules.dataset2 import unpack_label
from modules.models import RetinaFaceModel
from modules.anchor import prior_box
from modules.lr_scheduler import MultiStepWarmUpLR
from modules.losses2 import MultiBoxLoss

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
    
    logger = tf.get_logger()
    logger.disabled = True
    logger.setLevel(logging.FATAL)
    set_memory_growth()
    
    cfg = load_yaml(FLAGS.cfg_path)
    
    # define network
    model = RetinaFaceModel(cfg)
    
    
    # define prior box
    priors = prior_box((cfg['input_size'], cfg['input_size']),
                       cfg['min_sizes'], cfg['steps'], cfg['clip'])
    
    (train_dataset, train_num), (_, _) = load_dataset(cfg, priors)
    
    data1 = train_dataset.take(1)
    for input, label in data1:
        pred = model(input)
    model.summary()
    
    # define optimizer
    steps_per_epoch = train_num // cfg['batch_size']
    learning_rate = MultiStepWarmUpLR(
        initial_learning_rate=cfg['init_lr'], # 1e-2
        lr_steps=[e * steps_per_epoch for e in cfg['lr_decay_epoch']], # [50, 68]
        lr_rate=cfg['lr_rate'], # 0.1
        warmup_steps=cfg['warmup_epoch'] * steps_per_epoch, # 5
        min_lr=cfg['min_lr'] # 1e-3
    )
    
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=learning_rate, momentum=0.9, nesterov=True
    )
    
    # define losses function
    multi_box_loss = MultiBoxLoss()
    
    # load checkpoint
    checkpoint_dir = './checkpoints/' + cfg['sub_name']
    checkpoint = tf.train.Checkpoint(
        step=tf.Variable(0, name='step'),
        optimizer=optimizer,
        model=model
    )
    manager = tf.train.CheckpointManager(
        checkpoint=checkpoint,
        directory=checkpoint_dir,
        max_to_keep=3
    )
    
    if manager.latest_checkpoint:
        checkpoint.restore(manager.latest_checkpoint)
        print('[*] load ckpy from {} at step {}.'.format(
            manager.latest_checkpoint, checkpoint.step.numpy()
        ))
    else:
        print("[*] training from scratch.")
        
    # define training step function
    @tf.function
    def train_step(inputs, labels):
        with tf.GradientTape() as tape:
            predictions = model(inputs, training=True)
            
            losses = {}
            losses['reg'] = tf.reduce_sum(model.losses)
            losses['loc'], losses['landm'], losses['param'], losses['class'] = \
                multi_box_loss(labels, predictions)
            total_loss = tf.add_n([l for l in losses.values()])
        
        grads = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        
        return total_loss, losses
    
    # training loop
    summary_writer = tf.summary.create_file_writer('./logs/' + cfg['sub_name'])
    remain_steps = max(
        steps_per_epoch * cfg['epoch'] - checkpoint.step.numpy(), 0
    )
    print("remaining steps: {}".format(remain_steps))
    prog_bar = ProgressBar(
        steps_per_epoch, checkpoint.step.numpy() % steps_per_epoch
    )
    
    for inputs, labels in train_dataset.take(remain_steps):
        checkpoint.step.assign_add(1)
        steps = checkpoint.step.numpy()
        
        total_loss, losses = train_step(inputs, labels)
        prog_bar.update(
            "epoch={}/{}, loss={:.4f}, lr={:.1e}".format(
                ((steps - 1) // steps_per_epoch) + 1, cfg['epoch'],
                total_loss.numpy(), optimizer.lr(steps).numpy()
            )
        )
        
        if steps % 10 == 0:
            with summary_writer.as_default():
                tf.summary.scalar(
                    'loss/total_loss', total_loss, step=steps
                )
                
                for k, l in losses.items():
                    tf.summary.scalar(
                        'loss/{}'.format(k), l, step=steps
                    )
                
                tf.summary.scalar(
                    'learning_rate', optimizer.lr(steps), step=steps
                )
                
        
        if steps % 1000 == 0:
            
            with summary_writer.as_default():
                
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
                    
                    img_viz = img[idx].numpy()
                    
                    post_out_pred = post_process_pred(preds[idx], priors, cfg['variances'], FLAGS.iou_th, FLAGS.score_th)
                    
                    faceBox, landmarks, _, _, _ = unpack_label(post_out_pred, priors)
                    
                    img_viz = draw_result(img_viz, landmarks.numpy(), faceBox.numpy(), color=(255,255,0))
                    
                    post_out_gt = post_process_pred(labels[idx], priors, cfg['variances'], FLAGS.iou_th, FLAGS.score_th)
                
                    faceBoxGT, landmarksGT, _, _, _ = unpack_label(post_out_gt, priors)
                    img_viz = draw_result(img_viz, landmarksGT.numpy(), faceBoxGT.numpy(), color=(0,255,0))
            
                    res.append(img_viz)
                    
                restf = tf.convert_to_tensor(res, dtype=tf.uint8)
                tf.summary.image('pred/image', restf, step=steps)
                
        if steps % cfg['save_steps'] == 0:
            manager.save()
            print("[*] save ckpt file at {}".format(
                manager.latest_checkpoint
            ))
            
    manager.save()
    print("[*] training done! save ckpt file at {}".format(
        manager.latest_checkpoint
    ))
    

if __name__ == '__main__':
    app.run(main)