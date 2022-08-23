from absl import app
import tensorflow as tf
import os

from bfm.bfm import BFMModel
from modules.dataset2 import load_tfds_dataset, unpack_label
from modules.anchor2 import prior_box
from modules.utils2 import draw_landmarks

FILE_DIR = os.path.dirname(os.path.realpath(__file__))

def main(_):
    
    img_dim = 450
    batch_size = 4
    numBatchToTake = 2
    
    bfm = BFMModel(
        bfm_fp=os.path.join(FILE_DIR, "bfm", "bfm_noneck_v3.pkl"),
        shape_dim=40,
        exp_dim=10
    )
    
    # define prior box
    priors = prior_box((img_dim, img_dim),
                       [[16, 32], [64, 128], [256, 512]], 
                       [8, 16, 32], False)
    
    
    dataset = load_tfds_dataset(
        bfm=bfm, 
        tfds_name="the300wlp_tfds",
        batch_size=batch_size,
        img_dim=img_dim,
        priors=priors
    )
    
    resultDir = os.path.join(FILE_DIR, "result")
    
    for i, (inputs, labels) in enumerate(dataset.take(numBatchToTake)):
        
        for j, (image, label) in enumerate(zip(inputs, labels)):
            image = tf.clip_by_value(image, 0, 255)
            image = tf.cast(image, tf.uint8)
            
            faceBox, landmarks, _, _, _ = unpack_label(label, priors)
            
            if not os.path.exists(resultDir):
                os.mkdir(resultDir)
            
            outputPath = os.path.join(resultDir, "{}_{}.jpg".format(i, j))
            
            draw_landmarks(image.numpy(), landmarks.numpy(), faceBox.numpy(), 
                           img_dim, outputPath)
            
            print("Done drawing {}".format(outputPath))
            

    return

if __name__ == '__main__':
    app.run(main)
