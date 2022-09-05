"""the300wlp_tfds dataset."""

import tensorflow_datasets as tfds
import tensorflow as tf
import math
import numpy as np
import cv2

# TODO(the300wlp_tfds): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
"""

# TODO(the300wlp_tfds): BibTeX citation
_CITATION = """
"""

ORI_IMG_DIM = 450
NEW_IMG_DIM = 640

class The300wlpTfds(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for the300wlp_tfds dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.'
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(the300wlp_tfds): Specifies the tfds.core.DatasetInfo object
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset like images, labels ...
            'image': tfds.features.Image(shape=(NEW_IMG_DIM, NEW_IMG_DIM, 3)),
            'param': tfds.features.Tensor(shape=(62,), dtype=tf.float64),
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        supervised_keys=('image', 'param'),  # Set to `None` to disable
        homepage='https://dataset-homepage/',
        citation=_CITATION,
        disable_shuffling=False
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    # TODO(the300wlp_tfds): Downloads the data and defines the splits
    # path = dl_manager.download_and_extract('https://todo-data-url')

    dataset = tfds.load('the300w_lp')['train']

    # TODO(the300wlp_tfds): Returns the Dict[split names, Iterator[Key, Example]]
    return {
        'train': self._generate_examples(dataset),
    }

  def _generate_examples(self, dataset):
    """Yields examples."""
    # TODO(the300wlp_tfds): Yields (key, example) tuples from the dataset
    self.params = []

    for i, (sample) in enumerate(dataset):
      img = sample['image'].numpy()
      img = cv2.resize(img, (NEW_IMG_DIM, NEW_IMG_DIM), interpolation=cv2.INTER_AREA)
      pose_params = sample['pose_params'].numpy()
      
      shape_params = sample['shape_params'].numpy()
      exp_params = sample['exp_params'].numpy()
      
      resizeScale = NEW_IMG_DIM / ORI_IMG_DIM
      param = EncodeParams(pose_params, shape_params, exp_params, resizeScale)
      
      self.params.append(param)

      yield i, {
          'image': img,
          'param': param,
      }

    self.params = np.array(self.params)

    mean = np.mean(self.params, axis=0)
    std = np.std(self.params, axis=0)

    np.save('mean_62', mean)
    print("[*] saved mean_62.npy")
    np.save('std_62', std)
    print("[*] saved std_62.npy")
    
      
def RPYToRotationMatrix(phi, gamma, theta): # pitch, yaw, roll
    R_x = np.array([
        [1, 0, 0],
        [0, math.cos(phi), math.sin(phi)],
        [0, -math.sin(phi), math.cos(phi)]
    ])

    R_y = np.array([
        [math.cos(gamma), 0, -math.sin(gamma)],
        [0, 1, 0],
        [math.sin(gamma), 0, math.cos(gamma)]
    ])

    R_z = np.array([
        [math.cos(theta), math.sin(theta), 0],
        [-math.sin(theta), math.cos(theta), 0],
        [0, 0, 1]
    ])

    R = R_x @ R_y @ R_z

    return R

def EncodeParams(pose_params, shape_params, exp_params, resizeScale):
    # [rotations(3, 3), translation(3,), shape_params(40,), exp_params(10,)]: 62
    pitch = pose_params[0]
    yaw = pose_params[1]
    roll = pose_params[2]
    
    scale = pose_params[6]
    rotation = RPYToRotationMatrix(pitch, yaw, roll) * scale * resizeScale
    
    translation = np.array([pose_params[3], pose_params[4], pose_params[5]]) * resizeScale
    
    shape = shape_params[:40]
    exp = exp_params[:10]
    
    param = np.concatenate([rotation.reshape(-1), translation, shape, exp])
    
    # rotation_, translation_, shape_, exp_ = DecodeParams(param)
    
    return param

def GetMeanStd():

  mean_o_name = "the300wlp_tfds/mean_62"
  std_o_name = "the300wlp_tfds/std_62"

  split = 0.8
  
  # dataset = tfds.load('the300w_lp')['train']
  train_dataset = tfds.load('the300wlp_tfds_640',
                            data_dir='the300wlp_tfds',
                            split='train[:{}%]'.format(int(split*100)))
  train_data_num = int(train_dataset.cardinality().numpy())
  print("Load training data: {}".format(train_data_num))
  params = []

  for i, (sample) in enumerate(train_dataset):
    # img = sample['image'].numpy()
    # img = cv2.resize(img, (NEW_IMG_DIM, NEW_IMG_DIM), interpolation=cv2.INTER_AREA)

    # pose_params = sample['pose_params'].numpy()
    
    # shape_params = sample['shape_params'].numpy()
    # exp_params = sample['exp_params'].numpy()
    
    # resizeScale = NEW_IMG_DIM / ORI_IMG_DIM
    # param = EncodeParams(pose_params, shape_params, exp_params, resizeScale)

    param = sample['param'].numpy()

    params.append(param)

  params = np.array(params)
  mean = np.mean(params, axis=0)
  std = np.std(params, axis=0)

  np.save(mean_o_name, mean)
  print("[*] saved {}.npy".format(mean_o_name))
  np.save(std_o_name, std)
  print("[*] saved {}.npy".format(std_o_name))

if __name__ == "__main__":
  GetMeanStd()

  # ds = The300wlpTfds()
  # dl = tfds.download.DownloadManager
  # ds._split_generators(dl)