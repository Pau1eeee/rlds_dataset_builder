from typing import Iterator, Tuple, Any

import glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import pandas as pd
from PIL import Image
import os


class VlamDrone(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image': tfds.features.Image(
                            shape=(800, 848, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Drone camera RGB observation.',
                        ),
                        'state': tfds.features.Tensor(
                            shape=(4,),
                            dtype=np.float32,
                            doc='Drone state, consists of [vx, vy, vz, yaw_rate].',
                        )
                    }),
                    'action': tfds.features.Tensor(
                        shape=(4,),
                        dtype=np.float32,
                        doc='Drone action, consists of [vx, vy, vz, yaw_rate].',
                    ),
                    'discount': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Discount if provided, default to 1.'
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Reward if provided, 1 on final step for demos.'
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on first step of the episode.'
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode.'
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode if it is a terminal step, True for demos.'
                    ),
                    'language_instruction': tfds.features.Text(
                        doc='Language Instruction.'
                    ),
                    'language_embedding': tfds.features.Tensor(
                        shape=(512,),
                        dtype=np.float32,
                        doc='Kona language embedding. '
                            'See https://tfhub.dev/google/universal-sentence-encoder-large/5'
                    ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                }),
            }))

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        return {
            'train': self._generate_examples(path=r'dataset_1'),
        }

    def _generate_examples(self, path) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""

        csv_path = os.path.join(path, "dataset.csv")
        df = pd.read_csv(csv_path)

        instruction_path = os.path.join(path, "user_input.txt")
        with open(instruction_path, "r") as f:
            instruction = f.read().strip()

        # Language embedding
        language_embedding = self._embed([instruction])[0].numpy()

        episode = []
        for i, row in df.iterrows():
            # loading image
            img_path = os.path.join(path, f"img_{int(row.image_id):05d}.png")
            image = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)


            step = {
                'observation': {
                    'image': image,
                    'state': np.array([row.vx, row.vy, row.vz, row.yaw_rate], dtype=np.float32),
                },
                'action': np.array([row.vx, row.vy, row.vz, row.yaw_rate], dtype=np.float32),
                'discount': 1.0,
                'reward': float(i == len(df)-1),
                'is_first': i == 0,
                'is_last': i == len(df)-1,
                'is_terminal': i == len(df)-1,
                'language_instruction': instruction,
                'language_embedding': language_embedding,
            }
            episode.append(step)

        sample = {
            'steps': episode,
            'episode_metadata': {
                'file_path': csv_path
            }
        }


        # just one episode -> one key
        yield "episode_0", sample
