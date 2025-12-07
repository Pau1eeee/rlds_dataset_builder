from typing import Iterator, Tuple, Any

import glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import pandas as pd
from PIL import Image
import os
import shutil
from tqdm import tqdm


class VlamDrone(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version('1.1.0')
    RELEASE_NOTES = {
      '1.1.0': 'Initial release.',
    }

    DATASET_PATH = r"/mnt/d/Projekte/Vlam/vlam_drone_project/thws_dataset"

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
            'train': self._generate_examples(root_path=self.DATASET_PATH),
        }

    def _generate_examples(self, root_path: str) -> Iterator[Tuple[str, Any]]:
        """
        Generator that yields all episodes from a merged dataset CSV.

        Args:
            csv_path: Path to the merged dataset CSV containing multiple episodes.
        """
        df = pd.read_csv(os.path.join(root_path, "dataset.csv"))


        for episode_id, episode_df in df.groupby("episode_id"):
            episode = []


            # Instruction und Embedding
            language_instruction = str(episode_df.iloc[0]["instruction"])
            language_embedding = np.array(
                self._embed([language_instruction])[0], dtype=np.float32
            ).reshape(512,)

            for _, row in tqdm(episode_df.iterrows(), total=len(episode_df), desc=f"Episode {episode_id}"):
                img_name = f"img_{int(row.image_id):06d}.png"
                img_path = os.path.join(root_path, "images", img_name)
                image = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)

                # check state and action
                state = np.nan_to_num(
                    [float(row.vx), float(row.vy), float(row.vz), float(row.yaw_rate)],
                    nan=0.0
                ).astype(np.float32)

                step = {
                    "observation": {
                        "image": image,
                        "state": state,
                    },
                    "action": state.copy(),
                    "discount": 1.0,
                    "reward": float(bool(row.is_last)),
                    "is_first": bool(int(row.is_first)),
                    "is_last": bool(int(row.is_last)),
                    "is_terminal": bool(int(row.is_last)),
                    "language_instruction": language_instruction,
                    "language_embedding": language_embedding,
                }

                episode.append(step)

            sample = {
                "steps": episode,
                "episode_metadata": {
                    "file_path": os.path.join(root_path, "dataset.csv")
                }
            }

            yield f"episode_{episode_id}", sample
