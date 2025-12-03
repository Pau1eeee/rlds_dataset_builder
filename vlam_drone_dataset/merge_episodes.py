import os
import shutil
import pandas as pd
from tqdm import tqdm

def merge_dataset(root_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    output_img_dir = os.path.join(output_dir, "images")
    os.makedirs(output_img_dir, exist_ok=True)

    global_image_id = 0
    episode_id = 0
    merged_rows = []
    all_dirs = list(os.walk(root_path))
    for dirpath, dirnames, filenames in all_dirs:
        if "dataset.csv" in filenames and "user_input.txt" in filenames:

            df = pd.read_csv(os.path.join(dirpath, "dataset.csv"))

            # Instruction der Episode
            with open(os.path.join(dirpath, "user_input.txt"), "r") as f:
                instruction = f.read().strip()

            episode_length = len(df)

            # Jede Zeile ist ein Schritt
            for i, row in tqdm(df.iterrows(), total=len(df)):

                old_img_path = os.path.join(
                    dirpath,
                    f"img_{int(row.image_id):05d}.png",
                )

                new_img_name = f"img_{global_image_id:06d}.png"
                new_img_path = os.path.join(output_img_dir, new_img_name)

                shutil.copy2(old_img_path, new_img_path)

                merged_rows.append({
                    "episode_id": episode_id,
                    "step_index": i,
                    "image_id": global_image_id,
                    "vx": row.vx,
                    "vy": row.vy,
                    "vz": row.vz,
                    "yaw_rate": row.yaw_rate,
                    "instruction": instruction,

                    # Episoden-Grenzen markieren:
                    "is_first": int(i == 0),
                    "is_last": int(i == episode_length - 1),
                })

                global_image_id += 1

            episode_id += 1   # nächste Episode

    merged_df = pd.DataFrame(merged_rows)
    merged_df.to_csv(os.path.join(output_dir, "dataset.csv"), index=False)

merge_dataset(r"D:\Projekte\Vlam\vlam_drone_project\data", r"D:\Projekte\Vlam\vlam_drone_project\thws_dataset")