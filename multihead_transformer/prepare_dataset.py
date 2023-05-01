import json
import pandas as pd

from utils.dataset_utils import clean_captions, create_vocab, extract_embeddings, split_dataset, load_captions, load_captions_from_df, split_dataset_custom
from pathlib import Path

PROJ_DIR = Path.cwd().parent
DATA_DIR = PROJ_DIR / 'data'

if __name__ == "__main__":
    # Load the project pipeline configuration
    config_path = "config.json"
    with open(config_path, "r") as f:
        config = json.load(f)

    # Load and clean the loaded captions
    # dataset_path = config["dataset_path"]
    # with open(dataset_path, "r") as f:
    #     data = f.read()
    dataset_path = str(DATA_DIR / "full_master_updated.pkl")
    full_master_df = pd.read_pickle(dataset_path)
    cond = (full_master_df['IsUsefulSentence'] == 1) & (~full_master_df['Key steps'].isna())
    df = full_master_df[cond]
    # image2caption = load_captions(data)
    image2caption = load_captions_from_df(df)
    image2caption = clean_captions(image2caption)
    print(f"Length of image2caption: {len(image2caption)}")

    # Create and save dataset corpus vocabulary
    vocab = create_vocab(image2caption)
    # Extract GloVe embeddings for tokens present in the training set vocab
    extract_embeddings(config, vocab)

    # Save info regarding the dataset split elements
    # Paths to train, validation, test images
    # split_images_paths = list(config["split_images"].values())
    
    # Paths to train, validation, test files which will
    # contain all the image paths
    split_save_paths = list(config["split_save"].values())
    # split_dataset(image2caption, split_images_paths, split_save_paths)
    split_dataset_custom(image2caption, df, split_save_paths)
