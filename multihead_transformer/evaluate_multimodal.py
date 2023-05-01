import json

import torch
from decoder import CaptionDecoder
from tensorboardX import SummaryWriter
from dataloader import QuickRecipeDataset
from pathlib import Path

data_name = 'youcook_multihead_transformer'
PROJ_DIR = Path.cwd().parent
DATA_DIR = PROJ_DIR / 'data'
LOG_DIR = PROJ_DIR / 'logs'
LOG_SAVE_DIR = LOG_DIR / data_name
LOG_SAVE_DIR.mkdir(parents=True, exist_ok=True) 
MODEL_DIR = PROJ_DIR / 'models'
MODEL_SAVE_DIR = MODEL_DIR / data_name
MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True) 

# from trainer import train
from trainer_multimodal import generate_evaluation_predictions

def main():
    # Load the pipeline configuration file
    config_path = "config.json"
    with open(config_path, "r", encoding="utf8") as f:
        config = json.load(f)

    writer = SummaryWriter()
    use_gpu = config["use_gpu"] and torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu else "cpu")
    test_set = QuickRecipeDataset(config, config["split_save"]["test"], training=False, multimodal=True)

    encoder = None
    
    decoder = CaptionDecoder(config)
    decoder = decoder.to(device)

    if config["checkpoint"]["load"]:
        model_path = config["checkpoint"]["path"]
        checkpoint_full_path = f"{str(MODEL_SAVE_DIR)}/{model_path}"
        decoder.load_state_dict(torch.load(checkpoint_full_path))
    decoder.eval()

    generate_evaluation_predictions(test_set, encoder, decoder, config, device)


if __name__ == "__main__":
    main()
