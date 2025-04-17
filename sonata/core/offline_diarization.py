import os
import yaml
import logging
import shutil
from pathlib import Path
from typing import Dict, Optional
from huggingface_hub import hf_hub_download


def download_diarization_models(
    token: str, save_dir: str = "~/.sonata/models"
) -> Dict[str, str]:
    """
    Download diarization models and prepare them for offline use.

    Args:
        token: HuggingFace token with read permission
        save_dir: Directory to save models to

    Returns:
        Dictionary with paths to config and model files
    """
    # Expand user directory if needed (e.g., ~ to /home/user)
    save_dir = os.path.expanduser(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    logging.info(f"Downloading diarization models to {save_dir}")

    # Download configuration file
    config_path = hf_hub_download(
        repo_id="pyannote/speaker-diarization-3.1",
        filename="config.yaml",
        token=token,
        cache_dir=save_dir,
    )

    # Download model file
    model_path = hf_hub_download(
        repo_id="pyannote/segmentation-3.0",
        filename="pytorch_model.bin",
        token=token,
        cache_dir=save_dir,
    )

    # Create offline config by modifying original config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Convert to absolute path and update config
    model_absolute_path = str(Path(model_path).resolve())
    config["segmentation"] = model_absolute_path

    # Save modified config
    offline_config_path = os.path.join(save_dir, "offline_config.yaml")
    with open(offline_config_path, "w") as f:
        yaml.dump(config, f)

    logging.info(f"Diarization models downloaded and configured for offline use")
    logging.info(f"Config: {offline_config_path}")
    logging.info(f"Model: {model_path}")

    return {"config_path": offline_config_path, "model_path": model_path}


def verify_offline_models(config_path: str) -> bool:
    """
    Verify that offline diarization models exist and are correctly configured.

    Args:
        config_path: Path to offline configuration file

    Returns:
        True if models exist and are correctly configured, False otherwise
    """
    # Check if config file exists
    if not os.path.exists(config_path):
        logging.error(f"Config file {config_path} does not exist")
        return False

    # Load config file
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logging.error(f"Error loading config file: {e}")
        return False

    # Check if segmentation model path is specified and exists
    if "segmentation" not in config:
        logging.error(f"Config file does not contain segmentation model path")
        return False

    model_path = config["segmentation"]
    # Expand user directory if needed
    model_path = os.path.expanduser(model_path)

    if not os.path.exists(model_path):
        logging.error(f"Segmentation model {model_path} does not exist")
        return False

    return True


def clean_offline_models(save_dir: str = "~/.sonata/models") -> bool:
    """
    Remove downloaded offline diarization models.

    Args:
        save_dir: Directory containing models

    Returns:
        True if cleanup succeeded, False otherwise
    """
    save_dir = os.path.expanduser(save_dir)

    if not os.path.exists(save_dir):
        logging.warning(f"Directory {save_dir} does not exist, nothing to clean")
        return True

    try:
        shutil.rmtree(save_dir)
        logging.info(f"Removed offline diarization models from {save_dir}")
        return True
    except Exception as e:
        logging.error(f"Error removing offline diarization models: {e}")
        return False
