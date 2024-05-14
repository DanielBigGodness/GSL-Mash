import dotenv
import hydra
from omegaconf import DictConfig
import os

# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
dotenv.load_dotenv(override=True)



@hydra.main(config_path="configs/", config_name="train.yaml",version_base= "1.2") 
def main(config: DictConfig):

    from src import utils
    from src.training_pipeline import train

    # Applies optional utilities
    utils.extras(config)
    
    # Train model
    return train(config)


if __name__ == "__main__":
    main()
