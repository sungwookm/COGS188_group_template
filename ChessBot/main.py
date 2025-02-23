import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from models.transformer_chess import EncoderOnlyTransformer
from trainer.trainer import Trainer 
from trainer.dataset import ChessDataset


@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(config: DictConfig) -> None:
    print(OmegaConf.to_yaml(config))

    # Load dataset and dataloader
    dataset = ChessDataset(config.data.dataset_path)
    dataloader = DataLoader(dataset, batch_size=config.training.batch_size, shuffle=True, num_workers=4)

    # Initialize model and trainer
    model = EncoderOnlyTransformer(config.model)
    trainer = Trainer(config, model, dataloader)

    # Train and save the model
    trainer.train()
    trainer.save_model()


if __name__ == "__main__":
    main()
