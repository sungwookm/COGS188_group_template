import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.transformer_chess import EncoderOnlyTransformer
from trainer.dataset import ChessDataset


class Trainer:
    def __init__(self, config, model: nn.Module, dataloader: DataLoader):
        self.config = config
        self.device = torch.device(config.training.device if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.dataloader = dataloader
        
        # Optimizer and loss functions
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.training.learning_rate)
        self.criterion_moves = nn.CrossEntropyLoss()
        self.criterion_winrate = nn.BCELoss()

    def train(self):
        for epoch in range(self.config.training.num_epochs):
            self.model.train()
            total_loss = 0

            for batch in tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{self.config.training.num_epochs}"):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # Forward pass
                outputs = self.model(batch)
                '''
                moves = self.moves_head(boards)
                # shape of moves is (N, n_moves)
                winrate = self.winrate_head(boards) 
                # shape of winrate is (N, 1)
                return {
                    "move": moves,
                    "winrate": winrate
                }
                '''
                
                # Calculate loss TODO
                

                loss = move_loss + winrate_loss
                total_loss += loss.item()

                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            avg_loss = total_loss / len(self.dataloader)
            print(f"Epoch {epoch+1}/{self.config.training.num_epochs}, Loss: {avg_loss:.4f}")

    def save_model(self):
        torch.save(self.model.state_dict(), self.config.paths.model_save_path)
        print(f"Model saved to {self.config.paths.model_save_path}")
