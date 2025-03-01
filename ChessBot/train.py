import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import os

from models.transformer_chess import EncoderOnlyTransformer
from utils.mcts import train_model_with_mcts_regeneration
from utils.utils import visualize_training_history
from utils.mcts import train_model_with_predata


@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(config: DictConfig) -> None:
    """
    Main entry point for training the chess transformer model with MCTS self-play.
    This function initializes the model and runs the training process with game regeneration.
    """
    print(OmegaConf.to_yaml(config))
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    model = EncoderOnlyTransformer(config.model).to(device)
    
    # Initialize model weights
    model.init_weights()
    
    # Load model weights if continuing training
    if config.training.get("resume", False) and config.checkpoint.path:
        checkpoint = torch.load(config.checkpoint.path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model weights from {config.checkpoint.path}")
    
    # Data and training parameters
    games_dir = config.data.get("games_dir", "games/")
    training_data_dir = config.data.get("output_dir", "training_data/")
    num_games = config.data.get("num_games", 10)
    batch_size = config.data.get("batch_size", 32)
    max_moves = config.data.get("max_moves", 100)
    temperature = config.data.get("temperature", 1.0)
    top_k = config.data.get("top_k", 5)
    
    # Stockfish parameters
    stockfish_percentage = config.data.get("stockfish_percentage", 0)
    stockfish_path = config.stockfish.get("path", None)
    stockfish_elo = config.stockfish.get("elo", 1500)
    stockfish_depth = config.stockfish.get("depth", 5)
    
    # MCTS parameters
    use_mcts = config.mcts.get("enabled", False)
    mcts_simulations = config.mcts.get("simulations", 100)
    mcts_exploration = config.mcts.get("exploration", 1.0)
    mcts_dirichlet_noise = config.mcts.get("dirichlet_noise", False)
    
    # Training parameters
    num_epochs = config.training.get("epochs", 10)
    learning_rate = config.training.get("learning_rate", 1e-4)
    checkpoint_dir = config.training.get("checkpoint_dir", "checkpoints/")
    save_frequency = config.training.get("save_frequency", 10)
    
    '''# Train the model with MCTS game regeneration each epoch
    model, history = train_model_with_mcts_regeneration(
        model=model,
        num_epochs=num_epochs,
        games_dir=games_dir,
        training_data_dir=training_data_dir,
        num_games=num_games,
        batch_size=batch_size,
        max_moves=max_moves,
        device=device,
        temperature=temperature,
        top_k=top_k,
        learning_rate=learning_rate,
        checkpoint_dir=checkpoint_dir,
        save_frequency=save_frequency,
        stockfish_percentage=stockfish_percentage,
        stockfish_path=stockfish_path,
        stockfish_elo=stockfish_elo,
        stockfish_depth=stockfish_depth,
        use_mcts=use_mcts,
        mcts_simulations=mcts_simulations,
        mcts_exploration=mcts_exploration,
        mcts_dirichlet_noise=mcts_dirichlet_noise
    )'''

    model, history = train_model_with_predata(
        model=model,
        num_epochs=20,
        batch_size=32,
        device="cuda",  # or "cpu"
        predata_path="C:\\Users\\haoyan\\Documents\\COGS188_group_template-1\\ChessBot\\data\\GM_games_dataset.csv",
    )
    
    # Visualize training progress
    if config.training.get("visualize", True):
        visualize_training_history(history)
    
    print("Training completed successfully!")


if __name__ == "__main__":
    main()