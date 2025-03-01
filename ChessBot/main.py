import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import datetime
import os

from models.transformer_chess import EncoderOnlyTransformer
from utils.utils import play_game, save_game


@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(config: DictConfig) -> None:
    """
    Main entry point for the chess transformer application.
    This function sets up the model and runs a chess game based on the provided configuration.
    """
    print(OmegaConf.to_yaml(config))
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    model = EncoderOnlyTransformer(config.model).to(device)
    
    # Initialize model weights
    model.init_weights()
    
    # Load the latest model if available
    checkpoint_path = config.get("checkpoint_path", "checkpoints/latest_model.pt")
    
    if os.path.exists(checkpoint_path):
        try:
            print(f"Loading model from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Successfully loaded model (epoch {checkpoint.get('epoch', 'unknown')})")
            
            # Set model to evaluation mode
            model.eval()
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Continuing with initialized model")
    else:
        print(f"No model checkpoint found at {checkpoint_path}. Using initialized model.")
    
    # Game parameters
    opponent = config.game.get("opponent", "self")
    model_color = config.game.get("model_color", "white")
    max_moves = config.game.get("max_moves", 100)
    temperature = config.game.get("temperature", 1.0)
    top_k = config.game.get("top_k", 5)
    
    # Stockfish parameters (only used if opponent is stockfish)
    stockfish_path = config.stockfish.get("path", None)
    stockfish_elo = config.stockfish.get("elo", 1500)
    stockfish_depth = config.stockfish.get("depth", 5)
    
    # Run game
    game = play_game(
        model=model,
        opponent=opponent,
        model_color=model_color,
        stockfish_path=stockfish_path,
        stockfish_elo=stockfish_elo,
        stockfish_depth=stockfish_depth,
        max_moves=max_moves,
        device=device,
        temperature=temperature,
        top_k=top_k
    )
    
    # Save the game
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    opponent_name = "self" if opponent == "self" else f"stockfish_{stockfish_elo}"
    save_path = f"games/game_{model_color}_vs_{opponent_name}_{timestamp}.pgn"
    save_game(game, save_path)
    
    # Print a summary of the game
    print("\nGame summary:")
    print(f"Result: {game.headers['Result']}")
    print(f"Number of moves: {len(list(game.mainline_moves()))}")
    print(f"PGN file saved to: {save_path}")


if __name__ == "__main__":
    main()