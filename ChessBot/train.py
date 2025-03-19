import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
import torch
import os
import matplotlib.pyplot as plt
import datetime
import pickle
import chess.pgn
from models.transformer_chess import EncoderOnlyTransformer
from utils.mcts import train_model_with_mcts
# Import our new parallelized stockfish function


@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(config: DictConfig) -> None:
    """
    Main entry point for training the chess transformer model.
    This function initializes the model and runs the training process.
    """
    print(OmegaConf.to_yaml(config))
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    model = EncoderOnlyTransformer(config.model).to(device)
    
    # Initialize model weights
    model.init_weights()
    
    # Setup checkpoint loading for resuming training
    checkpoint_path = config.training.get("checkpoint_path", r'C:\Users\haoyan\Documents\COGS188_group_template-1\checkpoints\latest_model.pt')
    resume = config.training.get("resume", False)
    start_epoch = 0

    if resume:
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}")
            try:
                checkpoint = torch.load(checkpoint_path, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                start_epoch = checkpoint.get('epoch', 0)
                print(f"Successfully loaded model (epoch {start_epoch})")
                
                # Calculate remaining epochs
                epochs_to_train = config.training.get("epochs", 10) - start_epoch
                print(f"Resuming training from epoch {start_epoch} for {epochs_to_train} more epochs")
            except Exception as e:
                print(f"Error loading model: {e}")
                print("Starting with freshly initialized model")
        else:
            print(f"No checkpoint found at {checkpoint_path}. Starting with initialized model.")

    # Setup optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config.training.get("learning_rate", 1e-4),
        weight_decay=config.training.get("weight_decay", 1e-4)
    )
    
    # Load optimizer state if continuing training
    if resume and checkpoint_path and os.path.exists(checkpoint_path):
        try:
            # No need to reload checkpoint as it's already loaded above
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print(f"Loaded optimizer state from {checkpoint_path}")
        except Exception as e:
            print(f"Error loading optimizer state: {e}")

    # Training parameters
    num_epochs = config.training.get("epochs", 10)
    batch_size = config.data.get("batch_size", 32)
    checkpoint_dir = config.training.get("checkpoint_dir", "checkpoints")
    save_frequency = config.training.get("save_frequency", 5)
    
    # Create game history directory
    game_history_dir = os.path.join(checkpoint_dir, "game_history")
    os.makedirs(game_history_dir, exist_ok=True)
    
    # Training method selection
    train_method = config.training.get("method", "stockfish")
    
    if train_method == "stockfish":
        # Train with Stockfish vs Stockfish games
        print("Training with Stockfish vs Stockfish games (Parallel)")
        
        # Stockfish parameters
        stockfish_path = config.stockfish.get("path", None)
        if not stockfish_path:
            print("Error: Stockfish path not specified. Please set stockfish.path in the config.")
            return
            
        stockfish_games_per_epoch = config.stockfish.get("games_per_epoch", 10)
        
        # Get the number of parallel workers (defaults to None which will use CPU count)
        max_workers = config.stockfish.get("max_workers", None)
        print(f"Using {'CPU count' if max_workers is None else max_workers} worker threads for parallel game generation")
        
        history = train_model_with_stockfish_games(
            model=model,
            optimizer=optimizer,
            num_epochs=num_epochs,
            batch_size=batch_size,
            device=device,
            games_per_epoch=stockfish_games_per_epoch,
            stockfish_path=stockfish_path,
            checkpoint_dir=checkpoint_dir,
            save_frequency=save_frequency,
            start_epoch=start_epoch,
            max_workers=max_workers,
            regenerate_games = False
        )
    elif train_method == "mcts":
        # MCTS parameters
        mcts_games_per_epoch = config.mcts.get("games_per_epoch", 10)
        mcts_simulations = config.mcts.get("simulations", 100)
        mcts_temperature = config.mcts.get("temperature", 1.0)
        # Train with continuous game generation
        print("Training with MCTS self-play (continuous game generation)")
        num_games = config.mcts.get("num_games", 100)
        epochs_per_game = config.mcts.get("epochs_per_game", 1)
        
        history = train_model_with_mcts(
            model=model,
            optimizer=optimizer,
            num_games=num_games,
            epochs_per_game=epochs_per_game,
            batch_size=batch_size,
            device=device,
            simulations=mcts_simulations,
            temperature_init=mcts_temperature,
            save_path=checkpoint_dir,
            game_history_dir=game_history_dir
        )
    else:
        print(f"Error: Unknown training method '{train_method}'. Please use 'stockfish', 'mcts', or 'predata'.")
        return
    
    # Save final model
    final_checkpoint_path = os.path.join(checkpoint_dir, "final_model.pt")
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history
    }, final_checkpoint_path)
    print(f"Final model saved to {final_checkpoint_path}")
    
    # Also save as latest model for easy loading
    latest_path = os.path.join(checkpoint_dir, "latest_model.pt")
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history
    }, latest_path)
    print(f"Model also saved as latest model at {latest_path}")
    

if __name__ == "__main__":
    main()