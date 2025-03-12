import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
import torch
import os
import matplotlib.pyplot as plt

from models.transformer_chess import EncoderOnlyTransformer
from utils.mcts import train_model_with_mcts, train_model_with_mcts_regeneration



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
    
    # Setup optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config.training.get("learning_rate", 1e-4),
        weight_decay=config.training.get("weight_decay", 1e-4)
    )
    
    # Load optimizer state if continuing training
    if config.training.get("resume", False) and config.checkpoint.path:
        checkpoint = torch.load(config.checkpoint.path, map_location=device)
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("Loaded optimizer state")
    
    # Training parameters
    num_epochs = config.training.get("epochs", 10)
    batch_size = config.data.get("batch_size", 32)
    checkpoint_dir = config.training.get("checkpoint_dir", "checkpoints")
    save_frequency = config.training.get("save_frequency", 5)
    
    # MCTS parameters
    use_mcts = config.mcts.get("enabled", True)
    mcts_games_per_epoch = config.mcts.get("games_per_epoch", 10)
    mcts_simulations = config.mcts.get("simulations", 100)
    mcts_temperature = config.mcts.get("temperature", 1.0)
    mcts_exploration = config.mcts.get("exploration", 1.0)
    mcts_dirichlet_noise = config.mcts.get("dirichlet_noise", True)
    
    # Choose training method
    if use_mcts:
        train_method = config.mcts.get("train_method", "regeneration")
        
        if train_method == "regeneration":
            # Train with game regeneration at each epoch
            print("Training with MCTS self-play (game regeneration at each epoch)")
            history = train_model_with_mcts_regeneration(
                model=model,
                optimizer=optimizer,
                num_epochs=num_epochs,
                batch_size=batch_size,
                num_games_per_epoch=mcts_games_per_epoch,
                device=device,
                simulations=mcts_simulations,
                temperature=mcts_temperature,
                c_puct=mcts_exploration,
                checkpoint_dir=checkpoint_dir,
                save_frequency=save_frequency
            )
        else:
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
                save_path=checkpoint_dir
            )
    else:
        # Use preexisting training data
        predata_path = config.data.get("predata_path", None)
        if predata_path:
            print(f"Training with preexisting data from {predata_path}")
            from utils.mcts import train_model_with_predata
            
            history = train_model_with_predata(
                model=model,
                num_epochs=num_epochs,
                batch_size=batch_size,
                device=device,
                predata_path=predata_path
            )
        else:
            print("No training method selected. Please set mcts.enabled=True or provide a predata_path.")
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