import torch
import chess
import chess.pgn
import chess.engine
import datetime
import io
import numpy as np
import random
import os
import time
import subprocess
import platform
import shutil
import glob
from tqdm import tqdm

# Import UCI_MOVES mapping from the main file
from utils.paste import UCI_MOVES


def create_batch_from_board(board, device='cpu'):
    """
    Creates a batch dictionary from a chess.Board object for the EncoderOnlyTransformer model.
    
    Args:
        board (chess.Board): The chess board to create a batch from.
        device (str): The device to put the tensors on.
        
    Returns:
        dict: A batch dictionary containing board state information
    """
    # Map pieces to integers (our vocab)
    piece_to_idx = {
        'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,  # White pieces
        'p': 7, 'n': 8, 'b': 9, 'r': 10, 'q': 11, 'k': 12,  # Black pieces
        '.': 0  # Empty square
    }
    
    # Create a batch of size 1
    batch = {}
    
    # Turn encoding (white=0, black=1)
    batch["turns"] = torch.tensor([[0 if board.turn else 1]], dtype=torch.long, device=device)
    
    # Castling rights (0=no, 1=yes)
    batch["white_kingside_castling_rights"] = torch.tensor([[1 if board.has_kingside_castling_rights(chess.WHITE) else 0]], 
                                                          dtype=torch.long, device=device)
    batch["white_queenside_castling_rights"] = torch.tensor([[1 if board.has_queenside_castling_rights(chess.WHITE) else 0]], 
                                                           dtype=torch.long, device=device)
    batch["black_kingside_castling_rights"] = torch.tensor([[1 if board.has_kingside_castling_rights(chess.BLACK) else 0]], 
                                                          dtype=torch.long, device=device)
    batch["black_queenside_castling_rights"] = torch.tensor([[1 if board.has_queenside_castling_rights(chess.BLACK) else 0]], 
                                                           dtype=torch.long, device=device)
    
    # Board positions (64 squares)
    board_positions = []
    for square in range(64):
        piece = board.piece_at(square)
        if piece is None:
            board_positions.append(piece_to_idx['.'])
        else:
            board_positions.append(piece_to_idx[piece.symbol()])
    
    batch["board_positions"] = torch.tensor([board_positions], dtype=torch.long, device=device)
    
    return batch


def get_best_move(board, model, device='cpu', temperature=1.0, top_k=5):
    """
    Process a chess board through the model to get the best legal move.
    
    Args:
        board (chess.Board): The chess board to process.
        model (EncoderOnlyTransformer): The model to use for prediction.
        device (str): The device to put the tensors on.
        temperature (float): Temperature for softmax sampling (higher = more random).
        top_k (int): Number of top moves to consider.
        
    Returns:
        chess.Move: The selected move.
    """
    # Create batch from the board
    batch = create_batch_from_board(board, device)
    
    # Set model to evaluation mode
    model.eval()
    
    # Forward pass
    with torch.no_grad():
        outputs = model(batch)
    
    # Extract move logits
    move_logits = outputs["move"].squeeze(0)  # [1, 1970] -> [1970]
    
    # Get all legal moves
    legal_moves = list(board.legal_moves)
    
    # Filter for moves that are in our UCI_MOVES dictionary
    valid_move_indices = []
    valid_moves = []
    
    for i, move in enumerate(legal_moves):
        uci = move.uci()
        if uci in UCI_MOVES:
            valid_move_indices.append(UCI_MOVES[uci])
            valid_moves.append(move)
    
    if not valid_moves:
        # If no moves are in our dictionary, return a random legal move
        return random.choice(legal_moves)
    
    # Extract logits for valid moves
    valid_logits = move_logits[valid_move_indices]
    
    # Apply temperature and softmax to get probabilities
    if temperature > 0:
        valid_logits = valid_logits / temperature
    
    # Get top-k moves
    if len(valid_moves) > top_k:
        top_k_indices = torch.topk(valid_logits, top_k).indices
        top_k_valid_logits = valid_logits[top_k_indices]
        top_k_valid_moves = [valid_moves[i] for i in top_k_indices]
    else:
        top_k_valid_logits = valid_logits
        top_k_valid_moves = valid_moves
    
    # Apply softmax to get probabilities
    probs = torch.softmax(top_k_valid_logits, dim=0).cpu().numpy()
    
    # Sample from the probability distribution
    selected_idx = np.random.choice(len(top_k_valid_moves), p=probs)
    selected_move = top_k_valid_moves[selected_idx]
    
    return selected_move


def find_stockfish():
    """
    Attempt to find the Stockfish executable on the system.
    
    Returns:
        str or None: Path to Stockfish executable, or None if not found
    """
    # Check if stockfish is available in PATH
    stockfish_cmd = "stockfish"
    if platform.system() == "Windows":
        stockfish_cmd = "stockfish.exe"
    
    if shutil.which(stockfish_cmd):
        return stockfish_cmd
    
    # Common locations by platform
    if platform.system() == "Windows":
        common_paths = [
            r"C:\Program Files\stockfish\stockfish-windows-x86-64-avx2.exe",
            r"C:\Program Files (x86)\stockfish\stockfish.exe",
            r"stockfish.exe"
        ]
    elif platform.system() == "Darwin":  # macOS
        common_paths = [
            "/usr/local/bin/stockfish",
            "/opt/homebrew/bin/stockfish",
            "./stockfish"
        ]
    else:  # Linux/Unix
        common_paths = [
            "/usr/games/stockfish",
            "/usr/local/bin/stockfish",
            "./stockfish"
        ]
    
    for path in common_paths:
        if os.path.isfile(path):
            return path
    
    print("Stockfish not found. Please install Stockfish or provide the correct path.")
    return None


def play_game(model, opponent="self", model_color="white", stockfish_path=None, 
              stockfish_elo=1500, stockfish_depth=5, max_moves=100, device='cpu', 
              temperature=1.0, top_k=5):
    """
    Play a game with the model against itself or Stockfish.
    
    Args:
        model: The model to use for prediction.
        opponent (str): "self" or "stockfish" to determine the opponent.
        model_color (str): "white", "black", or "random" to determine which side the model plays.
        stockfish_path (str): Path to Stockfish executable (only needed for stockfish opponent).
        stockfish_elo (int): ELO rating for Stockfish (only used with Stockfish opponent).
        stockfish_depth (int): Search depth for Stockfish (only used with Stockfish opponent).
        max_moves (int): Maximum number of moves before declaring a draw.
        device (str): The device to put the tensors on.
        temperature (float): Temperature for move selection (higher = more random).
        top_k (int): Number of top moves to consider.
        
    Returns:
        chess.pgn.Game: The completed game.
    """
    # Initialize a new board
    board = chess.Board()
    
    # Create a game object to store the moves
    game = chess.pgn.Game()
    
    # Set up game metadata
    game.headers["Event"] = f"Chess Transformer vs {opponent.capitalize()}"
    game.headers["Site"] = "Local Machine"
    game.headers["Date"] = datetime.datetime.now().strftime("%Y.%m.%d")
    game.headers["Round"] = "1"
    
    # Randomly determine which side the model plays if specified
    if model_color == "random":
        model_plays_white = random.choice([True, False])
    else:
        model_plays_white = model_color.lower() == "white"
    
    if opponent == "self":
        game.headers["White"] = "ChessTransformer"
        game.headers["Black"] = "ChessTransformer"
    else:
        # Set up player names based on model's color
        game.headers["White"] = "ChessTransformer" if model_plays_white else f"Stockfish (ELO {stockfish_elo})"
        game.headers["Black"] = f"Stockfish (ELO {stockfish_elo})" if model_plays_white else "ChessTransformer"
    
    # Initialize Stockfish if opponent is stockfish
    engine = None
    if opponent == "stockfish":
        if not stockfish_path:
            stockfish_path = find_stockfish()
            if not stockfish_path:
                raise ValueError("Stockfish executable not found. Please provide a valid path.")
        
        try:
            engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
            
            # Configure Stockfish
            if stockfish_elo is not None:
                # Set ELO strength
                engine.configure({"UCI_LimitStrength": True, "UCI_Elo": stockfish_elo})
            
            #print(f"Playing against Stockfish (ELO: {stockfish_elo}, Depth: {stockfish_depth})")
        except Exception as e:
            raise Exception(f"Failed to start Stockfish engine: {e}")
    
    # Initialize node for adding moves
    node = game
    
    move_count = 0
    #print(f"Starting game: ChessTransformer {'(White)' if model_plays_white else '(Black)'} vs "
     #     f"{'ChessTransformer' if opponent == 'self' else 'Stockfish'} "
    #      f"{'(Black)' if model_plays_white else '(White)'}")
    
    try:
        # Main game loop
        while not board.is_game_over() and move_count < max_moves:
            current_turn_is_white = board.turn == chess.WHITE
            model_turn = (current_turn_is_white and model_plays_white) or (not current_turn_is_white and not model_plays_white)
            
            if opponent == "self" or model_turn:
                # Model's turn
                move = get_best_move(board, model, device, temperature, top_k)
                player = "Model"
            else:
                # Stockfish's turn
                result = engine.play(board, chess.engine.Limit(depth=stockfish_depth))
                move = result.move
                player = "Stockfish"
            
            # Add the move to the game
            node = node.add_variation(move)
            
            # Make the move on the board
            board.push(move)
            
            # Print the move
            side = "White" if current_turn_is_white else "Black"
            #print(f"Move {move_count + 1}: {move.uci()} (by {side} - {player})")
            
            move_count += 1
    
    finally:
        # Close the engine if it was initialized
        if engine:
            engine.quit()
    
    # Add the result to the game headers
    if board.is_checkmate():
        if board.turn == chess.WHITE:
            game.headers["Result"] = "0-1"
            #print("Black wins by checkmate")
        else:
            game.headers["Result"] = "1-0"
            #print("White wins by checkmate")
    elif board.is_stalemate() or board.is_insufficient_material() or board.is_fifty_moves() or board.is_repetition(3):
        game.headers["Result"] = "1/2-1/2"
        print("Game drawn")
    elif move_count >= max_moves:
        game.headers["Result"] = "1/2-1/2"
        print(f"Game drawn after reaching maximum moves ({max_moves})")
    
    return game


def self_play(model, max_moves=100, device='cpu', temperature=1.0, top_k=5):
    """
    Have the model play a game against itself.
    
    Args:
        model: The model to use for prediction.
        max_moves (int): Maximum number of moves before declaring a draw.
        device (str): The device to put the tensors on.
        temperature (float): Temperature for move selection (higher = more random).
        top_k (int): Number of top moves to consider.
        
    Returns:
        chess.pgn.Game: The completed game.
    """
    return play_game(
        model=model,
        opponent="self",
        max_moves=max_moves,
        device=device,
        temperature=temperature,
        top_k=top_k
    )


def save_game(game, filepath):
    """
    Save a chess game to a PGN file.
    
    Args:
        game (chess.pgn.Game): The game to save.
        filepath (str): Where to save the game.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Write the game to the file
    with open(filepath, "w") as f:
        exporter = chess.pgn.FileExporter(f)
        game.accept(exporter)
    
    print(f"Game saved to {filepath}")


def generate_selfplay_games(model, num_games=10, games_dir="games/", max_moves=100, 
                            device='cpu', temperature=1.0, top_k=5, progress_bar=True):
    """
    Generate multiple self-play games and save them to the specified directory.
    
    Args:
        model: The model to use for prediction.
        num_games (int): Number of self-play games to generate.
        games_dir (str): Directory to save the games in.
        max_moves (int): Maximum number of moves before declaring a draw.
        device (str): Device to run the model on.
        temperature (float): Temperature for move selection (higher = more random).
        top_k (int): Number of top moves to consider.
        progress_bar (bool): Whether to display a progress bar.
        
    Returns:
        list: List of file paths to the generated game files.
    """
    # Ensure games directory exists
    os.makedirs(games_dir, exist_ok=True)
    
    # Set model to evaluation mode
    model.eval()
    
    # Generate games
    game_files = []
    
    # Set up progress tracking
    iter_range = range(num_games)
    if progress_bar:
        try:
            iter_range = tqdm(iter_range, desc="Generating self-play games")
        except ImportError:
            print(f"Generating {num_games} self-play games...")
    else:
        print(f"Generating {num_games} self-play games...")
    
    for i in iter_range:
        # Play a game
        game = self_play(
            model=model,
            max_moves=max_moves,
            device=device,
            temperature=temperature,
            top_k=top_k
        )
        
        # Save the game
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(games_dir, f"selfplay_{i+1}_{timestamp}.pgn")
        save_game(game, save_path)
        
        game_files.append(save_path)
        
        # Small delay to ensure unique timestamps
        time.sleep(0.1)
    
    print(f"Generated {num_games} self-play games in {games_dir}")
    
    return game_files


def create_training_batches_from_selfplay(model=None, num_games=10, games_dir="games/", 
                                          generate_games=True, batch_size=32, 
                                          max_moves=100, device="cpu", temperature=1.0, 
                                          top_k=5):
    """
    Generate self-play games and/or create training batches from self-play game files.
    
    This function can:
    1. Generate new self-play games using the provided model
    2. Process existing self-play games from the games directory
    3. Or both generate and process games
    
    Args:
        model: The model to use for generating games (required if generate_games=True)
        num_games (int): Number of self-play games to generate (if generate_games=True)
        games_dir (str): Directory for game files
        generate_games (bool): Whether to generate new games
        batch_size (int): Size of training batches to create
        max_moves (int): Maximum moves per game when generating
        device (str): Device to place tensors on
        temperature (float): Temperature for move selection when generating
        top_k (int): Number of top moves to consider when generating
        
    Returns:
        list: List of batch dictionaries ready for training
    """
    import random
    
    # Ensure games_dir ends with a slash
    if not games_dir.endswith('/'):
        games_dir += '/'
    
    # Step 1: Generate new games if requested
    if generate_games:
        if model is None:
            raise ValueError("Model must be provided when generate_games=True")
        
        print(f"Generating {num_games} new self-play games...")
        generate_selfplay_games(
            model=model,
            num_games=num_games,
            games_dir=games_dir,
            max_moves=max_moves,
            device=device,
            temperature=temperature,
            top_k=top_k
        )
    
    # Step 2: Find all PGN files in the games directory
    pgn_files = glob.glob(f"{games_dir}*.pgn")
    
    # Filter for self-play games
    selfplay_files = []
    for file_path in pgn_files:
        # Use the filename to identify self-play games
        if "selfplay_" in os.path.basename(file_path) or "vs_self" in os.path.basename(file_path):
            selfplay_files.append(file_path)
    
    if not selfplay_files:
        raise ValueError(f"No self-play games found in {games_dir}")
    
    print(f"Found {len(selfplay_files)} self-play games for processing")
    
    # Process each game file to extract training examples
    all_examples = []
    
    for file_path in tqdm(selfplay_files, desc="Processing game files"):
        examples = process_game_file(file_path, device)
        all_examples.extend(examples)
    
    print(f"Extracted {len(all_examples)} training examples")
    
    # Shuffle examples for better training
    random.shuffle(all_examples)
    
    # Create batches
    batches = []
    
    for i in range(0, len(all_examples), batch_size):
        batch_examples = all_examples[i:i+batch_size]
        if len(batch_examples) == batch_size:  # Only use full batches
            batch = create_batch_from_examples(batch_examples, device)
            batches.append(batch)
    
    print(f"Created {len(batches)} training batches of size {batch_size}")
    
    return batches


def process_game_file(file_path, device="cpu"):
    """
    Process a single PGN game file to extract training examples.
    
    Args:
        file_path (str): Path to PGN file
        device (str): Device to place tensors on
        
    Returns:
        list: List of training examples from this game
    """
    examples = []
    
    with open(file_path) as pgn_file:
        # Read the game from the PGN file
        game = chess.pgn.read_game(pgn_file)
        
        if game is None:
            return examples
        
        # Determine the game result
        result = game.headers.get("Result", "*")
        if result == "1-0":
            game_result = 1.0  # White win
        elif result == "0-1":
            game_result = 0.0  # Black win
        else:
            game_result = 0.5  # Draw
        
        # Initialize board
        board = game.board()
        
        # Process each move in the game
        for move in game.mainline_moves():
            # Get the current board state before the move
            board_state = board.copy()
            
            # Create training example
            example = {
                "board": board_state,
                "move": move,
                "result": game_result,
                # For white, use the result directly; for black, use 1-result
                "target_result": game_result if board.turn == chess.WHITE else 1.0 - game_result
            }
            
            examples.append(example)
            
            # Make the move on the board
            board.push(move)
    
    return examples


def create_batch_from_examples(examples, device="cpu"):
    """
    Create a batch dictionary from a list of training examples.
    
    Args:
        examples (list): List of example dictionaries containing board, move, and result
        device (str): Device to place tensors on
        
    Returns:
        dict: Batch dictionary ready for model training
    """
    # Initialize batch dictionary
    batch = {
        "turns": [],
        "white_kingside_castling_rights": [],
        "white_queenside_castling_rights": [],
        "black_kingside_castling_rights": [],
        "black_queenside_castling_rights": [],
        "board_positions": [],
        "target_moves": [],
        "target_results": []
    }
    
    # Process each example
    for example in examples:
        board = example["board"]
        move = example["move"]
        
        # Convert board to batch entries
        board_batch = create_batch_from_board(board, device)
        
        # Add board features to batch
        batch["turns"].append(board_batch["turns"][0])
        batch["white_kingside_castling_rights"].append(board_batch["white_kingside_castling_rights"][0])
        batch["white_queenside_castling_rights"].append(board_batch["white_queenside_castling_rights"][0])
        batch["black_kingside_castling_rights"].append(board_batch["black_kingside_castling_rights"][0])
        batch["black_queenside_castling_rights"].append(board_batch["black_queenside_castling_rights"][0])
        batch["board_positions"].append(board_batch["board_positions"][0])
        
        # Add target move
        move_uci = move.uci()
        if move_uci in UCI_MOVES:
            batch["target_moves"].append(UCI_MOVES[move_uci])
        else:
            # Handle moves not in UCI_MOVES - use <move> token
            batch["target_moves"].append(UCI_MOVES["<move>"])
        
        # Add target result
        batch["target_results"].append(example["target_result"])
    
    # Convert lists to tensors
    batch["turns"] = torch.stack(batch["turns"])
    batch["white_kingside_castling_rights"] = torch.stack(batch["white_kingside_castling_rights"])
    batch["white_queenside_castling_rights"] = torch.stack(batch["white_queenside_castling_rights"])
    batch["black_kingside_castling_rights"] = torch.stack(batch["black_kingside_castling_rights"])
    batch["black_queenside_castling_rights"] = torch.stack(batch["black_queenside_castling_rights"])
    batch["board_positions"] = torch.stack(batch["board_positions"])
    batch["target_moves"] = torch.tensor(batch["target_moves"], dtype=torch.long, device=device)
    batch["target_results"] = torch.tensor(batch["target_results"], dtype=torch.float32, device=device)
    
    return batch


def save_training_batches(batches, output_dir="training_data/"):
    """
    Save training batches to disk for later use.
    
    Args:
        batches (list): List of batch dictionaries
        output_dir (str): Directory to save batches to
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save each batch as a separate file
    for i, batch in enumerate(batches):
        output_path = os.path.join(output_dir, f"batch_{i}.pt")
        torch.save(batch, output_path)
    
    print(f"Saved {len(batches)} batches to {output_dir}")


def load_training_batches(input_dir="training_data/", device="cpu"):
    """
    Load training batches from disk.
    
    Args:
        input_dir (str): Directory containing saved batches
        device (str): Device to place tensors on
        
    Returns:
        list: List of batch dictionaries
    """
    # Find all batch files
    batch_files = glob.glob(os.path.join(input_dir, "batch_*.pt"))
    batch_files.sort()  # Sort to ensure consistent order
    
    batches = []
    
    for file_path in batch_files:
        # Load batch
        batch = torch.load(file_path, map_location=device)
        batches.append(batch)
    
    print(f"Loaded {len(batches)} batches from {input_dir}")
    
    return batches


def train_model(model, batches, num_epochs=10, learning_rate=1e-4, checkpoint_dir="checkpoints/"):
    """
    Train the model using the provided batches.
    
    Args:
        model: The EncoderOnlyTransformer model to train
        batches (list): List of training batch dictionaries
        num_epochs (int): Number of epochs to train for
        learning_rate (float): Learning rate for optimization
        checkpoint_dir (str): Directory to save model checkpoints
        
    Returns:
        model: The trained model
        list: Training history (loss values)
    """
    import torch.nn as nn
    import torch.optim as optim
    from datetime import datetime
    
    # Ensure checkpoint directory exists
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Set model to training mode
    model.train()
    
    # Define loss function and optimizer
    move_criterion = nn.CrossEntropyLoss()
    result_criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Initialize training history
    history = {
        'epoch': [],
        'total_loss': [],
        'move_loss': [],
        'result_loss': []
    }
    
    # Set loss weights
    move_loss_weight = 1.0
    result_loss_weight = 0.2
    
    # Training loop
    print(f"Starting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Track losses for this epoch
        epoch_total_loss = 0.0
        epoch_move_loss = 0.0
        epoch_result_loss = 0.0
        
        # Process each batch
        for batch_idx, batch in enumerate(batches):
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(batch)
            
            # Calculate move loss
            move_logits = outputs["move"]
            move_targets = batch["target_moves"]
            move_loss = move_criterion(move_logits, move_targets)
            
            # Calculate result loss (win prediction)
            result_logits = outputs["winrate"].squeeze(-1)
            result_targets = batch["target_results"]
            result_loss = result_criterion(result_logits, result_targets)
            
            # Combine losses
            total_loss = move_loss_weight * move_loss + result_loss_weight * result_loss
            
            # Backward pass and optimization
            total_loss.backward()
            optimizer.step()
            
            # Update epoch losses
            epoch_total_loss += total_loss.item()
            epoch_move_loss += move_loss.item()
            epoch_result_loss += result_loss.item()
            
            # Print progress
            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(batches)}, "
                      f"Loss: {total_loss.item():.4f} (Move: {move_loss.item():.4f}, Result: {result_loss.item():.4f})")
        
        # Calculate average losses for the epoch
        avg_total_loss = epoch_total_loss / len(batches)
        avg_move_loss = epoch_move_loss / len(batches)
        avg_result_loss = epoch_result_loss / len(batches)
        
        # Update history
        history['epoch'].append(epoch + 1)
        history['total_loss'].append(avg_total_loss)
        history['move_loss'].append(avg_move_loss)
        history['result_loss'].append(avg_result_loss)
        
        # Print epoch summary
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{num_epochs} completed in {epoch_time:.2f}s, "
              f"Avg Loss: {avg_total_loss:.4f} (Move: {avg_move_loss:.4f}, Result: {avg_result_loss:.4f})")
        
        # Save checkpoint
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}_{timestamp}.pt")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_total_loss,
            'history': history
        }, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
    
    print("Training completed!")
    
    return model, history


def visualize_training_history(history):

    """
    Visualize the training history.
    
    Args:
        history (dict): Training history dictionary
    """
    try:
        import matplotlib.pyplot as plt
        
        # Create figure and subplots
        fig, axs = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot total loss
        axs[0].plot(history['epoch'], history['total_loss'], 'b-', label='Total Loss')
        axs[0].set_title('Total Loss vs. Epoch')
        axs[0].set_xlabel('Epoch')
        axs[0].set_ylabel('Loss')
        axs[0].legend()
        axs[0].grid(True)
        
        # Plot component losses
        axs[1].plot(history['epoch'], history['move_loss'], 'r-', label='Move Loss')
        axs[1].plot(history['epoch'], history['result_loss'], 'g-', label='Result Loss')
        axs[1].set_title('Component Losses vs. Epoch')
        axs[1].set_xlabel('Epoch')
        axs[1].set_ylabel('Loss')
        axs[1].legend()
        axs[1].grid(True)
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig('training_history.png')
        print("Training history visualization saved to training_history.png")
        
        # Display the plot if in an interactive environment
        plt.show()
        
    except ImportError:
        print("Matplotlib not available for visualizing training history")

def clear_games_directory(games_dir):
    """
    Delete all PGN files in the games directory to start with a clean slate.
    
    Args:
        games_dir (str): Directory containing PGN game files
    """
    import os
    import glob
    
    # Ensure games_dir ends with a slash
    if not games_dir.endswith('/'):
        games_dir += '/'
        
    # Find all PGN files in the games directory
    pgn_files = glob.glob(f"{games_dir}*.pgn")
    
    if pgn_files:
        print(f"Clearing {len(pgn_files)} existing game files from {games_dir}")
        for file_path in pgn_files:
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Error removing {file_path}: {e}")
    else:
        print(f"No existing game files found in {games_dir}")

def clear_training_data_directory(training_data_dir):
    """
    Delete all training data files in the training data directory.
    
    Args:
        training_data_dir (str): Directory containing training data files
    """
    import os
    import glob
    
    # Ensure training_data_dir ends with a slash
    if not training_data_dir.endswith('/'):
        training_data_dir += '/'
    
    # Find all training data files in the directory
    data_files = glob.glob(f"{training_data_dir}*.pt")
    
    if data_files:
        print(f"Clearing {len(data_files)} existing training data files from {training_data_dir}")
        for file_path in data_files:
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Error removing {file_path}: {e}")
    else:
        print(f"No existing training data files found in {training_data_dir}")

def generate_training_games(model, num_games=10, games_dir="games/", max_moves=100,
                          device='cpu', temperature=1.0, top_k=5, stockfish_percentage=0, 
                          stockfish_path=None, stockfish_elo=1500, stockfish_depth=5, progress_bar=True):
    """
    Generate training games with a specified percentage against Stockfish.
    
    Args:
        model: The model to use for prediction.
        num_games (int): Total number of games to generate.
        games_dir (str): Directory to save the games in.
        max_moves (int): Maximum number of moves before declaring a draw.
        device (str): Device to run the model on.
        temperature (float): Temperature for move selection (higher = more random).
        top_k (int): Number of top moves to consider.
        stockfish_percentage (float): Percentage (0-100) of games to play against Stockfish.
        stockfish_path (str): Path to Stockfish executable.
        stockfish_elo (int): ELO rating for Stockfish.
        stockfish_depth (int): Search depth for Stockfish.
        progress_bar (bool): Whether to display a progress bar.
        
    Returns:
        list: List of file paths to the generated game files.
    """
    import random
    import os
    from tqdm import tqdm
    import time
    import datetime
    
    # Ensure games directory exists
    os.makedirs(games_dir, exist_ok=True)
    
    # Set model to evaluation mode
    model.eval()
    
    # Calculate number of games for each type
    num_stockfish_games = int(num_games * (stockfish_percentage / 100))
    num_selfplay_games = num_games - num_stockfish_games
    
    print(f"Generating {num_selfplay_games} self-play games and {num_stockfish_games} games against Stockfish")
    
    game_files = []
    
    # Generate self-play games
    if num_selfplay_games > 0:
        selfplay_files = generate_selfplay_games(
            model=model,
            num_games=num_selfplay_games,
            games_dir=games_dir,
            max_moves=max_moves,
            device=device,
            temperature=temperature,
            top_k=top_k,
            progress_bar=progress_bar
        )
        game_files.extend(selfplay_files)
    
    # Generate games against Stockfish
    if num_stockfish_games > 0:
        if not stockfish_path:
            stockfish_path = find_stockfish()
            if not stockfish_path:
                print("Warning: Stockfish not found. Falling back to self-play games instead.")
                # Generate additional self-play games instead
                additional_selfplay_files = generate_selfplay_games(
                    model=model,
                    num_games=num_stockfish_games,
                    games_dir=games_dir,
                    max_moves=max_moves,
                    device=device,
                    temperature=temperature,
                    top_k=top_k,
                    progress_bar=progress_bar
                )
                game_files.extend(additional_selfplay_files)
            else:
                # Set up progress tracking for Stockfish games
                iter_range = range(num_stockfish_games)
                if progress_bar:
                    try:
                        iter_range = tqdm(iter_range, desc="Generating games against Stockfish")
                    except ImportError:
                        print(f"Generating {num_stockfish_games} games against Stockfish...")
                else:
                    print(f"Generating {num_stockfish_games} games against Stockfish...")
                
                for i in iter_range:
                    # Randomly decide model color for variety
                    model_color = random.choice(["white", "black"])
                    
                    # Play a game against Stockfish
                    game = play_game(
                        model=model,
                        opponent="stockfish",
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
                    save_path = os.path.join(
                        games_dir, 
                        f"stockfish_{model_color}_vs_stockfish_{stockfish_elo}_{timestamp}.pgn"
                    )
                    save_game(game, save_path)
                    
                    game_files.append(save_path)
                    
                    # Small delay to ensure unique timestamps
                    time.sleep(0.1)
    
    print(f"Generated {len(game_files)} total training games in {games_dir}")
    return game_files

def train_model_with_regeneration(model, num_epochs=10, games_dir="games/", training_data_dir="training_data/",
                                  num_games=10, batch_size=32, max_moves=100, device="cpu",
                                  temperature=1.0, top_k=5, learning_rate=1e-4, checkpoint_dir="checkpoints/",
                                  save_frequency=10, stockfish_percentage=0, stockfish_path=None, 
                                  stockfish_elo=1500, stockfish_depth=5):
    """
    Train the model with regeneration of games at each epoch, with option to include Stockfish games.
    
    Args:
        model: The EncoderOnlyTransformer model to train
        num_epochs (int): Number of epochs to train for
        games_dir (str): Directory for game files
        training_data_dir (str): Directory to save processed training data
        num_games (int): Number of games to generate per epoch
        batch_size (int): Size of training batches
        max_moves (int): Maximum moves per game when generating
        device (str): Device to run on
        temperature (float): Temperature for move selection
        top_k (int): Number of top moves to consider
        learning_rate (float): Learning rate for optimizer
        checkpoint_dir (str): Directory to save model checkpoints
        save_frequency (int): How often to save model checkpoints (in epochs)
        stockfish_percentage (float): Percentage (0-100) of games to play against Stockfish
        stockfish_path (str): Path to Stockfish executable
        stockfish_elo (int): ELO rating for Stockfish
        stockfish_depth (int): Search depth for Stockfish
        
    Returns:
        model: The trained model
        dict: Training history
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import os
    import time
    from datetime import datetime
    
    # Ensure directories exist
    os.makedirs(games_dir, exist_ok=True)
    os.makedirs(training_data_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Define loss function and optimizer
    move_criterion = nn.CrossEntropyLoss()
    result_criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Initialize training history
    history = {
        'epoch': [],
        'total_loss': [],
        'move_loss': [],
        'result_loss': []
    }
    
    # Set loss weights
    move_loss_weight = 1.0
    result_loss_weight = 0.2
    
    # Training loop
    print(f"Starting training for {num_epochs} epochs with game regeneration each epoch")
    if stockfish_percentage > 0:
        print(f"Including {stockfish_percentage}% games against Stockfish (ELO: {stockfish_elo})")
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
        
        # Clear games directory and generate new games for this epoch
        clear_games_directory(games_dir)
        clear_training_data_directory(training_data_dir)
        
        # Generate new games with potential Stockfish games
        print(f"Generating {num_games} new training games for epoch {epoch+1}")
        game_files = generate_training_games(
            model=model,
            num_games=num_games,
            games_dir=games_dir,
            max_moves=max_moves,
            device=device,
            temperature=temperature,
            top_k=top_k,
            stockfish_percentage=stockfish_percentage,
            stockfish_path=stockfish_path,
            stockfish_elo=stockfish_elo,
            stockfish_depth=stockfish_depth
        )
        
        # Process games into training batches
        print("Processing games into training batches")
        batches = []
        
        # Extract training examples from games
        all_examples = []
        for file_path in tqdm(game_files, desc="Processing game files"):
            examples = process_game_file(file_path, device)
            all_examples.extend(examples)
        
        print(f"Extracted {len(all_examples)} training examples")
        
        # Shuffle examples for better training
        random.shuffle(all_examples)
        
        # Create batches
        for i in range(0, len(all_examples), batch_size):
            batch_examples = all_examples[i:i+batch_size]
            if len(batch_examples) == batch_size:  # Only use full batches
                batch = create_batch_from_examples(batch_examples, device)
                batches.append(batch)
        
        print(f"Created {len(batches)} training batches of size {batch_size}")
        
        # Save batches if requested
        save_training_batches(batches, training_data_dir)
        
        # Set model to training mode
        model.train()
        
        # Track losses for this epoch
        epoch_total_loss = 0.0
        epoch_move_loss = 0.0
        epoch_result_loss = 0.0
        
        # Process each batch
        print(f"Training on {len(batches)} batches")
        for batch_idx, batch in enumerate(batches):
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(batch)
            
            # Calculate move loss
            move_logits = outputs["move"]
            move_targets = batch["target_moves"]
            move_loss = move_criterion(move_logits, move_targets)
            
            # Calculate result loss (win prediction)
            result_logits = outputs["winrate"].squeeze(-1)
            result_targets = batch["target_results"]
            result_loss = result_criterion(result_logits, result_targets)
            
            # Combine losses
            total_loss = move_loss_weight * move_loss + result_loss_weight * result_loss
            
            # Backward pass and optimization
            total_loss.backward()
            optimizer.step()
            
            # Update epoch losses
            epoch_total_loss += total_loss.item()
            epoch_move_loss += move_loss.item()
            epoch_result_loss += result_loss.item()
            
            # Print progress
            if (batch_idx + 1) % 10 == 0:
                print(f"Batch {batch_idx+1}/{len(batches)}, "
                      f"Loss: {total_loss.item():.4f} (Move: {move_loss.item():.4f}, Result: {result_loss.item():.4f})")
        
        # Calculate average losses for the epoch
        avg_total_loss = epoch_total_loss / max(len(batches), 1)
        avg_move_loss = epoch_move_loss / max(len(batches), 1)
        avg_result_loss = epoch_result_loss / max(len(batches), 1)
        
        # Update history
        history['epoch'].append(epoch + 1)
        history['total_loss'].append(avg_total_loss)
        history['move_loss'].append(avg_move_loss)
        history['result_loss'].append(avg_result_loss)
        
        # Print epoch summary
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{num_epochs} completed in {epoch_time:.2f}s, "
              f"Avg Loss: {avg_total_loss:.4f} (Move: {avg_move_loss:.4f}, Result: {avg_result_loss:.4f})")
        
        # Save checkpoint based on frequency
        if (epoch + 1) % save_frequency == 0 or epoch == num_epochs - 1:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}_{timestamp}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_total_loss,
                'history': history
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
            
            # Save latest model separately (overwrite)
            latest_model_path = os.path.join(checkpoint_dir, "latest_model.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_total_loss,
                'history': history
            }, latest_model_path)
            print(f"Latest model saved to {latest_model_path}")
    
    print("Training completed!")
    
    return model, history