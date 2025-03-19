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
import pickle
from tqdm import tqdm

# Import UCI_MOVES mapping from the main file
from utils.paste import UCI_MOVES

def play_game(model, opponent="self", model_color="white", stockfish_path=None, stockfish_elo=1500, 
              stockfish_depth=5, max_moves=100, device="cuda", temperature=1.0, top_k=5):
    """
    Play a game between the model and specified opponent.
    
    Args:
        model: Neural network model
        opponent (str): Either "self" for self-play or "stockfish" for playing against Stockfish
        model_color (str): Either "white" or "black", determines which side the model plays
        stockfish_path (str): Path to Stockfish engine executable
        stockfish_elo (int): ELO rating for Stockfish (only used if opponent is stockfish)
        stockfish_depth (int): Search depth for Stockfish (only used if opponent is stockfish)
        max_moves (int): Maximum number of moves before the game is called a draw
        device: Device to run the model on
        temperature (float): Temperature for move sampling
        top_k (int): Number of top moves to consider when sampling
        
    Returns:
        chess.pgn.Game: The completed game
    """
    # Initialize board
    board = chess.Board()
    
    # Initialize Stockfish if needed
    stockfish_engine = None
    if opponent == "stockfish" and stockfish_path:
        try:
            stockfish_engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
            # Set engine parameters
            stockfish_engine.configure({"Skill Level": min(20, stockfish_elo // 100)})
        except Exception as e:
            print(f"Error initializing Stockfish: {e}")
            print("Falling back to self-play")
            opponent = "self"
    
    # Determine when the model plays
    model_plays_white = model_color.lower() == "white"
    
    # Create game object for PGN
    game = chess.pgn.Game()
    game.headers["Event"] = "Model Game"
    game.headers["Site"] = "Local"
    game.headers["Date"] = datetime.datetime.now().strftime("%Y.%m.%d")
    
    if opponent == "stockfish":
        game.headers["White"] = "Model" if model_plays_white else f"Stockfish (ELO {stockfish_elo})"
        game.headers["Black"] = f"Stockfish (ELO {stockfish_elo})" if model_plays_white else "Model"
    else:
        game.headers["White"] = "Model (White)"
        game.headers["Black"] = "Model (Black)"
    
    # Play the game
    node = game
    move_count = 0
    
    while not board.is_game_over() and move_count < max_moves:
        # Determine whose turn it is
        is_white_turn = board.turn == chess.WHITE
        is_model_turn = (is_white_turn and model_plays_white) or (not is_white_turn and not model_plays_white)
        
        # Get the move
        if opponent == "self" or is_model_turn:
            # Model makes a move
            move = get_model_move(model, board, device, temperature, top_k)
        else:
            # Stockfish makes a move
            result = stockfish_engine.play(board, chess.engine.Limit(depth=stockfish_depth))
            move = result.move
        
        # Make the move
        if move:
            board.push(move)
            node = node.add_variation(move)
            move_count += 1
        else:
            # No legal moves or model couldn't decide
            break
    
    # Set game result
    if board.is_checkmate():
        result = "1-0" if board.turn == chess.BLACK else "0-1"
    elif board.is_stalemate() or board.is_insufficient_material() or move_count >= max_moves:
        result = "1/2-1/2"
    else:
        result = "*"
    
    game.headers["Result"] = result
    
    # Clean up Stockfish
    if stockfish_engine:
        stockfish_engine.quit()
    
    return game

def get_model_move(model, board, device, temperature=1.0, top_k=5):
    """
    Get a move from the model for the current board position.
    
    Args:
        model: Neural network model
        board (chess.Board): Current board state
        device: Device to run the model on
        temperature (float): Temperature for move sampling
        top_k (int): Number of top moves to consider when sampling
        
    Returns:
        chess.Move: Selected move
    """
    # Get legal moves
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None
    
    # Create model input
    with torch.no_grad():
        # Convert board to model input format
        board_tensor = torch.zeros(1, 64, dtype=torch.long)
        
        # Map pieces to indices
        piece_to_idx = {
            'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,  # White pieces
            'p': 7, 'n': 8, 'b': 9, 'r': 10, 'q': 11, 'k': 12  # Black pieces
        }
        
        # Fill board tensor
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                board_tensor[0, square] = piece_to_idx[piece.symbol()]
        
        # Turn
        turn_tensor = torch.tensor([1 if board.turn == chess.WHITE else 0], dtype=torch.long)
        
        # Castling rights
        w_kingside = torch.tensor([1 if board.has_kingside_castling_rights(chess.WHITE) else 0], dtype=torch.long)
        w_queenside = torch.tensor([1 if board.has_queenside_castling_rights(chess.WHITE) else 0], dtype=torch.long)
        b_kingside = torch.tensor([1 if board.has_kingside_castling_rights(chess.BLACK) else 0], dtype=torch.long)
        b_queenside = torch.tensor([1 if board.has_queenside_castling_rights(chess.BLACK) else 0], dtype=torch.long)
        
        # Create batch
        batch = {
            "board_positions": board_tensor.to(device),
            "turns": turn_tensor.to(device),
            "white_kingside_castling_rights": w_kingside.to(device),
            "white_queenside_castling_rights": w_queenside.to(device),
            "black_kingside_castling_rights": b_kingside.to(device),
            "black_queenside_castling_rights": b_queenside.to(device)
        }
        
        # Get model predictions
        predictions = model(batch)
        
        # Extract move probabilities
        move_probs = torch.softmax(predictions["move"], dim=-1).cpu().numpy()[0]
    
    # Convert legal moves to UCI format
    legal_moves_uci = [move.uci() for move in legal_moves]
    
    # Map legal moves to their indices in the output
    legal_indices = []
    legal_probs = []
    
    for i, move_uci in enumerate(legal_moves_uci):
        if move_uci in UCI_MOVES:
            idx = UCI_MOVES[move_uci]
            legal_indices.append(i)
            legal_probs.append(move_probs[idx])
    
    # If no legal moves are in the UCI_MOVES mapping, select randomly
    if not legal_indices:
        return random.choice(legal_moves)
    
    # Apply temperature
    if temperature > 0:
        legal_probs = np.array(legal_probs) ** (1/temperature)
        legal_probs = legal_probs / np.sum(legal_probs)
    else:
        # Greedy selection
        max_idx = np.argmax(legal_probs)
        return legal_moves[legal_indices[max_idx]]
    
    # Get top-k moves
    if len(legal_indices) > top_k:
        top_indices = np.argsort(legal_probs)[-top_k:]
        top_probs = legal_probs[top_indices]
        top_probs = top_probs / np.sum(top_probs)
        chosen_idx = np.random.choice(top_indices, p=top_probs)
    else:
        chosen_idx = np.random.choice(len(legal_indices), p=legal_probs)
    
    return legal_moves[legal_indices[chosen_idx]]

def save_game(game, save_path):
    """
    Save a chess game to a PGN file.
    
    Args:
        game (chess.pgn.Game): The game to save
        save_path (str): Path to save the PGN file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Check if file is a .pgn or .pkl
    if save_path.endswith('.pgn'):
        # Save as PGN
        with open(save_path, "w") as f:
            print(game, file=f)
    elif save_path.endswith('.pkl'):
        # Save as pickle for visualization
        pgn_string = str(game)
        with open(save_path, "wb") as f:
            pickle.dump(pgn_string, f)
    else:
        # Default to PGN
        with open(save_path, "w") as f:
            print(game, file=f)
            
    print(f"Game saved to {save_path}")