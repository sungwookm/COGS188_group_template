import chess
import numpy as np
import torch
import math
from utils.utils import create_batch_from_board
from utils.paste import UCI_MOVES


class MCTSNode:
    """
    Node in the Monte Carlo Search Tree
    """
    def __init__(self, board, parent=None, move=None, prior=0.0):
        self.board = board.copy()  # Chess board at this node
        self.parent = parent  # Parent node
        self.move = move  # Move that led to this node
        self.children = {}  # Dict mapping moves to child nodes
        self.visits = 0  # Number of visits to this node
        self.value_sum = 0.0  # Sum of values from this node
        self.prior = prior  # Prior probability from policy network
        self.expanded = False  # Whether this node has been expanded
    
    def value(self):
        """
        Get the average value (Q) of this node
        """
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits
    
    def ucb_score(self, parent_visits, exploration=1.0):
        """
        Calculate the UCB score for this node
        UCB = Q + U where U = exploration * prior * sqrt(parent_visits) / (1 + visits)
        """
        # Exploration term (U)
        u = exploration * self.prior * math.sqrt(parent_visits) / (1 + self.visits)
        # Exploitation term (Q) + exploration term (U)
        return self.value() + u
    
    def select_child(self, exploration=1.0):
        """
        Select the child with the highest UCB score
        Modified to avoid selecting moves that lead to draws
        """
        best_score = float('-inf')
        best_move = None
        
        for move, child in self.children.items():
            # Skip this move if it leads to a draw
            if child.board.is_game_over():
                result = child.board.result()
                if result == "1/2-1/2":
                    continue
            
            score = child.ucb_score(self.visits, exploration)
            if score > best_score:
                best_score = score
                best_move = move
        
        # If all moves lead to draws or we couldn't find a valid move,
        # fall back to the original selection method
        if best_move is None:
            for move, child in self.children.items():
                score = child.ucb_score(self.visits, exploration)
                if score > best_score:
                    best_score = score
                    best_move = move
        
        return best_move
    
    def expand(self, model, device='cpu'):
        """
        Expand this node by creating child nodes for all legal moves
        """
        if self.expanded:
            return
        
        # Get move probabilities from model
        batch = create_batch_from_board(self.board, device)
        with torch.no_grad():
            outputs = model(batch)
        
        move_logits = outputs["move"].squeeze(0)
        winrate_pred = outputs["winrate"].item()
        
        # Get all legal moves
        legal_moves = list(self.board.legal_moves)
        
        # If there are no legal moves, this is a terminal state
        if not legal_moves:
            self.expanded = True
            return
        
        # Filter for moves that are in UCI_MOVES
        valid_move_indices = []
        valid_moves = []
        
        for move in legal_moves:
            uci = move.uci()
            if uci in UCI_MOVES:
                valid_move_indices.append(UCI_MOVES[uci])
                valid_moves.append(move)
        
        if not valid_moves:
            # If no moves are in our dictionary, use all legal moves with uniform probabilities
            valid_moves = legal_moves
            priors = np.ones(len(valid_moves)) / len(valid_moves)
        else:
            # Extract logits for valid moves
            valid_logits = move_logits[valid_move_indices].cpu().numpy()
            # Convert logits to probabilities
            priors = np.exp(valid_logits) / np.sum(np.exp(valid_logits))
        
        # Create child nodes for each legal move
        for i, move in enumerate(valid_moves):
            child_board = self.board.copy()
            child_board.push(move)
            self.children[move] = MCTSNode(child_board, parent=self, move=move, prior=priors[i])
        
        self.expanded = True
    
    def backup(self, value):
        """
        Update this node with the result of a simulation
        """
        node = self
        turn = node.board.turn  # True for white, False for black
        
        while node is not None:
            node.visits += 1
            # Value is from the perspective of the current player
            # We need to negate it when moving up the tree to parent's perspective
            if turn == node.board.turn:
                node.value_sum += value
            else:
                node.value_sum += (1 - value)
            
            node = node.parent

class MCTS:
    """
    Monte Carlo Tree Search for chess move selection
    """
    def __init__(self, model, device='cpu', num_simulations=100, exploration=1.0, dirichlet_noise=False, temperature=1.0):
        self.model = model
        self.device = device
        self.num_simulations = num_simulations
        self.exploration = exploration
        self.dirichlet_noise = dirichlet_noise
        self.temperature = temperature
    
    def search(self, board):
        """
        Perform MCTS search from the given board state
        """
        # Create root node
        root = MCTSNode(board)
        
        # Add Dirichlet noise to the root to encourage exploration
        if self.dirichlet_noise:
            root.expand(self.model, self.device)
            self._add_dirichlet_noise(root)
        
        # Perform simulations
        for _ in range(self.num_simulations):
            node = root
            
            # Selection: descend tree to a leaf or unexpanded node
            while node.expanded and node.children:
                move = node.select_child(self.exploration)
                node = node.children[move]
            
            # Expansion: expand the node if it's not a terminal state
            if not node.expanded and not node.board.is_game_over():
                node.expand(self.model, self.device)
            
            # Evaluation: get a value estimate from the model or terminal state
            value = self._evaluate(node)
            
            # Backup: update values up the tree
            node.backup(value)
        
        # Return the best move based on visit counts
        return self._select_move(root)
    
    def _evaluate(self, node):
        """
        Evaluate the position in the given node
        """
        # If it's a terminal state, determine the value
        if node.board.is_game_over():
            result = node.board.result()
            if result == '1-0':  # White wins
                return 1.0 if node.board.turn == chess.WHITE else 0.0
            elif result == '0-1':  # Black wins
                return 0.0 if node.board.turn == chess.WHITE else 1.0
            else:  # Draw
                return 0.5
        
        # Otherwise, use the model to estimate the value
        batch = create_batch_from_board(node.board, self.device)
        with torch.no_grad():
            outputs = self.model(batch)
        
        winrate = outputs["winrate"].item()
        
        # Ensure winrate is from the perspective of the current player
        return winrate if node.board.turn == chess.WHITE else 1.0 - winrate
    
    def _add_dirichlet_noise(self, node):
        """
        Add Dirichlet noise to the prior probabilities of the root node's children
        """
        noise = np.random.dirichlet([0.3] * len(node.children))
        
        for i, (move, child) in enumerate(node.children.items()):
            child.prior = child.prior * 0.75 + noise[i] * 0.25
    
    def _select_move(self, root):
        """
        Select a move based on visit counts, with optional temperature
        Modified to avoid selecting moves that lead to draws
        """
        # Filter out moves that lead to immediate draws
        valid_moves = []
        valid_visits = []
        
        for move, child in root.children.items():
            if child.board.is_game_over():
                result = child.board.result()
                if result == "1/2-1/2":
                    continue
            valid_moves.append(move)
            valid_visits.append(child.visits)
        
        # If no valid moves (all lead to draws), use all moves
        if not valid_moves:
            valid_moves = list(root.children.keys())
            valid_visits = [child.visits for child in root.children.values()]
        
        visits = np.array(valid_visits)
        moves = valid_moves
        
        if self.temperature == 0 or len(moves) == 1:
            # Deterministic selection of the best move
            return moves[np.argmax(visits)]
        else:
            # Sample a move based on the visit distribution with temperature
            visits = visits ** (1 / max(self.temperature, 1e-8))
            visits = visits / np.sum(visits)
            return np.random.choice(moves, p=visits)

def get_move_with_mcts(board, model, device='cpu', num_simulations=100, exploration=1.0, temperature=1.0, dirichlet_noise=False):
    """
    Get the best move for a given board state using MCTS
    
    Args:
        board (chess.Board): The chess board
        model: The transformer model
        device (str): Device to run the model on
        num_simulations (int): Number of MCTS simulations to run
        exploration (float): Exploration parameter for UCB
        temperature (float): Temperature for move selection
        dirichlet_noise (bool): Whether to add Dirichlet noise at the root
        
    Returns:
        chess.Move: The selected move
    """
    model.eval()  # Set model to evaluation mode
    
    # Create and run MCTS
    mcts = MCTS(
        model=model,
        device=device,
        num_simulations=num_simulations,
        exploration=exploration,
        temperature=temperature,
        dirichlet_noise=dirichlet_noise
    )
    
    # Return the best move
    return mcts.search(board)

def get_best_move_enhanced(board, model, device='cpu', use_mcts=True, mcts_simulations=100, 
                          exploration=1.0, temperature=1.0, dirichlet_noise=False, top_k=5):
    """
    Enhanced version of get_best_move that can use either MCTS or the original method
    
    Args:
        board (chess.Board): The chess board to process
        model: The transformer model
        device (str): Device to run the model on
        use_mcts (bool): Whether to use MCTS for move selection
        mcts_simulations (int): Number of MCTS simulations to run
        exploration (float): Exploration parameter for UCB in MCTS
        temperature (float): Temperature for move selection
        dirichlet_noise (bool): Whether to add Dirichlet noise at the root in MCTS
        top_k (int): Number of top moves to consider (used only in non-MCTS mode)
        
    Returns:
        chess.Move: The selected move
    """
    if use_mcts:
        return get_move_with_mcts(
            board=board,
            model=model,
            device=device,
            num_simulations=mcts_simulations,
            exploration=exploration,
            temperature=temperature,
            dirichlet_noise=dirichlet_noise
        )
    else:
        # Use the original method
        from utils.utils import get_best_move
        return get_best_move(board, model, device, temperature, top_k)

def generate_mcts_training_games(model, num_games=10, games_dir="games/", max_moves=100,
                                device='cpu', temperature=1.0, top_k=5, 
                                use_mcts=True, mcts_simulations=100, exploration=1.0, dirichlet_noise=False,
                                stockfish_percentage=0, stockfish_path=None, 
                                stockfish_elo=1500, stockfish_depth=5, progress_bar=True):
    """
    Generate training games using MCTS with a specified percentage against Stockfish.
    
    Args:
        model: The model to use for prediction.
        num_games (int): Total number of games to generate.
        games_dir (str): Directory to save the games in.
        max_moves (int): Maximum number of moves before declaring a draw.
        device (str): Device to run the model on.
        temperature (float): Temperature for move selection (higher = more random).
        top_k (int): Number of top moves to consider (used when not using MCTS).
        use_mcts (bool): Whether to use MCTS for move selection.
        mcts_simulations (int): Number of MCTS simulations per move.
        exploration (float): Exploration parameter for MCTS.
        dirichlet_noise (bool): Whether to add Dirichlet noise at the root for MCTS.
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
    import time
    import datetime
    import chess
    import chess.pgn
    import chess.engine
    from tqdm import tqdm
    from utils.utils import find_stockfish, save_game
    
    # Ensure games directory exists
    os.makedirs(games_dir, exist_ok=True)
    
    # Set model to evaluation mode
    model.eval()
    
    # Calculate number of games for each type
    num_stockfish_games = int(num_games * (stockfish_percentage / 100))
    num_selfplay_games = num_games - num_stockfish_games
    
    mcts_text = "with MCTS" if use_mcts else ""
    print(f"Generating {num_selfplay_games} self-play games {mcts_text} and {num_stockfish_games} games against Stockfish")
    
    game_files = []
    
    # Generate self-play games with MCTS
    if num_selfplay_games > 0:
        # Set up progress tracking
        iter_range = range(num_selfplay_games)
        if progress_bar:
            try:
                iter_range = tqdm(iter_range, desc=f"Generating self-play games {mcts_text}")
            except ImportError:
                print(f"Generating {num_selfplay_games} self-play games {mcts_text}...")
        else:
            print(f"Generating {num_selfplay_games} self-play games {mcts_text}...")
        
        for i in iter_range:
            # Initialize a new board
            board = chess.Board()
            
            # Create a game object to store the moves
            game = chess.pgn.Game()
            
            # Set up game metadata
            game.headers["Event"] = f"Self-Play {mcts_text}"
            game.headers["Site"] = "Training"
            game.headers["Date"] = datetime.datetime.now().strftime("%Y.%m.%d")
            game.headers["Round"] = str(i+1)
            game.headers["White"] = f"ChessTransformer{' (MCTS)' if use_mcts else ''}"
            game.headers["Black"] = f"ChessTransformer{' (MCTS)' if use_mcts else ''}"
            
            # Initialize node for adding moves
            node = game
            
            move_count = 0
            
            # Main game loop
            while not board.is_game_over() and move_count < max_moves:
                # Use MCTS or standard move selection
                move = get_best_move_enhanced(
                    board=board,
                    model=model,
                    device=device,
                    use_mcts=use_mcts,
                    mcts_simulations=mcts_simulations,
                    exploration=exploration,
                    temperature=temperature,
                    dirichlet_noise=dirichlet_noise,
                    top_k=top_k
                )
                
                # Add the move to the game
                node = node.add_variation(move)
                
                # show the move being made
                #print("")
                
                # Make the move on the board
                board.push(move)
                
                move_count += 1
            
            # Add the result to the game headers
            if board.is_checkmate():
                if board.turn == chess.WHITE:
                    game.headers["Result"] = "0-1"
                else:
                    game.headers["Result"] = "1-0"
            elif board.is_stalemate() or board.is_insufficient_material() or board.is_fifty_moves() or board.is_repetition(3):
                game.headers["Result"] = "1/2-1/2"
            elif move_count >= max_moves:
                game.headers["Result"] = "1/2-1/2"
            
            # Save the game
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(games_dir, f"selfplay_{mcts_text.replace(' ', '_')}_{i+1}_{timestamp}.pgn")
            save_game(game, save_path)
            
            game_files.append(save_path)
            
            # Small delay to ensure unique timestamps
            time.sleep(0.1)
    
    # Generate games against Stockfish
    if num_stockfish_games > 0:
        if not stockfish_path:
            stockfish_path = find_stockfish()
            if not stockfish_path:
                print("Warning: Stockfish not found. Falling back to self-play games instead.")
                # Generate additional self-play games instead
                additional_selfplay_files = generate_mcts_training_games(
                    model=model,
                    num_games=num_stockfish_games,
                    games_dir=games_dir,
                    max_moves=max_moves,
                    device=device,
                    temperature=temperature,
                    top_k=top_k,
                    use_mcts=use_mcts,
                    mcts_simulations=mcts_simulations,
                    exploration=exploration,
                    dirichlet_noise=dirichlet_noise,
                    stockfish_percentage=0,  # No more Stockfish attempts
                    progress_bar=progress_bar
                )
                game_files.extend(additional_selfplay_files)
            else:
                # Set up progress tracking for Stockfish games
                iter_range = range(num_stockfish_games)
                if progress_bar:
                    try:
                        iter_range = tqdm(iter_range, desc=f"Generating games against Stockfish {mcts_text}")
                    except ImportError:
                        print(f"Generating {num_stockfish_games} games against Stockfish {mcts_text}...")
                else:
                    print(f"Generating {num_stockfish_games} games against Stockfish {mcts_text}...")
                
                for i in iter_range:
                    # Initialize a new board
                    board = chess.Board()
                    
                    # Create a game object to store the moves
                    game = chess.pgn.Game()
                    
                    # Randomly decide model color for variety
                    model_plays_white = random.choice([True, False])
                    
                    # Set up game metadata
                    game.headers["Event"] = f"Training vs Stockfish {mcts_text}"
                    game.headers["Site"] = "Training"
                    game.headers["Date"] = datetime.datetime.now().strftime("%Y.%m.%d")
                    game.headers["Round"] = str(i+1)
                    game.headers["White"] = f"ChessTransformer{' (MCTS)' if use_mcts else ''}" if model_plays_white else f"Stockfish (ELO {stockfish_elo})"
                    game.headers["Black"] = f"Stockfish (ELO {stockfish_elo})" if model_plays_white else f"ChessTransformer{' (MCTS)' if use_mcts else ''}"
                    
                    # Initialize Stockfish
                    try:
                        engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
                        if stockfish_elo is not None:
                            engine.configure({"UCI_LimitStrength": True, "UCI_Elo": stockfish_elo})
                    except Exception as e:
                        print(f"Error starting Stockfish: {e}")
                        continue
                    
                    # Initialize node for adding moves
                    node = game
                    
                    move_count = 0
                    
                    try:
                        # Main game loop
                        while not board.is_game_over() and move_count < max_moves:
                            current_turn_is_white = board.turn == chess.WHITE
                            model_turn = (current_turn_is_white and model_plays_white) or (not current_turn_is_white and not model_plays_white)
                            
                            if model_turn:
                                # Model's turn - use enhanced move selection with MCTS
                                move = get_best_move_enhanced(
                                    board=board,
                                    model=model,
                                    device=device,
                                    use_mcts=use_mcts,
                                    mcts_simulations=mcts_simulations,
                                    exploration=exploration,
                                    temperature=temperature,
                                    dirichlet_noise=dirichlet_noise,
                                    top_k=top_k
                                )
                            else:
                                # Stockfish's turn
                                result = engine.play(board, chess.engine.Limit(depth=stockfish_depth))
                                move = result.move
                            
                            # Add the move to the game
                            node = node.add_variation(move)
                            
                            # Make the move on the board
                            board.push(move)
                            
                            move_count += 1
                    
                    finally:
                        # Close the engine
                        engine.quit()
                    
                    # Add the result to the game headers
                    if board.is_checkmate():
                        if board.turn == chess.WHITE:
                            game.headers["Result"] = "0-1"
                        else:
                            game.headers["Result"] = "1-0"
                    elif board.is_stalemate() or board.is_insufficient_material() or board.is_fifty_moves() or board.is_repetition(3):
                        game.headers["Result"] = "1/2-1/2"
                    elif move_count >= max_moves:
                        game.headers["Result"] = "1/2-1/2"
                    
                    # Save the game
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    model_color = "white" if model_plays_white else "black"
                    save_path = os.path.join(
                        games_dir, 
                        f"stockfish_{model_color}_vs_stockfish_{stockfish_elo}_{mcts_text.replace(' ', '_')}_{timestamp}.pgn"
                    )
                    save_game(game, save_path)
                    
                    game_files.append(save_path)
                    
                    # Small delay to ensure unique timestamps
                    time.sleep(0.1)
    
    print(f"Generated {len(game_files)} total training games in {games_dir}")
    return game_files

def play_game_with_mcts(model, opponent="self", model_color="white", stockfish_path=None, 
                        stockfish_elo=1500, stockfish_depth=5, max_moves=100, device='cpu',
                        use_mcts=True, mcts_simulations=100, exploration=1.0, 
                        temperature=1.0, dirichlet_noise=False, top_k=5):
    """
    Play a game with the model against itself or Stockfish, using MCTS for move selection.
    
    Args:
        model: The model to use for prediction.
        opponent (str): "self" or "stockfish" to determine the opponent.
        model_color (str): "white", "black", or "random" to determine which side the model plays.
        stockfish_path (str): Path to Stockfish executable (only needed for stockfish opponent).
        stockfish_elo (int): ELO rating for Stockfish (only used with Stockfish opponent).
        stockfish_depth (int): Search depth for Stockfish (only used with Stockfish opponent).
        max_moves (int): Maximum number of moves before declaring a draw.
        device (str): The device to put the tensors on.
        use_mcts (bool): Whether to use MCTS for move selection.
        mcts_simulations (int): Number of MCTS simulations to run.
        exploration (float): Exploration parameter for UCB in MCTS.
        temperature (float): Temperature for move selection.
        dirichlet_noise (bool): Whether to add Dirichlet noise at the root in MCTS.
        top_k (int): Number of top moves to consider (used only in non-MCTS mode).
        
    Returns:
        chess.pgn.Game: The completed game.
    """
    import random
    import chess.pgn
    import chess.engine
    import datetime
    from utils.utils import find_stockfish
    
    # Initialize a new board
    board = chess.Board()
    
    # Create a game object to store the moves
    game = chess.pgn.Game()
    
    # Set up game metadata
    game.headers["Event"] = f"Chess Transformer {'with MCTS' if use_mcts else ''} vs {opponent.capitalize()}"
    game.headers["Site"] = "Local Machine"
    game.headers["Date"] = datetime.datetime.now().strftime("%Y.%m.%d")
    game.headers["Round"] = "1"
    
    # Randomly determine which side the model plays if specified
    if model_color == "random":
        model_plays_white = random.choice([True, False])
    else:
        model_plays_white = model_color.lower() == "white"
    
    if opponent == "self":
        game.headers["White"] = f"ChessTransformer{'(MCTS)' if use_mcts else ''}"
        game.headers["Black"] = f"ChessTransformer{'(MCTS)' if use_mcts else ''}"
    else:
        # Set up player names based on model's color
        game.headers["White"] = f"ChessTransformer{'(MCTS)' if use_mcts else ''}" if model_plays_white else f"Stockfish (ELO {stockfish_elo})"
        game.headers["Black"] = f"Stockfish (ELO {stockfish_elo})" if model_plays_white else f"ChessTransformer{'(MCTS)' if use_mcts else ''}"
    
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
            
            print(f"Playing against Stockfish (ELO: {stockfish_elo}, Depth: {stockfish_depth})")
        except Exception as e:
            raise Exception(f"Failed to start Stockfish engine: {e}")
    
    # Initialize node for adding moves
    node = game
    
    move_count = 0
    print(f"Starting game: ChessTransformer {'with MCTS' if use_mcts else ''} {'(White)' if model_plays_white else '(Black)'} vs "
          f"{'ChessTransformer with MCTS' if opponent == 'self' else 'Stockfish'} "
          f"{'(Black)' if model_plays_white else '(White)'}")
    
    try:
        # Main game loop
        while not board.is_game_over() and move_count < max_moves:
            current_turn_is_white = board.turn == chess.WHITE
            model_turn = (current_turn_is_white and model_plays_white) or (not current_turn_is_white and not model_plays_white)
            
            if opponent == "self" or model_turn:
                # Model's turn - use enhanced move selection with MCTS
                move = get_best_move_enhanced(
                    board=board,
                    model=model,
                    device=device,
                    use_mcts=use_mcts,
                    mcts_simulations=mcts_simulations,
                    exploration=exploration,
                    temperature=temperature,
                    dirichlet_noise=dirichlet_noise,
                    top_k=top_k
                )
                player = f"Model {'with MCTS' if use_mcts else ''}"
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
            print(f"Move {move_count + 1}: {move.uci()} (by {side} - {player})")
            
            move_count += 1
    
    finally:
        # Close the engine if it was initialized
        if engine:
            engine.quit()
    
    # Add the result to the game headers
    if board.is_checkmate():
        if board.turn == chess.WHITE:
            game.headers["Result"] = "0-1"
            print("Black wins by checkmate")
        else:
            game.headers["Result"] = "1-0"
            print("White wins by checkmate")
    elif board.is_stalemate() or board.is_insufficient_material() or board.is_fifty_moves() or board.is_repetition(3):
        game.headers["Result"] = "1/2-1/2"
        print("Game drawn")
    elif move_count >= max_moves:
        game.headers["Result"] = "1/2-1/2"
        print(f"Game drawn after reaching maximum moves ({max_moves})")
    
    return game

def train_model_with_mcts_regeneration(model, num_epochs=10, games_dir="games/", training_data_dir="training_data/",
                                       
                                      num_games=10, batch_size=32, max_moves=100, device="cpu",
                                      temperature=1.0, top_k=5, learning_rate=1e-4, checkpoint_dir="checkpoints/",
                                      save_frequency=10, stockfish_percentage=0, stockfish_path=None, 
                                      stockfish_elo=1500, stockfish_depth=5,
                                      use_mcts=True, mcts_simulations=100, mcts_exploration=1.0, 
                                      mcts_dirichlet_noise=False):
    """
    Train the model with regeneration of games at each epoch, using MCTS for self-play.
    
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
        top_k (int): Number of top moves to consider (used when not using MCTS)
        learning_rate (float): Learning rate for optimizer
        checkpoint_dir (str): Directory to save model checkpoints
        save_frequency (int): How often to save model checkpoints (in epochs)
        stockfish_percentage (float): Percentage (0-100) of games to play against Stockfish
        stockfish_path (str): Path to Stockfish executable
        stockfish_elo (int): ELO rating for Stockfish
        stockfish_depth (int): Search depth for Stockfish
        use_mcts (bool): Whether to use MCTS for self-play game generation
        mcts_simulations (int): Number of MCTS simulations per move
        mcts_exploration (float): Exploration parameter for MCTS
        mcts_dirichlet_noise (bool): Whether to add Dirichlet noise at MCTS root nodes
        
    Returns:
        model: The trained model
        dict: Training history
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import os
    import time
    import random
    from datetime import datetime
    from tqdm import tqdm
    from utils.utils import (
        clear_games_directory, 
        clear_training_data_directory, 
        save_training_batches,
        process_game_file,
        create_batch_from_examples
    )
    
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
    if use_mcts:
        print(f"Using MCTS for self-play (simulations={mcts_simulations}, exploration={mcts_exploration})")
    if stockfish_percentage > 0:
        print(f"Including {stockfish_percentage}% games against Stockfish (ELO: {stockfish_elo})")
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
        
        # Clear games directory and generate new games for this epoch
        clear_games_directory(games_dir)
        clear_training_data_directory(training_data_dir)
        
        # Generate new games with MCTS and potential Stockfish games
        print(f"Generating {num_games} new training games for epoch {epoch+1}")
        predata = True
        if predata:
            #TODO use the data folder to create the game files for training
            
            pass
        else:
            game_files = generate_mcts_training_games(
                model=model,
                num_games=num_games,
                games_dir=games_dir,
                max_moves=max_moves,
                device=device,
                temperature=temperature,
                top_k=top_k,
                use_mcts=use_mcts,
                mcts_simulations=mcts_simulations,
                exploration=mcts_exploration,
                dirichlet_noise=mcts_dirichlet_noise,
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

def train_model_with_predata(model, num_epochs=10, training_data_dir="training_data/",
                           batch_size=32, device="cpu", learning_rate=1e-4, 
                           checkpoint_dir="checkpoints/", save_frequency=10,
                           predata_path="data/GM_games_dataset.csv"):
    """
    Train the model using only pre-existing GM games data, loading the data once and reusing for all epochs.
    
    Args:
        model: The model to train
        num_epochs (int): Number of epochs to train for
        training_data_dir (str): Directory to save processed training data
        batch_size (int): Size of training batches
        device (str): Device to run on
        learning_rate (float): Learning rate for optimizer
        checkpoint_dir (str): Directory to save model checkpoints
        save_frequency (int): How often to save model checkpoints (in epochs)
        predata_path (str): Path to the CSV file containing GM games data
        sample_size (int, optional): Number of games to sample from the dataset. If None, use all games.
        
    Returns:
        model: The trained model
        dict: Training history
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import os
    import time
    import random
    import pandas as pd
    import io
    import chess.pgn
    from datetime import datetime
    from tqdm import tqdm
    from utils.utils import (
        process_game_file,
        create_batch_from_examples,
        save_game
    )
    
    # Ensure directories exist
    os.makedirs(training_data_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    temp_games_dir = os.path.join(training_data_dir, "temp_games")
    os.makedirs(temp_games_dir, exist_ok=True)
    
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
    
    print(f"Starting training for {num_epochs} epochs using pre-existing GM games")
    print(f"Loading data from {predata_path}")
    
    # Process the data once and reuse for all epochs
    try:
        # Load GM games from CSV file
        df = pd.read_csv(predata_path)
        
        # Check if 'pgn' column exists
        if 'pgn' not in df.columns:
            raise ValueError(f"CSV file at {predata_path} does not contain a 'pgn' column")
        
        print(f"Found {len(df)} games in the dataset")
        
        # Always use the full dataset
        print(f"Using all {len(df)} games from the dataset")
        
        # Save each PGN as a temporary file
        game_files = []
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Saving GM games to temporary files"):
            pgn_text = row['pgn']
            
            # Parse PGN
            pgn_io = io.StringIO(pgn_text)
            game = chess.pgn.read_game(pgn_io)
            
            if game is None:
                print(f"Warning: Could not parse game at index {idx}")
                continue
            
            # Save game to file
            game_file_path = os.path.join(temp_games_dir, f"gm_game_{idx}.pgn")
            save_game(game, game_file_path)
            
            game_files.append(game_file_path)
        
        print(f"Saved {len(game_files)} GM games to temporary files")
        
        # Process games into training batches
        print("Processing games into training batches")
        
        # Extract training examples from games
        all_examples = []
        for file_path in tqdm(game_files, desc="Processing game files"):
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
        
        # Clear temporary files
        for file_path in game_files:
            if os.path.exists(file_path):
                os.remove(file_path)
        
        if not batches:
            raise ValueError("No valid training batches could be created from the data")
        
        # Training loop - reusing the same batches for all epochs
        for epoch in range(num_epochs):
            start_time = time.time()
            
            print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
            
            # Set model to training mode
            model.train()
            
            # Track losses for this epoch
            epoch_total_loss = 0.0
            epoch_move_loss = 0.0
            epoch_result_loss = 0.0
            
            # Process each batch
            print(f"Training on {len(batches)} batches")
            
            # Shuffle the order of batches each epoch for better training
            random.shuffle(batches)
            
            for batch_idx, batch in enumerate(tqdm(batches, desc=f"Epoch {epoch+1}")):
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
                
                # Print progress periodically
                if (batch_idx + 1) % (max(len(batches) // 10, 1)) == 0:
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
    
    except Exception as e:
        print(f"Error during training: {e}")
        raise
    
    finally:
        # Clean up any remaining temporary files
        if os.path.exists(temp_games_dir):
            for filename in os.listdir(temp_games_dir):
                os.remove(os.path.join(temp_games_dir, filename))
            try:
                os.rmdir(temp_games_dir)
            except:
                pass
    
    print("Training completed successfully!")
    
    return model, history