import chess
import numpy as np
import torch
import math
import random
from utils.paste import UCI_MOVES
import os
from tqdm import tqdm
import datetime
import pickle
import io
import chess.engine
import chess.pgn

class MCTSNode:
    def __init__(self, board, parent=None, move=None, prior=0.0, position_history=None):
        """
        Initialize a node in the MCTS tree.
        
        Args:
            board (chess.Board): The chess board state at this node
            parent (MCTSNode): Parent node
            move (chess.Move): Move that led to this node
            prior (float): Prior probability from policy network
            position_history (dict): Dictionary tracking position occurrences
        """
        self.board = board.copy()  # Create a deep copy of the board
        self.parent = parent
        self.move = move
        self.children = []
        self.visits = 0
        self.value_sum = 0.0
        self.prior = prior
        self.expanded = False
        
        # Track position history to detect repetitions
        if position_history is None:
            self.position_history = {}
        else:
            self.position_history = position_history.copy()
        
        # Add current position to history
        current_position = self.board.fen().split(' ')[0]  # Just the piece positions
        if current_position in self.position_history:
            self.position_history[current_position] += 1
        else:
            self.position_history[current_position] = 1
    
    def is_leaf(self):
        """Check if the node is a leaf (not expanded yet)."""
        return not self.expanded
    
    def is_root(self):
        """Check if the node is the root of the tree."""
        return self.parent is None
    
    def is_terminal(self):
        """Check if the node represents a terminal state."""
        return self.board.is_game_over()
    
    def get_ucb_score(self, c_puct=1.0, repetition_penalty=0.5):
        """
        Calculate the UCB score for this node with repetition penalty.
        
        Args:
            c_puct (float): Exploration constant
            repetition_penalty (float): Penalty factor for repeated positions
            
        Returns:
            float: UCB score with repetition penalty
        """
        if self.visits == 0:
            return float('inf')
        
        # Exploitation term
        exploitation = self.value_sum / self.visits
        
        # Exploration term
        if self.parent:
            n_parent = self.parent.visits
            exploration = c_puct * self.prior * math.sqrt(n_parent) / (1 + self.visits)
        else:
            exploration = 0.0
        
        # Calculate repetition penalty
        current_position = self.board.fen().split(' ')[0]
        position_count = self.position_history.get(current_position, 0)
        
        # Apply penalty if position has been seen before
        repetition_factor = 1.0
        if position_count > 1:
            # The more times we've seen this position, the larger the penalty
            repetition_factor = 1.0 / (1.0 + repetition_penalty * (position_count - 1))
        
        return (exploitation + exploration) * repetition_factor
    
    def expand(self, model, device):
        """
        Expand the node by adding all possible child nodes.
        Uses the model to get move probabilities.
        
        Args:
            model: Neural network model for move prediction
            device: Device to run the model on (CPU/GPU)
            
        Returns:
            None
        """
        if self.is_terminal():
            return
        
        # Get all legal moves
        legal_moves = list(self.board.legal_moves)

        # find legal moves that do not result in a draw or any repetition
        for move in legal_moves.copy():  # Use copy() to safely modify during iteration
            copy_board = self.board.copy()
            copy_board.push(move)
            if copy_board.is_stalemate() or copy_board.is_insufficient_material() or copy_board.is_seventyfive_moves() or copy_board.is_fivefold_repetition():
                legal_moves.remove(move)
        
        if not legal_moves:
            return
        
        # Create inputs for the model
        model_input = create_batch_from_board(self.board, device)
        
        # Get model predictions
        with torch.no_grad():
            predictions = model(model_input)
            
        # Extract move probabilities
        move_probs = torch.softmax(predictions["move"], dim=-1).cpu().numpy()[0]
        
        # Create mapping for legal moves to their UCI indices
        legal_moves_uci = [str(move) for move in legal_moves]
        legal_moves_idx = [UCI_MOVES.get(move, -1) for move in legal_moves_uci]
        
        # Filter out any moves that weren't found in UCI_MOVES
        valid_moves = [(move, idx) for move, idx in zip(legal_moves, legal_moves_idx) if idx != -1]
        
        # Create mask for legal moves
        mask = np.zeros(len(move_probs))
        for _, idx in valid_moves:
            mask[idx] = 1
            
        # Apply mask and renormalize
        masked_probs = move_probs * mask
        sum_masked_probs = np.sum(masked_probs)
        
        if sum_masked_probs > 0:
            masked_probs = masked_probs / sum_masked_probs
        else:
            # If all legal moves have zero probability, use uniform distribution
            masked_probs = mask / np.sum(mask)
            
        # Create children nodes
        for move, idx in valid_moves:
            new_board = self.board.copy()
            new_board.push(move)
            prior = masked_probs[idx]
            self.children.append(MCTSNode(new_board, self, move, prior, self.position_history))
            
        self.expanded = True
    
    def select_child(self, c_puct=1.0, repetition_penalty=0.5):
        """
        Select the child node with the highest UCB score.
        
        Args:
            c_puct (float): Exploration constant
            repetition_penalty (float): Penalty factor for repeated positions
            
        Returns:
            MCTSNode: Selected child node
        """
        best_score = float('-inf')
        best_child = None
        
        for child in self.children:
            score = child.get_ucb_score(c_puct, repetition_penalty)
            if score > best_score:
                best_score = score
                best_child = child
                
        return best_child
    
    def update(self, value):
        """
        Update node statistics with the result of a simulation.
        
        Args:
            value (float): Result of the simulation from perspective of node's turn
        """
        self.visits += 1
        self.value_sum += value
        
    def get_value(self):
        """Get the average value of this node."""
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits
    
    def get_visit_count_distribution(self, temperature=1.0):
        """
        Get distribution over children based on visit counts.
        
        Args:
            temperature (float): Temperature parameter for controlling exploration
            
        Returns:
            list: Probability distribution over child nodes
        """
        visits = np.array([child.visits for child in self.children])
        
        if temperature == 0:
            # Deterministic selection of the best move
            best_idx = np.argmax(visits)
            probs = np.zeros_like(visits, dtype=np.float32)
            probs[best_idx] = 1.0
            return probs
        
        # Apply temperature
        visits_temp = visits ** (1.0 / temperature)
        
        # Normalize to get probabilities
        sum_visits = np.sum(visits_temp)
        if sum_visits > 0:
            return visits_temp / sum_visits
        
        # If all visits are 0 (shouldn't happen), use uniform distribution
        return np.ones_like(visits_temp) / len(visits_temp)


class MCTS:
    def __init__(self, model, device, simulations=800, c_puct=1.0, 
                 dirichlet_noise=True, dirichlet_alpha=0.3, noise_fraction=0.25,
                 repetition_penalty=0.5):
        """
        Initialize the MCTS algorithm.
        
        Args:
            model: Neural network model
            device: Device to run the model on
            simulations (int): Number of simulations to run
            c_puct (float): Exploration constant
            dirichlet_noise (bool): Whether to add Dirichlet noise at the root node
            dirichlet_alpha (float): Alpha parameter for Dirichlet distribution
            noise_fraction (float): Fraction of prior to be replaced with noise
            repetition_penalty (float): Penalty factor for repeated positions
        """
        self.model = model
        self.device = device
        self.simulations = simulations
        self.c_puct = c_puct
        self.dirichlet_noise = dirichlet_noise
        self.dirichlet_alpha = dirichlet_alpha
        self.noise_fraction = noise_fraction
        self.repetition_penalty = repetition_penalty
       
    def evaluate(self, node):
        """
        Evaluate the node using the model.
        
        Args:
            node (MCTSNode): Node to evaluate
            
        Returns:
            float: Evaluation score
        """
        if node.is_terminal():
            # Game is over
            result = node.board.result()
            
            if result == "1-0":
                # White wins
                return 1.0 if node.board.turn == chess.BLACK else -1.0
            elif result == "0-1":
                # Black wins
                return 1.0 if node.board.turn == chess.WHITE else -1.0
            else:
                # Draw
                return 0.0
        
        # Use model to evaluate position
        model_input = create_batch_from_board(node.board, self.device)
        
        with torch.no_grad():
            predictions = self.model(model_input)
            
        # Extract value prediction (from perspective of player to move)
        value = predictions["winrate"].item()
        
        # Apply repetition penalty
        current_position = node.board.fen().split(' ')[0]
        position_count = node.position_history.get(current_position, 0)
        
        # Penalize value if position has been seen before
        if position_count > 1:
            # This reduces value as repetition count increases
            penalty_factor = 1.0 / (1.0 + self.repetition_penalty * (position_count - 1))
            value = value * penalty_factor
        
        # Convert to [-1, 1] range
        return 2 * value - 1

    def search(self, board, temperature=1.0, visit_weight=0.7, value_weight=0.3):
            """
            Run MCTS search from the given board state.
            
            Args:
                board (chess.Board): Current board state
                temperature (float): Temperature for controlling exploration in move selection
                visit_weight (float): Weight given to visit counts in move selection (0-1)
                value_weight (float): Weight given to node values in move selection (0-1)
                
            Returns:
                chess.Move: Selected move
                list: Distribution over moves
            """
            # Create root node
            root = MCTSNode(board)
            
            # Expand root node
            root.expand(self.model, self.device)
            
            # Run simulations
            for _ in range(self.simulations):
                node = root
                search_path = [node]
                
                # Selection phase - traverse tree until we find a leaf node
                while not node.is_leaf() and not node.is_terminal():
                    node = node.select_child(self.c_puct, self.repetition_penalty)
                    search_path.append(node)
                
                # Expansion phase - expand the leaf node if it's not terminal
                if not node.is_terminal():
                    node.expand(self.model, self.device)
                    
                    # If the node has children after expansion, select one
                    if node.children:
                        child = node.select_child(self.c_puct, self.repetition_penalty)
                        search_path.append(child)
                        node = child
                
                # Simulation/Evaluation phase - use the model to evaluate the leaf
                value = self.evaluate(node)
                
                # Backup phase - update statistics of all nodes in the search path
                for node in reversed(search_path):
                    # Negate the value because the perspective alternates at each level
                    value = -value
                    node.update(value)
            
            # Select move based on score-adjusted distribution
            if not root.children:
                # No legal moves
                return None, None
            
            # Get distribution that balances visit counts and value estimates
            probs = root.get_visit_count_distribution(temperature)
            
            if temperature == 0:
                # Deterministic selection
                best_idx = np.argmax(probs)
                selected_node = root.children[best_idx]
            else:
                # Sample from distribution
                selected_idx = np.random.choice(len(root.children), p=probs)
                selected_node = root.children[selected_idx]
            
            # Return the selected move and the probability distribution
            return selected_node.move, [(child.move, prob) for child, prob in zip(root.children, probs)]


def create_batch_from_board(board, device):
    """
    Create model input batch from a chess board.
    
    Args:
        board (chess.Board): Chess board
        device: Device to create tensors on
        
    Returns:
        dict: Model input batch
    """
    # Board positions
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
    
    return batch

def get_best_move_mcts(board, model, device, temperature=1.0, simulations=800, 
                      repetition_penalty=0.5, visit_weight=0.7, value_weight=0.3):
    """
    Get the best move for the current board position using MCTS.
    
    Args:
        board (chess.Board): Current board state
        model: Neural network model
        device: Device to run the model on
        temperature (float): Temperature for move selection
        simulations (int): Number of MCTS simulations
        repetition_penalty (float): Penalty factor for repeated positions
        visit_weight (float): Weight given to visit counts in move selection (0-1)
        value_weight (float): Weight given to node values in move selection (0-1)
        
    Returns:
        chess.Move: Best move according to MCTS
    """
    mcts = MCTS(model, device, simulations=simulations, repetition_penalty=repetition_penalty)
    best_move, _ = mcts.search(board, temperature, visit_weight, value_weight)
    return best_move

def train_model_with_mcts(model, optimizer, num_games=100, epochs_per_game=1, batch_size=64, 
                          device="cuda", simulations=100, temperature_init=1.0, 
                          repetition_penalty=0.5, save_path="checkpoints", game_history_dir=None):
    """
    Train the model using MCTS self-play.
    
    Args:
        model: Neural network model
        optimizer: Optimizer for training
        num_games (int): Number of self-play games
        epochs_per_game (int): Number of training epochs per game
        batch_size (int): Batch size for training
        device: Device to run the model on
        simulations (int): Number of MCTS simulations per move
        temperature_init (float): Initial temperature for move selection
        repetition_penalty (float): Penalty factor for repeated positions
        save_path (str): Path to save model checkpoints
        game_history_dir (str): Path to save game history for visualization
        
    Returns:
        dict: Training history
    """
    history = {
        'policy_loss': [],
        'value_loss': [],
        'total_loss': [],
        'games': []
    }
    
    # Create directory for saving models
    os.makedirs(save_path, exist_ok=True)
    
    # Create directory for game history if not provided
    if game_history_dir is None:
        game_history_dir = os.path.join(save_path, "game_history")
    os.makedirs(game_history_dir, exist_ok=True)
    
    # Training data
    states = []
    policy_targets = []
    value_targets = []
    
    for game_idx in range(num_games):
        print(f"Playing game {game_idx + 1}/{num_games}")
        
        # Temperature annealing
        temperature = temperature_init * (0.9 ** (game_idx // 5))
        
        # Play a game
        mcts = MCTS(model, device, simulations=simulations, repetition_penalty=repetition_penalty)
        board = chess.Board()
        game_states = []
        game_policies = []
        
        move_count = 0
        position_history = {}  # Track position history
        
        while not board.is_game_over() and move_count < 500:
            # Track current position
            current_position = board.fen().split(' ')[0]
            if current_position in position_history:
                position_history[current_position] += 1
            else:
                position_history[current_position] = 1
                
            # Create a temporary node with the position history for search
            root = MCTSNode(board, position_history=position_history)
            
            # Get MCTS policy
            move, move_probs = mcts.search(board, temperature)
            
            if not move:
                # No legal moves
                break
            
            # Save state and policy
            state = create_batch_from_board(board, device)
            policy = torch.zeros(1, len(UCI_MOVES))
            
            for move_item, prob in move_probs:
                move_uci = move_item.uci()
                if move_uci in UCI_MOVES:
                    policy[0, UCI_MOVES[move_uci]] = prob
            
            game_states.append(state)
            game_policies.append(policy)
            
            # Make the move
            board.push(move)
            move_count += 1
        
        # Game result
        if board.is_checkmate():
            # Winner is opposite of the current turn
            winner = 1.0 if board.turn == chess.BLACK else -1.0
            print(f"Game {game_idx + 1} - Winner: {'Black' if winner == 1.0 else 'White'}")
        elif board.is_stalemate() or board.is_insufficient_material() or board.is_fifty_moves() or board.is_repetition():
            # Draw
            winner = 0.0
            print(f"Game {game_idx + 1} - Draw")
        else:
            # Draw due to max moves
            print(f"Game {game_idx + 1} - Max moves reached")
            winner = 0.0

        # PGN Conversion for game saving
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        pgn_game = chess.pgn.Game()
        pgn_game.headers["Event"] = "Self-Play"
        pgn_game.headers["Site"] = "Training"
        pgn_game.headers["Date"] = datetime.datetime.now().strftime("%Y.%m.%d")
        pgn_game.headers["Round"] = str(game_idx + 1)
        pgn_game.headers["White"] = "AI"
        pgn_game.headers["Black"] = "AI"
        pgn_game.headers["Result"] = board.result()
        
        # Add game moves to PGN
        node = pgn_game
        for move in board.move_stack:
            node = node.add_variation(move)
        
        pgn_string = str(pgn_game)
        
        # Save game for visualization
        pgn_pickle_path = os.path.join(game_history_dir, f"game_{game_idx+1}_{timestamp}.pkl")
        with open(pgn_pickle_path, "wb") as f:
            pickle.dump(pgn_string, f)
        
        print(f"Game saved for visualization at {pgn_pickle_path}")
        
        # Assign values to states based on the final result
        game_values = []
        
        for i in range(len(game_states)):
            # Alternate the sign for each move
            value = winner if (len(game_states) - i) % 2 == 1 else -winner
            game_values.append(torch.tensor([[value]]))
        
        # Add game data to training data
        states.extend(game_states)
        policy_targets.extend(game_policies)
        value_targets.extend(game_values)
        
        # Train the model on accumulated data
        if len(states) > batch_size:
            epoch_policy_losses = []
            epoch_value_losses = []
            epoch_total_losses = []
            
            for _ in range(epochs_per_game):
                # Shuffle data
                indices = list(range(len(states)))
                random.shuffle(indices)
                
                # Train in batches
                for i in range(0, len(indices), batch_size):
                    batch_indices = indices[i:i+batch_size]
                    
                    # Aggregate batch data
                    batch_states = {}
                    for key in states[0].keys():
                        batch_states[key] = torch.cat([states[j][key] for j in batch_indices], dim=0)
                    
                    batch_policies = torch.cat([policy_targets[j] for j in batch_indices], dim=0).to(device)
                    batch_values = torch.cat([value_targets[j] for j in batch_indices], dim=0).to(device)
                    
                    # Zero the gradients
                    optimizer.zero_grad()
                    
                    # Forward pass
                    predictions = model(batch_states)
                    
                    # Calculate loss
                    policy_logits = predictions["move"]
                    value_preds = predictions["winrate"]
                    
                    # Convert batch_values from [-1, 1] to [0, 1]
                    batch_values = (batch_values + 1) / 2
                    
                    # Policy loss (cross-entropy)
                    policy_loss = -torch.sum(batch_policies * torch.log_softmax(policy_logits, dim=1)) / batch_policies.size(0)
                    
                    # Value loss (MSE)
                    value_loss = torch.mean((value_preds - batch_values) ** 2)
                    
                    # Total loss
                    total_loss = policy_loss + value_loss
                    
                    # Backward pass and optimization
                    total_loss.backward()
                    optimizer.step()
                    
                    # Record losses
                    epoch_policy_losses.append(policy_loss.item())
                    epoch_value_losses.append(value_loss.item())
                    epoch_total_losses.append(total_loss.item())
            
            # Average losses for this epoch
            avg_policy_loss = sum(epoch_policy_losses) / len(epoch_policy_losses)
            avg_value_loss = sum(epoch_value_losses) / len(epoch_value_losses)
            avg_total_loss = sum(epoch_total_losses) / len(epoch_total_losses)
            
            history['policy_loss'].append(avg_policy_loss)
            history['value_loss'].append(avg_value_loss)
            history['total_loss'].append(avg_total_loss)
            history['games'].append(game_idx + 1)
            
            print(f"Game {game_idx + 1} - Policy Loss: {avg_policy_loss:.4f}, Value Loss: {avg_value_loss:.4f}, Total Loss: {avg_total_loss:.4f}")
        
        # Save model checkpoint every 10 games
        if (game_idx + 1) % 10 == 0:
            checkpoint_path = os.path.join(save_path, f"model_game_{game_idx + 1}.pt")
            torch.save({
                'epoch': game_idx + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': history['total_loss'][-1] if history['total_loss'] else None,
                'repetition_penalty': repetition_penalty
            }, checkpoint_path)
            print(f"Model saved to {checkpoint_path}")
    
    return history

def process_game_for_training(pgn_string, device="cpu"):
    """
    Process a PGN game to create training data.
    
    Args:
        pgn_string (str): PGN string representation of a chess game
        device (str): Device to create tensors on
        
    Returns:
        tuple: (states, policies, values) - Training data from the game
    """
    # Parse PGN
    pgn = chess.pgn.read_game(io.StringIO(pgn_string))
    if not pgn:
        return [], [], []
    
    # Get game result
    result = pgn.headers.get("Result", "*")
    
    if result == "1-0":
        final_score = 1.0  # White wins
    elif result == "0-1":
        final_score = -1.0  # Black wins
    elif result == "1/2-1/2":
        final_score = 0.0  # Draw
    else:
        return [], [], []  # Incomplete game, skip
    
    states = []
    policies = []
    values = []
    
    # Piece mapping for the model input
    piece_to_idx = {
        'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,  # White pieces
        'p': 7, 'n': 8, 'b': 9, 'r': 10, 'q': 11, 'k': 12  # Black pieces
    }
    
    # Replay the game and extract training data
    board = chess.Board()
    node = pgn
    
    # For each move, extract the board state, the move made, and the eventual outcome
    while node.variations:
        node = node.variations[0]  # Main line
        move = node.move
        move_uci = move.uci()
        
        # Create model input from current board state
        board_tensor = torch.zeros(1, 64, dtype=torch.long)
        
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
        state = {
            "board_positions": board_tensor.to(device),
            "turns": turn_tensor.to(device),
            "white_kingside_castling_rights": w_kingside.to(device),
            "white_queenside_castling_rights": w_queenside.to(device),
            "black_kingside_castling_rights": b_kingside.to(device),
            "black_queenside_castling_rights": b_queenside.to(device)
        }
        
        # Create one-hot policy vector for the move made
        policy = torch.zeros(1, len(UCI_MOVES))
        if move_uci in UCI_MOVES:
            policy[0, UCI_MOVES[move_uci]] = 1.0
        
        # Calculate value based on result from perspective of current player
        # For white: win=1, draw=0, loss=-1
        # For black: win=-1, draw=0, loss=1
        if board.turn == chess.WHITE:
            value = torch.tensor([[final_score]], device=device)
        else:
            value = torch.tensor([[-final_score]], device=device)
        
        # Add to training data
        states.append(state)
        policies.append(policy.to(device))
        values.append(value)
        
        # Make the move on the board
        board.push(move)
    
    return states, policies, values
