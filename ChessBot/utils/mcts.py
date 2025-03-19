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
import concurrent.futures
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
        
        # Add Dirichlet noise to root node
        if self.dirichlet_noise and root.children:
            noise = np.random.dirichlet([self.dirichlet_alpha] * len(root.children))
            for i, child in enumerate(root.children):
                child.prior = (1 - self.noise_fraction) * child.prior + self.noise_fraction * noise[i]
        
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
        probs = root.get_score_adjusted_distribution(temperature, visit_weight, value_weight)
        
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


def train_model_with_mcts_regeneration(model, optimizer, num_epochs=50, batch_size=64, num_games_per_epoch=10, 
                                      device="cpu", simulations=800, temperature=1.0, c_puct=1.0,
                                      repetition_penalty=0.5, checkpoint_dir="checkpoints", save_frequency=5,
                                      start_epoch=0, game_history_dir=None):
    """
    Train the model using MCTS self-play with game regeneration at each epoch.
    
    Args:
        model: Neural network model
        optimizer: Optimizer for training
        num_epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        num_games_per_epoch (int): Number of self-play games per epoch
        device: Device to run the model on
        simulations (int): Number of MCTS simulations per move
        temperature (float): Temperature for move selection
        c_puct (float): Exploration constant for MCTS
        repetition_penalty (float): Penalty factor for repeated positions
        checkpoint_dir (str): Directory to save model checkpoints
        save_frequency (int): How often to save model checkpoints
        start_epoch (int): Starting epoch number for resuming training
        game_history_dir (str): Directory to save games for visualization
        
    Returns:
        dict: Training history
    """
    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Create directory for game history if not provided
    if game_history_dir is None:
        game_history_dir = os.path.join(checkpoint_dir, "game_history")
    os.makedirs(game_history_dir, exist_ok=True)
    
    # Training history
    history = {
        'policy_loss': [],
        'value_loss': [],
        'total_loss': [],
        'epoch': []
    }
    
    for epoch in range(start_epoch, num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # Generate new games using current model
        states = []
        policy_targets = []
        value_targets = []
        
        for game_idx in tqdm(range(num_games_per_epoch), desc="Generating games"):
            # Create MCTS
            mcts = MCTS(model, device, simulations=simulations, c_puct=c_puct, repetition_penalty=repetition_penalty)
            
            # Play a game
            board = chess.Board()
            game_states = []
            game_policies = []
            
            move_count = 0
            position_history = {}  # Track position history
            
            while not board.is_game_over() and move_count < 2000:
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
                    break
                    
                # Save state and policy
                state = create_batch_from_board(board, device)
                policy = torch.zeros(1, len(UCI_MOVES), device=device)
                
                for m, prob in move_probs:
                    m_uci = m.uci()
                    if m_uci in UCI_MOVES:
                        policy[0, UCI_MOVES[m_uci]] = prob
                
                game_states.append(state)
                game_policies.append(policy)
                
                # Make the move
                board.push(move)
                move_count += 1
                
                # Break if game is taking too long
                if move_count >= 100:
                    temperature = max(0.1, temperature * 0.9)  # Reduce temperature to make moves more deterministic
            
            # Game result
            if board.is_checkmate():
                winner = 1.0 if board.turn == chess.BLACK else -1.0
                result_str = "1-0" if winner == -1.0 else "0-1"
                result_txt = f"Winner: {'White' if winner == -1.0 else 'Black'}"
            elif board.is_stalemate() or board.is_insufficient_material() or board.is_fifty_moves() or board.is_repetition():
                winner = 0.0
                result_str = "1/2-1/2"
                result_txt = "Draw"
            else:
                winner = 0.0
                result_str = "1/2-1/2"
                result_txt = "Draw (Max moves)"
            
            # Save game for visualization
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            pgn_game = chess.pgn.Game()
            pgn_game.headers["Event"] = "Self-Play"
            pgn_game.headers["Site"] = f"Training Epoch {epoch+1}"
            pgn_game.headers["Date"] = datetime.datetime.now().strftime("%Y.%m.%d")
            pgn_game.headers["Round"] = str(game_idx + 1)
            pgn_game.headers["White"] = "AI"
            pgn_game.headers["Black"] = "AI"
            pgn_game.headers["Result"] = result_str
            
            # Add game moves to PGN
            node = pgn_game
            for move in board.move_stack:
                node = node.add_variation(move)
            
            pgn_string = str(pgn_game)
            
            # Save game for visualization
            pgn_pickle_path = os.path.join(
                game_history_dir, 
                f"epoch_{epoch+1}_game_{game_idx+1}_{timestamp}.pkl"
            )
            with open(pgn_pickle_path, "wb") as f:
                pickle.dump(pgn_string, f)
            
            print(f"  Game {game_idx+1}: {result_txt} ({move_count} moves) - Saved at {os.path.basename(pgn_pickle_path)}")
            
            # Assign values to states based on the final result
            game_values = []
            
            for i in range(len(game_states)):
                value = winner if (len(game_states) - i) % 2 == 1 else -winner
                game_values.append(torch.tensor([[value]], device=device))
            
            # Add game data to training data
            states.extend(game_states)
            policy_targets.extend(game_policies)
            value_targets.extend(game_values)
            
        # Train on the generated data
        epoch_policy_losses = []
        epoch_value_losses = []
        epoch_total_losses = []
        
        # Convert training data to tensors
        num_samples = len(states)
        
        # Shuffle data
        indices = list(range(num_samples))
        random.shuffle(indices)
        
        # Train in batches
        num_batches = (num_samples + batch_size - 1) // batch_size
        for batch_idx in tqdm(range(num_batches), desc="Training"):
            # Get batch indices
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_samples)
            batch_indices = indices[start_idx:end_idx]
            if not batch_indices:
                continue
                
            # Prepare batch data
            batch_states = {}
            for key in states[0].keys():
                batch_states[key] = torch.cat([states[i][key] for i in batch_indices], dim=0)
            
            batch_policies = torch.cat([policy_targets[i] for i in batch_indices], dim=0)
            batch_values = torch.cat([value_targets[i] for i in batch_indices], dim=0)
            
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
            
            # Backward pass and optimize
            total_loss.backward()
            optimizer.step()
            
            # Record losses
            epoch_policy_losses.append(policy_loss.item())
            epoch_value_losses.append(value_loss.item())
            epoch_total_losses.append(total_loss.item())
        
        # Average losses for this epoch
        if epoch_policy_losses:
            avg_policy_loss = sum(epoch_policy_losses) / len(epoch_policy_losses)
            avg_value_loss = sum(epoch_value_losses) / len(epoch_value_losses)
            avg_total_loss = sum(epoch_total_losses) / len(epoch_total_losses)
            
            history['policy_loss'].append(avg_policy_loss)
            history['value_loss'].append(avg_value_loss)
            history['total_loss'].append(avg_total_loss)
            history['epoch'].append(epoch + 1)
            
            print(f"Epoch {epoch+1} - Policy Loss: {avg_policy_loss:.4f}, Value Loss: {avg_value_loss:.4f}, Total Loss: {avg_total_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % save_frequency == 0 or epoch == num_epochs - 1:
            checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_total_loss if epoch_policy_losses else None,
                'history': history,
                'repetition_penalty': repetition_penalty
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
            
            # Update latest model
            latest_path = os.path.join(checkpoint_dir, "latest_model.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_total_loss if epoch_policy_losses else None,
                'history': history,
                'repetition_penalty': repetition_penalty
            }, latest_path)
    
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

def generate_single_stockfish_game(game_idx, stockfish_path, game_history_dir,
                                  min_elo=1200, max_elo=2000, max_moves=100,
                                  min_tc=0.01, max_tc=0.05, vary_params=True):
    """
    Generate a single chess game played between two Stockfish engines.
    
    Args:
        game_idx (int): Index of the game
        stockfish_path (str): Path to Stockfish executable
        game_history_dir (str): Directory to save game history for visualization
        min_elo (int): Minimum ELO rating for Stockfish
        max_elo (int): Maximum ELO rating for Stockfish
        max_moves (int): Maximum number of moves per game
        min_tc (float): Minimum time control (seconds per move)
        max_tc (float): Maximum time control (seconds per move)
        vary_params (bool): Whether to vary Stockfish parameters between games
        
    Returns:
        tuple: (game_idx, pgn_string, result, move_count, elo1, elo2)
    """
    try:
        engine1 = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        engine2 = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        
        # Set varying parameters if requested
        if vary_params:
            elo1 = random.randint(min_elo, max_elo)
            elo2 = random.randint(min_elo, max_elo)
            tc1 = random.uniform(min_tc, max_tc)
            tc2 = random.uniform(min_tc, max_tc)
            
            # Configure engines with different skill levels
            skill_level1 = min(20, elo1 // 100)
            skill_level2 = min(20, elo2 // 100)
            
            engine1.configure({"Skill Level": skill_level1})
            engine2.configure({"Skill Level": skill_level2})
        else:
            # Use consistent parameters
            elo1 = max_elo
            elo2 = max_elo
            tc1 = max_tc
            tc2 = max_tc
        
        # Create a new board
        board = chess.Board()
        
        # Setup PGN game
        game = chess.pgn.Game()
        game.headers["Event"] = "Stockfish Training Game"
        game.headers["Site"] = "Training"
        game.headers["Date"] = datetime.datetime.now().strftime("%Y.%m.%d")
        game.headers["Round"] = str(game_idx + 1)
        game.headers["White"] = f"Stockfish (ELO {elo1})"
        game.headers["Black"] = f"Stockfish (ELO {elo2})"
        
        node = game
        move_count = 0
        
        # Play the game
        while not board.is_game_over() and move_count < max_moves:
            if board.turn == chess.WHITE:
                # White's move
                result = engine1.play(board, chess.engine.Limit(time=tc1))
            else:
                # Black's move
                result = engine2.play(board, chess.engine.Limit(time=tc2))
            
            move = result.move
            board.push(move)
            node = node.add_variation(move)
            move_count += 1
        
        # Set the result
        if board.is_checkmate():
            result = "1-0" if board.turn == chess.BLACK else "0-1"
        elif board.is_stalemate() or board.is_insufficient_material() or move_count >= max_moves:
            result = "1/2-1/2"
        else:
            result = "*"
            
        game.headers["Result"] = result
        
        # Convert game to PGN string
        pgn_string = str(game)
        
        # Save game for visualization
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        pgn_pickle_path = os.path.join(
            game_history_dir, 
            f"stockfish_game_{game_idx+1}_{timestamp}.pkl"
        )
        with open(pgn_pickle_path, "wb") as f:
            pickle.dump(pgn_string, f)
            
        return game_idx, pgn_string, result, move_count, elo1, elo2
            
    except Exception as e:
        print(f"Error generating game {game_idx+1}: {e}")
        return game_idx, None, "Error", 0, 0, 0
    finally:
        # Clean up engines
        if 'engine1' in locals(): engine1.quit()
        if 'engine2' in locals(): engine2.quit()

def generate_stockfish_games_parallel(num_games, stockfish_path, game_history_dir, 
                                     min_elo=1200, max_elo=2000, max_moves=100, 
                                     min_tc=0.01, max_tc=0.05, vary_params=True,
                                     max_workers=None):
    """
    Generate chess games played between Stockfish engines in parallel.
    
    Args:
        num_games (int): Number of games to generate
        stockfish_path (str): Path to Stockfish executable
        game_history_dir (str): Directory to save game history for visualization
        min_elo (int): Minimum ELO rating for Stockfish
        max_elo (int): Maximum ELO rating for Stockfish
        max_moves (int): Maximum number of moves per game
        min_tc (float): Minimum time control (seconds per move)
        max_tc (float): Maximum time control (seconds per move)
        vary_params (bool): Whether to vary Stockfish parameters between games
        max_workers (int): Maximum number of worker threads (None = auto)
        
    Returns:
        list: List of PGN game strings
    """
    os.makedirs(game_history_dir, exist_ok=True)
    games = []

    # clear the game history directory
    for file in os.listdir(game_history_dir):
        file_path = os.path.join(game_history_dir, file)
        if os.path.isfile(file_path):
            os.remove(file_path)
    
    # If max_workers is None, use the number of CPU cores
    if max_workers is None:
        import multiprocessing
        max_workers = multiprocessing.cpu_count()
        print(f"Using {max_workers} worker threads based on CPU count")
    
    # Create a progress bar for tracking
    progress_bar = tqdm(total=num_games, desc="Generating Stockfish games")
    
    # Function to update progress and process results
    def process_result(future):
        game_idx, pgn_string, result, move_count, elo1, elo2 = future.result()
        if pgn_string:
            games.append(pgn_string)
            progress_bar.set_postfix_str(f"Result={result} Moves={move_count} ELO: {elo1} vs {elo2}")
        progress_bar.update(1)
    
    # Use ThreadPoolExecutor to run games in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all game generation tasks
        futures = []
        for game_idx in range(num_games):
            future = executor.submit(
                generate_single_stockfish_game,
                game_idx,
                stockfish_path,
                game_history_dir,
                min_elo,
                max_elo,
                max_moves,
                min_tc,
                max_tc,
                vary_params
            )
            future.add_done_callback(process_result)
            futures.append(future)
        
        # Wait for all tasks to complete
        concurrent.futures.wait(futures)
    
    progress_bar.close()
    print(f"Generated {len(games)} games successfully")
    return games

def train_model_with_stockfish_games(model, optimizer, num_epochs, batch_size, device, 
                                    games_per_epoch, stockfish_path, checkpoint_dir, 
                                    save_frequency=5, start_epoch=0, max_workers=None, 
                                    regenerate_games=False):
    """
    Train the model using games played between Stockfish engines.
    
    Args:
        model: Neural network model
        optimizer: Optimizer for training
        num_epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        device: Device to run the model on
        games_per_epoch (int): Number of Stockfish games to use per epoch
        stockfish_path (str): Path to Stockfish executable
        checkpoint_dir (str): Directory to save model checkpoints
        save_frequency (int): How often to save model checkpoints
        start_epoch (int): Starting epoch number for resuming training
        max_workers (int): Maximum number of parallel workers for game generation
        regenerate_games (bool): Whether to regenerate games for each epoch
        
    Returns:
        dict: Training history
    """
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Create game history directory
    game_history_dir = os.path.join(checkpoint_dir, "stockfish_games")
    os.makedirs(game_history_dir, exist_ok=True)
    
    # Training history
    history = {
        'policy_loss': [],
        'value_loss': [],
        'total_loss': [],
        'epoch': []
    }
    
    # Generate or load games only once if not regenerating each epoch
    if not regenerate_games:
        # Check if we already have games saved
        games_cache_path = os.path.join(checkpoint_dir, "stockfish_games_cache.pkl")
        
        if os.path.exists(games_cache_path):
            print(f"Loading {games_per_epoch} cached Stockfish games...")
            try:
                with open(games_cache_path, "rb") as f:
                    pgn_games = pickle.load(f)
                
                # If we don't have enough games in the cache, generate more
                if len(pgn_games) < games_per_epoch:
                    print(f"Need more games. Have {len(pgn_games)}, need {games_per_epoch}")
                    additional_games = generate_stockfish_games_parallel(
                        num_games=games_per_epoch - len(pgn_games),
                        stockfish_path=stockfish_path,
                        game_history_dir=game_history_dir,
                        min_elo=1200,
                        max_elo=2800,
                        max_moves=200,
                        vary_params=True,
                        max_workers=max_workers
                    )
                    pgn_games.extend(additional_games)
                    
                    # Save the updated cache
                    with open(games_cache_path, "wb") as f:
                        pickle.dump(pgn_games, f)
            except Exception as e:
                print(f"Error loading cached games: {e}")
                print("Generating new games...")
                pgn_games = generate_stockfish_games_parallel(
                    num_games=games_per_epoch,
                    stockfish_path=stockfish_path,
                    game_history_dir=game_history_dir,
                    min_elo=1200,
                    max_elo=2800,
                    max_moves=200,
                    vary_params=True,
                    max_workers=max_workers
                )
                
                # Save the games cache
                with open(games_cache_path, "wb") as f:
                    pickle.dump(pgn_games, f)
        else:
            print(f"Generating {games_per_epoch} Stockfish games in parallel (one-time)...")
            pgn_games = generate_stockfish_games_parallel(
                num_games=games_per_epoch,
                stockfish_path=stockfish_path,
                game_history_dir=game_history_dir,
                min_elo=1200,
                max_elo=2800,
                max_moves=200,
                vary_params=True,
                max_workers=max_workers
            )
            
            # Save the games cache
            with open(games_cache_path, "wb") as f:
                pickle.dump(pgn_games, f)
    
    # Process games to create the training dataset (only needed once if not regenerating)
    all_states = []
    all_policies = []
    all_values = []
    
    for epoch in range(start_epoch, num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # Generate new games each epoch if regenerate_games is True
        if regenerate_games:
            print(f"Regenerating {games_per_epoch} Stockfish games for this epoch...")
            pgn_games = generate_stockfish_games_parallel(
                num_games=games_per_epoch,
                stockfish_path=stockfish_path,
                game_history_dir=game_history_dir,
                min_elo=1200,
                max_elo=2800,
                max_moves=200,
                vary_params=True,
                max_workers=max_workers
            )
            
            # Process games for training (needed each epoch when regenerating)
            print("Processing newly generated games for training...")
            all_states = []
            all_policies = []
            all_values = []
            
            for pgn_string in tqdm(pgn_games, desc="Processing games"):
                states, policies, values = process_game_for_training(pgn_string, device)
                all_states.extend(states)
                all_policies.extend(policies)
                all_values.extend(values)
        elif epoch == start_epoch:
            # Only process games once if we're not regenerating
            print("Processing games for training (one-time)...")
            
            for pgn_string in tqdm(pgn_games, desc="Processing games"):
                states, policies, values = process_game_for_training(pgn_string, device)
                all_states.extend(states)
                all_policies.extend(policies)
                all_values.extend(values)
        
        # Check if we have enough training data
        if len(all_states) < batch_size:
            print(f"Warning: Not enough training data. Got {len(all_states)} samples, need at least {batch_size}.")
            if len(all_states) == 0:
                continue
        
        # Train on the processed games
        epoch_policy_losses = []
        epoch_value_losses = []
        epoch_total_losses = []
        
        # Shuffle data
        indices = list(range(len(all_states)))
        random.shuffle(indices)
        
        # Train in batches
        num_batches = (len(indices) + batch_size - 1) // batch_size
        for batch_idx in tqdm(range(num_batches), desc="Training"):
            # Get batch indices
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(indices))
            batch_indices = indices[start_idx:end_idx]
            
            if not batch_indices:
                continue
                
            # Prepare batch data
            batch_states = {}
            for key in all_states[0].keys():
                batch_states[key] = torch.cat([all_states[i][key] for i in batch_indices], dim=0)
            
            batch_policies = torch.cat([all_policies[i] for i in batch_indices], dim=0)
            batch_values = torch.cat([all_values[i] for i in batch_indices], dim=0)
            
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
            policy_loss = -torch.sum(batch_policies * torch.log_softmax(policy_logits, dim=1)) / batch_policies.size(0) / 10
            
            # Value loss (MSE)
            value_loss = torch.mean((value_preds - batch_values) ** 2)
            
            # Total loss
            total_loss = policy_loss + value_loss
            
            # Backward pass and optimize
            total_loss.backward()
            optimizer.step()
            
            # Record losses
            epoch_policy_losses.append(policy_loss.item())
            epoch_value_losses.append(value_loss.item())
            epoch_total_losses.append(total_loss.item())

            print(f"Batch {batch_idx + 1}/{num_batches} - Policy Loss: {policy_loss.item():.4f}, Value Loss: {value_loss.item():.4f}, Total Loss: {total_loss.item():.4f}")
        
        # Average losses for this epoch
        if epoch_policy_losses:
            avg_policy_loss = sum(epoch_policy_losses) / len(epoch_policy_losses)
            avg_value_loss = sum(epoch_value_losses) / len(epoch_value_losses)
            avg_total_loss = sum(epoch_total_losses) / len(epoch_total_losses)
            
            history['policy_loss'].append(avg_policy_loss)
            history['value_loss'].append(avg_value_loss)
            history['total_loss'].append(avg_total_loss)
            history['epoch'].append(epoch + 1)
            
            print(f"Epoch {epoch+1} - Policy Loss: {avg_policy_loss:.4f}, Value Loss: {avg_value_loss:.4f}, Total Loss: {avg_total_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % save_frequency == 0 or epoch == num_epochs - 1:
            checkpoint_path = os.path.join(checkpoint_dir, f"stockfish_model_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_total_loss if epoch_policy_losses else None,
                'history': history
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
            
            # Update latest model
            latest_path = os.path.join(checkpoint_dir, "latest_model.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_total_loss if epoch_policy_losses else None,
                'history': history
            }, latest_path)
    
    return history