import pygame
import chess
import chess.svg
import torch
import io
import cairosvg
import time
import threading
import math
import numpy as np
import os
import sys
import argparse
from copy import deepcopy

# Add parent directory to path to be able to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import required modules from your existing code
from utils.paste import UCI_MOVES
from models.transformer_chess import EncoderOnlyTransformer
from utils.mcts import MCTS, MCTSNode, create_batch_from_board

class MCTSVisualizer:
    def __init__(self, model_path, board_size=600, stockfish_path=None):
        # Initialize PyGame
        pygame.init()
        pygame.display.set_caption("Chess MCTS Visualizer")
        
        # Screen dimensions
        self.board_size = board_size
        self.screen_width = board_size
        self.screen_height = board_size + 200  # Extra space for move information
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        
        # Fonts
        self.font_large = pygame.font.SysFont('Arial', 18, bold=True)
        self.font_small = pygame.font.SysFont('Arial', 16)
        import chess
        # Chess board and game state
        self.board = chess.Board()
        self.current_arrows = []
        self.top_moves = []
        self.is_searching = False
        self.search_thread = None
        self.stop_search = False
        
        # Load configuration
        self.config = self.load_config()
        
        # Load model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.model = self.load_model(model_path)
        
        # MCTS parameters from config
        self.mcts_simulations = 12000
        self.mcts_temperature = self.config.mcts.temperature if hasattr(self.config, 'mcts') else 1.0
        self.mcts_exploration = self.config.mcts.exploration if hasattr(self.config, 'mcts') else 1.0
        self.dirichlet_noise = self.config.mcts.dirichlet_noise if hasattr(self.config, 'mcts') else True
        self.dirichlet_alpha = self.config.mcts.dirichlet_alpha if hasattr(self.config, 'mcts') else 0.3
        self.noise_fraction = self.config.mcts.noise_fraction if hasattr(self.config, 'mcts') else 0.25
        
        print(f"MCTS Configuration: simulations={self.mcts_simulations}, "
              f"temperature={self.mcts_temperature}, exploration={self.mcts_exploration}")
        
        self.mcts = MCTS(
            model=self.model,
            device=self.device,
            simulations=self.mcts_simulations,
            c_puct=self.mcts_exploration,
            dirichlet_noise=self.dirichlet_noise,
            dirichlet_alpha=self.dirichlet_alpha,
            noise_fraction=self.noise_fraction
        )
        
        # Colors
        self.bg_color = (50, 50, 50)
        self.text_color = (255, 255, 255)
        self.highlight_color = (0, 120, 215)
        
        # Stockfish for comparison if available
        self.stockfish_engine = None
        if stockfish_path and os.path.exists(stockfish_path):
            try:
                import chess.engine
                self.stockfish_engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
                print(f"Stockfish loaded from {stockfish_path}")
            except Exception as e:
                print(f"Error loading Stockfish: {e}")
    
    def load_model(self, model_path):
        """Load the chess model from checkpoint"""
        from omegaconf import OmegaConf
        
        # Define default model config that matches your configuration
        config = {
            "embed_dim": 768,
            "num_heads": 12,
            "num_layers": 12,
            "ff_dim": 1024,
            "dropout": 0.1,
            "board_vocab_size": 13,     # 12 pieces + 1 empty square
            "moves_vocab_size": 1971,   # Based on UCI_MOVES length
            "pos_size": 69,             # 8x8 board + metadata
            "turn_size": 2,             # White/Black
            "castling_size": 2          # Can/Cannot castle
        }
        
        # Try to load model config from checkpoint first
        model_config = None
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'config' in checkpoint:
                model_config = checkpoint['config']
                print("Using model configuration from checkpoint")
        except:
            pass
            
        # If no config in checkpoint, use default
        if model_config is None:
            model_config = OmegaConf.create(config)
            print("Using default model configuration")
            
        model = EncoderOnlyTransformer(model_config).to(self.device)
        
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            print(f"Model loaded from {model_path}")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Using initialized model...")
            model.init_weights()
            return model
    
    def svg_to_surface(self, svg_data):
        """Convert SVG data to PyGame surface"""
        try:
            png_data = cairosvg.svg2png(bytestring=svg_data.encode("utf-8"), 
                                       output_width=self.board_size, 
                                       output_height=self.board_size)
            return pygame.image.load(io.BytesIO(png_data))
        except Exception as e:
            print(f"Error converting SVG: {e}")
            # Return a blank surface as fallback
            surface = pygame.Surface((self.board_size, self.board_size))
            surface.fill((255, 255, 255))
            return surface
    
    def get_board_svg(self):
        """Get the chess board as SVG with arrows for candidate moves"""
        arrows = []
        for move, score, visits, q_value in self.top_moves:
            # Scale arrow width based on score
            width = 5 + int(15 * score)
            # Use different colors for different moves
            color = "#0066cc" if move == self.top_moves[0][0] else "#cc0000"
            arrows.append(chess.svg.Arrow(
                move.from_square, move.to_square, color=color
            ))
        
        # Get SVG with arrows
        return chess.svg.board(
            board=self.board,
            arrows=arrows,
            size=self.board_size
        )
    
    def load_config(self):
        """Load default configuration"""
        from omegaconf import OmegaConf
        
        # Default configuration that matches your system
        default_config = {
            "model": {
                "embed_dim": 768,
                "num_heads": 12,
                "ff_dim": 1024,
                "dropout": 0.1,
                "num_layers": 12,
                "board_vocab_size": 13,
                "pos_size": 69,
                "turn_size": 2,
                "castling_size": 2,
                "moves_vocab_size": 1971
            },
            "game": {
                "opponent": "self",
                "model_color": "white",
                "max_moves": 1000,
                "temperature": 0.7,
                "top_k": 5
            },
            "stockfish": {
                "path": "C:\\Program Files\\stockfish\\stockfish-windows-x86-64-avx2.exe",
                "games_per_epoch": 12800,
                "max_workers": 24,
                "min_elo": 1200,
                "max_elo": 2800,
                "max_moves": 200,
                "min_tc": 5,
                "max_tc": 10,
                "vary_params": True
            },
            "mcts": {
                "enabled": True,
                "train_method": "continuous",
                "games_per_epoch": 10,
                "simulations": 800,
                "temperature": 1.0,
                "exploration": 1.0,
                "dirichlet_noise": True,
                "dirichlet_alpha": 0.3,
                "noise_fraction": 0.25,
                "num_games": 100,
                "epochs_per_game": 1
            },
            "checkpoint": {
                "path": "checkpoints/latest_model.pt"
            }
        }
        
        config = OmegaConf.create(default_config)
        
        # Try to load from file if it exists
        try:
            file_config = OmegaConf.load("configs/default.yaml")
            config = OmegaConf.merge(config, file_config)
            print("Loaded configuration from configs/default.yaml")
        except:
            print("Using default configuration")
        
        return config
    

    def run_mcts_search(self):
        """Run MCTS search in a separate thread"""
        self.is_searching = True
        self.stop_search = False
        
        # Prepare for new search
        self.top_moves = []
        
        # Track search progress
        node_counts = []
        start_time = time.time()
        last_update = start_time
        display_interval = 0.5  # Update display every 0.5 seconds
        
        # Initialize root node for search
        root = MCTSNode(self.board)
        
        # Expand root node
        root.expand(self.model, self.device)
        
        # Add Dirichlet noise to root if enabled
        if self.dirichlet_noise and root.children:
            noise = np.random.dirichlet([self.dirichlet_alpha] * len(root.children))
            for i, child in enumerate(root.children):
                child.prior = (1 - self.noise_fraction) * child.prior + self.noise_fraction * noise[i]
        
        # Run simulations
        for sim_idx in range(self.mcts_simulations):
            if self.stop_search:
                break
                
            # Selection phase
            node = root
            search_path = [node]
            
            while not node.is_leaf() and not node.is_terminal():
                node = node.select_child()
                search_path.append(node)
            
            # Expansion phase
            if not node.is_terminal():
                node.expand(self.model, self.device)
                
                if node.children:
                    child = node.select_child()
                    search_path.append(child)
                    node = child
            
            # Evaluation phase
            if node.is_terminal():
                # Game is over
                result = node.board.result()
                
                if result == "1-0":
                    value = 1.0 if node.board.turn == chess.BLACK else -1.0
                elif result == "0-1":
                    value = 1.0 if node.board.turn == chess.WHITE else -1.0
                else:
                    value = 0.0
            else:
                # Use model to evaluate position
                model_input = create_batch_from_board(node.board, self.device)
                
                with torch.no_grad():
                    predictions = self.model(model_input)
                value = predictions["winrate"].item()
                value = 2 * value - 1  # Convert to [-1, 1] range
            
            # Backup phase
            for node in reversed(search_path):
                value = -value  # Negate value for opponent's perspective
                node.update(value)
            
            # Update display periodically
            current_time = time.time()
            if current_time - last_update > display_interval:
                # Collect top moves
                self.update_top_moves(root)
                
                # Record node count
                node_counts.append(sim_idx + 1)
                
                # Trigger display update
                last_update = current_time
        
        # Final update after search is complete
        self.update_top_moves(root)
        print(f"MCTS search complete. Total time: {time.time() - start_time:.2f}s")
        self.is_searching = False
    
    def update_top_moves(self, root, visit_weight=0.6, value_weight=0.4):
        """
        Update the list of top moves based on current search state,
        balancing visit counts with Q-values.
        
        Args:
            root (MCTSNode): Root node of the search tree
            visit_weight (float): Weight given to visit counts (0-1)
            value_weight (float): Weight given to Q-values (0-1)
        """
        if not root.children:
            self.top_moves = []
            return
        
        # Calculate total visits for normalization
        total_visits = sum(child.visits for child in root.children)
        if total_visits == 0:
            self.top_moves = []
            return
            
        # Get all children with their weighted scores
        move_scores = []
        for child in root.children:
            # Only consider moves with at least one visit
            if child.visits == 0:
                continue
                
            # Calculate Q-value (from parent's perspective)
            q_value = child.value_sum / child.visits
            
            # Normalize visit count (0-1 range)
            visit_score = child.visits / total_visits
            
            # Normalize Q-value to 0-1 range (from [-1,1])
            norm_q_value = (q_value + 1) / 2
            
            # Calculate combined score
            combined_score = (visit_weight * visit_score) + (value_weight * norm_q_value)
            
            move_scores.append((child.move, combined_score, child.visits, q_value))
        
        # Sort by combined score (descending)
        move_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Take top 5 moves
        self.top_moves = move_scores[:5]
    
    def draw_board(self):
        """Draw the chess board with annotations"""
        # Fill background
        self.screen.fill(self.bg_color)
        
        # Get board SVG with arrows and convert to surface
        svg_data = self.get_board_svg()
        board_surface = self.svg_to_surface(svg_data)
        
        # Draw board
        self.screen.blit(board_surface, (0, 0))
        
        # Draw information panel
        self.draw_info_panel()
        
        # Update display
        pygame.display.flip()
    
    def draw_info_panel(self):
        """Draw the information panel with MCTS search results"""
        # Panel background
        panel_rect = pygame.Rect(0, self.board_size, self.screen_width, 200)
        pygame.draw.rect(self.screen, (30, 30, 30), panel_rect)
        pygame.draw.line(self.screen, (100, 100, 100), 
                        (0, self.board_size), (self.screen_width, self.board_size), 2)
        
        # Current position info
        turn_text = f"{'White' if self.board.turn == chess.WHITE else 'Black'} to move"
        turn_surface = self.font_large.render(turn_text, True, self.text_color)
        self.screen.blit(turn_surface, (20, self.board_size + 15))
        
        # Search status
        status_text = "Searching..." if self.is_searching else "Search complete" 
        status_surface = self.font_small.render(status_text, True, self.text_color)
        self.screen.blit(status_surface, (20, self.board_size + 40))
        
        # Explain scoring method
        scoring_text = "Moves scored by: 60% visits + 40% evaluation"
        scoring_surface = self.font_small.render(scoring_text, True, (180, 180, 180))
        self.screen.blit(scoring_surface, (225, self.board_size + 40))
        
        # Draw top moves
        self.draw_top_moves()
        
        # Instructions
        instructions = [
            "SPACE: Start/Stop search",
            "ENTER: Make top move",
            "ESC: Quit",
            "R: Reset board"
        ]
        
        for i, instruction in enumerate(instructions):
            instr_surface = self.font_small.render(instruction, True, (180, 180, 180))
            self.screen.blit(instr_surface, (self.screen_width - 220, self.board_size + 15 + i*25))
    
    def draw_top_moves(self):
        """Draw the list of top candidate moves with combined scores"""
        if not self.top_moves:
            empty_text = "No moves analyzed yet"
            empty_surface = self.font_small.render(empty_text, True, self.text_color)
            self.screen.blit(empty_surface, (20, self.board_size + 70))
            return
        
        # Header
        header_text = "Top Moves:   Move    Score    Visits    Q-Value"
        header_surface = self.font_small.render(header_text, True, self.highlight_color)
        self.screen.blit(header_surface, (20, self.board_size + 70))
        
        # Move list
        y_offset = self.board_size + 95
        for i, (move, score, visits, q_value) in enumerate(self.top_moves):
            # Choose color based on rank (best move highlighted)
            color = self.highlight_color if i == 0 else self.text_color
            
            # Format move info
            move_text = f"{i+1}. {move.uci()}     {score*100:.1f}%     {visits}     {q_value:.2f}"
            
            # Render and draw
            move_surface = self.font_small.render(move_text, True, color)
            self.screen.blit(move_surface, (30, y_offset + i*25))
    
    def start_search(self):
        """Start MCTS search in a separate thread"""
        if self.is_searching:
            return
            
        self.search_thread = threading.Thread(target=self.run_mcts_search)
        self.search_thread.daemon = True
        self.search_thread.start()
    
    def stop_search_thread(self):
        """Stop the running search thread"""
        if self.is_searching:
            self.stop_search = True
            self.search_thread.join(timeout=1.0)
            self.is_searching = False
    
    def make_move(self, move):
        """Make a move on the chess board"""
        if move in self.board.legal_moves:
            self.board.push(move)
            return True
        return False
    
    def handle_events(self):
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
                
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                    
                elif event.key == pygame.K_SPACE:
                    # Toggle search
                    if self.is_searching:
                        self.stop_search_thread()
                    else:
                        self.start_search()
                        
                elif event.key == pygame.K_RETURN:
                    # Make top move
                    if self.top_moves:
                        top_move = self.top_moves[0][0]
                        if self.make_move(top_move):
                            self.stop_search_thread()
                            self.top_moves = []
                            
                elif event.key == pygame.K_r:
                    # Reset board
                    self.stop_search_thread()
                    self.board = chess.Board()
                    self.top_moves = []
        
        return True
    
    def run(self):
        """Main loop"""
        running = True
        
        # Initial board display
        self.draw_board()
        
        while running:
            running = self.handle_events()
            
            # Update display regularly
            self.draw_board()
            
            # Limit frame rate
            pygame.time.delay(50)
        
        # Clean up
        self.stop_search_thread()
        pygame.quit()
        
        if self.stockfish_engine:
            self.stockfish_engine.quit()

def main():
    parser = argparse.ArgumentParser(description="Chess MCTS Visualizer")
    parser.add_argument("--model", type=str, default="checkpoints/latest_model.pt",
                      help="Path to model checkpoint")
    parser.add_argument("--size", type=int, default=600,
                      help="Board size in pixels")
    parser.add_argument("--stockfish", type=str, 
                      default="C:\\Program Files\\stockfish\\stockfish-windows-x86-64-avx2.exe",
                      help="Path to Stockfish executable")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                      help="Path to configuration file")
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        from omegaconf import OmegaConf
        config = OmegaConf.load(args.config)
        print(f"Loaded configuration from {args.config}")
    except Exception as e:
        print(f"Warning: Could not load config file: {e}")
        print("Using default values")
        config = None
    
    # Use config values if available, otherwise use command line arguments
    model_path = config.checkpoint.path if config and hasattr(config, 'checkpoint') else args.model
    stockfish_path = config.stockfish.path if config and hasattr(config, 'stockfish') else args.stockfish
    
    # Create and run visualizer
    visualizer = MCTSVisualizer(
        model_path=model_path,
        board_size=args.size,
        stockfish_path=stockfish_path
    )
    
    visualizer.run()

if __name__ == "__main__":
    main()