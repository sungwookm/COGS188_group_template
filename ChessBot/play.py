import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import chess
import pygame
import sys
import os
import time
from datetime import datetime

from models.transformer_chess import EncoderOnlyTransformer
from utils.utils import get_best_move


# Define colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
LIGHT_SQUARE = (240, 217, 181)
DARK_SQUARE = (181, 136, 99)
HIGHLIGHT = (124, 252, 0, 128)  # Light green with transparency
LAST_MOVE = (255, 255, 0, 128)  # Yellow with transparency
CHECK = (255, 0, 0, 128)  # Red with transparency
SELECTION = (65, 105, 225, 128)  # Royal blue with transparency

# Define board dimensions
BOARD_SIZE = 480
SQUARE_SIZE = BOARD_SIZE // 8
INFO_PANEL_WIDTH = 240
WINDOW_SIZE = (BOARD_SIZE + INFO_PANEL_WIDTH, BOARD_SIZE)

# Load piece images
def load_pieces():
    pieces = {}
    piece_chars = ['p', 'n', 'b', 'r', 'q', 'k', 'P', 'N', 'B', 'R', 'Q', 'K']
    
    for piece in piece_chars:
        # Check assets directory first
        asset_path = os.path.join('assets', 'pieces', f'{piece}.png')
        if os.path.exists(asset_path):
            pieces[piece] = pygame.image.load(asset_path)
        else:
            # Try to create a default piece representation
            color = WHITE if piece.isupper() else BLACK
            s = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
            s.fill((0, 0, 0, 0))  # Transparent
            
            # Draw the piece shape
            if piece.lower() == 'p':
                pygame.draw.circle(s, color, (SQUARE_SIZE//2, SQUARE_SIZE//2), SQUARE_SIZE//4)
            elif piece.lower() == 'n':
                points = [(SQUARE_SIZE//4, SQUARE_SIZE//4), (3*SQUARE_SIZE//4, SQUARE_SIZE//4), 
                         (3*SQUARE_SIZE//4, 3*SQUARE_SIZE//4), (SQUARE_SIZE//4, 3*SQUARE_SIZE//4)]
                pygame.draw.polygon(s, color, points)
            elif piece.lower() == 'b':
                pygame.draw.polygon(s, color, [(SQUARE_SIZE//2, SQUARE_SIZE//6), 
                                             (SQUARE_SIZE//6, 5*SQUARE_SIZE//6), 
                                             (5*SQUARE_SIZE//6, 5*SQUARE_SIZE//6)])
            elif piece.lower() == 'r':
                pygame.draw.rect(s, color, (SQUARE_SIZE//4, SQUARE_SIZE//4, SQUARE_SIZE//2, SQUARE_SIZE//2))
            elif piece.lower() == 'q':
                pygame.draw.circle(s, color, (SQUARE_SIZE//2, SQUARE_SIZE//2), SQUARE_SIZE//3)
                pygame.draw.polygon(s, color, [(SQUARE_SIZE//2, SQUARE_SIZE//6), 
                                             (SQUARE_SIZE//6, 5*SQUARE_SIZE//6), 
                                             (5*SQUARE_SIZE//6, 5*SQUARE_SIZE//6)], 2)
            elif piece.lower() == 'k':
                pygame.draw.circle(s, color, (SQUARE_SIZE//2, SQUARE_SIZE//2), SQUARE_SIZE//3)
                pygame.draw.line(s, color, (SQUARE_SIZE//2, SQUARE_SIZE//6), 
                               (SQUARE_SIZE//2, 5*SQUARE_SIZE//6), 3)
                pygame.draw.line(s, color, (SQUARE_SIZE//6, SQUARE_SIZE//2), 
                               (5*SQUARE_SIZE//6, SQUARE_SIZE//2), 3)
            
            pieces[piece] = s
    
    # Resize images to fit the squares
    for piece in pieces:
        pieces[piece] = pygame.transform.scale(pieces[piece], (SQUARE_SIZE, SQUARE_SIZE))
    
    return pieces


class ChessGame:
    def __init__(self, model=None, player_color='white', model_thinking_time=1.0, device='cpu', temperature=0.5, top_k=3):
        pygame.init()
        pygame.display.set_caption('Chess vs. Model')
        
        # Set up the display
        self.screen = pygame.display.set_mode(WINDOW_SIZE)
        self.clock = pygame.time.Clock()
        
        # Game state
        self.board = chess.Board()
        self.selected_square = None
        self.valid_moves = []
        self.last_move = None
        self.game_over = False
        self.model = model
        self.player_color = chess.WHITE if player_color.lower() == 'white' else chess.BLACK
        self.model_thinking_time = model_thinking_time
        self.device = device
        self.temperature = temperature
        self.top_k = top_k
        
        # Load pieces
        self.pieces = load_pieces()
        
        # Game history for PGN
        self.move_history = []
        
        # Status messages
        self.status_message = "Your move" if self.board.turn == self.player_color else "Model thinking..."
        self.last_status_update = time.time()
        
        # If model starts, make the first move
        if self.board.turn != self.player_color and self.model is not None:
            self.make_model_move()

    def draw_board(self):
        # Draw the chess board
        for row in range(8):
            for col in range(8):
                # Determine square color
                color = LIGHT_SQUARE if (row + col) % 2 == 0 else DARK_SQUARE
                
                # Draw square
                pygame.draw.rect(self.screen, color, (col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))
                
                # Get piece at this square
                square = chess.square(col, 7 - row)  # Convert to chess.Square
                piece = self.board.piece_at(square)
                
                # Draw piece if present
                if piece:
                    piece_img = self.pieces[piece.symbol()]
                    self.screen.blit(piece_img, (col * SQUARE_SIZE, row * SQUARE_SIZE))
        
        # Highlight selected square
        if self.selected_square is not None:
            col = chess.square_file(self.selected_square)
            row = 7 - chess.square_rank(self.selected_square)
            s = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
            s.fill(SELECTION)
            self.screen.blit(s, (col * SQUARE_SIZE, row * SQUARE_SIZE))
        
        # Highlight valid moves
        for move in self.valid_moves:
            col = chess.square_file(move.to_square)
            row = 7 - chess.square_rank(move.to_square)
            s = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
            s.fill(HIGHLIGHT)
            self.screen.blit(s, (col * SQUARE_SIZE, row * SQUARE_SIZE))
        
        # Highlight last move
        if self.last_move:
            for square in [self.last_move.from_square, self.last_move.to_square]:
                col = chess.square_file(square)
                row = 7 - chess.square_rank(square)
                s = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
                s.fill(LAST_MOVE)
                self.screen.blit(s, (col * SQUARE_SIZE, row * SQUARE_SIZE))
        
        # Highlight check
        if self.board.is_check():
            king_square = self.board.king(self.board.turn)
            col = chess.square_file(king_square)
            row = 7 - chess.square_rank(king_square)
            s = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
            s.fill(CHECK)
            self.screen.blit(s, (col * SQUARE_SIZE, row * SQUARE_SIZE))

    def draw_info_panel(self):
        # Info panel background
        pygame.draw.rect(self.screen, WHITE, (BOARD_SIZE, 0, INFO_PANEL_WIDTH, BOARD_SIZE))
        pygame.draw.line(self.screen, BLACK, (BOARD_SIZE, 0), (BOARD_SIZE, BOARD_SIZE), 2)
        
        # Set up font
        font = pygame.font.SysFont('Arial', 16)
        
        # Game status
        status_text = font.render(f"Status: {self.status_message}", True, BLACK)
        self.screen.blit(status_text, (BOARD_SIZE + 10, 10))
        
        # Turn information
        turn_text = font.render(f"Turn: {'White' if self.board.turn else 'Black'}", True, BLACK)
        self.screen.blit(turn_text, (BOARD_SIZE + 10, 40))
        
        # Piece counts
        w_pawns = len(self.board.pieces(chess.PAWN, chess.WHITE))
        w_knights = len(self.board.pieces(chess.KNIGHT, chess.WHITE))
        w_bishops = len(self.board.pieces(chess.BISHOP, chess.WHITE))
        w_rooks = len(self.board.pieces(chess.ROOK, chess.WHITE))
        w_queens = len(self.board.pieces(chess.QUEEN, chess.WHITE))
        
        b_pawns = len(self.board.pieces(chess.PAWN, chess.BLACK))
        b_knights = len(self.board.pieces(chess.KNIGHT, chess.BLACK))
        b_bishops = len(self.board.pieces(chess.BISHOP, chess.BLACK))
        b_rooks = len(self.board.pieces(chess.ROOK, chess.BLACK))
        b_queens = len(self.board.pieces(chess.QUEEN, chess.BLACK))
        
        # Material difference (simplified)
        w_material = w_pawns + 3*w_knights + 3*w_bishops + 5*w_rooks + 9*w_queens
        b_material = b_pawns + 3*b_knights + 3*b_bishops + 5*b_rooks + 9*b_queens
        material_diff = w_material - b_material
        
        material_text = font.render(f"Material: {'White +' if material_diff > 0 else 'Black +' if material_diff < 0 else 'Even'} {abs(material_diff)}", True, BLACK)
        self.screen.blit(material_text, (BOARD_SIZE + 10, 70))
        
        # Move history (last few moves)
        history_text = font.render("Move History:", True, BLACK)
        self.screen.blit(history_text, (BOARD_SIZE + 10, 100))
        
        # Show last 10 moves
        y_offset = 130
        move_pairs = []
        for i in range(0, len(self.move_history), 2):
            if i+1 < len(self.move_history):
                move_pairs.append(f"{i//2+1}. {self.move_history[i]} {self.move_history[i+1]}")
            else:
                move_pairs.append(f"{i//2+1}. {self.move_history[i]}")
        
        for i, move_pair in enumerate(move_pairs[-10:]):
            move_text = font.render(move_pair, True, BLACK)
            self.screen.blit(move_text, (BOARD_SIZE + 10, y_offset + i*20))
        
        # Game result if game is over
        if self.game_over:
            if self.board.is_checkmate():
                result = "Black wins!" if self.board.turn == chess.WHITE else "White wins!"
            elif self.board.is_stalemate() or self.board.is_insufficient_material():
                result = "Draw!"
            else:
                result = "Game Over"
                
            result_font = pygame.font.SysFont('Arial', 24, bold=True)
            result_text = result_font.render(result, True, BLACK)
            self.screen.blit(result_text, (BOARD_SIZE + 10, BOARD_SIZE - 60))
            
            restart_text = font.render("Press 'R' to restart", True, BLACK)
            self.screen.blit(restart_text, (BOARD_SIZE + 10, BOARD_SIZE - 30))

    def handle_click(self, pos):
        if self.game_over or self.board.turn != self.player_color:
            return
            
        x, y = pos
        
        # Only handle clicks on the chess board
        if x >= BOARD_SIZE:
            return
            
        col = x // SQUARE_SIZE
        row = y // SQUARE_SIZE
        square = chess.square(col, 7 - row)  # Convert to chess.Square
        
        # If a square is already selected, try to make a move
        if self.selected_square is not None:
            move = chess.Move(self.selected_square, square)
            
            # Check for promotion
            if (self.board.piece_at(self.selected_square) and 
                self.board.piece_at(self.selected_square).piece_type == chess.PAWN and
                chess.square_rank(square) in [0, 7]):
                # Default promotion to queen
                move = chess.Move(self.selected_square, square, promotion=chess.QUEEN)
            
            # Check if the move is legal
            if move in self.board.legal_moves:
                self.make_move(move)
                
                # If game not over, let model make its move
                if not self.game_over and self.model is not None:
                    self.make_model_move()
            
            # Clear selection
            self.selected_square = None
            self.valid_moves = []
            
        else:
            # Select the square if it has a piece of the player's color
            piece = self.board.piece_at(square)
            if piece and piece.color == self.player_color:
                self.selected_square = square
                # Find valid moves for this piece
                self.valid_moves = [move for move in self.board.legal_moves if move.from_square == square]

    def make_move(self, move):
        # Execute the move
        self.board.push(move)
        self.last_move = move
        
        # Update move history
        san_move = self.board.san(move)
        self.move_history.append(san_move)
        
        # Check game state
        if self.board.is_game_over():
            self.game_over = True
            if self.board.is_checkmate():
                self.status_message = "Checkmate!"
            elif self.board.is_stalemate():
                self.status_message = "Stalemate!"
            elif self.board.is_insufficient_material():
                self.status_message = "Draw by insufficient material!"
            elif self.board.is_fifty_moves():
                self.status_message = "Draw by fifty move rule!"
            elif self.board.is_repetition():
                self.status_message = "Draw by repetition!"
        else:
            self.status_message = "Model thinking..." if self.board.turn != self.player_color else "Your move"

    def make_model_move(self):
        self.status_message = "Model thinking..."
        pygame.display.flip()
        
        # Simulate thinking time
        start_time = time.time()
        
        # Get model's move
        move = get_best_move(self.board, self.model, self.device, self.temperature, self.top_k)
        
        # Ensure minimum thinking time for UX
        elapsed = time.time() - start_time
        if elapsed < self.model_thinking_time:
            time.sleep(self.model_thinking_time - elapsed)
            
        # Make the move
        self.make_move(move)

    def restart_game(self):
        self.board = chess.Board()
        self.selected_square = None
        self.valid_moves = []
        self.last_move = None
        self.game_over = False
        self.move_history = []
        self.status_message = "Your move" if self.board.turn == self.player_color else "Model thinking..."
        
        # If model starts, make the first move
        if self.board.turn != self.player_color and self.model is not None:
            self.make_model_move()

    def save_game(self):
        import chess.pgn
        
        # Create a game object
        game = chess.pgn.Game()
        
        # Set headers
        game.headers["Event"] = "Human vs ChessTransformer"
        game.headers["Site"] = "Local"
        game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
        game.headers["Round"] = "1"
        game.headers["White"] = "Human" if self.player_color == chess.WHITE else "ChessTransformer"
        game.headers["Black"] = "ChessTransformer" if self.player_color == chess.WHITE else "Human"
        
        # Set the result
        if self.game_over:
            if self.board.is_checkmate():
                game.headers["Result"] = "0-1" if self.board.turn == chess.WHITE else "1-0"
            else:
                game.headers["Result"] = "1/2-1/2"
        else:
            game.headers["Result"] = "*"
        
        # Create a new board to replay the moves
        replay_board = chess.Board()
        
        # Add all moves as variations
        node = game
        for move in self.board.move_stack:
            node = node.add_variation(move)
            replay_board.push(move)
        
        # Save to file
        os.makedirs("games", exist_ok=True)
        filename = f"games/human_vs_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pgn"
        
        with open(filename, "w") as f:
            f.write(str(game))
            
        print(f"Game saved to {filename}")
        
        return filename

    def run(self):
        running = True
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    if not self.game_over:
                        self.save_game()
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left mouse button
                        self.handle_click(event.pos)
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:  # Restart game
                        if self.game_over:
                            self.restart_game()
                    elif event.key == pygame.K_s:  # Save game
                        self.save_game()
                        self.status_message = "Game saved!"
                        self.last_status_update = time.time()
            
            # Clear the screen
            self.screen.fill(WHITE)
            
            # Draw the board and pieces
            self.draw_board()
            
            # Draw the info panel
            self.draw_info_panel()
            
            # Update the display
            pygame.display.flip()
            
            # Limit to 60 FPS
            self.clock.tick(60)
            
            # Reset status message after a delay
            if time.time() - self.last_status_update > 3 and self.status_message == "Game saved!":
                self.status_message = "Your move" if self.board.turn == self.player_color else "Model thinking..."
                self.last_status_update = time.time()
        
        pygame.quit()


@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(config: DictConfig) -> None:
    """
    Main entry point for playing against the chess transformer model.
    """
    print(OmegaConf.to_yaml(config))
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model if path provided
    model = None
    if config.model_path:
        try:
            # Create model
            model = EncoderOnlyTransformer(config.model).to(device)
            
            # Load weights
            checkpoint = torch.load(config.model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            print(f"Loaded model weights from {config.model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            if config.require_model:
                sys.exit(1)
            else:
                print("Continuing without model (pieces will move randomly)")
    elif config.require_model:
        print("No model path provided and require_model=True. Exiting.")
        sys.exit(1)
    
    # Game settings
    player_color = config.player_color
    model_thinking_time = config.model_thinking_time
    temperature = config.temperature
    top_k = config.top_k
    
    # Create and run the game
    game = ChessGame(
        model=model,
        player_color=player_color,
        model_thinking_time=model_thinking_time,
        device=device,
        temperature=temperature,
        top_k=top_k
    )
    
    print("Game started!")
    print("Controls:")
    print("  - Click on a piece to select it, then click on a destination square to move")
    print("  - Press 'R' to restart the game (when game is over)")
    print("  - Press 'S' to save the current game state to a PGN file")
    
    game.run()


if __name__ == "__main__":
    main()