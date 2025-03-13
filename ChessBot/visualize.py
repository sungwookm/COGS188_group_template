import pygame
import chess
import chess.pgn
import chess.svg
import io
import time
import cairosvg
import pickle
import sys

# Make sure to download cairosvg using: conda install -c conda-forge cairo

def svg_to_surface(svg_data, size):
    # Convert the SVG string to PNG bytes using cairosvg
    png_data = cairosvg.svg2png(bytestring=svg_data.encode("utf-8"), output_width=size, output_height=size)
    # Create a BytesIO stream from the PNG data and load it into a PyGame image
    return pygame.image.load(io.BytesIO(png_data))

def visualize_game_from_pickle(pkl_filename, delay=1.0, board_size=500):
    with open(pkl_filename, "rb") as f:
        pgn_string = pickle.load(f)
    pgn_io = io.StringIO(pgn_string)
    game = chess.pgn.read_game(pgn_io)
    board = game.board()

    # Initialize PyGame
    pygame.init()
    screen = pygame.display.set_mode((board_size, board_size))
    pygame.display.set_caption("Chess Game Visualization")
    clock = pygame.time.Clock()

    def display_board(b):
        # Render the board as an SVG and convert to a PyGame surface
        svg_data = chess.svg.board(board=b, size=board_size)
        surface = svg_to_surface(svg_data, board_size)
        screen.blit(surface, (0, 0))
        pygame.display.flip()
        
    # Display initial board state
    display_board(board)
    time.sleep(delay)
    for move in game.mainline_moves():
        board.push(move)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
        display_board(board)
        time.sleep(delay)
        clock.tick(60)
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        clock.tick(60)
    pygame.quit()

def main():
    # Change this variable to the desired pickle file path.
    DEFAULT_PKL_FILE = "checkpoints/game_history/game_1_pgn.pkl"
    
    if len(sys.argv) > 1:
        pkl_file = sys.argv[1]
    else:
        pkl_file = DEFAULT_PKL_FILE

    print(f"Visualizing game from: {pkl_file}")
    visualize_game_from_pickle(pkl_file)

if __name__ == "__main__":
    main()
