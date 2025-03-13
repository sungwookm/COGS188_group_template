import pygame
import chess
import chess.pgn
import chess.svg
import io
import time
import cairosvg


# Make sure to download cairosvg using: conda install -c conda-forge cairo

def svg_to_surface(svg_data, size):
    # Convert the SVG string to PNG bytes using cairosvg
    png_data = cairosvg.svg2png(bytestring=svg_data.encode("utf-8"), output_width=size, output_height=size)
    # Create a BytesIO stream from the PNG data and load it into a PyGame image
    return pygame.image.load(io.BytesIO(png_data))

def visualize_game_from_pgn(pgn_text, delay=1.0, board_size=500):
    # Parse the PGN string
    pgn_io = io.StringIO(pgn_text)
    game = chess.pgn.read_game(pgn_io)
    board = game.board()

    # Initialize PyGame
    pygame.init()
    window_size = (board_size, board_size)
    screen = pygame.display.set_mode(window_size)
    pygame.display.set_caption("Chess Game Visualization")
    clock = pygame.time.Clock()

    def display_board(current_board):
        # Render the board as an SVG and convert to a PyGame surface
        svg_data = chess.svg.board(board=current_board, size=board_size)
        surface = svg_to_surface(svg_data, board_size)
        screen.blit(surface, (0, 0))
        pygame.display.flip()

    # Display initial board state
    display_board(board)
    time.sleep(delay)

    # Replay moves from the game
    for move in game.mainline_moves():
        board.push(move)
        # Check for window close events during replay
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
        display_board(board)
        time.sleep(delay)
        clock.tick(60)

    # Wait until the user closes the window
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        clock.tick(60)
    pygame.quit()

def main():
    # Example PGN of a game.
    pgn_game = """
[Event "Example Game"]
[Site "Local"]
[Date "2025.03.12"]
[Round "1"]
[White "Alice"]
[Black "Bob"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 8. c3 O-O 9. h3 1-0
    """
    visualize_game_from_pgn(pgn_game, delay=1.0, board_size=500)

if __name__ == "__main__":
    main()
