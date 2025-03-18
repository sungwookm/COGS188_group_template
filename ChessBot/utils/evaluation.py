import chess
import chess.engine
import math

class EloEvaluator:
    def __init__(self, stockfish_path="stockfish", initial_elo=1200, stockfish_elo=2500, k_factor=32):
        self.engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        self.base_elo = initial_elo
        self.stockfish_elo = stockfish_elo
        self.k_factor = k_factor

    def expected_score(self, player_elo, opponent_elo):
        return 1 / (1 + 10 ** ((opponent_elo - player_elo) / 400))

    def update_elo(self, result, opponent_elo):
        expected = self.expected_score(self.base_elo, opponent_elo)
        self.base_elo += self.k_factor * (result - expected)
        return self.base_elo

    def play_game(self, ai_bot, stockfish_level=8):
        board = chess.Board()
        self.engine.configure({"Skill Level": stockfish_level})

        while not board.is_game_over():
            if board.turn == chess.WHITE:
                move = ai_bot.get_move(board)
            else:
                move = self.engine.play(board, chess.engine.Limit(time=0.1)).move
            board.push(move)

        result = board.result()
        if result == "1-0":
            return 1
        elif result == "0-1":
            return 0
        else:
            return 0.5

    def evaluate_ai(self, ai_bot, num_games=10):
        wins, losses, draws = 0, 0, 0

        for _ in range(num_games):
            result = self.play_game(ai_bot)
            new_elo = self.update_elo(result, self.stockfish_elo)

            if result == 1:
                wins += 1
            elif result == 0.5:
                draws += 1
            else:
                losses += 1

            print(f"Game {_+1}: {'Win' if result == 1 else 'Draw' if result == 0.5 else 'Loss'} | Updated Elo: {new_elo}")

        print(f"\nFinal AI Elo after {num_games} games: {self.base_elo}")
        self.engine.quit()