import h5py
import torch
from torch.utils.data import Dataset


class ChessDataset(Dataset):
    def __init__(self, file_path: str):
        self.file_path = file_path
        
        # Load the dataset
        with h5py.File(self.file_path, 'r') as f:
            self.board_positions = f['encoded']['board_position'][:]
            self.turns = f['encoded']['turn'][:]
            self.white_ks_castling = f['encoded']['white_kingside_castling_rights'][:]
            self.white_qs_castling = f['encoded']['white_queenside_castling_rights'][:]
            self.black_ks_castling = f['encoded']['black_kingside_castling_rights'][:]
            self.black_qs_castling = f['encoded']['black_queenside_castling_rights'][:]
            self.moves = f['encoded']['moves'][:]
            self.lengths = f['encoded']['length'][:]

    def __len__(self):
        return len(self.moves)

    def __getitem__(self, idx: int) -> dict:
        return {
            "board_positions": torch.tensor(self.board_positions[idx], dtype=torch.long),
            "turns": torch.tensor(self.turns[idx], dtype=torch.long),
            "white_kingside_castling_rights": torch.tensor(self.white_ks_castling[idx], dtype=torch.long),
            "white_queenside_castling_rights": torch.tensor(self.white_qs_castling[idx], dtype=torch.long),
            "black_kingside_castling_rights": torch.tensor(self.black_ks_castling[idx], dtype=torch.long),
            "black_queenside_castling_rights": torch.tensor(self.black_qs_castling[idx], dtype=torch.long),
            "moves": torch.tensor(self.moves[idx], dtype=torch.long),
            "lengths": torch.tensor(self.lengths[idx], dtype=torch.long)
        }
