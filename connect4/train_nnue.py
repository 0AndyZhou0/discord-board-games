from pathlib import Path

import numpy as np

from nnue.connect4_nnue_wrapper import Connect4NNUEWrapper

if __name__ == "__main__":
    path = Path(__file__).parent / "nnue"
    bad_dataset = np.load(path / "data" / "bad_dataset_fixed.npy", allow_pickle=True)

    nnue_wrapper = Connect4NNUEWrapper()
    nnue_wrapper.load_model(path / "models" / "best.pt")

    large_batch_size = 100000
    large_batch_number = len(bad_dataset) // large_batch_size
    for i in range(large_batch_number):
        print(f"Large batch: {i+1}/{large_batch_number}")
        batch = np.random.default_rng().choice(len(bad_dataset), size=large_batch_size)
        batch = [bad_dataset[i] for i in batch]
        nnue_wrapper.train(batch, epochs=10, batch_size=256)
        # nnue_wrapper.train(first_8_ply, epochs=10, batch_size=128)
        nnue_wrapper.save_model(path / "models" / f"best{i+1}.pt")