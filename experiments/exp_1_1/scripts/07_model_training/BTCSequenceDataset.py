import torch
from torch.utils.data import Dataset

class BTCSequenceDataset(Dataset):
    def __init__(self, x, y, seq_size):
        self.seq_size = seq_size
        self.features = x.values
        self.targets = y.values.reshape(-1)

    # Wie viele Sequenzen können aus den Daten gebaut werden
    def __len__(self):
        return len(self.features) - self.seq_size + 1

    def __getitem__(self, idx):
        # 48h-Fenster
        x_window = self.features[idx : idx + self.seq_size]
        #Vorhersage target für letzten Datensatz im Fenster
        y_value = self.targets[idx + self.seq_size -1]

        x_tensor = torch.tensor(x_window, dtype=torch.float32)
        y_tensor = torch.tensor(y_value, dtype=torch.long)

        return x_tensor, y_tensor
