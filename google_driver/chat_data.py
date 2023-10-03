
from torch.utils.data import Dataset
import json

class ChatData(Dataset):
    def __init__(self, path: str, tokenizer):
        self.data = json.load(open(path, "r"))

        self.X = []
        for i in self.data:
            for j in i['dialog']:
                self.X.append(j['text'])

        for idx, i in enumerate(self.X):
            try:
                self.X[idx] = "<startofstring> " + i + " <bot>: " + self.X[idx + 1] + " <endofstring>"
            except:
                break

        for i in self.data:
            for j in i['dialog']:
                self.X.append(j['text'])

        total_samples = len(self.X)  # Calculate the total number of samples
        print("total_samples", total_samples)
        # define samples amount
        self.X = self.X[:500]
        print("Here is the self.X[0] i wanna check:")
        print(self.X[0])

        self.X_encoded = tokenizer(self.X, return_tensors="pt", max_length=30, padding="max_length", truncation=True)
        self.input_ids = self.X_encoded['input_ids']
        self.attention_mask = self.X_encoded['attention_mask']

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attention_mask[idx]
