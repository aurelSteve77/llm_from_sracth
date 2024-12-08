from random import shuffle

import tiktoken
import torch

from torch.utils.data import Dataset, DataLoader


class GPTDatasetV1(Dataset):

    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt, allowed_special={'<|endoftext|>'})
        print(len(token_ids))

        for i in range(0, len(token_ids)- max_length, stride):
            input_chunk = token_ids[i:i+max_length]
            target_chunk = token_ids[i+1:i+max_length+1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))


    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]



def create_data_loader_v1(txt, batch_size = 4, max_length=128, stride=64, shuffle=True, drop_last=True, num_workers=0):

    tokenizer = tiktoken.get_encoding('gpt2')
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return data_loader

if __name__ == '__main__':

    with open('../../data/raw/kari_hillsong.txt', 'r') as f:
        raw_test = f.read()

    max_length = 4
    data_loader = create_data_loader_v1(raw_test, batch_size=8, max_length=max_length, stride=max_length, shuffle=False)
    data_iter = iter(data_loader)
    inputs, targets = next(data_iter)
    print(f'Tokens IDS:\n {inputs}')
    print(f'Shape of input tensor: {inputs.shape}')

    tokenizer = tiktoken.get_encoding('gpt2')
    vocab_size = tokenizer.n_vocab + 1
    embedding_dim = 256

    embedding_layer = torch.nn.Embedding(vocab_size, embedding_dim)
    embedded_input = embedding_layer(inputs)
    print(f'Embedded input shape:\n {embedded_input.shape}')


    context_length = max_length
    pos_embedding_layer = torch.nn.Embedding(context_length, embedding_dim)
    pos_embedded_input = pos_embedding_layer(torch.arange(context_length))
    print(f'Positional embedded input shape:\n {pos_embedded_input.shape}')

    input_embedded = embedded_input + pos_embedded_input
    print(f'Input embedded shape:\n {input_embedded.shape}')
