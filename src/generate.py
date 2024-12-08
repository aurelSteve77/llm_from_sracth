import torch
from torch import nn

from src.models.gpt import GPTModel


def generate_text(model, idx, max_new_tokens, context_size):

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, : ]
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)

    return idx

def text_to_token_ids(text, tokenizer):

    # encode text
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})

    # convert to tensor
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) # add batch dimension

    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):

    # flatten tensor
    token_ids = token_ids.squeeze(0)

    # decode token ids
    decoded_text = tokenizer.decode(token_ids.tolist())

    return decoded_text


def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):

    for _ in range(max_new_tokens):

        idx_cond = idx[:, -context_size:]

        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[ :, -1, :]

        if top_k:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(
                logits < min_val,
                torch.tensor(float('-inf')).to(logits.device),
                logits
            )

        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        if idx_next == eos_id:
            break

        idx = torch.cat((idx, idx_next), dim=1)

    return idx



if __name__ == '__main__':

    import tiktoken

    GPT_CONFIG_124M = {
        "vocab_size": 50257,  # Vocabulary size
        "context_length": 256,  # Context length
        "emb_dim": 768,  # Embedding dimension
        "n_heads": 12,  # Number of attention heads
        "n_layers": 12,  # Number of layers
        "drop_rate": 0.1,  # Dropout rate
        "qkv_bias": False  # Query-Key-Value bias
    }

    tokenizer = tiktoken.get_encoding('gpt2')

    text = "La vie est belle et vaut"
    encoded_tensor = text_to_token_ids(text, tokenizer)
    model = GPTModel(GPT_CONFIG_124M)
    model.eval()

    out = generate_text(model, encoded_tensor, 7, context_size=GPT_CONFIG_124M["context_length"])
    print(f"Output: ", out)
    print("Output length", len(out[0]))
    #res = model(batch)

    print(token_ids_to_text(out, tokenizer))

    #print(res.size())

    #total_params = sum(p.numel() for p in model.parameters())
    #print(f"Total number of parameters: {total_params:,}")

