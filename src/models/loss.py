import torch

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0,1), target_batch.flatten())
    return loss


def calc_loss_loader(data_loader, model, device, num_batches = None):
    total_loss = 0
    if len(data_loader) == 0:
        return float('nan')

    elif num_batches is None:
        num_batches = len(data_loader)

    else :
        num_batches = min(len(data_loader), num_batches)


    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch=input_batch, target_batch=target_batch, model=model, device=device)
            total_loss = total_loss + loss.item()
        else:
            break

    return total_loss / num_batches