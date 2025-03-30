from tqdm import tqdm
import time
import torch

from src.eval.evaluate import AverageMeter, accuracy


def train_fn(model, optimizer, criterion, loader, device):
    """
    Training method
    :param model: model to train
    :param optimizer: optimization algorithm
    :criterion: loss function
    :param loader: data loader for either training or testing set
    :param device: torch device
    :return: (accuracy, loss) on the data
    """
    model.train()
    score = AverageMeter()
    losses = AverageMeter()
    batch_time = AverageMeter()

    end = time.time()

    with tqdm(loader, desc="Training", unit="batch") as t:
        for images, labels in t:
            # Data to device
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # Forward pass
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():  # AMP support to speed up training
                logits = model(images)
                loss = criterion(logits, labels)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)  # Gradient clipping
            optimizer.step()

            # Metrics
            acc = accuracy(logits, labels)
            batch_size = images.size(0)
            losses.update(loss.item(), batch_size)
            score.update(acc.item(), batch_size)
            batch_time.update(time.time() - end)
            end = time.time()

            # Progress bar update
            t.set_postfix({
                'Loss': f'{losses.avg:.4f}',
                'Acc': f'{score.avg:.2f}%',
                'Time': f'{batch_time.avg:.3f}s/b'
            })

    print(f'\n* Train - Loss: {losses.avg:.4f} | Acc: {score.avg:.2f}%')
    return score.avg, losses.avg
