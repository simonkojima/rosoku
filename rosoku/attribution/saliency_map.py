import torch


def saliency_map(model, dataloader, device, class_index=1):
    model.eval()

    for batch in iter(dataloader):
        data, target = batch

        _, n_chans, n_times = data.shape

        break

    saliency = torch.zeros((n_chans, n_times), device=device)

    for batch in iter(dataloader):
        data, target = batch
        mask = target == class_index
        data = data[mask]

        data = data.to(device)
        data.requires_grad = True

        output = model(data)
        output = torch.sum(output, dim=0) / output.shape[0]

        output_cls = output[class_index]
        # If output is not a scalar, consider using torch.sum(output).backward()
        output_cls.backward()
        # Assuming data.grad is not None and has the same shape as data
        if data.grad is not None:
            saliency += data.grad.abs().sum(dim=0)  # Sum over the batch
        else:
            raise ValueError("data.grad is None")

    saliency = saliency / len(dataloader.dataset)

    saliency = saliency.cpu().numpy()

    return saliency
