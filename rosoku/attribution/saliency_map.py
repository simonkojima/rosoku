import torch


def saliency_map(model, dataloader, device, class_index=1):
    model.eval()

    for batch in iter(dataloader):
        data, target = batch

        _, n_chans, n_times = data.shape

        break

    saliency = torch.zeros((n_chans, n_times), device=device)

    cnt = 0
    for batch in iter(dataloader):
        data, target = batch

        mask = target == class_index
        data = data[mask]

        # saliency should be scaled with cnt, istead of len(dataloader.dataset)?
        cnt += data.shape[0]

        data = data.to(device)
        data.requires_grad = True

        output = model(data)
        output = torch.sum(output, dim=0) / output.shape[0]

        output_cls = output[class_index]
        output_cls.backward()

        # Assuming data.grad is not None and has the same shape as data
        if data.grad is not None:
            saliency += data.grad.abs().sum(dim=0)  # Sum over the batch
        else:
            raise ValueError("data.grad is None")

    saliency = saliency / len(dataloader.dataset)

    saliency = saliency.cpu().numpy()

    return saliency


def saliency_temporal(saliency):
    saliency_temporal = saliency.sum(axis=0)
    saliency_temporal = saliency_temporal / saliency_temporal.sum() * 100

    return saliency_temporal


def saliency_spatial(saliency):
    saliency_spatial = saliency.sum(axis=1)
    saliency_spatial = saliency_spatial / saliency_spatial.sum() * 100

    return saliency_spatial
