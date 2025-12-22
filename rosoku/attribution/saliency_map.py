import numpy as np
import torch


def saliency_map(model, dataloader, device, class_index=0):
    """
    Compute a class-conditional gradient-based saliency map.

    This function computes an input-gradient saliency map by accumulating the
    absolute gradients of the model output for a specified class with respect
    to the input EEG signals. Only samples whose labels equal ``class_index``
    contribute to the saliency map.

    The resulting saliency map has shape ``(n_channels, n_times)`` and represents
    the average absolute input gradient per sample for the target class. This
    function is used internally by :func:`rosoku.deeplearning` to provide a simple
    post-hoc interpretability analysis of trained deep-learning models.

    Parameters
    ----------
    model : torch.nn.Module
        Trained PyTorch model. The model is set to evaluation mode internally
        via ``model.eval()``. The forward pass is expected to return a tensor
        of shape ``(batch_size, n_classes)`` containing class-wise scores
        or logits.

    dataloader : torch.utils.data.DataLoader
        DataLoader yielding batches of ``(data, target)`` pairs.
        ``data`` must have shape ``(batch_size, n_channels, n_times)``, and
        ``target`` must contain integer class labels.

    device : {"cpu", "cuda"}
        Device on which gradients and model evaluation are performed.

    class_index : int, optional
        Index of the target class for which the saliency map is computed.
        Only samples with ``target == class_index`` are used (default: 0).

    Returns
    -------
    saliency : numpy.ndarray, shape (n_channels, n_times)
        Class-conditional saliency map obtained by averaging the absolute
        input gradients over all samples belonging to ``class_index``.

    Raises
    ------
    ValueError
        If no samples corresponding to ``class_index`` are found in the dataset.
    ValueError
        If ``data.grad`` is ``None`` after backpropagation.

    Notes
    -----
    - Gradients are computed with respect to the input ``data`` by enabling
      ``requires_grad=True`` and backpropagating the averaged model output
      for the target class.
    - For each batch, gradients are summed over samples in the batch and then
      accumulated across batches.
    - This implementation assumes that at least one sample of the target class
      is present in each batch where gradients are computed. If a batch contains
      no samples of ``class_index``, the behavior depends on the model and may
      result in runtime errors.

    See Also
    --------
    rosoku.deeplearning :
        Deep-learning pipeline that optionally computes saliency maps during
        test-time evaluation.

    Examples
    --------
    Compute a saliency map for class 1::

        sal = saliency_map(
            model=model,
            dataloader=test_loader,
            device="cuda",
            class_index=1,
        )
    """
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

        # saliency should be scaled with cnt, instead of len(dataloader.dataset)?
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

    if cnt == 0:
        raise ValueError(f"No samples found for class_index={class_index}")

    saliency = saliency / cnt

    saliency = saliency.cpu().numpy()

    return saliency


def saliency_temporal(saliency):
    """
    Compute a normalized temporal saliency profile from a saliency map.

    This function collapses a channel–time saliency map into a one-dimensional
    temporal profile by summing saliency values across channels at each time
    point. The resulting temporal saliency is then normalized so that its total
    sum equals 100, allowing interpretation as a percentage contribution over
    time.

    Parameters
    ----------
    saliency : array-like, shape (n_channels, n_times)
        Channel–time saliency map, typically returned by
        :func:`saliency_map`. Each entry represents the importance of a specific
        channel and time point.

    Returns
    -------
    saliency_temporal : numpy.ndarray, shape (n_times,)
        One-dimensional temporal saliency profile, normalized such that the
        sum over time equals 100.

    Notes
    -----
    - The temporal saliency is computed as::

          saliency_temporal[t] = sum_c saliency[c, t]

      followed by normalization::

          saliency_temporal = saliency_temporal / sum_t saliency_temporal[t] * 100

    - This normalization allows the temporal profile to be interpreted as the
      relative contribution (in percent) of each time point to the overall
      saliency.

    Examples
    --------
    Compute a temporal saliency profile from a saliency map::

        sal = saliency_map(model, dataloader, device="cuda", class_index=0)
        sal_t = saliency_temporal(sal)
    """
    saliency = np.array(saliency)
    saliency_temporal = saliency.sum(axis=0)
    saliency_temporal = saliency_temporal / saliency_temporal.sum() * 100

    return saliency_temporal


def saliency_spatial(saliency):
    """
    Compute a normalized spatial saliency profile from a saliency map.

    This function collapses a channel–time saliency map into a one-dimensional
    spatial profile by summing saliency values across time for each channel.
    The resulting spatial saliency is then normalized so that its total sum
    equals 100, allowing interpretation as a percentage contribution across
    channels.

    Parameters
    ----------
    saliency : array-like, shape (n_channels, n_times)
        Channel–time saliency map, typically returned by
        :func:`saliency_map`. Each entry represents the importance of a specific
        channel and time point.

    Returns
    -------
    saliency_spatial : numpy.ndarray, shape (n_channels,)
        One-dimensional spatial saliency profile, normalized such that the
        sum over channels equals 100.

    Notes
    -----
    - The spatial saliency is computed as::

          saliency_spatial[c] = sum_t saliency[c, t]

      followed by normalization::

          saliency_spatial = saliency_spatial / sum_c saliency_spatial[c] * 100

    - This normalization allows the spatial profile to be interpreted as the
      relative contribution (in percent) of each channel to the overall
      saliency.

    Examples
    --------
    Compute a spatial saliency profile from a saliency map::

        sal = saliency_map(model, dataloader, device="cuda", class_index=0)
        sal_s = saliency_spatial(sal)
    """
    saliency = np.array(saliency)
    saliency_spatial = saliency.sum(axis=1)
    saliency_spatial = saliency_spatial / saliency_spatial.sum() * 100

    return saliency_spatial
