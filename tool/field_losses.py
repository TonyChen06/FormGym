import math
import torch
import torch.distributed as dist
import torch.nn.functional as F


def coordinate_weights(labels, first, last):
    mask = (labels >= first) & (labels <= last)
    if not torch.all(mask.sum(-1) == 4):
        raise ValueError("Expected four coordinate tokens per example")
    # Each group gets half the global-batch loss; padding remains in the other group.
    return (mask.float() * (0.5 * len(labels) / mask.sum())
            + (~mask).float() * (0.5 * len(labels) / (~mask).sum()))


def region_targets(boxes, grid_size=24):
    edges = torch.arange(grid_size + 1, device=boxes.device, dtype=boxes.dtype) / grid_size
    boxes = boxes.clamp(0, 1)
    dx = (torch.minimum(boxes[:, 2:3], edges[1:])
          - torch.maximum(boxes[:, 0:1], edges[:-1])).clamp_min(0)
    dy = (torch.minimum(boxes[:, 3:4], edges[1:])
          - torch.maximum(boxes[:, 1:2], edges[:-1])).clamp_min(0)
    area = (dy[:, :, None] * dx[:, None, :]).flatten(1)
    if not torch.all(area.sum(-1) > 0):
        raise ValueError("Answer boxes must have positive area inside the image")
    return area / area.sum(-1, keepdim=True)


class FieldLoss(torch.nn.Module):
    def __init__(self, model, spatial_weight=0.0):
        super().__init__()
        self.model, self.spatial_weight = model, spatial_weight
        self.head = None
        if spatial_weight:
            assert model.config.vision_config.image_feature_source == [
                "spatial_avg_pool", "temporal_avg_pool"]
            # Initialize the head without changing the backbone's training RNG.
            with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
                torch.manual_seed(42)
                self.head = torch.nn.Linear(model.config.text_config.d_model, 1).to(
                    next(model.parameters()).device)

    def forward(self, ids, pixels, labels, weights, boxes):
        output = self.model(input_ids=ids, pixel_values=pixels, labels=labels)
        loss = output.loss
        if weights is not None:
            ce = F.cross_entropy(output.logits.flatten(0, 1), labels.flatten(),
                                 reduction="none").reshape_as(labels)
            loss = (ce * weights).sum() / len(labels)
        if self.head is not None:
            targets = region_targets(boxes)
            # Florence's pooled image token precedes its 24x24 spatial tokens.
            features = output.encoder_last_hidden_state[:, 1:577]
            logits = self.head(features).squeeze(-1)
            spatial = F.kl_div(logits.log_softmax(-1), targets, reduction="batchmean")
            loss = loss + self.spatial_weight * (spatial / math.log(576))
        return loss


def batches(loader):
    if not dist.is_initialized():
        yield from loader
        return
    # Pad and shuffle once globally, before splitting examples between GPUs.
    iterator = iter(loader) if dist.get_rank() == 0 else None
    for _ in range(len(loader)):
        batch = next(iterator) if iterator is not None else None
        metadata = [({k: (tuple(v.shape), v.dtype) for k, v in batch[0].items()},
                     batch[1:]) if batch is not None else None]
        dist.broadcast_object_list(metadata, src=0)
        shapes, rest = metadata[0]
        inputs = {}
        for key, (shape, dtype) in shapes.items():
            tensor = batch[0][key].cuda() if batch is not None else torch.empty(
                shape, dtype=dtype, device="cuda")
            dist.broadcast(tensor, src=0)
            inputs[key] = tensor
        yield inputs, *rest


def training_loss(objective, inputs, labels, weights, boxes):
    if not dist.is_initialized():
        return objective(inputs["input_ids"], inputs["pixel_values"], labels, weights, boxes)
    indices = list(range(len(labels)))[dist.get_rank()::dist.get_world_size()]
    scale = dist.get_world_size() * len(indices) / len(labels)
    indices = indices or [0]  # Empty ranks join collectives with zero gradient.
    loss = objective(inputs["input_ids"][indices], inputs["pixel_values"][indices],
                     labels[indices], weights[indices] if weights is not None else None,
                     boxes[indices]) * scale
    reported = loss.detach().clone()
    dist.all_reduce(reported)
    return loss + (reported / dist.get_world_size() - loss.detach())
