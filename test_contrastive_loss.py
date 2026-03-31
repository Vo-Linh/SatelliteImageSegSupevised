"""Standalone script to run prototype_contrastive_loss with dummy data."""

import torch
import torch.nn.functional as F


def prototype_contrastive_loss(features,
                               prototypes,
                               labels,
                               num_classes,
                               num_prototypes_per_class=1,
                               temperature=0.07,
                               ignore_index=255):
    """Prototype-based contrastive loss (InfoNCE) with multi-prototype support."""
    K = num_prototypes_per_class

    valid = labels != ignore_index
    if valid.sum() == 0:
        return torch.tensor(0.0, device=features.device, requires_grad=True)

    feats = F.normalize(features[valid], dim=1)
    feats = torch.nan_to_num(feats, nan=0.0)
    labs = labels[valid]
    proto_norm = F.normalize(prototypes, dim=1)
    proto_norm = torch.nan_to_num(proto_norm, nan=0.0)

    sim = torch.mm(feats, proto_norm.t()) / temperature

    if K == 1:
        log_probs = F.log_softmax(sim, dim=1)
        loss = -log_probs[torch.arange(len(labs), device=labs.device), labs]
    else:
        sim_3d = sim.view(-1, num_classes, K)
        sim_class, _ = sim_3d.max(dim=2)
        log_probs = F.log_softmax(sim_class, dim=1)
        loss = -log_probs[torch.arange(len(labs), device=labs.device), labs]

    return loss.mean()


def main():
    num_classes = 19
    feature_dim = 256
    num_pixels = 5000
    temperature = 0.07
    ignore_index = 255

    print("=== Single prototype per class (K=1) ===")
    features = torch.randn(num_pixels, feature_dim)
    prototypes = torch.randn(num_classes, feature_dim)
    labels = torch.randint(0, num_classes, (num_pixels,))

    loss = prototype_contrastive_loss(
        features, prototypes, labels,
        num_classes=num_classes,
        num_prototypes_per_class=1,
        temperature=temperature,
        ignore_index=ignore_index,
    )
    print(f"Loss: {loss.item():.4f}")

    # --- Multi-prototype per class (K=3) ---
    K = 3
    print(f"\n=== Multi-prototype per class (K={K}) ===")
    prototypes_multi = torch.randn(num_classes * K, feature_dim)
    loss_multi = prototype_contrastive_loss(
        features, prototypes_multi, labels,
        num_classes=num_classes,
        num_prototypes_per_class=K,
        temperature=temperature,
        ignore_index=ignore_index,
    )
    print(f"Loss: {loss_multi.item():.4f}")

    # --- With some ignored labels ---
    print("\n=== With ignored pixels (ignore_index=255) ===")
    labels_with_ignore = labels.clone()
    labels_with_ignore[:1000] = ignore_index  # 20% ignored
    loss_ignore = prototype_contrastive_loss(
        features, prototypes, labels_with_ignore,
        num_classes=num_classes,
        num_prototypes_per_class=1,
        temperature=temperature,
        ignore_index=ignore_index,
    )
    print(f"Loss (with 20% ignored): {loss_ignore.item():.4f}")

    # --- Verify gradient flow ---
    print("\n=== Gradient check ===")
    feats_grad = torch.randn(num_pixels, feature_dim, requires_grad=True)
    loss_grad = prototype_contrastive_loss(
        feats_grad, prototypes, labels,
        num_classes=num_classes,
        num_prototypes_per_class=1,
        temperature=temperature,
    )
    loss_grad.backward()
    print(f"Loss: {loss_grad.item():.4f}")
    print(f"Gradient norm: {feats_grad.grad.norm().item():.4f}")
    print(f"Gradient has NaN: {torch.isnan(feats_grad.grad).any().item()}")


if __name__ == "__main__":
    main()
