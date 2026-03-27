# ---------------------------------------------------------------
# DAPCN-SSL-DLV3Plus Base Configuration
# DeepLabV3+ variant of DAPCN-SSL
#
# Uses DAPCN_SSL_DLV3Plus module (from dapcn_ssl_dlv3plus.py),
# which is a development clone of DAPCN_SSL dedicated to
# DeepLabV3+ + ResNet-101 experiments.
# ---------------------------------------------------------------

uda = dict(
    type='DAPCN_SSL_DLV3Plus',
    # --- Self-training (EMA teacher) ---
    alpha=0.999,
    pseudo_threshold=0.968,
    pseudo_weight_ignore_top=0,
    pseudo_weight_ignore_bottom=0,
    mix='class',
    blur=True,
    color_jitter_strength=0.2,
    color_jitter_probability=0.2,
    debug_img_interval=1000,
    # --- SSL-specific ---
    pseudo_label_warmup_iters=1000,
    # --- Prototype-based pseudo-label correction ---
    proto_correction=True,
    proto_correction_alpha=0.5,
    proto_correction_start_iter=1000,
    # --- DAPCN loss weights ---
    boundary_lambda=0.3,
    proto_lambda=0.1,
    # --- Boundary configuration ---
    boundary_loss_mode='affinity',
    boundary_mode='sobel',
    apply_boundary_on_target=True,
    apply_proto_on_target=True,
    hybrid_binary_weight=0.5,
    ignore_index=255,
    # --- DynamicAnchorModule ---
    dynamic_anchor=dict(
        type='DynamicAnchorModule',
        max_groups=96,
        temperature=0.1,
        num_iters=3,
        init_method='xavier',
        min_quality=0.1,
        use_quality_gate=True,
        use_mask_predictor=False,
        ema_decay=0.0,
    ),
    # --- DAPGLoss ---
    dapg_loss=dict(
        type='DAPGLoss',
        margin=0.3,
        lambda_inter=0.5,
        lambda_quality=0.1,
        loss_weight=1.0,
    ),
    # --- AffinityBoundaryLoss ---
    affinity_loss=dict(
        type='AffinityBoundaryLoss',
        temperature=0.5,
        scale=2,
        num_neighbors=4,
        ignore_index=255,
        loss_weight=1.0,
    ),
)
use_ddp_wrapper = True
