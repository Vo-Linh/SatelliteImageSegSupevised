# UNetFormer DAPCN Workflow Diagram
# Config: configs/unetformer/unetformer_dapcn_after_fusion_cityscapes.py
# Generated for visualization

from graphviz import Digraph
import os

def create_unetformer_dapcn_workflow():
    """Create a comprehensive workflow diagram for UNetFormer with DAPCN."""
    
    dot = Digraph(comment='UNetFormer DAPCN Workflow', format='png')
    dot.attr(rankdir='TB', size='20,20', dpi='150')
    dot.attr('node', shape='box', style='rounded,filled', fontname='Helvetica', fontsize='10')
    dot.attr('edge', fontname='Helvetica', fontsize='9')
    
    # Color scheme
    colors = {
        'input': '#E8F4FD',      # Light blue
        'backbone': '#D4EDDA',    # Light green
        'decoder': '#FFF3CD',     # Light yellow
        'dapcn': '#F8D7DA',       # Light red
        'loss': '#E2D4F0',        # Light purple
        'output': '#D1ECF1',      # Cyan
    }
    
    # ==================== INPUT ====================
    with dot.subgraph(name='cluster_input') as c:
        c.attr(label='Input', style='dashed', color='blue')
        c.node('input_img', 'Input Image\n(B, 3, 512, 512)', fillcolor=colors['input'])
    
    # ==================== BACKBONE (UNetFormer) ====================
    with dot.subgraph(name='cluster_backbone') as c:
        c.attr(label='UNetFormer Backbone\n(Hybrid CNN-Transformer)', style='dashed', color='green')
        
        # Initial embedding
        c.node('patch_embed', 'Patch Embedding\nConv4x4 + LayerNorm\n(B, H/4×W/4, 64)', 
               fillcolor=colors['backbone'])
        
        # Encoder stages
        c.node('stage0', 'Stage 0: 2 Transformer Blocks\nWindow Attention + Global Tokens\n(B, H/4×W/4, 64)', 
               fillcolor=colors['backbone'])
        c.node('stage1', 'Stage 1: 2 Blocks + PatchMerge\n(B, H/8×W/8, 128)', 
               fillcolor=colors['backbone'])
        c.node('stage2', 'Stage 2: 6 Blocks + PatchMerge\n(B, H/16×W/16, 256)', 
               fillcolor=colors['backbone'])
        c.node('stage3', 'Stage 3: 2 Blocks + PatchMerge\n(B, H/32×W/32, 512)', 
               fillcolor=colors['backbone'])
    
    # ==================== DECODER (Feature Pyramid) ====================
    with dot.subgraph(name='cluster_decoder') as c:
        c.attr(label='Feature Pyramid Decoder', style='dashed', color='orange')
        
        c.node('proj0', 'Proj: 64→256\n(B, H/8×W/8, 256)', fillcolor=colors['decoder'])
        c.node('proj1', 'Proj: 128→256\n(B, H/16×W/16, 256)', fillcolor=colors['decoder'])
        c.node('proj2', 'Proj: 256→256\n(B, H/32×W/32, 256)', fillcolor=colors['decoder'])
        c.node('proj3', 'Proj: 512→256\n(B, H/32×W/32, 256)', fillcolor=colors['decoder'])
        
        c.node('fuse3', 'Upsample + Concat + Conv\n(B, H/32×W/32, 256)', fillcolor=colors['decoder'])
        c.node('fuse2', 'Upsample + Concat + Conv\n(B, H/16×W/16, 256)', fillcolor=colors['decoder'])
        c.node('fuse1', 'Upsample + Concat + Conv\n(B, H/8×W/8, 256)', fillcolor=colors['decoder'])
        
        c.node('final_conv', 'Final Conv\n(B, 256, H/8, W/8)', fillcolor=colors['decoder'])
    
    # ==================== DAPCN MODULE (After Fusion) ====================
    with dot.subgraph(name='cluster_dapcn') as c:
        c.attr(label='DAPCN Module (After Fusion)\nDynamic Anchor Prototype-Guided Learning', 
                style='dashed', color='red')
        
        c.node('da_feat', 'Feature Selection\nC=256 (fused)', fillcolor=colors['dapcn'])
        
        c.node('dynamic_anchor', 'DynamicAnchorModule\nMax Groups: 64\nTemperature: 0.1\nIters: 3', 
               fillcolor=colors['dapcn'], shape='component')
        
        with dot.subgraph(name='cluster_em') as em:
            em.attr(label='EM Refinement Loop', style='dotted')
            em.node('e_step', 'E-step: Soft Assignment\nsim = feats @ proto.T / τ\nassign = softmax(sim)', 
                   fillcolor=colors['dapcn'])
            em.node('m_step', 'M-step: Update Prototypes\nproto = assign.T @ feats / sizes\nnormalize', 
                   fillcolor=colors['dapcn'])
        
        c.node('quality_gate', 'Quality Gate\nFilter: quality ≥ 0.1\nMin 1 prototype kept', 
               fillcolor=colors['dapcn'])
        
        c.node('prototypes', 'Learnable Prototypes\n(K, 256) nn.Parameter\nPersistent across batches', 
               fillcolor=colors['dapcn'])
    
    # ==================== OUTPUT HEAD ====================
    with dot.subgraph(name='cluster_head') as c:
        c.attr(label='Segmentation Head', style='dashed', color='cyan')
        c.node('cls_seg', 'Classification\nConv + Upsample\n(B, 19, H, W)', fillcolor=colors['output'])
        c.node('seg_logits', 'Segmentation Logits\n19 Classes (Cityscapes)', fillcolor=colors['output'])
    
    # ==================== LOSSES ====================
    with dot.subgraph(name='cluster_losses') as c:
        c.attr(label='Multi-Task Losses', style='dashed', color='purple')
        
        c.node('loss_ce', 'CrossEntropy Loss\nλ = 1.0\nStandard Segmentation', fillcolor=colors['loss'])
        c.node('loss_boundary', 'Boundary Loss\nλ = 0.3\nSobel Edge Detection\nBinary BCE', 
               fillcolor=colors['loss'])
        c.node('loss_dapg', 'DAPGLoss\nλ = 0.1\nIntra + Inter + Quality\nMargin: 0.3', 
               fillcolor=colors['loss'])
        c.node('loss_contrastive', 'Contrastive Loss\nλ = 0.1\nInfoNCE\nTemp: 0.07', 
               fillcolor=colors['loss'])
        
        c.node('total_loss', 'Total Loss\nL = Lce + 0.3×Lboundary + 0.1×Ldapg + 0.1×Lcontrastive', 
               fillcolor=colors['loss'], shape='ellipse', style='bold,filled')
    
    # ==================== EDGES ====================
    # Input to backbone
    dot.edge('input_img', 'patch_embed')
    dot.edge('patch_embed', 'stage0')
    
    # Encoder stages
    dot.edge('stage0', 'stage1')
    dot.edge('stage1', 'stage2')
    dot.edge('stage2', 'stage3')
    
    # Encoder to decoder projections
    dot.edge('stage0', 'proj0')
    dot.edge('stage1', 'proj1')
    dot.edge('stage2', 'proj2')
    dot.edge('stage3', 'proj3')
    
    # Decoder fusion
    dot.edge('proj3', 'fuse3')
    dot.edge('proj2', 'fuse2')
    dot.edge('fuse3', 'fuse2')
    dot.edge('proj1', 'fuse1')
    dot.edge('fuse2', 'fuse1')
    dot.edge('proj0', 'fuse1')
    
    # Final conv
    dot.edge('fuse1', 'final_conv')
    
    # DAPCN path (after fusion)
    dot.edge('final_conv', 'da_feat')
    dot.edge('da_feat', 'dynamic_anchor')
    dot.edge('dynamic_anchor', 'e_step')
    dot.edge('e_step', 'm_step')
    dot.edge('m_step', 'quality_gate')
    dot.edge('prototypes', 'dynamic_anchor', style='dashed', label='persistent')
    
    # Classification head
    dot.edge('final_conv', 'cls_seg')
    dot.edge('cls_seg', 'seg_logits')
    
    # Losses
    dot.edge('seg_logits', 'loss_ce')
    dot.edge('seg_logits', 'loss_boundary')
    dot.edge('quality_gate', 'loss_dapg')
    dot.edge('quality_gate', 'loss_contrastive', style='dashed')
    
    dot.edge('loss_ce', 'total_loss')
    dot.edge('loss_boundary', 'total_loss')
    dot.edge('loss_dapg', 'total_loss')
    dot.edge('loss_contrastive', 'total_loss')
    
    # Legend
    with dot.subgraph(name='cluster_legend') as legend:
        legend.attr(label='Legend', style='dotted')
        legend.node('legend_input', 'Input', fillcolor=colors['input'])
        legend.node('legend_backbone', 'Backbone', fillcolor=colors['backbone'])
        legend.node('legend_decoder', 'Decoder', fillcolor=colors['decoder'])
        legend.node('legend_dapcn', 'DAPCN', fillcolor=colors['dapcn'])
        legend.node('legend_output', 'Output', fillcolor=colors['output'])
        legend.node('legend_loss', 'Loss', fillcolor=colors['loss'])
    
    return dot


def create_simplified_workflow():
    """Create a simplified high-level workflow diagram."""
    
    dot = Digraph(comment='UNetFormer DAPCN Simplified', format='png')
    dot.attr(rankdir='LR', size='16,10', dpi='150')
    dot.attr('node', shape='box', style='rounded,filled', fontname='Helvetica', fontsize='11')
    
    colors = {
        'input': '#E8F4FD',
        'backbone': '#D4EDDA',
        'decoder': '#FFF3CD',
        'dapcn': '#F8D7DA',
        'head': '#D1ECF1',
        'loss': '#E2D4F0',
    }
    
    # High-level components
    dot.node('input', 'Input Image\n(3×512×512)', fillcolor=colors['input'])
    dot.node('backbone', 'UNetFormer Backbone\n\n4-Stage Hierarchical\nCNN + Transformer\n\nOutput: 4-scale features\n[64, 128, 256, 512]', 
             fillcolor=colors['backbone'])
    dot.node('decoder', 'Feature Pyramid\nDecoder\n\nMulti-scale fusion\nU-Net style skip\n\nOutput: (256, H/8, W/8)', 
             fillcolor=colors['decoder'])
    dot.node('dapcn', 'DAPCN Module\n(after fusion)\n\nDynamic Anchors: 64\nEM Refinement: 3 iters\nQuality Gating\n\nLearns class prototypes', 
             fillcolor=colors['dapcn'])
    dot.node('head', 'Segmentation Head\n\nClassification\nUpsample to full res\n\nOutput: (19, H, W)', 
             fillcolor=colors['head'])
    dot.node('loss', 'Multi-Task Loss\n\nCE: λ=1.0\nBoundary: λ=0.3\nDAPG: λ=0.1\nContrastive: λ=0.1', 
             fillcolor=colors['loss'])
    
    # Edges
    dot.edge('input', 'backbone', penwidth='2')
    dot.edge('backbone', 'decoder', penwidth='2')
    dot.edge('decoder', 'dapcn', penwidth='2', label='fused\nfeature', fontsize='9')
    dot.edge('decoder', 'head', penwidth='2')
    dot.edge('dapcn', 'loss', style='dashed', color='red', label='prototypes')
    dot.edge('head', 'loss', penwidth='2', label='predictions')
    
    return dot


def create_training_flow():
    """Create a training flow diagram showing loss computation."""
    
    dot = Digraph(comment='DAPCN Training Flow', format='png')
    dot.attr(rankdir='TB', size='14,16', dpi='150')
    dot.attr('node', shape='box', style='rounded,filled', fontname='Helvetica', fontsize='10')
    
    colors = {
        'data': '#E8F4FD',
        'forward': '#D4EDDA',
        'dapcn': '#F8D7DA',
        'loss': '#E2D4F0',
        'grad': '#FFF3CD',
    }
    
    dot.node('data', 'Input Batch\nImages + Ground Truth\n(B, 3, H, W) + (B, H, W)', fillcolor=colors['data'])
    
    dot.node('forward', 'Forward Pass\nUNetFormer + Decoder\n\nOutputs:\n- seg_logits (B, 19, H, W)\n- fused_features (B, 256, H/8, W/8)', 
             fillcolor=colors['forward'])
    
    dot.node('da_forward', 'Dynamic Anchor Forward\n\nInput: fused_features\n- EM refinement (3 iters)\n- Quality filtering\n\nOutputs:\n- assignments (N, K\')\n- prototypes (K\', 256)\n- quality (K\',)', 
             fillcolor=colors['dapcn'])
    
    dot.node('loss_ce', 'Compute L_ce\nCrossEntropy(seg_logits, gt)', fillcolor=colors['loss'])
    dot.node('loss_bound', 'Compute L_boundary\nSobel edges + BCE', fillcolor=colors['loss'])
    dot.node('loss_dapg', 'Compute L_dapg\nIntra + Inter + Quality', fillcolor=colors['loss'])
    dot.node('loss_cont', 'Compute L_contrastive\nInfoNCE (after warmup)', fillcolor=colors['loss'])
    
    dot.node('total', 'Total Loss\nL = L_ce + 0.3×L_boundary + 0.1×L_dapg + 0.1×L_contrastive', 
             fillcolor=colors['loss'], shape='ellipse', style='bold,filled')
    
    dot.node('backward', 'Backward Pass\nGradients flow back:\n- Network weights\n- Prototype parameters\n- Quality network', fillcolor=colors['grad'])
    
    dot.node('update', 'Optimizer Step\nAdamW updates all parameters', fillcolor=colors['grad'])
    
    # Edges
    dot.edge('data', 'forward')
    dot.edge('forward', 'da_forward', label='fused_features')
    
    dot.edge('forward', 'loss_ce', label='seg_logits')
    dot.edge('forward', 'loss_bound', label='seg_logits')
    dot.edge('da_forward', 'loss_dapg', label='assignments\nprototypes\nquality')
    dot.edge('da_forward', 'loss_cont', label='prototypes')
    
    dot.edge('loss_ce', 'total')
    dot.edge('loss_bound', 'total')
    dot.edge('loss_dapg', 'total')
    dot.edge('loss_cont', 'total')
    
    dot.edge('total', 'backward')
    dot.edge('backward', 'update')
    
    return dot


if __name__ == '__main__':
    print("Generating UNetFormer DAPCN workflow diagrams...")
    
    # Create output directory
    output_dir = 'workflow_diagrams'
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate detailed workflow
    print("1. Creating detailed architecture diagram...")
    detailed = create_unetformer_dapcn_workflow()
    detailed.render(f'{output_dir}/unetformer_dapcn_detailed', cleanup=True)
    print(f"   Saved: {output_dir}/unetformer_dapcn_detailed.png")
    
    # Generate simplified workflow
    print("2. Creating simplified high-level diagram...")
    simplified = create_simplified_workflow()
    simplified.render(f'{output_dir}/unetformer_dapcn_simplified', cleanup=True)
    print(f"   Saved: {output_dir}/unetformer_dapcn_simplified.png")
    
    # Generate training flow
    print("3. Creating training flow diagram...")
    training = create_training_flow()
    training.render(f'{output_dir}/unetformer_dapcn_training', cleanup=True)
    print(f"   Saved: {output_dir}/unetformer_dapcn_training.png")
    
    print("\n✓ All diagrams generated successfully!")
    print(f"\nFiles saved in: {output_dir}/")
    print("  - unetformer_dapcn_detailed.png (Full architecture)")
    print("  - unetformer_dapcn_simplified.png (High-level view)")
    print("  - unetformer_dapcn_training.png (Training flow)")
