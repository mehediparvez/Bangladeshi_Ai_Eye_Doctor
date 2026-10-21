"""
Bangladeshi AI Eye Doctor - Standalone Demo
Loads the trained EfficientNet-B0 model and runs inference on sample images.
Produces a visualization of predictions and model metrics.

Usage:
    python run_demo.py
"""
import torch
import torch.nn as nn
import timm
from torchvision import transforms
from PIL import Image
import os
import json
import sys

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

DISEASE_CLASSES = [
    'Central Serous Chorioretinopathy',
    'Diabetic Retinopathy',
    'Disc Edema',
    'Glaucoma',
    'Healthy',
    'Macular Scar',
    'Myopia',
    'Pterygium',
    'Retinal Detachment',
    'Retinitis Pigmentosa'
]

DISEASE_CLASSES_FULL = [
    'Central Serous Chorioretinopathy_Color_Fundus',
    'Diabetic Retinopathy',
    'Disc Edema',
    'Glaucoma',
    'Healthy',
    'Macular Scar',
    'Myopia',
    'Pterygium',
    'Retinal Detachment',
    'Retinitis Pigmentosa'
]


class EfficientNetB0Classifier(nn.Module):
    def __init__(self, num_classes=10, dropout=0.3):
        super().__init__()
        self.model = timm.create_model(
            'efficientnet_b0', pretrained=False,
            num_classes=num_classes, drop_rate=dropout
        )
        self.num_classes = num_classes

    def forward(self, x):
        return self.model(x)


def load_model(checkpoint_path):
    model = EfficientNetB0Classifier(num_classes=len(DISEASE_CLASSES), dropout=0.3)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)

    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        if not any(k.startswith('model.') for k in state_dict.keys()):
            state_dict = {'model.' + k: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)

    model.to(DEVICE)
    model.eval()
    return model


def predict_image(model, image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = Image.open(image_path).convert('RGB')
    tensor = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)[0]
    return probs.cpu().numpy()


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, 'final_model_for_deployment.pth')
    metrics_path = os.path.join(base_dir, 'final_model_metrics.json')
    img_dir = os.path.join(base_dir, 'eye_images')

    print(f"Device: {DEVICE}")
    print("Loading model...")
    model = load_model(model_path)
    print("Model loaded.\n")

    with open(metrics_path) as f:
        metrics = json.load(f)

    print("=" * 60)
    print("  Bangladeshi AI Eye Doctor - Model Demo")
    print("=" * 60)
    print(f"  Model: EfficientNet-B0")
    print(f"  Accuracy: {metrics['accuracy']:.2f}%")
    print(f"  Weighted F1: {metrics['weighted_f1']:.4f}")
    print(f"  Macro F1: {metrics['macro_f1']:.4f}")
    print(f"  CV F1 Mean: {metrics['cv_f1_mean']:.4f} +/- {metrics['cv_f1_std']:.4f}")
    print("=" * 60)

    print("\nPredictions on sample images:")
    print("-" * 60)
    for fname in sorted(os.listdir(img_dir)):
        fpath = os.path.join(img_dir, fname)
        probs = predict_image(model, fpath)
        top_idx = probs.argmax()
        top_prob = probs[top_idx]
        print(f"  {fname:35s} -> {DISEASE_CLASSES[top_idx]:30s} ({top_prob:.1%})")
    print("-" * 60)

    if '--no-plot' not in sys.argv:
        try:
            create_visualization(model, metrics, img_dir, base_dir)
        except Exception as e:
            print(f"\nVisualization skipped: {e}")


def create_visualization(model, metrics, img_dir, base_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    fig = plt.figure(figsize=(20, 14))
    fig.suptitle('Bangladeshi AI Eye Doctor - Model Dashboard', fontsize=18, fontweight='bold', y=0.98)

    # 1. Per-class F1 scores
    ax1 = fig.add_subplot(2, 3, 1)
    report = metrics['classification_report']
    class_names_short = [
        'CSC', 'DR', 'Disc Ed.', 'Glauc.', 'Healthy',
        'Mac. Scar', 'Myopia', 'Pteryg.', 'Ret. Det.', 'RP'
    ]
    f1_scores = [report[c]['f1-score'] for c in DISEASE_CLASSES_FULL]
    colors = plt.cm.RdYlGn([f / 1.0 for f in f1_scores])
    bars = ax1.barh(class_names_short, f1_scores, color=colors, edgecolor='gray', linewidth=0.5)
    ax1.set_xlim(0, 1.05)
    ax1.set_xlabel('F1 Score')
    ax1.set_title('Per-Class F1 Scores', fontweight='bold')
    for bar, val in zip(bars, f1_scores):
        ax1.text(val + 0.01, bar.get_y() + bar.get_height() / 2, f'{val:.2f}',
                 va='center', fontsize=8)
    ax1.invert_yaxis()

    # 2. Precision vs Recall scatter
    ax2 = fig.add_subplot(2, 3, 2)
    precisions = [report[c]['precision'] for c in DISEASE_CLASSES_FULL]
    recalls = [report[c]['recall'] for c in DISEASE_CLASSES_FULL]
    supports = [report[c]['support'] for c in DISEASE_CLASSES_FULL]
    max_support = max(supports)
    sizes = [40 + 200 * (s / max_support) for s in supports]
    scatter = ax2.scatter(recalls, precisions, s=sizes, c=f1_scores,
                          cmap='RdYlGn', edgecolors='gray', linewidth=0.5, vmin=0.5, vmax=1.0)
    for i, name in enumerate(class_names_short):
        ax2.annotate(name, (recalls[i], precisions[i]), fontsize=7,
                     textcoords='offset points', xytext=(5, 5))
    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision vs Recall (size = support)', fontweight='bold')
    ax2.set_xlim(0.4, 1.05)
    ax2.set_ylim(0.4, 1.05)

    # 3. Class distribution
    ax3 = fig.add_subplot(2, 3, 3)
    bars3 = ax3.bar(class_names_short, supports, color='steelblue', edgecolor='gray', linewidth=0.5)
    ax3.set_ylabel('Number of Samples')
    ax3.set_title('Test Set Class Distribution', fontweight='bold')
    ax3.tick_params(axis='x', rotation=45)
    for bar, val in zip(bars3, supports):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                 str(int(val)), ha='center', fontsize=7)

    # 4. Sample predictions grid
    ax4 = fig.add_subplot(2, 3, (4, 5))
    ax4.set_axis_off()
    ax4.set_title('Sample Image Predictions', fontweight='bold', pad=15)
    images = sorted(os.listdir(img_dir))
    n_imgs = len(images)
    cols = min(n_imgs, 5)
    rows = (n_imgs + cols - 1) // cols
    for idx, fname in enumerate(images):
        fpath = os.path.join(img_dir, fname)
        probs = predict_image(model, fpath)
        top_idx = probs.argmax()
        top_prob = probs[top_idx]
        img = Image.open(fpath).convert('RGB')
        r, c = divmod(idx, cols)
        inner_ax = fig.add_axes([
            0.05 + c * 0.12,
            0.05 + (rows - 1 - r) * 0.22,
            0.10, 0.18
        ])
        inner_ax.imshow(img)
        inner_ax.set_axis_off()
        true_label = fname.rsplit('.', 1)[0].rstrip('0123456789')
        pred_label = DISEASE_CLASSES[top_idx]
        color = 'green' if pred_label.startswith(true_label.rstrip()) else 'red'
        inner_ax.set_title(f'{pred_label}\n{top_prob:.0%}', fontsize=7,
                           color=color, fontweight='bold')

    # 5. Summary metrics
    ax5 = fig.add_subplot(2, 3, 6)
    ax5.set_axis_off()
    summary_text = (
        f"Model Summary\n"
        f"{'=' * 35}\n"
        f"Architecture:  EfficientNet-B0\n"
        f"Classes:       {len(DISEASE_CLASSES)}\n"
        f"Accuracy:      {metrics['accuracy']:.2f}%\n"
        f"Weighted F1:   {metrics['weighted_f1']:.4f}\n"
        f"Macro F1:      {metrics['macro_f1']:.4f}\n"
        f"CV F1 (3-fold):{metrics['cv_f1_mean']:.4f} +/- {metrics['cv_f1_std']:.4f}\n"
        f"{'=' * 35}\n"
        f"Best class:    Pterygium (F1=1.00)\n"
        f"Weakest class: CSC (F1=0.66)\n"
        f"Total test:    {int(report['weighted avg']['support'])} images"
    )
    ax5.text(0.1, 0.95, summary_text, transform=ax5.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = os.path.join(base_dir, 'model_dashboard.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\nDashboard saved to: {out_path}")


if __name__ == '__main__':
    main()
