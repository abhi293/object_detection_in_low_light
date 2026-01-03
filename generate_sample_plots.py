"""
Example: Generate sample visualizations to demonstrate functionality
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend

from utils.visualization import (
    plot_training_curves,
    plot_loss_breakdown,
    create_training_summary,
    plot_confusion_matrix
)

def create_sample_history():
    """Create sample training history for demonstration"""
    epochs = list(range(1, 21))
    
    # Simulate training curves with realistic patterns
    train_loss = [2.5 - 0.1 * i + 0.05 * np.sin(i) for i in epochs]
    val_loss = [2.6 - 0.09 * i + 0.08 * np.sin(i) for i in epochs]
    
    # Simulate component losses
    illum_loss = [0.8 - 0.03 * i for i in epochs]
    resto_loss = [0.6 - 0.02 * i for i in epochs]
    detect_loss = [1.1 - 0.05 * i for i in epochs]
    
    # Simulate mAP improvement
    mAP = [0.1 + 0.03 * i + 0.01 * np.sin(i) for i in epochs]
    mAP = [min(0.75, m) for m in mAP]  # Cap at 0.75
    
    # Learning rate schedule
    lr = [1e-4 * (0.95 ** i) for i in epochs]
    
    history = {
        'epochs': epochs,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'train_illum_loss': illum_loss,
        'train_resto_loss': resto_loss,
        'train_detect_loss': detect_loss,
        'val_mAP': mAP,
        'learning_rate': lr,
        'class_names': ['Bicycle', 'Boat', 'Bottle', 'Bus', 'Car', 'Cat', 
                       'Chair', 'Cup', 'Dog', 'Motorbike', 'People', 'Table'],
        'val_AP_per_class': [[0.1 + 0.03 * epoch + 0.02 * np.random.random() 
                             for _ in range(12)] for epoch in epochs]
    }
    
    return history


def create_sample_confusion_matrix():
    """Create sample confusion matrix"""
    num_classes = 12
    cm = np.random.randint(0, 50, size=(num_classes, num_classes))
    
    # Make diagonal dominant (correct predictions)
    for i in range(num_classes):
        cm[i, i] = np.random.randint(100, 200)
    
    return cm


def main():
    print("="*60)
    print("Generating Sample Visualizations")
    print("="*60)
    
    # Create sample data
    print("\n📊 Creating sample training history...")
    history = create_sample_history()
    
    print("📊 Creating sample confusion matrix...")
    class_names = ['Bicycle', 'Boat', 'Bottle', 'Bus', 'Car', 'Cat', 
                   'Chair', 'Cup', 'Dog', 'Motorbike', 'People', 'Table']
    cm = create_sample_confusion_matrix()
    
    # Generate visualizations
    print("\n🎨 Generating visualizations...")
    
    print("  1/4 Training curves...")
    plot_training_curves(history, save_path='sample_training_curves.png')
    
    print("  2/4 Loss breakdown...")
    plot_loss_breakdown(history, save_path='sample_loss_breakdown.png')
    
    print("  3/4 Training summary...")
    create_training_summary(history, save_path='sample_training_summary.png')
    
    print("  4/4 Confusion matrix...")
    plot_confusion_matrix(cm, class_names, save_path='sample_confusion_matrix.png')
    
    print("\n" + "="*60)
    print("✅ Sample visualizations generated successfully!")
    print("="*60)
    print("\nGenerated files:")
    print("  • sample_training_curves.png")
    print("  • sample_loss_breakdown.png")
    print("  • sample_training_summary.png")
    print("  • sample_confusion_matrix.png")
    print("\nThese demonstrate the visualization capabilities.")
    print("="*60)


if __name__ == '__main__':
    main()
