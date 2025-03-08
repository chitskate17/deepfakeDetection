from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model performance.

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels

    Returns:
        Dictionary with evaluation metrics
    """
    # Get predictions
    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()

    # Calculate metrics
    conf_matrix = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    # Calculate ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Real', 'Fake'],
                yticklabels=['Real', 'Fake'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()

    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('roc_curve.png')
    plt.show()

    # Return evaluation metrics
    eval_metrics = {
        'accuracy': report['accuracy'],
        'precision': report['1']['precision'],
        'recall': report['1']['recall'],
        'f1_score': report['1']['f1-score'],
        'auc': roc_auc,
        'confusion_matrix': conf_matrix
    }

    return eval_metrics


def visualize_predictions(model, X_test, y_test, file_paths, num_samples=5):
    """
    Visualize model predictions on sample images.

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        file_paths: List of file paths corresponding to test samples
        num_samples: Number of samples to visualize
    """
    # Get predictions
    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()

    # Randomly select samples
    indices = np.random.choice(len(X_test), num_samples, replace=False)

    # Plot samples
    plt.figure(figsize=(15, 3 * num_samples))

    for i, idx in enumerate(indices):
        # Get sample
        sample = X_test[idx] if len(X_test.shape) == 4 else X_test[idx][0]
        true_label = 'Fake' if y_test[idx] == 1 else 'Real'
        pred_label = 'Fake' if y_pred[idx] == 1 else 'Real'
        confidence = y_pred_proba[idx][0] if y_pred[idx] == 1 else 1 - y_pred_proba[idx][0]

        # Display sample
        plt.subplot(num_samples, 1, i + 1)
        plt.imshow(sample)

        # Add prediction info
        title = f"True: {true_label}, Predicted: {pred_label} (Confidence: {confidence:.2f})"
        color = 'green' if y_test[idx] == y_pred[idx] else 'red'
        plt.title(title, color=color)
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('prediction_visualization.png')
    plt.show()


def analyze_failure_cases(model, X_test, y_test):
    """
    Analyze cases where the model fails.

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels

    Returns:
        Tuple of false positives and false negatives
    """
    # Get predictions
    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()

    # Find failure cases
    false_positive_indices = np.where((y_test == 0) & (y_pred == 1))[0]
    false_negative_indices = np.where((y_test == 1) & (y_pred == 0))[0]

    print(f"Number of false positives: {len(false_positive_indices)}")
    print(f"Number of false negatives: {len(false_negative_indices)}")

    # Return failure cases
    return false_positive_indices, false_negative_indices