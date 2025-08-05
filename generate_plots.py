import matplotlib.pyplot as plt
import pandas as pd


def plot_real_vs_predicted(y_real, y_pred, ax=None, **kwargs):
    """
    Plot real vs predicted values.
    
    Parameters:
    y_real (pd.Series): Actual target values.
    y_pred (np.ndarray): Predicted target values.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    ax.axline((0, 0), slope=1, color='red', linestyle='--')
    ax.scatter(y_real, y_pred, **kwargs)
    ax.set_xlabel('Actual Values', fontsize=15)
    ax.set_ylabel('Predicted Values', fontsize=15)
    ax.grid(True)
    ax.legend(fontsize=13)
    ax.tick_params(labelsize=13)

    return ax


if __name__ == "__main__":
    # Load data from training workflow
    model_evaluations_train = pd.read_excel('training_data/model_evaluations_train.xlsx')
    model_evaluations_test = pd.read_excel('training_data/model_evaluations_test.xlsx')
    model_feature_importances = pd.read_excel('training_data/model_feature_importances.xlsx')
    model_feature_importances_std = pd.read_excel('training_data/model_feature_importances_std.xlsx')
    model_predictions_train = pd.read_excel('training_data/model_predictions_train.xlsx')
    model_predictions_test = pd.read_excel('training_data/model_predictions_test.xlsx')

    for model_name in ['adaboost', 'gradientboost', 'bayesian', 'neuralnetwork']:
        y_train = model_predictions_train['y_train']
        y_train_pred = model_predictions_train[model_name]
        y_test = model_predictions_test['y_test']
        y_test_pred = model_predictions_test[model_name]

        # Plot real vs predicted values
        ax = plot_real_vs_predicted(y_train, y_train_pred, label='Training Data', color='blue')
        ax = plot_real_vs_predicted(y_test, y_test_pred, ax=ax, label='Test Data', color='green')
        fig = ax.get_figure()
        fig.tight_layout()
        fig.savefig(f"images/{model_name}_real_vs_predicted.png", dpi=300)
        print(f"Plot saved to ./images/{model_name}_real_vs_predicted.png", end='\n' + '-'*40 + '\n')
