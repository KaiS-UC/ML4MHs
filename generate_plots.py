import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import pandas as pd
import numpy as np


MODEL_NAMES = {
    'adaboost': 'AdaBoost',
    'gradientboost': 'Gradient Boosting',
    'bayesian': 'Bayesian Ridge',
    'neuralnetwork': 'Neural Network'
}

FEATURE_NAMES = {
    'Heat_of_Formation_kJperMolH2': 'Heat of Formation (kJ/mol H2)',
    'LnEquilibrium_Pressure_25C': 'Ln Equilibrium Pressure (25°C)',
    'ave:atomic_radius': 'Average Atomic Radius',
    'ave:covalent_radius_pyykko': 'Average Covalent Radius (Pyykko)',
    'ave:gs_est_bcc_latcnt': 'Average Ground State BCC Lattice Constant',
    'ave:vdw_radius_mm3': 'Average VDW Radius',
    'var:gs_energy': 'Variance of Ground State Energy',
    'var:num_p_valence': 'Variance of No. of p Valence Electrons',
    'average_hydrogenation_energy': 'Average Hydrogenation Energy (kJ/mol H2)',
}

EVALUATION_NAMES = {
    'mean_absolute_error': 'Mean Absolute Error (MAE)',
    'mean_squared_error': 'Mean Squared Error (MSE)',
    'root_mean_squared_error': 'Root Mean Squared Error (RMSE)',
    'r2_score': 'R² Score',
    'explained_variance_score': 'Explained Variance Score',
    'median_absolute_error': 'Median Absolute Error'
}


def plot_feature_correlations(data, ax=None, **kwargs):
    """
    Plot feature correlations in the dataset.
    
    Parameters:
    data (pd.DataFrame): DataFrame containing the features.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 10))

    # Plot the correlation matrix (lower triangle only)
    corr = data.corr()
    corr = corr.where(np.tril(np.ones(corr.shape), k=-1).astype(bool))  # Mask the upper triangle
    cax = ax.matshow(corr, cmap='coolwarm', norm=TwoSlopeNorm(0, -1, 1), **kwargs)
    plt.colorbar(cax)

    # put the correlation values on the matrix
    for (i, j), value in np.ndenumerate(corr):
        if not np.isnan(value):
            ax.text(j, i, f"{value:-.2f}", ha='center', va='center', fontsize=12, color='black')

    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(len(data.columns)))
    ax.set_yticks(np.arange(len(data.columns)))
    labels = [FEATURE_NAMES[col] for col in data.columns]
    ax.set_xticklabels(labels, rotation=90, fontsize=13)
    ax.set_yticklabels(labels, fontsize=13)

    return ax


def plot_evaluation_metric(model_names, values, ax=None, metric_name=None, **kwargs):
    """
    Plot evaluation metrics for different models.
    
    Parameters:
    model_names (list): Names of the models.
    values (pd.DataFrame): DataFrame containing the evaluation metric values.
    metric_name (str): Name of the metric to plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 8))

    model_names = [MODEL_NAMES[model] for model in model_names]
    ax.bar(model_names, values, **kwargs)
    ax.set_xticklabels(model_names, rotation=30, ha='right', fontsize=13)
    ax.set_ylabel(EVALUATION_NAMES[metric_name], fontsize=15)
    ax.tick_params(labelsize=13)

    return ax


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
    ax.set_aspect('equal', 'datalim')
    ax.grid(True)
    ax.legend(fontsize=13)
    ax.tick_params(labelsize=13)

    return ax


def plot_feature_importance(feature_names, feature_importances, feature_importances_std=None,
                            ax=None, normalize=True, sort=True, **kwargs):
    """
    Plot feature importance.
    
    Parameters:
    feature_names (list): Names of the features.
    feature_importances (np.ndarray): Importance scores for each feature.
    feature_importances_std (np.ndarray, optional): Standard deviation of the importance scores.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    if normalize:
        factor = np.sum(feature_importances)
        feature_importances = feature_importances / factor
        if feature_importances_std is not None:
            feature_importances_std = feature_importances_std / factor

    if sort:
        indices = np.argsort(feature_importances)
        feature_names = feature_names[indices]
        feature_importances = feature_importances[indices]
        if feature_importances_std is not None:
            feature_importances_std = feature_importances_std[indices]

    features = [FEATURE_NAMES[name] for name in feature_names]
    ax.barh(features, feature_importances, xerr=feature_importances_std, capsize=5, **kwargs)
    ax.tick_params(labelsize=13)
    ax.set_xlim(left=0)
    ax.set_xlabel('Feature Importance', fontsize=13)

    return ax


if __name__ == "__main__":
    for target in ['Heat_of_Formation_kJperMolH2', 'LnEquilibrium_Pressure_25C']:
        # Plot feature correlations in the training data
        training_data = pd.read_excel(f'training_data/training_data_{target}_final-features.xlsx')
        training_data.drop('Composition_Formula', axis=1, inplace=True)
        ax = plot_feature_correlations(training_data)
        fig = ax.get_figure()
        fig.tight_layout()
        fig.savefig(f"images/feature_correlations_{target}.png", dpi=300)
        print(f"Plot saved to ./images/feature_correlations_{target}.png", end='\n' + '-'*40 + '\n')
        plt.close(fig)

        # Plot evaluation metrics
        for dataset in ['train', 'test']:
            model_evaluations = pd.read_excel(f'models/model_evaluations_{dataset}_{target}.xlsx')
            for i in range(len(model_evaluations.iloc[:, 0])):
                metric_name = model_evaluations.iloc[i, 0]
                values = model_evaluations.iloc[i, 1:]
                ax = plot_evaluation_metric(model_evaluations.columns[1:], values, metric_name=metric_name)
                fig = ax.get_figure()
                fig.tight_layout()
                fig.savefig(f"images/{target}_model-evaluation_{metric_name}_{dataset}.png", dpi=300)
                print(f"Plot saved to ./images/{target}_model-evaluation_{metric_name}_{dataset}.png", end='\n' + '-'*40 + '\n')
                plt.close(fig)

        # Load other data from training workflow
        model_feature_importances = pd.read_excel(f'models/model_feature_importances_{target}.xlsx')
        model_feature_importances_std = pd.read_excel(f'models/model_feature_importances_std_{target}.xlsx')
        model_predictions_train = pd.read_excel(f'models/model_predictions_train_{target}.xlsx')
        model_predictions_test = pd.read_excel(f'models/model_predictions_test_{target}.xlsx')

        for model_name in ['adaboost', 'gradientboost', 'bayesian', 'neuralnetwork']:
            # Plot real vs predicted values
            y_train = model_predictions_train['y_train']
            y_train_pred = model_predictions_train[model_name]
            y_test = model_predictions_test['y_test']
            y_test_pred = model_predictions_test[model_name]

            ax = plot_real_vs_predicted(y_train, y_train_pred, label='Training Data', color='blue')
            ax = plot_real_vs_predicted(y_test, y_test_pred, ax=ax, label='Test Data', color='green')
            fig = ax.get_figure()
            fig.tight_layout()
            fig.savefig(f"images/{model_name}_{target}_real_vs_predicted.png", dpi=300)
            print(f"Plot saved to ./images/{model_name}_{target}_real_vs_predicted.png", end='\n' + '-'*40 + '\n')
            plt.close(fig)

            # Plot feature importance
            feature_names = model_feature_importances.iloc[:, 0]
            feature_importances = model_feature_importances[model_name]
            feature_importances_std = model_feature_importances_std[model_name]
            ax = plot_feature_importance(feature_names, feature_importances, feature_importances_std)
            fig = ax.get_figure()
            fig.tight_layout()
            fig.savefig(f"images/{model_name}_{target}_feature_importance.png", dpi=300)
            print(f"Plot saved to ./images/{model_name}_{target}_feature_importance.png", end='\n' + '-'*40 + '\n')
            plt.close(fig)
