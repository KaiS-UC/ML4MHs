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
    #labels = [FEATURE_NAMES[col] for col in data.columns]
    labels = [col for col in data.columns]
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
    # List of targets
    targets = [
        'Heat_of_Formation_kJperMolH2',
        'LnEquilibrium_Pressure_25C'
    ]

    # List of trainingdatasets
    training_datasets = [
        'Heat_of_Formation_kJperMolH2_reduced-features',
        'LnEquilibrium_Pressure_25C_reduced-features',
        'all-targets_reduced-features',
        'all-targets_common-features'
        ]

    for dataset in training_datasets:
        # Load training data
        training_data = pd.read_excel(f'training_data/training_data_{dataset}.xlsx')
        training_data.drop('Composition_Formula', axis=1, inplace=True)

        # Plot feature correlations
        ax = plot_feature_correlations(training_data)
        fig = ax.get_figure()
        fig.tight_layout()

        # Save the plot
        fig.savefig(f"images/feature_correlations_{dataset}.png", dpi=300)
        print(f"Plot saved to ./images/feature_correlations_{dataset}.png", end='\n' + '-'*40 + '\n')
        plt.close(fig)

        # Determine the target based on the dataset name
        if 'Heat_of_Formation_kJperMolH2' in dataset:
            target = ['Heat_of_Formation_kJperMolH2']
        elif 'LnEquilibrium_Pressure_25C' in dataset:
            target = ['LnEquilibrium_Pressure_25C']
        else:
            target = targets  # For datasets that don't correspond to a single target

        for t in target:
            # Load predictions from training workflow
            model_predictions_train = pd.read_excel(f'models/predictions_train_{dataset}_{t}.xlsx')
            model_predictions_test = pd.read_excel(f'models/predictions_test_{dataset}_{t}.xlsx')

            # Load feature importance data
            model_feature_importances = pd.read_excel(f'models/importances_{dataset}_{t}.xlsx')
            model_feature_importances_std = pd.read_excel(f'models/importances_std_{dataset}_{t}.xlsx')

            for model_name in model_predictions_train.columns[1:]:
                # Plot real vs predicted values
                y_train = model_predictions_train['y_train']
                y_train_pred = model_predictions_train[model_name]
                y_test = model_predictions_test['y_test']
                y_test_pred = model_predictions_test[model_name]

                ax = plot_real_vs_predicted(y_train, y_train_pred, label='Training Data', color='blue')
                ax = plot_real_vs_predicted(y_test, y_test_pred, ax=ax, label='Test Data', color='green')
                fig = ax.get_figure()
                fig.tight_layout()
                fig.savefig(f"images/{dataset}_{t}_real_vs_predicted_{model_name}.png", dpi=300)
                print(f"Plot saved to ./images/{dataset}_{t}_real_vs_predicted_{model_name}.png", end='\n' + '-'*40 + '\n')
                plt.close(fig)

                # Plot feature importance
                feature_names = model_feature_importances.iloc[:, 0]
                feature_importances = model_feature_importances[model_name]
                feature_importances_std = model_feature_importances_std[model_name]
                ax = plot_feature_importance(feature_names, feature_importances, feature_importances_std)
                fig = ax.get_figure()
                fig.tight_layout()
                fig.savefig(f"images/{dataset}_{t}_feature_importance_{model_name}.png", dpi=300)
                print(f"Plot saved to ./images/{dataset}_{t}_feature_importance_{model_name}.png", end='\n' + '-'*40 + '\n')
                plt.close(fig)
