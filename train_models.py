import time
import pickle
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
from sklearn.inspection import permutation_importance


MODEL_PARAM_GRIDS = {
    'adaboost': {
        'estimator__max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
        'n_estimators': [10, 50, 100, 500, 1000, 5000, 10000],
        'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.5, 1.0],
    },
    'gradientboost': {
        'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
        'n_estimators': [10, 50, 100, 500, 1000, 5000, 10000],
        'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.5, 1.0],
    },
    'bayesian': {
        'alpha_1': [1e-6, 1e-5, 1e-4, 1e-3],
        'alpha_2': [1e-6, 1e-5, 1e-4, 1e-3],
        'lambda_1': [1e-6, 1e-5, 1e-4, 1e-3],
        'lambda_2': [1e-6, 1e-5, 1e-4, 1e-3],
    },
    'neuralnetwork': {
        'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50), (10, 10, 10)],
        'activation': ['relu', 'tanh'],
        'alpha': [1e-5, 1e-4, 0.001, 0.01, 0.1, 1.0, 10.0],
        'learning_rate_init': [0.001, 0.01, 0.1, 1.0],
    }
}


def load_data(file_path):
    """
    Load data from an excel file, but drop the composition column (Comp).
    
    Parameters:
    file_path (str): Path to the excel file.
    
    Returns:
    pd.DataFrame: Loaded data as a DataFrame.
    """
    df = pd.read_excel(file_path)
    df.drop(['Composition_Formula'], axis=1, inplace=True)
    return df[df.notna().all(axis=1)]  # Ensure all rows are complete before returning


def split_and_scale(X, y):
    """
    Split the data into training and testing sets, and scale the features.
    
    Parameters:
    X (pd.DataFrame): Features.
    y (pd.Series): Target variable.
    
    Returns:
    X_train, X_test, y_train, y_test: Split and scaled data.
    """
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test


def get_base_model(model_name):
    """
    Get the base model based on the model name.
    
    Parameters:
    model_name (str): Name of the model to retrieve.
    
    Returns:
    model: The base model corresponding to the model name.
    """
    if model_name == 'adaboost':
        # Create an AdaBoost regressor
        base_estimator = DecisionTreeRegressor()
        model = AdaBoostRegressor(estimator=base_estimator, random_state=42)

    elif model_name == 'gradientboost':
        # Create a Gradient Boosting regressor
        model = GradientBoostingRegressor(random_state=42)

    elif model_name == 'bayesian':
        # Create a Bayesian Ridge regressor
        model = BayesianRidge()

    elif model_name == 'neuralnetwork':
        # Create a Multi-layer Perceptron regressor
        model = MLPRegressor(random_state=42, max_iter=5000)

    else:
        raise ValueError(f"Model {model_name} is not recognized.")
    
    return model, MODEL_PARAM_GRIDS[model_name]


def train_model(X_train, y_train, base_model, param_grid, **kwargs):
    """
    Train a model using GridSearchCV to find the best parameters.
    
    Parameters:
    X_train (pd.DataFrame): Training features.
    y_train (pd.Series): Training target variable.
    base_model: The base model to be trained.
    param_grid (dict): Dictionary of parameters to search over.
    
    Returns:
    estimator: Fitted best model.
    """
    # Setup GridSearchCV and fit
    grid_search = GridSearchCV(estimator=base_model, param_grid=param_grid, n_jobs=-1, **kwargs)
    grid_search.fit(X_train, y_train)

    # Get the best parameters from GridSearchCV
    best_params = grid_search.best_params_
    print("Best parameters found: ", best_params)

    return grid_search.best_estimator_


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model on the test set.
    
    Parameters:
    model: The trained model to evaluate.
    X_test (pd.DataFrame): Test features.
    y_test (pd.Series): Test target variable.
    
    Returns:
    evaluation_metrics: A dictionary containing evaluation metrics.
    predictions: np.ndarray of predictions made by the model.
    """
    y_pred = model.predict(X_test)
    evaluation_metrics = {
        'r2_score': r2_score(y_test, y_pred),
        'mean_absolute_error': mean_absolute_error(y_test, y_pred),
        'root_mean_squared_error': root_mean_squared_error(y_test, y_pred)
    }
    return evaluation_metrics, y_pred


def feature_importance(model, X_test, y_test):
    """
    Calculate feature importance for the model.
    
    Parameters:
    model: The trained model.
    X_test (pd.DataFrame): Features in the test set.
    y_test (pd.Series): Target variable in the test set.
    
    Returns:
    pd.Series: Feature importances.
    """
    imp = permutation_importance(model, X_test, y_test, scoring='r2', n_repeats=10, random_state=42, n_jobs=-1)
    return imp.importances_mean, imp.importances_std


if __name__ == "__main__":
    # script variables
    target = 'Heat_of_Formation_kJperMolH2'  # Name of the target variable column in the data file
    data_file = f"training_data/training_data_{target}_final-features.xlsx"  # Path to the data file

    # Load the data
    data = load_data(data_file)
    
    # Separate features and target variable
    X = data.drop(target, axis=1)
    y = data[target]
    
    # Split and scale the data
    X_train, X_test, y_train, y_test = split_and_scale(X, y)
    
    # train all available models
    model_evaluations_train = {}
    model_evaluations_test = {}
    model_predictions_train = {'y_train': y_train}
    model_predictions_test = {'y_test': y_test}
    model_feature_importances = {}
    model_feature_importances_std = {}

    for model_name in MODEL_PARAM_GRIDS.keys():
        print(f"Training model: {model_name}")

        # Get the base model and parameter grid
        base_model, param_grid = get_base_model(model_name)
    
        # Start timing the training process
        t0 = time.time()

        # Train the model
        best_model = train_model(X_train, y_train, base_model, param_grid, cv=5, scoring='r2')

        # End timing the training process
        t1 = time.time()
        print(f"Training time for {model_name}: {time.strftime('%H:%M:%S', time.gmtime(t1 - t0))}")
    
        # Evaluate the model
        train_scores, y_train_pred = evaluate_model(best_model, X_train, y_train)
        test_scores, y_test_pred = evaluate_model(best_model, X_test, y_test)

        # save results
        model_evaluations_train[model_name] = train_scores
        model_evaluations_test[model_name] = test_scores
        model_predictions_train[model_name] = y_train_pred
        model_predictions_test[model_name] = y_test_pred

        # print results
        print(f"Evaluation scores for {model_name}:")
        print(f"  Training R^2: {train_scores['r2_score']:.2f}")
        print(f"  Test R^2: {test_scores['r2_score']:.2f}")
        print(f"  Training MAE: {train_scores['mean_absolute_error']:.2f}")
        print(f"  Test MAE: {test_scores['mean_absolute_error']:.2f}")
        print(f"  Training RMSE: {train_scores['root_mean_squared_error']:.2f}")
        print(f"  Test RMSE: {test_scores['root_mean_squared_error']:.2f}")

        # Save the model
        model_filename = f"models/{model_name}_{target}_best_model.pkl"
        with open(model_filename, 'wb') as f:
            pickle.dump(best_model, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Model saved to {model_filename}")

        # Calculate feature importance
        feature_importances, feature_importances_std = feature_importance(best_model, X_test, y_test)
        model_feature_importances[model_name] = feature_importances
        model_feature_importances_std[model_name] = feature_importances_std
        print(f"Feature importances for {model_name}:")
        for i in feature_importances.argsort()[::-1]:
            print(f"    {X.columns[i]:<28}: {feature_importances[i]:.3f} ± {feature_importances_std[i]:.3f}")

        # Separate outputs for clarity
        print('-' * 40)

    # Save the model evaluations, feature importances, and predictions
    model_evaluations_train = pd.DataFrame(model_evaluations_train)
    model_evaluations_test = pd.DataFrame(model_evaluations_test)
    model_feature_importances = pd.DataFrame(model_feature_importances, index=X.columns)
    model_feature_importances_std = pd.DataFrame(model_feature_importances_std, index=X.columns)
    model_predictions_train = pd.DataFrame(model_predictions_train)
    model_predictions_test = pd.DataFrame(model_predictions_test)

    model_evaluations_train.to_excel(f'training_data/model_evaluations_train_{target}.xlsx')
    model_evaluations_test.to_excel(f'training_data/model_evaluations_test_{target}.xlsx')
    model_feature_importances.to_excel(f'training_data/model_feature_importances_{target}.xlsx')
    model_feature_importances_std.to_excel(f'training_data/model_feature_importances_std_{target}.xlsx')
    model_predictions_train.to_excel(f'training_data/model_predictions_train_{target}.xlsx')
    model_predictions_test.to_excel(f'training_data/model_predictions_test_{target}.xlsx')

    print("All models trained and evaluated successfully.")
