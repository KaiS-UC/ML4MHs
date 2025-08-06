import pandas as pd
import numpy as np
from xenonpy.datatools import preset
from xenonpy.descriptor import Compositions
from composition_parser import parse_composition
from mp_hydrogenation_energy import MPHydrogenationEnergy
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV


def load_dataset(file_path, target=None):
    """
    Load hydrogenation data from a CSV file. Treat Composition_Formula as a feature and choose a target variable if specified.
    
    Parameters:
    file_path (str): Path to the CSV file.
    target (str): Name of the target variable column. If None, no target variable is set.
    
    Returns:
    pd.Series: Loaded compositions.
    pd.DataFrame: DataFrame of features derived from compositions.
    pd.Series or None: Target variable if specified, otherwise None.
    """
    df = pd.read_csv(file_path)
    if 'Composition_Formula' not in df.columns:
        raise ValueError("The dataset must contain a 'Composition_Formula' column.")
    compositions = df['Composition_Formula']
    return compositions, get_features_from_compositions(compositions), df[target] if target else None


def get_features_from_compositions(compositions):
    """
    Convert compositions to features using xenonpy.
    
    Parameters:
    compositions (pd.Series): Series of composition formulas.
    
    Returns:
    pd.DataFrame: DataFrame of features derived from compositions.
    """
    composition_data = [parse_composition(comp) for comp in compositions]
    calculator = Compositions()
    return calculator.transform(composition_data)


def lasso_feature_selection(X, y, **kwargs):
    """
    Perform Lasso feature selection on the dataset.
    
    Parameters:
    X (pd.DataFrame): DataFrame of features.
    y (pd.Series): Target variable.
    
    Returns:
    pd.DataFrame: DataFrame of selected features.
    """
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Perform Lasso feature selection
    lasso = LassoCV(**kwargs)
    lasso.fit(X_scaled, y)
    model = SelectFromModel(lasso, prefit=True)

    # Print some information about the selected features
    X_selected = X.iloc[:, model.get_support()]
    print(f"Selected {X_selected.shape[1]} features out of {X.shape[1]} total features.")
    print(f"Best alpha: {lasso.alpha_}")

    return X_selected


if __name__ == "__main__":
    datafile = 'training_data/ML_HYDPARK_v5.csv'
    target = 'LnEquilibrium_Pressure_25C'

    # Load dataset
    preset.sync('elements_completed')
    compositions, X, y = load_dataset(datafile, target)

    # Save full dataset before feature selection
    data_full = pd.concat([compositions, X, y], axis=1)
    data_full.to_excel(f'training_data/training_data_{target}_all-features.xlsx', index=False)
    print(f"Loaded {len(compositions)} compositions with {X.shape[1]} features.")

    # Select features
    X_selected = lasso_feature_selection(X, y, cv=5, alphas=100, #np.logspace(-6, 6, 100),
                                         max_iter=10000, random_state=42, n_jobs=-1)

    # Save reduced dataset
    data_reduced = pd.concat([compositions, X_selected, y], axis=1)
    data_reduced.to_excel(f'training_data/training_data_{target}_reduced-features.xlsx', index=False)
    print("The selected features are:")
    for col in X_selected.columns:
        print(f"    {col}")

    # Add average hydrogenation energy (from materialsproject data) as a feature
    mp_data = MPHydrogenationEnergy()
    mp_avg_hydrogenation = mp_data.get_average_hydrogenation_energy(compositions)
    mp_avg_hydrogenation = pd.Series(mp_avg_hydrogenation, name='average_hydrogenation_energy')

    # Save final dataset with average hydrogenation energy
    final_data = pd.concat([compositions, X_selected, mp_avg_hydrogenation, y], axis=1)
    final_data.to_excel(f'training_data/training_data_{target}_final-features.xlsx', index=False)
    print("Final dataset with average hydrogenation energy saved.")
