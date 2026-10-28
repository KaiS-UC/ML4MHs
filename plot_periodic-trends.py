from xenonpy.datatools import preset
from mp_hydrogenation_energy import HYDROGENATION_DATA
import pandas as pd
import periodic_trends as pt


if __name__ == "__main__":
    # Load xenonpy preset data
    ds_xenonpy = preset.elements_completed

    # Load hydrogenation data from mp_hydrogenation_energy.py
    ds_hydrogenation = pd.DataFrame.from_dict(HYDROGENATION_DATA, orient='index')
    ds_hydrogenation.rename(columns={'energy': 'hydrogenation_energy'}, inplace=True)

    # Merge the two datasets on the element symbol
    ds = pd.concat([ds_xenonpy, ds_hydrogenation['hydrogenation_energy']], axis=1)

    # Present user with a list of available properties to plot
    print("Available properties to plot:")
    for i, prop in enumerate(ds.columns):
        print(f"{i}: {prop}")

    # Prompt user to select a property
    prop_index = int(input("Enter the index of the property you want to plot: "))
    property_name = ds.columns[prop_index]
    ds_filtered = ds.reset_index(names='Element')

    # Remove elements from the plot?
    continue_plotting = True
    while continue_plotting:
        # Plot the selected property
        pt.plotter(ds_filtered, 'Element', property_name)

        # Ask user if they want to remove any elements from the plot
        remove_elements = input("Do you want to remove any elements from the plot? (y/n): ").strip().lower()
        if remove_elements == 'y':
            elements_to_remove = input("Enter all the elements to remove (including previously removed ones), separated by commas: ").strip().split(',')
            ds_filtered = ds.drop(elements_to_remove, errors='ignore').reset_index(names='Element')
        else:
            continue_plotting = False
