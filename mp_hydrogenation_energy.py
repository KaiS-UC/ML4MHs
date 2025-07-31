#!/usr/bin/env python3

from scipy import constants
import numpy as np
from composition_parser import parse_composition


API_KEY = "YOUR_API_KEY_HERE"  # replace with your Materials Project API key
eV2kJ = constants.e * constants.N_A / 1000

# hydrogenation data retrieved from mp on 20. Feb. 2024
HYDROGENATION_DATA = {
    'n': {'energy': None, 'composition': None},
    'H': {'energy': None, 'composition': None},
    'He': {'energy': None, 'composition': None},
    'Li': {'energy': -188.86507354144595, 'composition': 'Li1H1'},
    'Be': {'energy': -47.722999037803206, 'composition': 'Be6H12'},
    'B': {'energy': -34.416581373341394, 'composition': 'B72H88'},
    'C': {'energy': -60.88865710558825, 'composition': 'H4C1'},
    'N': {'energy': -89.75944028735731, 'composition': 'H12N4'},
    'O': {'energy': -361.1933493215136, 'composition': 'H24O12'},
    'F': {'energy': -665.5329315300394, 'composition': 'H2F2'},
    'Ne': {'energy': None, 'composition': None},
    'Na': {'energy': -113.04330901812295, 'composition': 'Na1H1'},
    'Mg': {'energy': -81.83473082327006, 'composition': 'Mg2H4'},
    'Al': {'energy': -18.94510924025365, 'composition': 'Al2H6'},
    'Si': {'energy': -1.9821513653485188, 'composition': 'Si4H16'},
    'P': {'energy': 7.538399910044685, 'composition': 'P4H12'},
    'S': {'energy': -49.33316200949742, 'composition': 'H16S8'},
    'Cl': {'energy': -314.3472681195151, 'composition': 'H2Cl2'},
    'Ar': {'energy': None, 'composition': None},
    'K': {'energy': -111.75768707447257, 'composition': 'K1H1'},
    'Ca': {'energy': -193.03286206869322, 'composition': 'Ca4H8'},
    'Sc': {'energy': -213.6457903873732, 'composition': 'Sc1H2'},
    'Ti': {'energy': -158.3193983033738, 'composition': 'Ti4H4'},
    'V': {'energy': -98.7899747360739, 'composition': 'V8H4'},
    'Cr': {'energy': 0.6600969221082975, 'composition': 'Cr1H2'},
    'Mn': {'energy': -48.875291680001865, 'composition': 'Mn29H2'},
    'Fe': {'energy': -25.956522883159106, 'composition': 'Fe1H1'},
    'Co': {'energy': -20.046850322392604, 'composition': 'Co1H1'},
    'Ni': {'energy': -4.294529086582011, 'composition': 'Ni1H1'},
    'Cu': {'energy': 37.90154762739619, 'composition': 'Cu2H2'},
    'Zn': {'energy': 6.517341685127955, 'composition': 'Zn4H24'},
    'Ga': {'energy': 137.64943814811033, 'composition': 'Ga2H2'},
    'Ge': {'energy': -8.58823672116816, 'composition': 'Ge4H16'},
    'As': {'energy': 187.0327829125361, 'composition': 'As4H12'},
    'Se': {'energy': 186.77154318217225, 'composition': 'H2Se1'},
    'Br': {'energy': -126.33476269202097, 'composition': 'H2Br2'},
    'Kr': {'energy': None, 'composition': None},
    'Rb': {'energy': -90.87751175317104, 'composition': 'Rb1H1'},
    'Sr': {'energy': -187.44579255772075, 'composition': 'Sr4H8'},
    'Y': {'energy': -231.4576981360738, 'composition': 'Y1H2'},
    'Zr': {'energy': -189.7012551494218, 'composition': 'Zr4H4'},
    'Nb': {'energy': -72.35933668002033, 'composition': 'Nb2H2'},
    'Mo': {'energy': 152.04402840366788, 'composition': 'Mo2H2'},
    'Tc': {'energy': 516.8846930534243, 'composition': 'Tc3H1'},
    'Ru': {'energy': None, 'composition': None},
    'Rh': {'energy': 19.609872112619588, 'composition': 'H1Rh1'},
    'Pd': {'energy': -34.48346090494239, 'composition': 'H2Pd2'},
    'Ag': {'energy': 46.65968782223652, 'composition': 'Ag2H2'},
    'Cd': {'energy': 145.54352212252445, 'composition': 'Cd2H2'},
    'In': {'energy': 145.82511938859264, 'composition': 'In2H2'},
    'Sn': {'energy': 12.126949767662753, 'composition': 'Sn2H8'},
    'Sb': {'energy': 175.00767896083636, 'composition': 'Sb1H3'},
    'Te': {'energy': None, 'composition': None},
    'I': {'energy': -181.0214028314492, 'composition': 'H2I6'},
    'Xe': {'energy': None, 'composition': None},
    'Cs': {'energy': -97.34280860691592, 'composition': 'Cs1H1'},
    'Ba': {'energy': -175.0476082898735, 'composition': 'Ba4H8'},
    'La': {'energy': -215.9720845498788, 'composition': 'La1H2'},
    'Ce': {'energy': -175.31338386925148, 'composition': 'Ce4H11'},
    'Pr': {'energy': -207.57316825183204, 'composition': 'Pr1H2'},
    'Nd': {'energy': -213.31462884740586, 'composition': 'Nd1H2'},
    'Pm': {'energy': -188.5719252122253, 'composition': 'Pm2H2'},
    'Sm': {'energy': -220.66829244626422, 'composition': 'Sm1H2'},
    'Eu': {'energy': -198.76890565610486, 'composition': 'Eu4H8'},
    'Gd': {'energy': -213.7336726048573, 'composition': 'Gd1H2'},
    'Tb': {'energy': -231.65268437890887, 'composition': 'Tb1H2'},
    'Dy': {'energy': -231.91848844338807, 'composition': 'Dy1H2'},
    'Ho': {'energy': -231.47777528640907, 'composition': 'Ho1H2'},
    'Er': {'energy': -230.6795780239832, 'composition': 'Er1H2'},
    'Tm': {'energy': -231.56218017252334, 'composition': 'Tm1H2'},
    'Yb': {'energy': None, 'composition': None},
    'Lu': {'energy': -227.41841464367192, 'composition': 'Lu1H2'},
    'Hf': {'energy': -156.96053957039737, 'composition': 'Hf1H2'},
    'Ta': {'energy': -102.36973390520174, 'composition': 'Ta2H1'},
    'W': {'energy': 15.438436600626495, 'composition': 'H12W2'},
    'Re': {'energy': 96.18149884760072, 'composition': 'Re2H6'},
    'Os': {'energy': 724.1532649109729, 'composition': 'H1Os3'},
    'Ir': {'energy': 163.7274054303202, 'composition': 'H2Ir2'},
    'Pt': {'energy': 82.09484024304373, 'composition': 'H12Pt3'},
    'Au': {'energy': None, 'composition': None},
    'Hg': {'energy': None, 'composition': None},
    'Tl': {'energy': 137.73725797174632, 'composition': 'Tl2H2'},
    'Pb': {'energy': 186.45628723412986, 'composition': 'H2Pb2'},
    'Bi': {'energy': 160.9242580020032, 'composition': 'Bi1H3'},
    'Po': {'energy': None, 'composition': None},
    'At': {'energy': None, 'composition': None},
    'Rn': {'energy': None, 'composition': None},
    'Fr': {'energy': None, 'composition': None},
    'Ra': {'energy': None, 'composition': None},
    'Ac': {'energy': -198.08446355499476, 'composition': 'Ac1H2'},
    'Th': {'energy': -163.24602249978278, 'composition': 'Th1H2'},
    'Pa': {'energy': -108.79683179785734, 'composition': 'Pa8H24'},
    'U': {'energy': -118.7828995671511, 'composition': 'U8H24'},
    'Np': {'energy': -151.44141828668555, 'composition': 'Np1H2'},
    'Pu': {'energy': -175.3598320506705, 'composition': 'Pu1H2'},
    'Am': {'energy': None, 'composition': None},
    'Cm': {'energy': None, 'composition': None},
    'Bk': {'energy': None, 'composition': None},
    'Cf': {'energy': None, 'composition': None},
    'Es': {'energy': None, 'composition': None},
    'Fm': {'energy': None, 'composition': None},
    'Md': {'energy': None, 'composition': None},
    'No': {'energy': None, 'composition': None},
    'Lr': {'energy': None, 'composition': None},
    'Rf': {'energy': None, 'composition': None},
    'Db': {'energy': None, 'composition': None},
    'Sg': {'energy': None, 'composition': None},
    'Bh': {'energy': None, 'composition': None},
    'Hs': {'energy': None, 'composition': None},
    'Mt': {'energy': None, 'composition': None},
    'Ds': {'energy': None, 'composition': None},
    'Rg': {'energy': None, 'composition': None},
    'Cn': {'energy': None, 'composition': None},
    'Nh': {'energy': None, 'composition': None},
    'Fl': {'energy': None, 'composition': None},
    'Mc': {'energy': None, 'composition': None},
    'Lv': {'energy': None, 'composition': None},
    'Ts': {'energy': None, 'composition': None},
    'Og': {'energy': None, 'composition': None}
}


def get_hydrogenation_energy(element):
    """
    Function to grab hydrogenation energy from materialsproject
    """
    # use stored results, if mp_api is not installed
    try:
        from mp_api.client import MPRester
    except ImportError:
        energy = HYDROGENATION_DATA[element]["energy"]
        composition = HYDROGENATION_DATA[element]["composition"]
        return energy, composition

    # get all relevant entries
    with MPRester(API_KEY) as mpr:
        docs = mpr.materials.summary.search(chemsys=f"{element}-H",
                                            fields=["formation_energy_per_atom",
                                                    "composition"])

    # calculate formation energy per mole of H2 (in kJ/(mol H2))
    energies = []
    compositions = []
    for doc in docs:
        E_tot = doc.formation_energy_per_atom * doc.composition.num_atoms
        E_H2  = E_tot / (doc.composition.get("H") / 2)
        E_H2_kJ = eV2kJ * E_H2
        energies.append(E_H2_kJ)
        compositions.append(doc.composition.to_pretty_string())

    # select composition with lowest hydrogenation energy
    if len(energies) > 0:
        hydrogenation_energy = np.amin(energies)
        composition = compositions[np.argmin(energies)]
    else:
        hydrogenation_energy = None
        composition = None
    return hydrogenation_energy, composition


def get_average_hydrogenation_energy(composition):
    """
    Calculate average hydrogenation energy of an alloy from elemental
    hydrogenation energy weighted by atomic fraction.
    """
    try:
        # parse composition
        comp_dict = parse_composition(composition)
    except TypeError:
        # maybe list or other iterable of compositions?
        try:
            avg = [get_average_hydrogenation_energy(comp) for comp in composition]
            return avg
        except TypeError:
            print("Unable to parse composition", composition)
            return None

    # calculate weighted sum
    weighted_sum = 0
    for key in comp_dict.keys():
        # return None if no energy in dataset (otherwise error)
        if HYDROGENATION_DATA[key]["energy"] is None:
            return None
        # add to weighted sum
        weighted_sum += comp_dict[key] * HYDROGENATION_DATA[key]["energy"]

    # normalize weights -> weighted average
    avg = weighted_sum / np.sum([v for v in comp_dict.values()])

    return avg
