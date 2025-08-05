#!/usr/bin/env python

import re
#import periodictable


FORMULA_RE = r'([A-Z][a-z]?)|(\.\d+)|(\d+\.\d+)|(\d+)|(\(.*\))'
#MASSES = {el.symbol: el.mass for el in periodictable.elements}


def parse_composition(s):
    """
    This function parses a string and returns a dictionary containing
    the chemical elements in the string and their amounts as key-value
    pairs.
    """
    # split up into components
    formula = [m.group() for m in re.finditer(FORMULA_RE, s)
               if re.finditer(FORMULA_RE, s)]

    # iterate through components and add to resulting dictionary
    result = {}
    i = 0
    while i < len(formula):
        element = formula[i]

        # find out number first
        try:
            # element has number attached to it
            num = float(formula[i+1])
            i += 2
        except (ValueError, IndexError):
            # no number attached - assumed to be 1
            num = 1.0
            i += 1

        # check if element is subformula in brackets
        if element[0] == "(" and element[-1] == ")":
            # formula inside formula -> parse substring
            el_dict = parse_composition(element[1:-1])
            # multiply by number after the bracket
            for key in el_dict.keys():
                el_dict[key] *= num
        else:
            # no subformula -> simple dict with number
            el_dict = {element: num}

        # add to result
        for key in el_dict:
            if key in result.keys():
                result[key] += el_dict[key]
            else:
                result[key] = el_dict[key]

    # return dictionary
    return result


# def mass_per_atom(s):
#     """
#     This function calculates the mass per atom of a given composition string
#     """
#     # get weights from composition parser
#     weights = parse_composition(s)

#     # calculate total mass and normalize by number of atoms
#     mass = sum([weights[el]*MASSES[el] for el in weights.keys()])
#     mpa = mass / sum(weights.values())

#     return mpa


if __name__ == "__main__":
    import sys
    comp = sys.argv[1]
    print(f"Parsing composition string: {comp}")
    print(parse_composition(comp))
#    print(f"Average mass per atom: {mass_per_atom(comp)}")
