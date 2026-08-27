#!/usr/bin/env python

import re


def parse_composition(s):
    """
    This function parses a string and returns a dictionary containing
    the chemical elements in the string and their amounts as key-value
    pairs.
    """

    # regular expression for parsing a composition string
    #   case 1: bracket with number after it: (A2B)2
    #   case 2: element with number after it: Aa2
    # composition strings **cannot** be nested, e.g. (A2(B3C)2)2
    FORMULA_RE = r'([A-Z][a-z]?|\(.*?\))(\d*\.?\d*)'

    # iterate through matches and build dictionary
    composition = {}
    for element, num in re.findall(FORMULA_RE, s):
        # convert number to float, default to 1 if no number is present
        if num == '':
            num = 1.0
        else:
            num = float(num)

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
            if key in composition.keys():
                composition[key] += el_dict[key]
            else:
                composition[key] = el_dict[key]

    # return dictionary
    return composition


if __name__ == "__main__":
    import sys
    comp = sys.argv[1]
    print(f"Parsing composition string: {comp}")
    print(parse_composition(comp))
