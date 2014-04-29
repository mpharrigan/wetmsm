__author__ = 'harrigan'

import argparse


class Attrib(object):
    def __init__(self):
        self.varname = None
        self.long_name = None
        self.short_name = None
        self.metavar = None
        self.dtype = None
        self.defval = None
        self.helptxt = None


def get_attribute_help(docstring):
    docdict = dict()
    linesplits = docstring.splitlines()
    for line in linesplits:
        line = line.strip()
        if line.startswith(':attr'):
            colon_split = line.split(':')[1:]
            name_split = colon_split[0].split()
            if len(name_split) != 2:
                continue

            docdict[name_split[1]] = colon_split[1]
    return docdict


def generate_atributes(var_dict, max_short):
    attr_short_names = set()
    for attr in var_dict:

        ao = Attrib()

        # Ignore private
        if attr.startswith('_'):
            continue

        # Figure out short name
        underscores = attr.split('_')
        maxl = max(len(us) for us in underscores)

        for short_i in range(1, min(maxl, max_short)):
            attr_short_name = ''.join([us[:short_i] for us in underscores])
            if attr_short_name not in attr_short_names:
                attr_short_names.add(attr_short_name)
                break
        else:
            attr_short_name = None

        # Add it if possible
        if attr_short_name is not None:
            ao.short_name = '--{}'.format(attr_short_name)

        # Long name is just variable name
        ao.varname = attr
        ao.long_name = '--{}'.format(attr)
        ao.metavar = underscores[-1]

        # Figure out datatype
        defval = var_dict[attr]
        if defval is not None:
            if defval.__class__ == type:
                ao.dtype = defval
                ao.defval = None
            elif hasattr(defval, '__call__'):
                # exclude functions
                continue
            else:
                ao.dtype = defval.__class__
                ao.defval = defval
        else:
            continue

        yield ao


def parsify(c_obj, max_short=3):
    """Take an object and make all of its attributes command line options."""

    parser = argparse.ArgumentParser(description=c_obj.__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    docdict = get_attribute_help(c_obj.__doc__)
    for ao in generate_atributes(vars(c_obj), max_short):

        try:
            ao.helptxt = docdict[ao.varname]
        except KeyError:
            ao.helptxt = "[help]"


        # Condition on whether we could find a valid short name
        if ao.short_name is not None:
            parser.add_argument(ao.long_name, ao.short_name,
                                metavar=ao.metavar, default=ao.defval,
                                type=ao.dtype, help=ao.helptxt)
        else:
            parser.add_argument(ao.long_name, metavar=ao.metavar,
                                default=ao.defval, type=ao.dtype,
                                help=ao.helptxt)

    c_obj = parser.parse_args(namespace=c_obj)
    return c_obj