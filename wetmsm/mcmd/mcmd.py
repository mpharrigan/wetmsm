__author__ = 'harrigan'

import argparse
import inspect

# Maximum length of the short names (one-dash arguments)
MAXSHORT = 3


class Parsable(object):
    """Mixin to make as parsable."""
    _is_parsable = True


class Attrib(object):
    """Container object for command line attributes."""

    def __init__(self):
        self.varname = None
        self.long_name = None
        self.short_name = None
        self.metavar = None
        self.dtype = None
        self.defval = None
        self.helptxt = None
        self.required = False


def get_attribute_help(docstring):
    """From a docstring, get help strings

    We expect lines of the form
    :param [atrribute_name]: help text

    :param docstring: The docstring to parse
    :returns: A dictionary of (attribute, help_text)
    """
    docdict = dict()
    linesplits = docstring.splitlines()
    for line in linesplits:
        line = line.strip()
        if line.startswith(':param'):
            colon_split = line.split(':')[1:]
            name_split = colon_split[0].split()
            if len(name_split) != 2:
                continue

            docdict[name_split[1]] = colon_split[1]
    return docdict


def generate_atributes(c_obj):
    """Yield attributes by iterating over a dictionary of class attributes.

    :param c_obj: Class to parse
    """
    initargs = inspect.getargspec(c_obj.__init__)
    n_defaults = len(initargs.defaults)
    n_attr = len(initargs.args)

    attr_short_names = set()
    for i, attr in enumerate(initargs.args):

        if attr == 'self':
            continue

        ao = Attrib()

        # Figure out short name
        underscores = attr.split('_')
        maxl = max(len(us) for us in underscores)

        for short_i in range(1, min(maxl, MAXSHORT)):
            attr_short_name = ''.join([us[:short_i] for us in underscores])
            if attr_short_name not in attr_short_names:
                attr_short_names.add(attr_short_name)
                break
        else:
            attr_short_name = None

        # Add it if possible
        if attr_short_name is not None:
            ao.short_name = '-{}'.format(attr_short_name)

        # Long name is just variable name
        ao.varname = attr
        ao.long_name = '--{}'.format(attr)
        ao.metavar = underscores[-1]

        # Figure out datatype and default value
        if (n_attr - i) <= n_defaults:
            defval = initargs.defaults[i - (n_attr - n_defaults)]
            ao.dtype = defval.__class__
            ao.defval = defval
        else:
            # TODO: How to specify datatype?
            ao.defval = None
            ao.dtype = str
            ao.required = True
        yield ao


def add_subparsers(c_obj, ap_parser):
    """Recurse down subparsers.

    :param c_obj: Find subparsers for this class
    :param ap_parser: Add subparsers to this argparse.ArgumentParser

    :returns: True if we added subparsers
    """
    subclasses = c_obj.__subclasses__()
    if len(subclasses) > 0:
        ap_subparsers = ap_parser.add_subparsers()
        for sub_cobj in subclasses:
            ap_sp = ap_subparsers.add_parser(sub_cobj._subcommand_shortname,
                                             formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            add_to_parser(sub_cobj, ap_sp)

        return True
    else:
        return False


def add_to_parser(c_obj, parser):
    """Add all attributes from an object to a given argparse.ArgumentParser.

    :param c_obj: Add attributes from this class
    :param parser: Add attributes to this argparse.ArgumentParser
    """

    if not add_subparsers(c_obj, parser):
        # Only set_defaults for 'leaf' subparsers
        parser.set_defaults(c_leaf=c_obj)

    docdict = get_attribute_help(c_obj.__doc__)
    for ao in generate_atributes(c_obj):

        try:
            ao.helptxt = docdict[ao.varname]
        except KeyError:
            ao.helptxt = "[help]"


        # Condition on whether we could find a valid short name
        if ao.short_name is not None:
            parser.add_argument(ao.long_name, ao.short_name,
                                metavar=ao.metavar, default=ao.defval,
                                type=ao.dtype, help=ao.helptxt,
                                required=ao.required)
        else:
            parser.add_argument(ao.long_name, metavar=ao.metavar,
                                default=ao.defval, type=ao.dtype,
                                help=ao.helptxt, required=ao.required)


def parsify(c_obj):
    """Take an object and make all of its attributes command line options.

    :param c_obj: A class to parse
    """

    first_doc_line = c_obj.__doc__.splitlines()[0]

    parser = argparse.ArgumentParser(description=first_doc_line,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_to_parser(c_obj, parser)
    args = parser.parse_args()

    # Prepare arguments for init
    init_dict = vars(args)
    c_leaf = args.c_leaf
    del init_dict['c_leaf']

    c_inst = c_leaf(**init_dict)
    return c_inst

