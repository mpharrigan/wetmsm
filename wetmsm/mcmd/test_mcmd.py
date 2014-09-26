__author__ = 'harrigan'

import mcmd
import glob


class WriteDirectoryListing(mcmd.Parsable):
    """List files and write them in a directory.

    :param out_fn: Where to write the file
    :param glob_str: How to glob files
    :param limit: Max number of files or -1 for all

    """

    def __init__(self, out_fn, glob_str='data/*.txt', limit=-1):
        self.out_fn = out_fn
        self.glob_str = glob_str
        self.limit = limit

    def main(self):
        fns = glob.glob(self.glob_str)
        limit = self.limit
        if 0 < limit < len(fns):
            fns = fns[:limit]

        with open(self.out_fn, 'w') as f:
            f.write('\n'.join(fns))


class WriteOnlyPart(WriteDirectoryListing):
    """Write only filename or dirname."""

    _subcommand_shortname = 'writeonly'

    def __init__(self, out_fn, glob_str='sample/*.txt', limit=-1,
                 which='dirname'):
        pass


def parse():
    c_inst = mcmd.parsify(WriteDirectoryListing)
    c_inst.main()


if __name__ == "__main__":
    parse()