"""Write PBS Jobs for various calculations."""

__author__ = 'harrigan'

PBS_HEADER = """#!/bin/bash
#PBS -l nodes=1:ppn=1
#PBS -l walltime={hours}:00:00
#PBS -l mem=8gb
#PBS -j oe
#PBS -M harrigan@stanford.edu
#PBS -m ae

: ${{PBS_O_WORKDIR="$PWD"}}
cd $PBS_O_WORKDIR
export PBS_O_WORKDIR
export OMP_NUM_THREADS=1

"""


from mcmd import mcmd

class MakeJobsCommand(mcmd.Parsable):

    def main(self):
        print "Hello world"


def parse():
    p = mcmd.parsify(MakeJobsCommand)
    p.main()

if __name__ == "__main__":
    parse()
