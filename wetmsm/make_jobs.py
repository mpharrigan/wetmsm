"""Write PBS Jobs for various calculations.

Author: Matthew Harrigan
"""
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from future.builtins import *
PBS_HEADER = """#!/bin/bash
#PBS -l nodes=1:ppn=1
#PBS -l walltime={hours}:00:00
#PBS -l mem=8gb
#PBS -j oe
#PBS -M harrigan@stanford.edu
#PBS -m ae
#PBS -o {work_dir}

: ${{PBS_O_WORKDIR="$PWD"}}
cd $PBS_O_WORKDIR
export PBS_O_WORKDIR
export OMP_NUM_THREADS=1

"""

SHELLS_JOB = PBS_HEADER + """

# Do not run this file from this directory.

python -m wetmsm.shells -sw {shell_width} -tt {traj_top} -ns {n_shells} \\
    -tf {work_dir}/{traj_fn} --solute_indices_fn {solute_indices_fn} \\
    --solvent_indices_fn {solvent_indices_fn} \\
    -cof {work_dir}/{counts_out_fn} -aof {work_dir}/{assign_out_fn}

"""

# Optionally put a qsub command here to turn output
# into a script that submits jobs
SUBMIT_LINE = "{jobfn}"

import glob
import os
import logging

from . import mcmd


log = logging.getLogger()
log.addHandler(logging.StreamHandler())
log.setLevel(logging.INFO)


def _files_exist(base_dir, *fns):
    """Return true only if all files do not exist

    :param base_dir: Filenames relative to here
    :param fns: Filenames
    """
    for fn in fns:
        if not os.path.exists(os.path.join(base_dir, fn)):
            return False
    else:
        return True


class MakeJobsCommand(mcmd.Parsable):
    """Make jobs for solvent fingerprinting. One job / trajectory

    :param traj_glob: A glob string for finding trajectories
    """


    def __init__(self, traj_glob='data/SYS*/0_centered.xtc',
                 solute_indices_fn='solute_indices.dat',
                 solvent_indices_fn='solvent_indices.dat', traj_top='',
                 counts_out_fn='{traj_fn}.count.h5',
                 assign_out_fn='{traj_fn}.assign.h5',
                 job_out_fn='{traj_fn}.shell.job'):
        self.traj_glob = traj_glob
        self.solute_indices_fn = solute_indices_fn
        self.solvent_indices_fn = solvent_indices_fn
        self.traj_top = traj_top
        self.counts_out_fn = counts_out_fn
        self.assign_out_fn = assign_out_fn
        self.job_out_fn = job_out_fn

    def get_trajs(self):
        for traj_fn in glob.iglob(self.traj_glob):
            dirname = os.path.dirname(traj_fn)
            basename = os.path.basename(traj_fn)
            yield dirname, basename


class MakeShellsJobsCommand(MakeJobsCommand):
    """Subcommand for making solvent shells jobs.

    :param n_shells: Number of shells
    :param shell_width: Width of each shell
    """

    _subcommand_shortname = 'shells'

    def __init__(self, n_shells=5, shell_width=0.2, **kwargs):
        self.n_shells = n_shells
        self.shell_width = shell_width

        super().__init__(**kwargs)

    def main(self):
        submit_lines = []

        for dn, fn in self.get_trajs():
            # Make a job filename
            jobfn = os.path.join(dn, self.job_out_fn.format(traj_fn=fn))

            fmt = dict()
            fmt['n_shells'] = self.n_shells
            fmt['shell_width'] = self.shell_width
            fmt['counts_out_fn'] = self.counts_out_fn.format(traj_fn=fn)
            fmt['assign_out_fn'] = self.assign_out_fn.format(traj_fn=fn)
            fmt['traj_top'] = self.traj_top
            fmt['traj_fn'] = fn
            fmt['work_dir'] = dn
            fmt['solute_indices_fn'] = self.solute_indices_fn
            fmt['solvent_indices_fn'] = self.solvent_indices_fn
            fmt['hours'] = 24

            # Check if it exists
            if _files_exist(dn, fmt['counts_out_fn'], fmt['assign_out_fn']):
                log.warn("Output files exist. Skipping %s",
                         os.path.join(dn, fn))
                continue

            # Write the job file
            with open(jobfn, 'w') as job_f:
                job_f.write(SHELLS_JOB.format(**fmt))

            # Keep track for submit script
            submit_lines += [SUBMIT_LINE.format(jobfn=jobfn)]

            # Tell the world
            log.info('Created job for %s', os.path.join(dn, fn))

        # Put a newline at end of file
        submit_lines += ['']
        with open('shells.job.list', 'w') as sub_f:
            sub_f.write('\n'.join(submit_lines))


def parse():
    """Parse command line arguments."""
    p = mcmd.parsify(MakeJobsCommand)
    p.main()


if __name__ == "__main__":
    parse()
