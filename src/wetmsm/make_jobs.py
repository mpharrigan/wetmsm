"""Write PBS Jobs for various calculations."""

__author__ = 'harrigan'

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

SHELLS_JOB_FN = "shells-{traj_basename}.job"

SUBMIT_LINE = "mqsub {jobfn}"

from mcmd import mcmd
import glob
import os
import stat
import logging

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


class MakeShellsJobsCommand(mcmd.Parsable):
    """Subcommand for making solvent shells jobs.

    :attrib n_shells: Number of shells
    :attrib shell_width: Width of each shell
    """

    n_shells = 6
    shell_width = 0.3

    def main(self, mk_job_cmd):
        submit_lines = []

        for dn, fn in mk_job_cmd.get_trajs():
            # Make a job filename
            jobfn = os.path.join(dn, SHELLS_JOB_FN.format(traj_basename=fn))

            fmt = dict()
            fmt['n_shells'] = self.n_shells
            fmt['shell_width'] = self.shell_width
            fmt['counts_out_fn'] = mk_job_cmd.counts_out_fn.format(traj_fn=fn)
            fmt['assign_out_fn'] = mk_job_cmd.assign_out_fn.format(traj_fn=fn)
            fmt['traj_top'] = mk_job_cmd.traj_top
            fmt['traj_fn'] = fn
            fmt['work_dir'] = dn
            fmt['solute_indices_fn'] = mk_job_cmd.solute_indices_fn
            fmt['solvent_indices_fn'] = mk_job_cmd.solvent_indices_fn
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
        with open('submit.sh', 'w') as sub_f:
            sub_f.write('\n'.join(submit_lines))

        # Make executable
        st = os.stat('submit.sh')
        os.chmod('submit.sh', st.st_mode | stat.S_IEXEC)


class MakeJobsCommand(mcmd.Parsable):
    """Make jobs for solvent fingerprinting. One job / trajectory

    :attrib traj_glob: A glob string for finding trajectories
    """

    _subparsers = {MakeShellsJobsCommand: 'shells'}
    traj_glob = 'data/SYS*/0_centered.xtc'
    solute_indices_fn = 'solute_indices.dat'
    solvent_indices_fn = 'solvent_indices.dat'
    traj_top = str
    counts_out_fn = 'shells-{traj_fn}_count.h5'
    assign_out_fn = 'shells-{traj_fn}_assign.h5'

    def main(self):
        # Call subcommand
        subc = self.c_obj()
        subc.main(self)

    def get_trajs(self):
        for traj_fn in glob.iglob(self.traj_glob):
            dirname = os.path.dirname(traj_fn)
            basename = os.path.basename(traj_fn)
            yield dirname, basename


def parse():
    """Parse command line arguments."""
    p = mcmd.parsify(MakeJobsCommand)
    p.main()


if __name__ == "__main__":
    parse()
