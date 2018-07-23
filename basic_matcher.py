import argparse
import json
import os
import pickle
import privileged_residues
import sys

from os import makedirs, path
from multiprocessing import Pool, freeze_support, cpu_count

from privileged_residues import PrivilegedResidues
from privileged_residues.postproc import filter_clash_minimize

import pyrosetta
from pyrosetta.rosetta.core.select.residue_selector import \
    ResidueIndexSelector, SecondaryStructureSelector, TrueResidueSelector


_bidentate_types = ['sc_sc', 'sc_scbb', 'sc_bb']
_network_types = ['acceptor_acceptor', 'acceptor_donor', 'donor_acceptor',
                  'donor_donor']

def make_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("PDBFile", type=str, help="pose to match against")

    parser.add_argument("-o", "--outpath", dest="outpath", type=str, required=True,
                        help="Output path")

    parser.add_argument("--residues", dest="residues", type=int, default=None,
                        nargs="+", help="Residue indices")

    parser.add_argument("--clash-cutoff", dest="clash_cutoff", type=float,
                        default=35., help="Tolerance for clash checking")

    parser.add_argument("--n-cutoff", dest="n_cutoff", type=int, default=10,
                        help="Number of interactions to isolate")

#     parser.add_argument("--params", dest="params", type=str, nargs="*",
#                         help="Additional parameter files")

    parser.add_argument("--bidentates", dest="bidentates", type=str,
                        default=_bidentate_types, nargs="*",
                        help="Type of bidentate interactions to consider. " +
                             "Allowed values are: 'sc_sc', 'sc_scbb', " +
                             "'sc_bb'.")
    parser.add_argument("--networks", dest="networks", type=str,
                        default=_network_types, nargs="*",
                        help="Type of network arrangements to consider. " +
                             "Allowed values are: 'acceptor_acceptor', "
                             "'acceptor_donor', 'donor_acceptor', " +
                             "'donor_donor'.")

    feature_parser = parser.add_mutually_exclusive_group(required=False)
    feature_parser.add_argument('--reduced-output', dest='reduced_output', action='store_true')
    feature_parser.add_argument('--no-reduced-output', dest='reduced_output', action='store_false')
    parser.set_defaults(reduced_output=False)

    return parser

def process(pr, p, args, out):
    selector = ResidueIndexSelector()

    for residue in args.residues:
        selector.append_index(residue)

    hits = pr.search(p, args.bidentates + args.networks, selector)

    prev = -1
    buffer = []
    limit = args.n_cutoff if args.reduced_output else 0

    for (hash, pose) in hits:
        if (hash != prev):
            for n, (_, minimized) in enumerate(filter_clash_minimize(p, buffer, clash_cutoff=args.clash_cutoff, limit=limit)):
                print("Match found! Pair hash: %d, match number: %d" % (prev, n))
                minimized.dump_pdb(path.join(out, "result_pair%d_num%d.pdb" % (prev, n)))

            prev = hash
            buffer = []
        else:
            buffer.append((hash, pose))

    if (buffer):
        for n, (_, minimized) in enumerate(filter_clash_minimize(p, buffer, clash_cutoff=args.clash_cutoff, limit=limit)):
            print("Match found! Pair hash: %d, match number: %d" % (prev, n))
            minimized.dump_pdb(path.join(out, "result_pair%d_num%d.pdb" % (prev, n)))

        buffer = []

def main(argv):
    parser = make_parser()
    args = parser.parse_args()

    pdb = path.expanduser(args.PDBFile)
    assert(path.exists(pdb))

    pr = PrivilegedResidues()
    p = pyrosetta.pose_from_file(pdb)

    for bidentate in args.bidentates:
        if bidentate not in _bidentate_types:
            raise ValueError('"{}" is not a valid option for '.format(
                                bidentate) + ' --bidentates. ' +
                             'Valid options are: "{}"'.format('", "'.join(
                                _bidentate_types)))
    for network in args.networks:
        if network not in _network_types:
            raise ValueError('"{}" is not a valid option for '.format(
                                network) + ' --networks. ' +
                             'Valid options are: "{}"'.format('", "'.join(
                                _network_types)))

    # TODO: allow an XML file that configures a ResidueSelector to be used
    # selector = TrueResidueSelector()

    # this should be trigguered by a flag. Fortunately, by factoring the
    # business end of this script into its own function, another script could
    # easily call that function and pass a pre-configured ResidueSelector in.
    # selector = SecondaryStructureSelector("L")

    out = path.expanduser(args.outpath)
    makedirs(out, exist_ok=True)
    p.dump_pdb(path.join(out, "ori.pdb"))

    if (args.residues is None):
        args.residues = range(1, len(p) + 1)

    process(pr, p, args, out)

if __name__ == "__main__":
    freeze_support()
    main(sys.argv)

