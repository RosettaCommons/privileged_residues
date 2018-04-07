#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `privileged_residues` package."""

import pytest

from click.testing import CliRunner

from privileged_residues import privileged_residues
from privileged_residues import cli
from privileged_residues.privileged_residues import HAVE_PYROSETTA

@pytest.fixture
def response():
    """Sample pytest fixture.

    See more at: http://doc.pytest.org/en/latest/fixture.html
    """
    # import requests
    # return requests.get('https://github.com/audreyr/cookiecutter-pypackage')


def test_content(response):
    """Sample pytest test function with the pytest fixture as an argument."""
    # from bs4 import BeautifulSoup
    # assert 'GitHub' in BeautifulSoup(response.content).title.string


def test_command_line_interface():
    """Test the CLI."""
    runner = CliRunner()
    result = runner.invoke(cli.main)
    assert result.exit_code == 0
    assert 'privileged_residues.cli.main' in result.output
    help_result = runner.invoke(cli.main, ['--help'])
    assert help_result.exit_code == 0
    assert '--help  Show this message and exit.' in help_result.output


def _test_sc_bb_bidentate_placement(fname):
    import pyrosetta
    import privileged_residues
    from privileged_residues.privileged_residues import _init_pyrosetta as init
    import numpy as np
    from numpy.testing import assert_allclose
    from privileged_residues import bidentify as bd

    init()

    atoms = privileged_residues.process_networks.fxnl_groups['CA_'].atoms
    cart_resl, ori_resl, cart_bound = 0.1, 2.0, 16.0
    
    ref_pose = pyrosetta.pose_from_file(fname)
    ref_coords = np.stack([np.array([*ref_pose.residues[2].xyz(atom)]) for atom in atoms])

    pairs, ht = privileged_residues.find_privileged_interactions_in_pose(ref_pose)
    hits = bd.look_up_interactions(pairs, ht, cart_resl, ori_resl, cart_bound)

    for i, hit in enumerate(hits):
        coords = np.stack([np.array([*hit.residues[1].xyz(atom)]) for atom in atoms])
        try:
            assert_allclose(coords, ref_coords, atol=0.5)
            return
        except AssertionError:
            continue
    assert(False)


@pytest.mark.skipif('not HAVE_PYROSETTA')
def test_sc_bb_bidentate_placement():
    _test_sc_bb_bidentate_placement('sc_bb_example_with_bidentate.pdb')


@pytest.mark.skipif('not HAVE_PYROSETTA')
def test_sc_bb_bidentate_placement_transformed():
    _test_sc_bb_bidentate_placement('transformed_example.pdb')
    
