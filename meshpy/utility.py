# -*- coding: utf-8 -*-
"""
This module implements some basic functions that are used in the meshpy
application.
"""

# Python modules.
import subprocess
import os
import numpy as np
import xml.etree.ElementTree as ET
import warnings

# Meshpy modules.
from . import mpy, find_close_nodes, find_close_nodes_binning


def get_git_data(repo):
    """Return the hash and date of the current git commit."""
    out_sha = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=repo,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    out_date = subprocess.run(['git', 'show', '-s', '--format=%ci'], cwd=repo,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    if not out_sha.returncode + out_date.returncode == 0:
        return None, None
    else:
        sha = out_sha.stdout.decode('ascii').strip()
        date = out_date.stdout.decode('ascii').strip()
        return sha, date


# Set the git version in the global configuration object.
mpy.git_sha, mpy.git_date = get_git_data(
    os.path.dirname(os.path.realpath(__file__)))


def flatten(data):
    """Flatten out all list items in data."""
    flatten_list = []
    if type(data) == list:
        for item in data:
            flatten_list.extend(flatten(item))
        return flatten_list
    else:
        return [data]


def get_close_nodes(nodes, binning=mpy.binning, nx=mpy.binning_n_bin,
        ny=mpy.binning_n_bin, nz=mpy.binning_n_bin, eps=mpy.eps_pos,
        return_nodes=True):
    """
    Find nodes that are close to each other.

    Args
    ----
    nodes: list(Node)
        Nodes that are checked for partners.
    binning: bool
        If binning should be used.
    nx, ny, nz: int
        Number of bins in the directions.
    eps: double
        Spherical value that the nodes have to be within, to be identified
        as overlapping.
    return_nodes: bool
        If true, the Node objects are returned, otherwise the index of the node
        objects in the list nodes. This option is mainly used for testing.

    Return
    ----
    partner_nodes: list(list(Node)), list(int)
        A list of lists with partner nodes.
    """

    # Get array of coordinates.
    coords = np.zeros([len(nodes), 3])
    for i, node in enumerate(nodes):
        coords[i, :] = node.coordinates

    if len(nodes) > mpy.binning_max_nodes_brute_force and not mpy.binning:
        warnings.warn('The function get_close_nodes is called directly '
            + 'with {} nodes, for performance reasons the '.format(len(nodes))
            + 'function find_close_nodes_binning should be used!')

    # Get list of closest pairs.
    if mpy.binning:
        has_partner, n_partner = find_close_nodes_binning(coords, nx, ny, nz,
            eps)
    else:
        has_partner, n_partner = find_close_nodes(coords, eps=eps)

    if return_nodes:
        # Create list with nodes.
        partner_nodes = [[] for i in range(n_partner)]
        for i, node in enumerate(nodes):
            if not has_partner[i] == -1:
                partner_nodes[has_partner[i]].append(node)

        return partner_nodes
    else:
        # Return the partner list.
        return has_partner, n_partner


def xml_to_dict(xml, tol_float):
    """Convert a XML to a nested dictionary."""

    # Get and sort keys.
    keys = xml.keys()
    keys.sort()

    # Get string for this XML element.
    string = '<' + xml.tag
    if 'Name' in keys:
        # If there is a key "Name" put this one first.
        index = keys.index('Name')
        if index == 0:
            pass
        else:
            keys[0], keys[index] = keys[index], keys[0]
    for key in keys:
        string += ' '
        string += key
        string += '="'
        string += xml.get(key)
        string += '"'
    string += '>'

    # Get data for this item.
    xml_dict = {}
    n_childs = len(xml.getchildren())
    is_text = not xml.text.strip() == ''
    if n_childs > 0 and is_text:
        raise ValueError('The text is not empty and there are children. This '
            + 'case should not happen!')
    elif n_childs > 0:
        # Add a child xml construct.
        for child in xml.getchildren():
            key, value = xml_to_dict(child, tol_float)
            xml_dict[key] = value
    elif is_text:
        # Add data.
        data = xml.text.split('\n')
        if tol_float is None:
            data_new = [line.strip() for line in data
                if not line.strip() == '']
        else:
            data_new = []
            for line in data:
                if line.strip() == '':
                    continue
                for number in line.strip().split(' '):
                    if np.abs(float(number)) < tol_float:
                        data_new.append('0.0')
                    else:
                        data_new.append(number)
        data_string = '\n'.join(data_new)
        xml_dict[''] = data_string

    # Return key for this item and all child items.
    return string, xml_dict


def xml_dict_to_string(item):
    """The nested XML dictionary to a string."""

    # Sort the keys.
    keys = list(item.keys())
    keys.sort()

    # Return the keys and the values.
    string = ''
    for key in keys:
        if key == '':
            string += item[key]
        else:
            # Add content.
            string += key
            string += '\n'
            string += xml_dict_to_string(item[key])
            string += '\n'

            # Get the name of the section from the key.
            section = key[1:].split(' ')[0]
            if section[-1] == '>':
                section = section[:-1]
            string += '</{}>\n'.format(section)

    # Return the value.
    return string.strip()


def compare_xml(path1, path2, tol_float=None):
    """
    Compare the xml files at path1 and path2.

    Args
    ----
    tol_float: None / float
        If it is None, the numbers are not changed.
        If it is a number, the nubers in the xml file are set to 0 when the
        absolute value is smaller that tol_float.
    """

    # Check that both arguments are paths and exist.
    if not (os.path.isfile(path1) and os.path.isfile(path2)):
        raise ValueError('The paths given are not ok!')

    tree1 = ET.parse(path1)
    tree2 = ET.parse(path2)

    key, value = xml_to_dict(tree1.getroot(), tol_float)
    string1 = xml_dict_to_string({key: value})
    hash1 = hash(string1)

    key, value = xml_to_dict(tree2.getroot(), tol_float)
    string2 = xml_dict_to_string({key: value})
    hash2 = hash(string2)

    if hash1 == hash2:
        return True, None, None
    else:
        return False, string1, string2
