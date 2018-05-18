
# python packages
import subprocess
import os


def get_section_string(section_name):
    """ Return the string for a section in the dat file. """
    return ''.join(['-' for i in range(80-len(section_name))]) + section_name


def get_git_sha(repo):
    """
    Return the hash of the last git commit.
    """
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo).decode('ascii').strip()
    return sha

    
# version number of beamgen
__VERSION__ = get_git_sha(os.path.dirname(os.path.realpath(__file__)))
