#

import subprocess
import os

def get_git_sha(repo):
    """
    Return the hash of the last git commit.
    """
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo).decode('ascii').strip()
    return sha

    
# version number of beamgen
__VERSION__ = get_git_sha(os.path.dirname(os.path.realpath(__file__)))
