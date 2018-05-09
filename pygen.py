#!/usr/bin/python3

import re






class InputSection(object):
    """
    TODO
    """
    
    def __init__(self, name):
        """
        TODO
        """
        
        # extract the name of the surface
        name = name.strip()
        start = re.search(r'[^-]', name).start()
        self.name = name[start:]

    
    def __str__(self):
        return self.name









filename = 'input/block-small.dat'
#filename = 'input/large-file-alex/confhex8_h12.dat'

new_lines = []

with open(filename) as infile:
    for line in infile:
#         if 'MAT 1 EAS none KINTYP nln' in line:
#             new_lines.append(line.replace('MAT 1 EAS none KINTYP nln', 'MAT 1 KINEM nonlinear EAS none'))
#         else:
#             new_lines.append(line)
        if line.startswith('----------'):
            a = InputSection(line)
            print(a)

# with open('input/large-file-alex/confhex8_h12_new.dat', 'w') as outfile:
#     for line in new_lines:
#         outfile.write(line)


print('END')
