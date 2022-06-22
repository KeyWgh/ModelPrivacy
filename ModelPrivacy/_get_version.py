"""Get the current package version from the configuration file."""
import re
from pathlib import Path
root = str(Path(__file__).absolute().parent.parent)
with open(root+'/setup.cfg', 'r') as f:
    content = f.read()
    p = re.compile('version = ([^\s]*)')
    __version__ = re.findall(p, content)[0]
del re, Path, content, p

if __name__ == '__main__':
    print(f'Version = {__version__}')
