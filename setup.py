#!/usr/bin/env python
"""
Copyright 2020
    Sigvald Marholm <marholm@marebakken.com>
    Diako Darian <diakod@math.uio.no>

This file is part of PUNC.

PUNC is free software: you can redistribute it and/or modify it under the terms
of the GNU General Public License as published by the Free Software Foundation,
either version 3 of the License, or (at your option) any later version.

PUNC is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
PUNC. If not, see <http://www.gnu.org/licenses/>.
"""

from setuptools import setup
from io import open # Necessary for Python 2.7

with open('README.rst', encoding='utf-8') as f:
    long_description = f.read()

with open('.version') as f:
    version = f.read().strip()

setup(name='punc',
      version=version,
      description='Particles-in-Unstructured-Cells',
      long_description=long_description,
      author=['Sigvald Marholm', 'Diako Darian'],
      author_email='marholm@marebakken.com',
      url='https://github.com/puncproject/punc.git',
      packages=['punc'],
      install_requires=['numpy',
                        'scipy',
                        'matplotlib',
                        'fenics>=2019.1,<2019.2',
                        'mshr',
                        'metaplot',
                        'tasktimer'],
      entry_points = {'console_scripts': ['punc = punc.object_interaction:run']},
      license='GPL',
      classifiers=[
        'Programming Language :: Python :: 3.7'
        ]
     )

