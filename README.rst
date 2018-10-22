PUNC
====

**NB: PUNC is replaced by** `PUNC++`_.

.. _`PUNC++`: https://github.com/puncproject/PUNCpp

*Particles-in-UNstructured-Cells* (PUNC) is a program that simulate plasmas in the kinetic regime through *Particle-In-Cell* (PIC) modelling on an unstructured mesh, using the *Finite Element Method* (FEM). It is written in Python and centered around FEniCS, which makes it particularly flexible and suited for rapid prototyping of novel methods.

PUNC supports 1D, 2D and 3D simulations of arbitrarily many species. It is particularly well suited for plasma-object interaction studies, with a flexible framework for handling arbitrary circuits of objects.

Contributors
------------

Principal authors:

- `Sigvald Marholm`_
- `Diako Darian`_

Contributors and mentors:

- `Mikael Mortensen`_
- `Richard Marchand`_
- `Wojciech J. Miloch`_

.. _`Sigvald Marholm`: mailto:sigvald@marebakken.com
.. _`Diako Darian`: mailto:diakod@math.uio.no
.. _`Mikael Mortensen`: mailto:mikael.mortensen@gmail.com
.. _`Richard Marchand`: mailto:rmarchan@ualberta.ca
.. _`Wojciech J. Miloch`: mailto:w.j.miloch@fys.uio.no

Installation
------------

The following dependencies must be installed prior to using PUNC:

- Python_ 3
- FEniCS_ 2018.1.0 (stable)
- TaskTimer_ 1

In addition, FEniCS must be compiled with *at least* the following optional dependencies:

- hdf5_
- hypre_
- PETSc_
- petsc4py_
- matplotlib_

It is crucial that the correct version of FEniCS is used. For more on installing these dependencies, see their official pages. For Arch Linux, the arch-fenics-packages_ repository can also be used. It contains full installation instructions for FEniCS and its dependencies.

Instead of installing PUNC to a system directory, we add its directory to ```PYTHONPATH``` so it is easier to tamper with for rapid prototyping. An example installation can look like the following::

    cd ~ # Or other parent folder
    git clone --recurse-submodules https://github.com/sigvaldm/punc
    echo "export PYTHONPATH=\"$PYTHONPATH:$(pwd)/punc\"" >> ~/.bashrc # Or .zshrc for Zsh, etc.

Note that the subfolder ```punc/mesh``` is a Git submodule and will be empty if submodules are not initialized. The ```--recurse-submodules``` flag should take care of this.

In addition we recommend the following tools for pre- and post-processing:

- Gmsh_
- ParaView_

.. _FEniCS: https://fenicsproject.org
.. _Python: https://www.python.org
.. _TaskTimer: https://github.com/sigvaldm/TaskTimer
.. _arch-fenics-packages: https://github.com/sigvaldm/arch-fenics-packages
.. _petsc4py: https://bitbucket.org/petsc/petsc4py/src/master/
.. _matplotlib: https://matplotlib.org/
.. _hdf5: https://support.hdfgroup.org/HDF5/
.. _hypre: https://computation.llnl.gov/projects/hypre-scalable-linear-solvers-multigrid-methods
.. _PETSc: http://www.mcs.anl.gov/petsc/
.. _Gmsh: http://gmsh.info/
.. _ParaView: https://www.paraview.org/


Getting Started
---------------

PUNC is like a library of objects and methods which can be glued together like Lego pieces in small scripts to carry out various FEM-PIC simulations. This library is located in ```punc/punc``` and further documentation is available in the docstrings in the source files.

A simulation program for plasma-object interaction is ```punc/simulations/object_interaction.py``` (other simulation scripts can also be found in this folder). ```object_interaction.py``` can be run like an executable::

    cd punc/simulations
    ./object_interaction.py laframboise.cfg.py

where ```laframboise.cfg.py``` is a configuration file where you specify the plasma parameters and simulation parameters. It is fully Python scriptable, which means you can for instance compute the thermal velocity (needed by PUNC) from temperature, etc (which parameters are used can readily be seen at the top of the ```object_interaction.py``` script).

The configuration file also specifies which mesh to use. The attached configuration file ```laframboise.cfg.py``` looks for a collection of XML files starting with ```laframboise_sphere_in_cub_res1```. This is the mesh format used by FEniCS and the mesh is not included in the repository. However, a suite of simulation geometries (```.geo``` files) for Gmsh, including the one needed for the example, is available in the folder ```punc/mesh```. The mesh can be created from the file ```laframboise_sphere_in_cub_res1.geo``` using Gmsh, and converted to FEniCS XML format using FEniCS's built-in command line tool::

    dolfin-convert laframboise_sphere_in_cube_res1.msh  laframboise_sphere_in_cube_res1.xml

Field quantities is written to ```*.pvd``` files which can readily be analyzed using e.g. ParaView, and history data is written for each time-step to ```history.dat```. This can be visualized using the supplied ```monitor.py``` script.
