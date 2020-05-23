PUNC
====

**NB: See** `PUNC++`_ **for the faster C++ version**.

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

The easiest way to install PUNC is through Anaconda_::

    $ git clone https://www.github.com/puncproject/punc
    $ cd punc/envs
    $ conda env create -f punc.yml
    $ conda activate punc
    $ cd ..
    $ pip install -e .

PUNC is unfortunately not yet available from PyPI or as a conda package, so it must be cloned and installed as described above. The ``-e`` flag means that PUNC is installed in "editable" mode or "developer's" mode, and can be ommitted if you do not intend to edit the PUNC source code.

If you choose to install PUNC any other way, make sure to install the correct version of FEniCS_, and with *at least* the following optional dependencies:

- hdf5_
- hypre_
- PETSc_
- petsc4py_
- matplotlib_

In addition, we recommend the following tools for pre- and post-processing:

- Gmsh_
- ParaView_

.. _FEniCS: https://fenicsproject.org
.. _petsc4py: https://bitbucket.org/petsc/petsc4py/src/master/
.. _matplotlib: https://matplotlib.org/
.. _hdf5: https://support.hdfgroup.org/HDF5/
.. _hypre: https://computation.llnl.gov/projects/hypre-scalable-linear-solvers-multigrid-methods
.. _PETSc: http://www.mcs.anl.gov/petsc/
.. _Gmsh: http://gmsh.info/
.. _ParaView: https://www.paraview.org/
.. _Anaconda: https://www.anaconda.com/

Getting Started
---------------

PUNC consists of a Python library, as well as an executable program. The library can be imported as `import punc` in Python, and contains classes and functions that can be glued together like Lego to make highly customized FEM-PIC simulations. The library is documented in terms of docstrings. Beware that PUNC is highly experimental, and not for production use. The command line program ``punc`` is a complete program for plasma-object interaction simulations. It can for instance be run as follows::

    $ cd punc/simulations
    $ punc laframboise.cfg.py

where ```laframboise.cfg.py``` is a configuration file where you specify the plasma parameters and simulation parameters. It is fully Python scriptable, which means you can for instance compute the thermal velocity (needed by PUNC) from temperature, etc. Specific variables in the configuration file is treated as input. What these are, is for now only documented in the top of ```punc/object_interaction.py``` (which is, in fact, the ``punc`` command).

The configuration file also specifies which mesh to use. The attached configuration file ```laframboise.cfg.py``` looks for a collection of XML files starting with ```laframboise_sphere_in_cub_res1```. This is the mesh format used by FEniCS and the mesh is not included in the repository. However, a suite of simulation geometries (```.geo``` files) for Gmsh, including the one needed for the example, is available in the folder ```punc/mesh```. The mesh can be created from the file ```laframboise_sphere_in_cub_res1.geo``` using Gmsh, and converted to FEniCS XML format using FEniCS's built-in command line tool::

    dolfin-convert laframboise_sphere_in_cube_res1.msh  laframboise_sphere_in_cube_res1.xml

Field quantities is written to ```*.pvd``` files which can readily be analyzed using e.g. ParaView, and history data is written for each time-step to ```history.dat```. This can be visualized using the supplied ```monitor.py``` script.
