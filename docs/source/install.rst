.. _install:

=================
Installing genheas
=================


Normal installation
===================

to install the latest version of  genheas:

.. code-block:: bash

  pip3 install --user --upgrade pip

.. code-block:: bash

  echo ‘PATH=$HOME/.local/bin:$PATH’ >> ~/.bashrc

On the other hand, you can install it in a *virtualenv*

Installation for contributors
=============================
To contribute to the project, you need to clone the repository:

+ Clone it: ``git clone https://github.com/CLEANit/pyHEA``.
+ Create virtualenv and activate it:

.. code-block:: bash

  virtualenv venv --python=python3
  # activate virtualenv (you need to do that every time)
  source venv/bin/activate

+ Install (dev) dependencies : ``pip install-dependencies-dev``.
+ Finally, “install” the pakage: ``pip install -e .``
+ Don’t forget to create a separate branch to implement your changes (see `the contribution part <contributing.html>`_)

You can launch the tests series with ``tox``
