=================
Contributing
=================


Fork on GitHub and clone
----------------------------


- Create a writeable fork on GitHub and clone it to implement your changes



Building HTML Documentation
----------------------------

genheas uses   Sphinx (https://docs.readthedocs.io/en/stable/intro/getting-started-with-sphinx.html) to generate the HTMLK documentation


- Install Sphinx:

.. code-block:: bash

    $ pip install sphinx

- Edit the module.rst. Add the name of the module if you have created new module


.. code-block:: bash

    $ cd  docs
    $ vi module.rst


- Run sphinx-apidoc

sphinx-apidoc is a tool for automatic generation of Sphinx sources that, using the autodoc extension,
document a whole package in the
style of other automatic API documentation tools.

.. code-block:: bash

    $ cd  docs
    $  sphinx-apidoc  -f -o source/ ../

- Use  tox to generate the documentation  see `install.rst <https://github.com/CLEANit/genheas/blob/master/docs/source/installg.rst>`_



Getting Started with Sphinx
----------------------------
This is just for information  please do not run the following command. The docs directory already exist

- Create a directory inside your project to hold your docs:

.. code-block:: bash

    $ cd /path/to/project
    $ mkdir docs

- Run sphinx-quickstart in there:

.. code-block:: bash

    $ cd  docs
    $ sphinx-quickstart

This quick start will walk you through creating the basic configuration;
in most cases, you can just accept the defaults. When it’s done, you’ll have an index.rst,
a conf.py and some other files. Add these to revision control.

Now, edit your index.rst and add some information about your project. Include as much detail as you like

pre-commit install
git commit -m "Release v0.1.6" pyproject.toml

For the contribution

# language: python
# python:
# - 3.6
# env:
#   global:
#   - secure: "<encrypted MYPYPI_USER=username>"
#   - secure: "<encrypted MYPYPI_PASS=p@ssword>"
# before_install:
# - pip install poetry
# install:
# - poetry install
# script:
# - poetry run pre-commit install
# - poetry run pre-commit
# - poetry run flake8 my_package test
# - poetry run coverage run --source=my_package -m unittest discover -b
# before_deploy:
# - poetry config repositories.mypypi http://mypypi.example.com/simple
# - poetry config http-basic.mypypi $MYPYPI_USER $MYPYPI_PASS
# - poetry build -f sdist
# deploy:
#   provider: script
#   script: poetry publish -r mypypi
#   skip_cleanup: true
#   on:
#     tags: true



