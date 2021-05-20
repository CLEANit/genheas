.. _install:

==================
Installing genheas
==================


Normal installation
===================

- Install poetry (https://github.com/python-poetry/poetry)

    + osx / linux / bashonwindows install instructions
        .. sourcecode:: bash

            $ curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/install-poetry.py | python -
            or
            $ pip install poetry

    + windows powershell install instructions
        .. sourcecode:: bash

            $ (Invoke-WebRequest -Uri https://raw.githubusercontent.com/python-poetry/poetry/master/install-poetry.py -UseBasicParsing).Content | python -
            or
            $ pip install poetry

- Once Poetry is installed you can execute the following:

.. sourcecode:: bash

    $ poetry --version

    $ poetry self update

- Clone the repo

.. sourcecode:: bash

    $ git clone  https://github.com/CLEANit/genheas

    $ cd genheas

- install the packages

.. sourcecode:: bash

    $genheas peotry install

    $genheas peotry check
    # $genheas poetry run pytest

    $genheas poetry build


+ Listing the current configuration

    .. sourcecode:: bash

        $genheas poetry config --list

    which will give you something similar to this

    .. sourcecode:: bash

        cache-dir = "/path/to/cache/directory"
        virtualenvs.create = true
        virtualenvs.in-project = null
        virtualenvs.path = "{cache-dir}/virtualenvs"  # /path/to/cache/directory/virtualenvs



Installation for contributors
=============================

