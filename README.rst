genheas
=======

genheas (Generate High Entropy Alloys Structures) is a  neural evolution structures (NESs) generation methodology
combining artificial neural networks (ANNs) and evolutionary algorithms (EAs) to generate High Entropy Alloys (HEAs).


Support and Documentation
-------------------------
see docs for documentation, reporting bugs, and getting support.



Installation
-------------------------

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

    $genheas poetry run pytest

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


Usage
-------------------------

Here is the schematic of the workflow to generate HEAs structures:




- train.y

    use small cell to train the model

.. figure:: docs/source/images/workflow.png
   :align: center

- generate.py

    use the trained model to general large cell

.. figure:: docs/source/images/gen_configuration.png
   :align: center



- Therefore, it should looks like:


    1- Train the model  and generate structure
        - Edit the configuration file both training and generation part

        .. sourcecode:: bash
            $genheas cd genheas
            $genheas/genheas  vi parameters.yml

        - run
        .. sourcecode:: bash

            $genheas/genheas poetry run python main.py

    2- Only train a model
        - Edit the configuration file : training part

        .. sourcecode:: bash

            $genheas/genheas vi parameters.yml

        - run
        .. sourcecode:: bash

            $genheas/genheas poetry run python train.py

    3 - Using a pre-trained model to generate cell
        - Edit the configuration file : generation part

        .. sourcecode:: bash

            $genheas/genheas vi parameters.yml

        - run
        .. sourcecode:: bash

            $genheas/genheas poetry run python generate.py

Developing and Contributing
---------------------------
See
`contributing.md <https://https://github.com/CLEANit/genheas/docs/source/contributing.rst>`_
for guidelines on running tests, adding features, coding style, and updating
documentation when developing in or contributing to genheas


Authors
-------

Conrard Tetsassi