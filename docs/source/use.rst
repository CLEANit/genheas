=================
How to use genheas
=================

Overview
========

Generate High Entropy Alloys Structures

+ A  neural evolution structures (NESs) generation methodology
combining artificial neural networks (ANNs) and evolutionary algorithms (EAs) to generate High Entropy Alloys (HEAs)



genheas workflow
================

Here is the schematic of the workflow to generate HEAs structures:



+ train.y

    use small structure to train the model


.. figure:: ./images/workflow.png
   :align: center


+ generate.py

    use the trained model to general big structures


.. figure:: ./images/gen_configuration.png
   :align: center

Therefore, it should looks like:

.. sourcecode:: bash

    # create subdirectory:
    mkdir new_directory
    cd new_directory

    # create the inputs parameter file:
    touch  parameters.yml

    # run
    run.py


genheas
=====
