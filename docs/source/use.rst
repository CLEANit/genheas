=================
How to use PyHEA
=================

Overview
========

PyHEA is a series of python codes and  scripts for High Entropy Alloys (HEAs)

+ hea-gen : A  Combination of  Artificial Neural Network (ANN) and a Genetic
    Algorithm (GA) to generate HEAs structure
+ pyhea : ML model to predict HEAs properties


hea-gen workflow
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
    hea_gen.py


pyhea
=====
