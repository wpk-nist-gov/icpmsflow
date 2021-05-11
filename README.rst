=========
icpmsflow
=========


.. image:: https://img.shields.io/pypi/v/icpmsflow.svg
        :target: https://pypi.python.org/pypi/icpmsflow

.. image:: https://img.shields.io/travis/wpk-nist-gov/icpmsflow.svg
        :target: https://travis-ci.com/wpk-nist-gov/icpmsflow

.. image:: https://readthedocs.org/projects/icpmsflow/badge/?version=latest
        :target: https://icpmsflow.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status




Package to work with icpms data


* Free software: NIST license
* Documentation: https://icpmsflow.readthedocs.io.


Installation
------------

This project is not yet available via conda or on pypi.  The recommended route is to install most dependencies via conda, then pip install directly from github.  For this, do the following:

If you'd like to create an isolated environment:

.. code-block:: console

    $ conda create -n {env-name} python=3.8

Activate the environment you'd like to install to with:

.. code-block:: console

   $ conda activate {env-name}

Install required dependencies with:

.. code-block:: console

   $ conda install -n {env-name} setuptools numpy pandas openpyxl scipy holoviews bokeh jupyter


Note that `jupyter` is not strictly required, but assumed.  You can also use `ipykernel` to just install the kernel for the target environment.


Finally, install `neutron_analysis` in the active environment do:

.. code-block:: console

   $ pip install git+https://github.com/wpk-nist-gov/icpmsflow.git@develop



Example usage
-------------

See demo notebook : `demo <notebooks/example_usage.ipynb>`_

Example dataset : `example <https://github.com/wpk-nist-gov/icpmsflow/blob/develop/notebooks/example.tgz?raw=true>_`


Features
--------

* TODO

Credits
-------

This package was created with Cookiecutter_ and the `wpk-nist-gov/cookiecutter-pypackage`_ Project template forked from `audreyr/cookiecutter-pypackage`_.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`wpk-nist-gov/cookiecutter-pypackage`: https://github.com/wpk-nist-gov/cookiecutter-pypackage
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
