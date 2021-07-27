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

This project is available on pypi and conda in two flavors.  The core package includes only the
packages needed to perform the data analysis.  This can be installed from pip with

.. code-block:: console

   $ pip install icpmsflow

or from conda with

.. code-block:: console

   $ conda install -c wpk-nist icpmsflow


To install the optional plotting/jupyter dependencies, use either

.. code-block:: console

   $ pip install icpmsflow[all]

or

.. code-block:: console

   $ conda install -c wpk-nist icpmsflow-all


If you prefer to install from github, use either

.. code-block:: console

   $ pip install git+https://github.com/wpk-nist-gov/icpmsflow.git@develop

for basic install, or


.. code-block:: console

   $ pip install 'git+https://github.com/wpk-nist-gov/icpmsflow.git@develop#egg=icpmsflow[all]'


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
