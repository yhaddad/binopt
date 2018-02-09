===============================
binopt
===============================


.. image:: https://img.shields.io/pypi/v/binopt.svg
        :target: https://pypi.python.org/pypi/binopt


.. image:: https://img.shields.io/travis/yhaddad/binopt.svg
        :target: https://travis-ci.org/yhaddad/binopt


.. image:: https://readthedocs.org/projects/binopt/badge/?version=latest
        :target: https://binopt.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status


.. image:: https://pyup.io/repos/github/yhaddad/binopt/shield.svg
     :target: https://pyup.io/repos/github/yhaddad/binopt/
     :alt: Updates

.. image:: https://zenodo.org/badge/86721620.svg
   :target: https://zenodo.org/badge/latestdoi/86721620


This package is aiming to categorize labeled data in terms of a global figure of merit. In high energy physics, categorization of collision data is done by maximizing the discovery significance. This package run on unbinned binary datasets.

installation
************
Install like any other python package::

pip install binopt --user

or::

git clone git@github.com:yhaddad/binopt.git
cd binopt/
pip install .

Getting started
***************

.. code-block:: python

   sevent = 1000
   bevent = 10000
   X = np.concatenate((
                expit(np.random.normal(+2.0, 2.0, sevent)),
                expit(np.random.normal(-0.5, 2.0, bevent))
   ))
   Y = np.concatenate((
                np.ones(sevent),
                np.zeros(bevent)
   ))
   W = np.concatenate((np.ones(sevent), np.ones(bevent)))

   binner = binopt.optimize_bin(
                nbins=3, range=[0, 1],
                drop_last_bin=True,
                fix_upper=True,
                fix_lower=False,
                use_kde_density=True
   )
   opt = binner.fit(
                X, Y, sample_weights=W,
                method="Nelder-Mead",
                breg=None, fom="AMS2"
   )

   print "bounds : ", opt.x
   print "signif : ", binner.binned_score(opt.x)
   print "Nsig   : ", binner.binned_stats(opt.x)[0]
   print "Nbkg   : ", binner.binned_stats(opt.x)[1]


* Free software: GNU General Public License v3
* Documentation: https://binopt.readthedocs.io.


Features
--------

* TODO

Credits
---------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
