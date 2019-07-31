.. vim: set fileencoding=utf-8 :
.. Tue Nov  7 16:30:33 CET 2017

.. image:: https://img.shields.io/badge/docs-stable-yellow.svg
   :target: https://www.idiap.ch/software/bob/docs/bob/bob.db.siw/stable/index.html
.. image:: https://img.shields.io/badge/docs-latest-orange.svg
   :target: https://www.idiap.ch/software/bob/docs/bob/bob.db.siw/master/index.html
.. image:: https://gitlab.idiap.ch/bob/bob.db.siw/badges/master/build.svg
   :target: https://gitlab.idiap.ch/bob/bob.db.siw/commits/master
.. image:: https://gitlab.idiap.ch/bob/bob.db.siw/badges/master/coverage.svg
   :target: https://gitlab.idiap.ch/bob/bob.db.siw/commits/master
.. image:: https://img.shields.io/badge/gitlab-project-0000c0.svg
   :target: https://gitlab.idiap.ch/bob/bob.db.siw
.. image:: https://img.shields.io/pypi/v/bob.db.siw.svg
   :target: https://pypi.python.org/pypi/bob.db.siw


=================================
 SIW Database Access in Bob
=================================

This package is part of the signal-processing and machine learning toolbox
Bob_. This package provides an interface to the `SIW`_ database.
The original data files need to be downloaded separately.

If you use this database, please cite the following publication::

    @inproceedings{learning-deep-models-for-face-anti-spoofing-binary-or-auxiliary-supervision,
      author = { Yaojie Liu* and Amin Jourabloo* and Xiaoming Liu },
      title = { Learning Deep Models for Face Anti-Spoofing: Binary or Auxiliary Supervision },
      booktitle = { In Proceeding of IEEE Computer Vision and Pattern Recognition },
      address = { Salt Lake City, UT },
      month = { June },
      year = { 2018 },
    }


Installation
------------

Complete Bob's `installation`_ instructions. Then, to install this package,
run::

  $ conda install bob.db.siw


Contact
-------

For questions or reporting issues to this software package, contact our
development `mailing list`_.


.. Place your references here:
.. _bob: https://www.idiap.ch/software/bob
.. _installation: https://www.idiap.ch/software/bob/install
.. _mailing list: https://www.idiap.ch/software/bob/discuss
.. _SIW: http://cvlab.cse.msu.edu/spoof-in-the-wild-siw-face-anti-spoofing-database.html
