.. vim: set fileencoding=utf-8 :

.. _bob.db.siw:

=================================
 SIW Database Access in Bob
=================================

This package provides an interface to the `SIW`_ - a mobile face
presentation attack database with real-world variations database. The original
data files need to be downloaded separately.After you have downloaded the
dataset, you need to configure bob.db.siw to find the dataset::

    $ bob config set bob.db.siw.directory /path/to/downloaded/dataset

If you use this database, please cite the following publication::

    @inproceedings{learning-deep-models-for-face-anti-spoofing-binary-or-auxiliary-supervision,
      author = { Yaojie Liu* and Amin Jourabloo* and Xiaoming Liu },
      title = { Learning Deep Models for Face Anti-Spoofing: Binary or Auxiliary Supervision },
      booktitle = { In Proceeding of IEEE Computer Vision and Pattern Recognition },
      address = { Salt Lake City, UT },
      month = { June },
      year = { 2018 },
    }

Package Documentation
---------------------

.. automodule:: bob.db.siw
.. _SIW: http://cvlab.cse.msu.edu/siw-spoof-in-the-wild-database.html

