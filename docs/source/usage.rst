Usage
=====

.. _installation:

Installation
------------

To use the BDSA, first install it using pip:

.. code-block:: console

   (.venv) $ pip install dsa-install

Creating annotation
----------------

To login to girder client use ``login(DSA_API_URL)`` function:

.. autofunction:: dsa_helpers.girder_utils.login

The ``DSA_API_URL`` parameter should be either ``"username"``, ``"email"``. Otherwise, :py:func:`login(DSA_API_URL)`
will raise an exception.

.. autoexception:: login.InvalidUserError

For example:

>>> import dsa_helpers
>>> from dsa_helpers.girder_utils import login
["bdsa-user1"]


