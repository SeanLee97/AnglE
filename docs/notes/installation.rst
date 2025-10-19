⬇️ Installation
================================

Install AnglE quickly and start building powerful sentence embeddings.

----

📦 Installation Methods
====================================

Method 1: Using uv (Recommended)
------------------------------------

For fast installation, use `uv <https://github.com/astral-sh/uv>`_ package manager:

.. code-block:: bash

    uv pip install -U angle-emb

.. tip::
   ``uv`` is significantly faster than pip for dependency resolution and installation.

----

Method 2: Manual Installation
------------------------------------

Install directly from the GitHub repository:

.. code-block:: bash

    git clone https://github.com/SeanLee97/AnglE.git
    cd AnglE
    python -m pip install -e .

The ``-e`` flag installs in editable mode, useful for development.
