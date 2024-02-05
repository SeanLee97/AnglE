🚀 Quick Start
================================

A few lines of code to generate sentence embeddings using AnglE.

1) Install the latest AnglE as follows:

.. code-block:: bash

    python -m pip install -U angle_emb


Other installation methods, please refer to the `Installation` section.

2) Load pretrained models and encode text.

.. code-block:: python

    from angle_emb import AnglE
    
    angle = AnglE.from_pretrained('WhereIsAI/UAE-Large-V1').cuda()
    vec = angle.encode("I'll take a break.")