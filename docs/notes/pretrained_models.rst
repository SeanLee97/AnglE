🏛️ Official Pretrained Models
================================



BERT-based models:
------------------------------

+------------------------------------+-------------+-------------------+--------------------------+
| 🤗 HF                              | Max Tokens  | Pooling Strategy  | Scenario                 |
+====================================+=============+===================+==========================+
| `WhereIsAI/UAE-Large-V1`_          | 512         | cls               | English, General purpose |
+------------------------------------+-------------+-------------------+--------------------------+
| `WhereIsAI/UAE-Code-Large-V1`_     | 512         | cls               | Code Similarity          |
+------------------------------------+-------------+-------------------+--------------------------+
| `WhereIsAI/pubmed-angle-base-en`_  | 512         | cls               | Medical Similarity       |
+------------------------------------+-------------+-------------------+--------------------------+
| `WhereIsAI/pubmed-angle-large-en`_ | 512         | cls               | Medical Similarity       |
+------------------------------------+-------------+-------------------+--------------------------+

.. _WhereIsAI/UAE-Large-V1: https://huggingface.co/WhereIsAI/UAE-Large-V1
.. _WhereIsAI/UAE-Code-Large-V1: https://huggingface.co/WhereIsAI/UAE-Code-Large-V1
.. _WhereIsAI/pubmed-angle-base-en: https://huggingface.co/WhereIsAI/pubmed-angle-base-en
.. _WhereIsAI/pubmed-angle-large-en: https://huggingface.co/WhereIsAI/pubmed-angle-large-en


LLM-based models:
------------------------------

+------------------------------------+-----------------------------+------------------+--------------------------+------------------+---------------------------------+
| 🤗 HF (lora weight)                | Backbone                    | Max Tokens       | Prompts                  | Pooling Strategy | Scenario                        |
+====================================+=============================+==================+==========================+==================+=================================+
| `SeanLee97/angle-llama-13b-nli`_   | NousResearch/Llama-2-13b-hf | 4096             | ``Prompts.A``            | last             | English, Similarity Measurement |
+------------------------------------+-----------------------------+------------------+--------------------------+------------------+---------------------------------+
| `SeanLee97/angle-llama-7b-nli-v2`_ | NousResearch/Llama-2-7b-hf  | 4096             | ``Prompts.A``            | last             | English, Similarity Measurement |
+------------------------------------+-----------------------------+------------------+--------------------------+------------------+---------------------------------+

.. _SeanLee97/angle-llama-13b-nli: https://huggingface.co/SeanLee97/angle-llama-13b-nli
.. _SeanLee97/angle-llama-7b-nli-v2: https://huggingface.co/SeanLee97/angle-llama-7b-nli-v2


📢 More pretrained models are coming soon!