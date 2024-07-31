.. AnglE documentation master file, created by
   sphinx-quickstart on Sun Jan 21 20:04:06 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

AnglE 📐
=================================

.. image:: https://img.shields.io/badge/Arxiv-2309.12871-yellow.svg?style=flat-square
   :target: https://arxiv.org/abs/2309.12871

.. image:: https://img.shields.io/pypi/v/angle_emb?style=flat-square
   :alt: PyPI version
   :target: https://pypi.org/project/angle_emb/

.. image:: https://img.shields.io/pypi/dm/angle_emb?style=flat-square
   :alt: PyPI version
   :target: https://pypi.org/project/angle_emb/

.. image:: https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/angle-optimized-text-embeddings/semantic-textual-similarity-on-sick-r-1
   :target: https://paperswithcode.com/sota/semantic-textual-similarity-on-sick-r-1?p=angle-optimized-text-embeddings

.. image:: https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/angle-optimized-text-embeddings/semantic-textual-similarity-on-sts16
   :target: https://paperswithcode.com/sota/semantic-textual-similarity-on-sts16?p=angle-optimized-text-embeddings

.. image:: https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/angle-optimized-text-embeddings/semantic-textual-similarity-on-sts15
   :target: https://paperswithcode.com/sota/semantic-textual-similarity-on-sts15?p=angle-optimized-text-embeddings

.. image:: https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/angle-optimized-text-embeddings/semantic-textual-similarity-on-sts14
   :target: https://paperswithcode.com/sota/semantic-textual-similarity-on-sts14?p=angle-optimized-text-embeddings

.. image:: https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/angle-optimized-text-embeddings/semantic-textual-similarity-on-sts13
   :target: https://paperswithcode.com/sota/semantic-textual-similarity-on-sts13?p=angle-optimized-text-embeddings

.. image:: https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/angle-optimized-text-embeddings/semantic-textual-similarity-on-sts12
   :target: https://paperswithcode.com/sota/semantic-textual-similarity-on-sts12?p=angle-optimized-text-embeddings

.. image:: https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/angle-optimized-text-embeddings/semantic-textual-similarity-on-sts-benchmark
   :target: https://paperswithcode.com/sota/semantic-textual-similarity-on-sts-benchmark?p=angle-optimized-text-embeddings

📢 **Train/Infer Powerful Sentence Embeddings with AnglE.**

This library is from the paper `Angle-optimized Text Embeddings <https://arxiv.org/abs/2309.12871>`_ .
It allows you to train state-of-the-art BERT/LLM-based sentence embeddings with just a few lines of code. 
AnglE is also a general sentence embedding inference framework, allowing for infering a variety of transformer-based sentence embeddings.



✨ Features
--------------


**Loss**:

1) 📐 AnglE loss
2) ⚖ Contrastive loss
3) 📏 CoSENT loss
4) ☕️ Espresso loss (previously known as 2DMSE)

**Backbones**:

1) BERT-based models (BERT, RoBERTa, ELECTRA, ALBERT, etc.)
2) LLM-based models (LLaMA, Mistral, Qwen, etc.)
3) Bi-directional LLM-based models (LLaMA, Mistral, Qwen, OpenELMo, etc.. refer to: https://github.com/WhereIsAI/BiLLM)

**Training**:

1) Single-GPU training
2) Multi-GPU training


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   notes/quickstart.rst
   notes/installation.rst
   notes/tutorial.rst
   notes/pretrained_models.rst
   notes/training.rst
   notes/evaluation.rst
   notes/citation.rst


.. toctree::
   :maxdepth: 1
   :caption: APIs:
