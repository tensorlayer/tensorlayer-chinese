API - 自然语言处理
======================

自然语言处理与词向量。

.. automodule:: tensorlayer.nlp

.. autosummary::

   generate_skip_gram_batch

   sample
   sample_top

   SimpleVocabulary
   Vocabulary
   process_sentence
   create_vocab

   simple_read_words
   read_words
   read_analogies_file
   build_vocab
   build_reverse_dictionary
   build_words_dataset
   save_vocab

   words_to_word_ids
   word_ids_to_words

   basic_tokenizer
   create_vocabulary
   initialize_vocabulary
   sentence_to_token_ids
   data_to_token_ids

   moses_multi_bleu

训练嵌入矩阵的迭代函数
------------------------------

.. autofunction:: generate_skip_gram_batch


抽样方法
-------------------

简单抽样
^^^^^^^^^^^^^^^^^^^
.. autofunction:: sample

从top k中抽样
^^^^^^^^^^^^^^^^^^^^
.. autofunction:: sample_top


词的向量表示
-------------------------------

词汇类 (class)
^^^^^^^^^^^^^^^^^

Simple vocabulary class
^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: SimpleVocabulary

Vocabulary class
^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: Vocabulary

Process sentence
^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: process_sentence

Create vocabulary
^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: create_vocab


从文件中读取文本
----------------

Simple read file
^^^^^^^^^^^^^^^^^^
.. autofunction:: simple_read_words

Read file
^^^^^^^^^^^^^^^^^^
.. autofunction:: read_words

从文件中读取类比题目
--------------------------------------------------------

.. autofunction:: read_analogies_file

建立词汇表、文本与ID转换字典及文本ID化
--------------------------------------------------------

为单词到ID建立字典
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: build_vocab

为ID到单词建立字典
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: build_reverse_dictionary

建立字典，统计表等
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: build_words_dataset

保存词汇表
^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: save_vocab

文本转ID，ID转文本
--------------------------------------------------------
These functions can be done by ``Vocabulary`` class.

单词到ID
^^^^^^^^^^
.. autofunction:: words_to_word_ids

ID到单词
^^^^^^^^^^^
.. autofunction:: word_ids_to_words

机器翻译相关函数
---------------------------

文本ID化
^^^^^^^^^^^^^^^^
.. autofunction:: basic_tokenizer

建立或读取词汇表
^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: create_vocabulary
.. autofunction:: initialize_vocabulary

文本转ID，ID转文本
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: sentence_to_token_ids
.. autofunction:: data_to_token_ids


衡量指标
---------------------------

BLEU
^^^^^^^^^^^^^^^^^^^
.. autofunction:: moses_multi_bleu
