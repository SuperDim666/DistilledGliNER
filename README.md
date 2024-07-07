<head>
  <link rel="stylesheet" type="text/css" href="styles.css">
</head>
<a class="top-link hide" href="#top">↑</a>
<a name="top"></a>

# Knowledge Distillation targeted at GliNER
by [Zixiang Xu](mailto:zxu635@gatech.edu), [Runmin Ma](mailto:rma308@gatech.edu), [Yiwei Wang](mailto:ywang3607@gatech.edu)

## Category
* [Abstract](#abstract)
* [Prior Related Work](#priorrelatedwork)
* [Abstract](#abstract)
* [Abstract](#abstract)
* [Abstract](#abstract)
* [Abstract](#abstract)
* [Abstract](#abstract)
* [Abstract](#abstract)
* [Abstract](#abstract)
* [References](#references)

## <a id="abstract"></a>Abstract 
Knowledge Distillation with GliNER explores the usage and effectiveness of knowledge distillation in improving the efficiency of large Natural Language Processing (NLP) models. The GliNER model ([Zaratiana et al., 2023](#ref4)) focuses on Named Entity Recognition tasks that aims to identify any entity type in sentences using a bidirectional transformer encoder similar to BERT model. By leveraging large pre-trained GliNER model, our study compares the performance of smaller Long short-term memory networks (LSTM) trained with distilled knowledge from GliNER against the performance of those trained with raw data. Through comprehensive testing, knowledge distillation shows benefits and confidence in transferring the capabilities of a large and complex NLP model to smaller and more efficient models.

### <a id="introduction"></a>1. Introduction
The need and quest for more efficient and accurate models is constant. Knowledge distillation can be a great help where a smaller model learns from a larger and more complex model. In our work, we investigate the effectiveness of knowledge distillation in enhancing the performance of smaller natural language processing models, particularly focusing on named entity recognition tasks on lightweight LSTM model. We compare the performance of smaller models trained directly on raw data against the same models trained with knowledge data distilled from output probability distribution of the larger pre-trained GliNER model.

The pre-trained GliNER model ([Zaratiana et al., 2023](#ref4)) has been trained on extensive data and has demonstrated strong performance in zero-shot evaluations on various Named Entity Recognition (NER) benchmarks. Named Entity Recognition has significant importance in various Natural Language Processing applications. On the other hand, the smaller models are chosen to represent different paradigms in sequence modeling. The two smaller models are compared based on multiple metrics including model accuracy, computational efficiency, and parameter count. By examining their performance, we aim to discern the advantages and limitation of knowledge distillation in the context of natural language processing tasks.

This approach provides insights into knowledge distillation techniques for improving NLP models and balancing between model size and performance. Exploring the impact of knowledge distillation can be helpful for advancement and developing more scalable models. We will evaluate how well knowledge distillation works by comparing the accuracy of models trained with and without distillation and see how they compare against the large model’s performance. This comparison will provide insights into the utility of knowledge distillation in transferring the capabilities of a large, complex model to smaller, more efficient models.

### <a id="priorrelatedwork"></a>2. Prior Related Work


## <a id="references"></a>References
<a id="ref1"></a>Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. 2015. [Distilling the knowledge in a neural network](http://arxiv.org/abs/1503.02531).

<a id="ref2"></a>Ahmad Rashid, Vasileios Lioutas, Abbas Ghaddar, and Mehdi Rezagholizadeh. 2020. [Towards zero-shot knowledge distillation for natural language processing](http://arxiv.org/abs/2012.15495).

<a id="ref3"></a>Ziqing Yang, Yiming Cui, Zhipeng Chen, Wanxiang Che, Ting Liu, Shijin Wang, and Guoping Hu. 2020. [Textbrewer: An open-source knowledge distillation toolkit for natural language processing](https://doi.org/10.18653/v1/2020.acl-demos.2). In *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics: System Demonstrations*. Association for Computational Linguistics.

<a id="ref4"></a>Urchade Zaratiana, Nadi Tomeh, Pierre Holat, and Thierry Charnois. 2023. [Gliner: Generalist model for named entity recognition using bidirectional transformer](http://arxiv.org/abs/2311.08526).

<a class="top-link hide" href="#top">↑</a>
<a name="top"></a>