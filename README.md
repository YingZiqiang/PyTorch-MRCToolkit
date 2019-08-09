# PyTorch Machine Reading Comprehension Toolkit
## Introduction
**The PyTorch Machine Reading Comprehension (PyTorch-MRC)** toolkit, which was rewritten on the basis of Sogou Machine Reading Comprehension (SMRC), was designed for the fast and efficient development of modern machine comprehension models, including both published models and original prototypes.

## Need Teammates!
The whole project is written and maintained by me alone, so I hope that some friends who like NLP and are interested in MRC will work with me to maintain it. Please contact me by email at yingzq0116@163.com.

## Toolkit Architecture

## Installation

## Quick Start

## Modules
1. `data`
    - vocabulary.py: Vocabulary building, word/char index mapping and pretrained word embedding building.
    - batch_generator.py: Mapping words and tags to indices and building them by [*PyTorch Dataset*](https://pytorch.org/docs/stable/data.html?highlight=dataset#torch.utils.data.Dataset), padding length-variable features dynamically, transforming all of the features into tensors, and batching them by [*PyTorch DataLoader*](https://pytorch.org/docs/stable/data.html?highlight=dataloader#torch.utils.data.DataLoader).
2. `dataset`
    - squad.py: Dataset reader and evaluator (from official code) for SQuAD 1.1
3. `examples`
    - Examples for running different models, where the specified data path should provided to run the examples
4. `model`
    - Base class and subclasses of models, where any model should inherit the base class
    - Built-in models such as BiDAF, R-Net and QANet
5. `nn`
    - attention.py: Attention functions such as BiAttention, Trilinear and MultiHeadAttention
    - layers: commonly used layers in PyTorch Machine Reading Comprehension, such as VariationalDropout, Highway and PointerNetwork
    - recurrent: Special wrappers for LSTM and GRU
    - similarity\_function.py: Similarity functions for attention, such as dot_product, trilinear, and symmetric_nolinear
    - util: some useful utility functions such as sequence_mask, weighted_sum and masked_softmax
6. `utils`
    - tokenizer.py: Tokenizers that can be used for both English and Chinese
    - feature_extractor: Extracting linguistic features used in some papers, e.g., POS, NER, and Lemma

## Custom Model and Dataset

## Performance

### F1/EM score on SQuAD 1.1 dev set
| Model | toolkit implementation | original paper|
| --- | --- | ---|
|BiDAF | 77.8/68.1  | 77.3/67.7 |
|R-Net(sogou) | 79.0/70.5 | 79.5/71.1 |
|R-Net(hkust) | 78.3/69.8 | 79.5/71.1 |
|IARNN-Word | - | - |
|IARNN-hidden | - | - |
|DrQA | - | 78.8/69.5  |
|FusionNet | - | 82.5/74.1  |
|QANet | - | 82.7/73.6  |
|BERT-Base | - | 88.5/80.8 |

### F1/EM score on SQuAD 2.0 dev set

### F1 score on CoQA dev set

## Contact information
For help or issues using this toolkit, please submit a GitHub issue or by email yingzq0116@163.com.

## Additional information
When implementing the MRC model, **sometimes I didn't follow the paper reproduction model completely**, because some parts of the paper were not clear to me or I didn't think they play a decisive role. So here's a description. Next I'll list the changes I've made.

## Reference Code
- [sogou MRCToolkit](https://github.com/sogou/SMRCToolkit)
- [allenai bi-att-flow](https://github.com/allenai/bi-att-flow)
- [BiDAF-pytorch](https://github.com/galsang/BiDAF-pytorch.git)

## Reference Paper
- [Match-LSTM](https://arxiv.org/pdf/1608.07905.pdf)
- [BIDAF](https://arxiv.org/pdf/1611.01603.pdf)
- [R-NET](https://www.microsoft.com/en-us/research/wp-content/uploads/2017/05/r-net.pdf)
- [Highway Networks](https://arxiv.org/pdf/1505.00387.pdf)
- [CNN for Sentence Classification](https://arxiv.org/pdf/1408.5882.pdf)
- [Pointer Networks](https://arxiv.org/pdf/1506.03134.pdf)
- [Variational Dropout](https://arxiv.org/pdf/1512.05287.pdf)

## License
