# Model Privacy 

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Introduction

We develop a new statistical framework named *Model Privacy* to study model stealing attacks and defenses. This package provides implementation of common attack and defense algorithms, along with the novel proposed defense mechanisms.  


## Installation

### Prerequisites
See `requirements.txt`.


### Build from source
```bash
git clone https://github.com/KeyWgh/ModelPrivacy.git
cd ModelPrivacy
python setup.py install
```


## Quick Start

All paper results can be reproduced by corresponding python files located under the main folder. In particular, Figure 1 in the main paper and Figure 2 nad 3 in the supplementary document are performed by `poly_exp.py` and plotted by `plot_result.ipynb`; Figure 4 in the supplementary document are performed by `krr_exp.py` and plotted by `plot_result.ipynb`; Figure 3 and 4 in the main paper and Figure 5-10 in the supplementary document are performed by `high_dim_exp.py` and plotted by `plot_result.ipynb`; Figure 5 and 6 in the main paper are performed by `pretrain.py`, `steal_bert.py` and plotted by `sentence_classify.ipynb`; Figure 11 and 12 in the supplementary document are performed by `mnist.py` and plotted by `mnist.ipynb`.


## Documentation
Read the Docs: TBA

## Authors
Ganghua Wang <ganghua@uchicago.edu> <keywgh@gmail.com>

## References

@article{wang2026model,
  title={Model Privacy: A Unified Framework for Understanding Model Stealing Attacks and Defenses},
  author={Wang, Ganghua and Yang, Yuhong and Ding, Jie},
  journal={Submitted to JRSSB, under minor revision},
  year={2026},
}

## License
This software is released under MIT License.