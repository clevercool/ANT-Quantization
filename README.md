# Artifact Evaluation for ANT [MICRO'22] and OliVe [ISCA'23]
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7002115.svg)](https://doi.org/10.5281/zenodo.7002115)

## Citation
If you use ANT or OliVe in your research, please cite our paper:
```bibtex
@inproceedings{guo2022ant,
  title={ANT: Exploiting Adaptive Numerical Data Type for Low-bit Deep Neural Network Quantization},
  author={Guo, Cong and Zhang, Chen and Leng, Jingwen and Liu, Zihan and Yang, Fan and Liu, Yunxin and Guo, Minyi and Zhu, Yuhao},
  booktitle={2022 55th IEEE/ACM International Symposium on Microarchitecture (MICRO)},
  pages={1414--1433},
  year={2022},
  organization={IEEE}
}
```

```bibtex
@article{guo2023olive,
    author = {Guo, Cong and Tang, Jiaming and Hu, Weiming and Leng, Jingwen and Zhang, Chen and Yang, Fan and Liu, Yunxin and Guo, Minyi and Zhu, Yuhao},
    title = {OliVe: Accelerating Large Language Models via Hardware-friendly Outlier-Victim Pair Quantization},
    year = {2023},
    eprint = {arXiv:2304.07493},
    doi = {10.1145/3579371.3589038},
}
```


This repository contains the source code for reproducing the experiments in the paper `"ANT: Exploiting Adaptive Numerical Data Type for Low-bit Deep Neural Network Quantization"` at MICRO'22 and `"OliVe: Accelerating Large Language Models via Hardware-friendly Outlier-Victim Pair Quantization"` at ISCA'23.

## Project Structure

`ant_quantization` contains the ANT framework with PyTorch.

`olive_quantization` contains the OliVe framework with PyTorch.

`ant_simulator` contains the performance and energy evaluation of ANT. 

## License
Licensed under an Apache-2.0 license.
