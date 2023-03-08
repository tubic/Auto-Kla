<div align="center">
<br>
<a href="#"><img src="https://user-images.githubusercontent.com/24775272/210609554-e5918fe0-c845-493d-b362-4ff9cb473c57.png"/> </a>
  <h4 align="center">A novel web server to discriminate lysine lactylation sites using automated machine learning.</h4>
</div>

<p align="center">
  <a href="#">
    <img src="https://img.shields.io/badge/python-3.7%20%7C%203.8%20%7C%203.9-blue">
  </a>
  <a href="#">
    <img src="https://img.shields.io/badge/AutoGluon-0.5.2-blue">
  </a>
  <a href="./LICENSE">
      <img src="https://img.shields.io/badge/license-Apache%202.0-green">
  </a>
</p>

## Intruduction
Recently, lysine lactylation (Kla), a novel post-translational modification (PTM), which can be stimulated by lactate, has been found to regulate gene expression and life activities. Therefore, it is imperative to identify Kla sites accurately. Currently, mass spectrometry technology is the fundamental method to identify PTM sites. However, it would be expensive and time-consuming for most researchers to achieve this through experiments alone. Here, we propose a novel web server, Auto-Kla, for accurately predicting Kla sites in gastric cancer cells based on automated machine learning and transformer-based model. Our model achieves reliable performance and outperforms another recently published model in 10-fold cross-validation. To investigate the generalizability and transferability of our approach, we evaluated the performance of our models trained on two other widely studied types of PTM, including phosphorylation sites in host cells infected with SARS-CoV-2 and lysine crotonylation sites. The results show that our models achieve a close or better performance than current outstanding models. We anticipate this method to become a useful analytical tool for PTM prediction research and provide a reference for the future development of related models. The web server in this study are freely available at http://tubic.org/Kla.

## Requirements

The project mainly uses two python packages, `biopython` and `AutoGluon v0.5.2`. Biopython is used to read amino acid sequences, and AutoGluon is used to quickly build a deep learning model for PTM recognition. AutoGluon requires python version 3.7, 3.8, or 3.9. We recommend using the pre-compiled binary wheels available on PyPI. For more details on the specific version numbers of required packages, see <a href='./Requirements.txt'>Requirements.txt</a>. This file is the result of using the command line `pip3 freeze > Requirements.txt` after creating a virtual environment in conda and installing all required packages.

- biopython installation :
```biopython
pip3 install biopython
```

-  AutoGluon installation : select the corresponding command line to install the required version according to your hardware.

    - **CPU:**
   
    ```
    pip3 install -U pip
    pip3 install -U setuptools wheel

    # CPU version of pytorch has smaller footprint - see installation instructions in
    # pytorch documentation - https://pytorch.org/get-started/locally/
    pip3 install torch==1.12+cpu torchvision==0.13.0+cpu torchtext==0.13.0 -f https://download.pytorch.org/whl/cpu/torch_stable.html

    pip3 install autogluon==0.5.2
    ```
    
    - **GPU:**

    ```
    pip3 install -U pip
    pip3 install -U setuptools wheel

    # Install the proper version of PyTorch following https://pytorch.org/get-started/locally/
    pip3 install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchtext==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu113

    pip3 install autogluon==0.5.2
    ```
    - **Sanity check that your installation is valid and can detect your GPU via testing in python:**
    ```
    import torch
    print(torch.cuda.is_available())  # Should be True
    print(torch.cuda.device_count())  # Should be > 0
    ```
## Getting Started

After installing the required dependencies, you can quickly start with the following command
```
git clone https://github.com/bioinformatica/Auto-Kla.git
```
<!-- TABLE OF CONTENTS -->
<details open>
  <summary>Description of files</summary>
  <ol>
    <li>
      <a href="./Codes">Codes</a>
      <ol>
        <li><a href="./Codes/predict.py">predict.py</a></li>
        <li><a href="./Codes/confusion_matrix.py">confusion_matrix.py</a></li>
        <li><a href="./Codes/t-SNE.py">t-SNE.py</a></li>
      </ol>
    </li>
    <li>
      <a href="./Datasets">Datasets</a> 　(10-fold cross-validation data sets of the three PTMs used in the manuscript)
      <ul>
        <li><a href="./Datasets/lactylation">lactylation</a></li>
        <li><a href="./Datasets/crotonylation">crotonylation</a></li>
        <li><a href="./Datasets/phosphorylation">phosphorylation</a></li>
      </ul>
    </li>
    <li>
      <a href="./Confusion matrix">Confusion matrix</a> 　(Confusion matrix for 10-fold cross-validation results)
      <ul>
        <li><a href="./Confusion matrix/lactylation">lactylation</a></li>
        <li><a href="./Confusion matrix/crotonylation">crotonylation</a></li>
        <li><a href="./Confusion matrix/phosphorylation">phosphorylation</a></li>
      </ul>
    </li>
    <li>
      <a href="./t-SNE">t-SNE</a> 　(Visualize test data sets using t-SNE)
      <ul>
        <li><a href="./t-SNE/lactylation.png">lactylation</a></li>
        <li><a href="./t-SNE/crotonylation.png">crotonylation</a></li>
        <li><a href="./t-SNE/phosphorylation.png">phosphorylation</a></li>
      </ul>
    </li>
  </ol>
</details>

## Running the test

Download models from <a href="https://1drv.ms/u/s!AriBkEcpGFzJk1_uB4eUqDbeRnQe?e=jYKUMc">network disk</a>, running following command to decompress lactylation model.

```
unzip -v lactylation_model.zip
```
<br>

Running following command will use model_9 to predict the `example.fasta` sequences file. 

Before running, please make sure that the model file exists in the current path.
```
python predict.py
```
<br>

Running following command will use model_9 to predict `test.fasta` and get a confusion matrix to evaluate the prediction results.

```
python confusion_matrix.py
```

<img src="https://user-images.githubusercontent.com/24775272/212626675-527a70f1-a307-4919-adc4-9700960a1333.png" width=550 >

<br>

Running following command allow us to obtain amino acid sequences' embedding vectors extracted from intermediate neural network representation.

```
python t-SNE.py
``` 

<img src="https://user-images.githubusercontent.com/24775272/212627657-67e26288-3366-4781-a9bd-94b0b717aec3.png" width=500px >

## Evaluation metrics on the test set
- lactylation

| Model|  SEN (%)|  SPE (%)|  PRE (%)|  ACC (%)|  MCC (%)|  AUROC (%)|
| --- | --- |--- |--- |--- |--- |--- |
| Auto-Kla| **70.31±5.36**|93.25±2.22|  **51.60±5.97**| **91.21±1.58**| **55.40±2.33**| **92.34±0.62**|
| DeepKla |  53.95±12.71| **93.30±3.91**| 47.10±7.37| 89.80±2.50| 44.20±2.23| 87.73±1.29|

<br>

- crotonylation

| Model|  SEN (%)|  SPE (%)|  PRE (%)|  ACC (%)|  MCC (%)|  AUROC (%)|
| --- | --- |--- |--- |--- |--- |--- |
|Auto-Kcr | 83.84±2.14| 86.70±1.37| **86.35±1.00**| **85.27±0.55**| **70.62±1.00**| **92.84±0.22**|
|Adapt-Kcr| **84.9**| 85.4| 85.0| **85.2**| **70.6**| 92.4|
|BERT-Kcr |83.8|  80.1| 83.2| 82.0| 64.0| 90.5|
|Deep-Kcr|  63.03|  **87.09**|  83.00|  75.09|  51.63|  85.91|

<br>

- phosphorylation

| Model|  SEN (%)|  SPE (%)|  PRE (%)|  ACC (%)|  MCC (%)|  AUROC (%)|
| --- | --- |--- |--- |--- |--- |--- |
|Auto-ST  |**88.15±2.26**|  78.58±3.20| **80.55±2.04**| **83.36±0.73**| **67.13±1.28**| **91.79±0.20**|
|Adapt-ST|  80.90 |**85.72**| - |83.32| 66.70|  91.20|
|DeepIPs  |79.61  |83.50| - |80.63  |63.16| 89.37|
|Bert-ST  |80.07  |74.60  |-  |79.84  |60.00| 88.90|
|DeepPSP  |76.65| 83.78 |-  |80.21  |60.58| 87.62|
|MusiteDeep2020|  82.95|  78.96 |-  |80.95| 61.96|  88.67|
|MusiteDeep2017 |78.87| 81.46 |-  |80.17  |60.35| 87.98|



## Citation
If you use Auto-Kla in your work, please cite the following paper:

Fei-liao Lai, Feng Gao* (2023) Auto-Kla: A novel web server to discriminate lysine lactylation sites using automated machine learning. Briefings in Bioinformatics,  bbad0770.

BibTeX entry:

```bibtex
@article{auto-kla,
  title={Auto-Kla: A novel web server to discriminate lysine lactylation sites using automated machine learning},
  author={Fei-liao Lai, Feng Gao},
  journal={Briefings in Bioinformatics},
  year={2023}
}
```

## License
>You can check out the full license [here](./LICENSE)
>
This project is licensed under the terms of the **Apache 2.0** license.
