### Dataset Description
The ZINC SMILES 1k test set contains 1k SMILES molecules derived from a subset of the ZINC-250k dataset (37,386 building blocks) and 115 SynFormer reaction templates for synthetic pathway generation benchmarking.

We generated synthetic pathways by randomly applying the SynFormer reactions to the ZINC-250k building blocks and then selecting the resulting product molecules as the 1k test set.

This dataset is ready for commercial/non-commercial use.

### Dataset Owner(s)
NVIDIA Corporation

### Dataset Creation Date
Feb 3rd, 2025

### License/Terms of Use
GOVERNING TERMS: This dataset is governed by the Creative Commons Attribution 4.0 International License (CC-BY-4.0) (https://creativecommons.org/licenses/by/4.0/legalcode.en). 

ADDITIONAL INFORMATION: This dataset contains information from ZINC-250k (https://www.kaggle.com/datasets/basu369victor/zinc250k), which is made available
under the Open Database License (ODbL) (https://opendatacommons.org/licenses/odbl/1-0/).


### Intended Usage
ReaSyn uses a ZINC SMILES 1k to benchmark the model on synthetic pathway generation . This dataset can be used by developers in the academic or pharmaceutical industries who would like to test the performance of the models that predict molecular synthesis pathways. 

### Dataset Characterization
Data Collection Method: Synthetic<br>
Labeling Method: N/A

### Dataset Format
ZINC entries are recorded in the SMILES format. It’s a text-based format for representing the structure of chemical molecules using short ASCII strings.

### Dataset Quantification
This test set contains 1k SMILES molecules. 
The total data storage is approximately 59 KB.

### Reference(s)
ZINC is provided by the Irwin and Shoichet Laboratories in the Department of Pharmaceutical Chemistry at the University of California, San Francisco (UCSF). ZINC-250k is a smaller, standardized subset of the main ZINC database that first appeared in a publication titled "Automatic Chemical Design Using a Data-Driven Continuous Representation of Molecules".

SynFormer reaction templates were introduced in the publication Gao, W., Luo, S., & Coley, C. W. (2024). Generative Artificial Intelligence for Navigating Synthesizable Chemical Space. arXiv. https://arxiv.org/abs/2410.03494

ReaSyn is releasing a test set that was created, based on these datasets. 

* ZINC
* Sterling, T., & Irwin, J. J. (2015). ZINC 15 – Ligand Discovery for Everyone. Journal of Chemical Information and Modeling, 55(11), 2324–2337. https://doi.org/10.1021/acs.jcim.5b00559
* Gómez-Bombarelli, R., Wei, J. N., Duvenaud, D., Hernández-Lobato, J. M., Sánchez-Lengeling, B., Sheberla, D., Aguilera-Iparraguirre, J., Hirzel, T. D., Adams, R. P., & Aspuru-Guzik, A. (2016). Automatic Chemical Design Using a Data-Driven Continuous Representation of Molecules. arXiv. https://arxiv.org/abs/1610.02415
* Gao, W., Luo, S., & Coley, C. W. (2024). Generative Artificial Intelligence for Navigating Synthesizable Chemical Space. arXiv. https://arxiv.org/abs/2410.03494


### Ethical Considerations
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications.  When downloaded or used in accordance with our terms of service, developers should work with their internal model team to ensure this model meets requirements for the relevant industry and use case and addresses unforeseen product misuse.   

Please report security vulnerabilities or NVIDIA AI Concerns [here](https://www.nvidia.com/en-us/support/submit-security-vulnerability/).
