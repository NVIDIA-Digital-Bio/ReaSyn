### Dataset Description
ZINC250k is a subset of the ZINC database, which is frequently used in machine learning and computational drug discovery. ZINC is a free database of commercially-available compounds for virtual screening. It contains over 230M purchasable compounds in ready-to-dock, text formats.

We are releasing a subset of ZINC250k, comprising 37,386 building blocks used for model benchmarking.

This dataset is ready for commercial/non-commercial use.

### Dataset Owner(s)
ZINC is provided by the Irwin and Shoichet Laboratories in the Department of Pharmaceutical Chemistry at the University of California, San Francisco (UCSF). 
ZINC-250k is a smaller, standardized subset of the main ZINC database that first appeared in a publication titled "Automatic Chemical Design Using a Data-Driven Continuous Representation of Molecules".

ReaSyn release is re-distributing a subset of ZINC-250k. 

### Dataset Creation Date
Feb 2, 2018

### License/Terms of Use
This dataset is governed by the Open Database License (ODbL) (https://opendatacommons.org/licenses/odbl/1-0/). 

### Intended Usage
ReaSyn uses a subset of the ZINC-250k to build a benchmark for synthetic pathway generation. This dataset can be used by developers in the academic or pharmaceutical industries who want to predict synthesis pathways for molecules and who build artificial intelligence applications to perform synthetic pathway generation.

### Dataset Characterization
Data Collection Method: Human<br>
Labeling Method: N/A

### Dataset Format
ZINC entries are recorded in the SMILES format. It’s a text-based format for representing the structure of chemical molecules using short ASCII strings

### Dataset Quantification
ReaSyn creates a subset of 37,386 building blocks by filtering the ZINC-250k dataset for molecules that have 18 or less heavy atoms. 

The total data storage is approximately 1.1 MB.

### Reference(s)
* ZINC250k
* Sterling, T., & Irwin, J. J. (2015). ZINC 15 – Ligand Discovery for Everyone. Journal of Chemical Information and Modeling, 55(11), 2324–2337. https://doi.org/10.1021/acs.jcim.5b00559
* Gómez-Bombarelli, R., Wei, J. N., Duvenaud, D., Hernández-Lobato, J. M., Sánchez-Lengeling, B., Sheberla, D., Aguilera-Iparraguirre, J., Hirzel, T. D., Adams, R. P., & Aspuru-Guzik, A. (2016). Automatic Chemical Design Using a Data-Driven Continuous Representation of Molecules. arXiv. https://arxiv.org/abs/1610.02415

### Ethical Considerations
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications.  When downloaded or used in accordance with our terms of service, developers should work with their internal model team to ensure this model meets requirements for the relevant industry and use case and addresses unforeseen product misuse.   

Please report security vulnerabilities or NVIDIA AI Concerns [here](https://www.nvidia.com/en-us/support/submit-security-vulnerability/).
