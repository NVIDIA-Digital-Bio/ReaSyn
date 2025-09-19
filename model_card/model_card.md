## Model Overview

### Description

ReaSyn is a model for predicting the synthesis pathway, reaction steps from reactants to final product(s), for a target product molecule. When the target molecule cannot be synthesized directly using known reaction steps, ReaSyn will generate the pathways for the most structurally similar, synthesizable analog of the target molecule.The model uses an encoder-decoder transformer architecture and a chain-of-reaction notation, where a full synthetic pathway is represented as a text sequence. This approach allows the model to achieve SOTA performance in tasks like synthesis planning and incorporating synthesizability into goal-directed molecular property optimization.

This model is ready for commercial use. 

### License/Terms of Use

GOVERNING TERMS: Use of this model is governed by the [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/). ReaSyn source code is licensed under [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0).

Deployment Geography: Global

Use Case: <br>
ReaSyn is a model for predicting the synthetic pathway, reaction steps from reactants to final product(s), for a target product molecule. The model can be used in the pharmaceutical and chemical industries and in academic research to identify how to synthesize a molecule, help chemists in planning a first time synthesis of a molecule, the optimization of an existing synthesis pathway, or the filtering of candidate molecules based on ease of synthesis. <br>

Release Date: <br>
Github 09/23/2025 via https://github.com/NVIDIA-Digital-Bio/ReaSyn <br>
NGC 09/23/2025 via https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara/resources/reasyn?version=1.0 <br> 

### References
Research paper: “Rethinking Molecule Synthesizability with Chain-of-Reaction”, LINK

### Model Architecture

Architecture Type: Encoder-decoder
Network Architecture: Encoder-decoder Transformer
ReaSyn utilizes an Encoder-decoder Transformer architecture which takes a molecular SMILES as input and outputs its synthetic pathway. Encoder contains 6 layers and decoder contains 10 layers. Both encoder and decoder have a hidden size of 768, 16 attention heads, and a feed-forward dimension of 4096.

The total number of parameters in ReaSyn is 166M.

### Input

Input Types: Text<br>
Input Formats: SMILES string<br>
Input Parameters: One-Dimensional (1D)<br>
Other Properties Related to Input: Maximum input length is 256 tokens.

### Output

Output Types: Text<br>
Output Formats: Chain-of-Reaction sequence (molecular synthetic pathway)<br>
Output Parameters: One-Dimensional (1D)<br>
Other Properties Related to Output: Maximum output length is 768 tokens.

Our AI models are designed and/or optimized to run on NVIDIA GPU-accelerated systems. By leveraging NVIDIA’s hardware (e.g. GPU cores) and software frameworks (e.g., CUDA libraries), the model achieves faster training and inference times compared to CPU-only solutions.


### Software Integration

Runtime Engine: Torch<br>
Supported Hardware Microarchitecture Compatibility: NVIDIA Ampere<br>
Preferred Operating System: Linux, Windows

The integration of foundation and fine-tuned models into AI systems requires additional testing using use-case-specific data to ensure safe and effective deployment. Following the V-model methodology, iterative testing and validation at both unit and system levels are essential to mitigate risks, meet technical and functional requirements, and ensure compliance with safety and ethical standards before deployment.

### Model Versions

ReaSyn v1


## Training and Evaluation Datasets

### Training Datasets

SynFormer Reaction Templates<br>
Link: https://github.com/wenhao-gao/synformer/blob/main/data/rxn_templates/comprehensive.txt<br>
Data Modality: Text<br>
Text Training Data Size: 1 Billion to 10 Trillion Tokens<br>
Data Collection Method by dataset: Human<br>
Labeling Method by dataset: Automated<br>
Properties: 115 molecular reaction templates in the SMARTS format

Building Blocks in Enamine US Stock retrieved in October 2023<br>
Link: https://enamine.net/building-blocks/building-blocks-catalog<br>
Data Modality: Text<br>
Text Training Data Size: 1 Billion to 10 Trillion Tokens<br>
Data Collection Method by dataset: Human<br>
Labeling Method by dataset: N/A<br>
Properties: 115 molecular reaction templates in the SMARTS format

### Evaluation Dataset

Enamine REAL Test Set<br>
Link: https://github.com/wenhao-gao/synformer/blob/main/data/enamine_smiles_1k.txt<br>
https://enamine.net/compound-collections/real-compounds/real-database<br>
Data Collection Method by dataset: Human<br>
Labeling Method by dataset: N/A<br>
Properties: Randomly selected 1k test molecules from Enamine REAL to evaluate synthesizable molecule reconstruction.<br>

ChEMBL Test Set<br>
Link: https://github.com/wenhao-gao/synformer/blob/main/data/chembl_filtered_1k.txt<br>
https://www.ebi.ac.uk/chembl<br>
Data Collection Method by dataset: Human<br>
Labeling Method by dataset: N/A<br>
Properties: Randomly selected 1k test molecules from ChEMBL to evaluate synthesizable molecule reconstruction.<br>

ZINC250k Test Set<br>
Link: https://www.kaggle.com/datasets/basu369victor/zinc250k<br>
Data Collection Method by dataset: Synthetic<br>
Labeling Method by dataset: N/A <br>
Properties: Randomly selected 1k test molecules from ZINC250k to evaluate synthesizable molecule reconstruction.<br>

### Inference

Engine: Torch<br>
Test Hardware: Ampere / NVIDIA A100

### Ethical Considerations

NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications.  When downloaded or used in accordance with our terms of service, developers should work with their internal model team to ensure this model meets requirements for the relevant industry and use case and addresses unforeseen product misuse.

Users are responsible for ensuring the physical properties of model-generated molecules are appropriately evaluated and comply with applicable safety regulations and ethical standards.

For more detailed information on ethical considerations for this model, please see the Model Card++ Explainability, Bias, Safety & Security, and Privacy Subcards.

Please report security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail).
