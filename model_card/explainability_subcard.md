# Explainability Subcard
Field                                                                                                  |  Response
:------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------
Intended Application(s) & Domain(s)                                                                   |  Molecular drug discovery and design
Model Type                                                                                            |  Molecular synthesis pathway generation
Intended Users                                                                                        |  Developers in the academic or pharmaceutical industries who want to predict synthesis pathways for molecules and who build artificial intelligence applications to perform property guided molecule optimization and novel molecule generation.
Output                                                                                                |  Text
Describe how the model works                                                                          |  ReaSyn uses a Transformer encoder-decoder architecture and requires a "target molecule" as its input (SMILES format). The model then generates a synthetic pathway for a synthesizable molecule that that molecule, or, if needed, an analog of the input molecule.
Name the adversely impacted groups this has been tested to deliver comparable outcomes regardless of  |  Not Applicable
Technical Limitations                                                                    |  The model's reasoning steps, while incorporating reactants, reaction types, and intermediate products, do not account for other important reaction information like environmental conditions or yields. This could be a limitation in real-world drug discovery scenarios.
Verified to have met prescribed quality standards?  |  Yes
Performance Metrics                                                                                   |  Reconstruction rate, Similarity, Diversity (Pathway), Diversity (BB).
Potential Known Risks                                                                                 |  The framework, while effective for generating drug candidates, also has the possibility of generating synthetic pathways for toxic drugs. This requires an additional scheme to be adopted to filter out harmful molecules during the generation search.
Licensing & Terms of Use                                                                                             |  Use of the model is governed by the [NVIDIA Open Model License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/).
