# Dissertation-project

[Live demo!](https://huggingface.co/spaces/Amrita25/Agent)

## Agent description
## Tools
- Lipinski tool calculates:
  * Molecular weight
  * LogP (lipophilicity)
  * Hydrogen bond donors/acceptors
  * Polar surface area
  * Rotatable bonds
  * Aromatic rings
  * Undersirable functional groups 
  * Evaluates drug-likeness by calcualting Quantitative Estimate of Drug-Likeness (QED)  
 - Substitution
   * Generates structural analogues using systematic structural modification
   * Evaluates analogue quality using QED
   * Enables rapid exploration of chemcial space 
- Pharmacophore
  * Compares pharmacophore features between molecules
  * Identifies shared functional features
  * Evaluates structural similarity using pharmacophore overlap scores
  * Assesses potential binding efficiency 
 
## AI agent workflow
  * Implements a structured AI agent capable of:
  * Selecting appropriate chemiformatics tools based on query context
  * Integrating tool outputs into scientifically accurate responses
  * Revising outputs using tool-verifired molecular data
  * Minimising hallucination through structures prompt design

## Prompt Engineering Optimisation 
- Systematically improved prompt quality by:
  * constraining model output to verifired tool results
  * ENforcing logical consistency
  * Minimising the generation of fabricated chemical properties
  * Improving scientific accuracy and clarity
- Tool integrated prompts
  * Generate analogues of the molecule and find the best one and test the pharmacophore for the best one.
  * Evaluate pharmacophore features and Lipinskis compliance for drug-likeness.
  * Analyse how substituent changes influence pharmacophore mapping and binding efficiency
  * Investigate how fluorine substitution affects Lipinski's properties such as logP and molecualr weight
  * Determine how fluorine substitution influences Lipinskis hydrogen bond donors/acceptors
  * Assess substituion pattern affects on Lipinski parameters across fluorobezene isomer
  * Explain how fluorine substitution alters pharmacophore aromatic and hydrophobic regions
  * Examine fluorine substitution influence on pharmacophore H bond acceptor and aromatic centroid positions

    ## Temperature Sensitivity Analysis
    - Evaluates the effect of model temperature on scientific accuracy:
      * 0.1 = Highly accurate, deterministic, factual
      * 0.2 = Accurate with slight variability
      * 0.6 = Balanced reasoning and felxibility
      * <0.6 = Increased creativity but higher error rate
     - Demonstrates lower temperature settings provide the most reliable scientific outputs.
   
   ## Technologies Used
  - Python
  - RDKit
  - Hugging face Transformers
  - Large Language models (LLMs)
  - Cheminformatics tools
  - Prompt engineering frameworks
 
  ## Aplications
  - This work is relevent to:
    * Drug discovery
    * Computational chemistry
    * Cheminformatics
    * Molecualr modelling
    * AI- assissted scientific research

  





