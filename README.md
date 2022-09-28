## 1. Abstract
There are codes, processed data and trained models for submission paper "Predicting Protein-Ligand Binding Affinity via Joint Global-Local Interaction Modeling" in ICDM 2022. 

## 2. Processed data
Limited to the size of attachments, we only upload PDBbind 2016 core dataset and CSAR-HiQ dataset to test the trained models. 
 - PDBbind v2016 core set: ./data/2016_core_data
 - CSAR-HiQ (including set1, set2): ./data/CSAR_HiQ_data

## 3. Code
 - The code for our GLI model in ./models/global_local_interaction_model.py.
 - The code for related baselines models in ./models/baseline_models.py
 - The code for evaluating and test in ./test_model.py

## 4. Environments
 - cuda: 11.0
 - GPU: V100
 - Packages: The required python package are listed in requirements.txt

## 5. Trained models and test command
In our submission, we used 10 fold-cross validation to evaluate our GLI framework performance. 
We provide some trained models in ./trained_models which were trained in PDBbind v2016 refined dataset, and tested in PDBbind v2016 core dataset and CASR-HiQ dataset.

To indicate the GLI framework based on different models:
 - GLI-0: Taking GAT+GCN as the model in chemical info embedding module.
 - GLI-1: Taking GIN as the model in chemical info embedding module.
 - GLI-2: Taking GCN2 as the model in chemical info embedding module.

To indicate the GLI framework with different modules:
 - GLI-\*-c: Including chemical info embedding module.
 - GLI-\*-cg: Including chemical info embedding module, global interaction module.
 - GLI-\*-cl: Including chemical info embedding module, local interaction module.
 - GLI-\*-cgl: Including chemical info embedding module, global interaction module, local interaction module.

### 5.1 test models GLI-0-c, GLI-1-c, GLI-2-c
Command: 

python test_model.py GLI-0-c  
python test_model.py GLI-1-c  
python test_model.py GLI-2-c  


### 5.2 test models GLI-0-cg, GLI-1-cg, GLI-2-cg
Command: 

python test_model.py GLI-0-cg  
python test_model.py GLI-1-cg  
python test_model.py GLI-2-cg


### 5.3 test models GLI-0-cl, GLI-1-cl, GLI-2-cl
Command: 

python test_model.py GLI-0-cl  
python test_model.py GLI-1-cl  
python test_model.py GLI-2-cl


### 5.4 test models GLI-0-cgl, GLI-1-cgl, GLI-2-cgl
Command: 

python test_model.py GLI-0-cgl  
python test_model.py GLI-1-cgl  
python test_model.py GLI-2-cgl

