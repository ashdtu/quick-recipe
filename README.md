
## Dataset
Please download data and model checkpoint files from [drive](https://drive.google.com/drive/folders/1hnY0ZlavaA_N8vZRS7_y9BKCQkXEff6I?usp=share_link). Extract it in data/ and model/ directories respectively. Our processed YouCook2 dataset in Pickle format is available at `data/full_master_updated.pkl`.

### Acknowledgement
This codebase has been initialised from the [repo](https://github.com/frankxu2004/cooking-procedural-extraction) for the paper "A Benchmark for Structured Procedural Knowledge Extraction from Cooking Videos". 


Project Organization
------------

    ├── data               <- Any datasets that need to be saved
    ├── logs               <- Logs from model training runs
    ├── models             <- Save any model weights here
    ├── notebooks          <- Notebooks for running experiments
    ├── scripts            <- Youcook scripts
    ├── src                <- Model training and evaluation code
        ├── models         <- Model classes
        ├── processing     <- Data processing scripts
    

------------

### Experiment wise Notebooks

**KeyClip Selection**
DistilBERT-Classifier(with and without Context): `notebooks/DistilBERT-keyclip.ipynb` 
Self-Attention (Text and Multi-modal experiments): `notebooks/self-attention-keyclip.ipynb`
Creating Sentence Embeddings(MiniLM-L6, DistilBERT): `notebooks/create-sentence-embeddings.ipynb`

**Knowledge Extraction**
