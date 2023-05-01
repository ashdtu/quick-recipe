
## Dataset
Please download data and model checkpoint files from [drive](https://drive.google.com/drive/folders/1hnY0ZlavaA_N8vZRS7_y9BKCQkXEff6I?usp=share_link). Extract it in data/ and model/ directories respectively. Once extracted in data/, our processed YouCook2 dataset in Pickle format will be available at `data/full_master_updated.pkl`.

### Acknowledgement
This codebase has been initialised from the [repo](https://github.com/frankxu2004/cooking-procedural-extraction) for the paper "A Benchmark for Structured Procedural Knowledge Extraction from Cooking Videos". 


Project Organization
------------

    ├── data               <- Any datasets that need to be saved
    ├── logs               <- Logs from model training runs
    ├── models             <- Save any model weights here
    ├── notebooks          <- Notebooks for running experiments
    ├── scripts            <- Youcook scripts
    ├── attn-lstm          <- Attention-LSTM training and evaluation scripts
    ├── src                <- Model training and evaluation code
        ├── models         <- Model classes
        ├── processing     <- Data processing scripts
    

------------

### Experiment wise Notebooks

**Feature Re-alignment**
- Feature Re-alignment notebook: `notebooks/yc2_feature_alignment_v2.ipynb`

**KeyClip Selection**
- DistilBERT-Classifier(with and without Context): `notebooks/DistilBERT-keyclip.ipynb` 
- Self-Attention (Text and Multi-modal experiments): `notebooks/self-attention-keyclip.ipynb`
- Creating Sentence Embeddings(MiniLM-L6, DistilBERT): `notebooks/create-sentence-embeddings.ipynb`
- Independent Visual Features: `notebooks/yc2_cnn_visual_only.ipynb`
- Unified Visual Features: `notebooks/yc2_visual_unified_feat.ipynb`
- Self-attention on visual features: `notebooks/yc2_self_attn.ipynb`

**Knowledge Extraction**
- T5 Small: `notebooks/2_1_Knowledge_Extraction_T5small.ipynb`
- T5 Base: `notebooks/2_2_Knowledge_Extraction_T5base.ipynb`
- BART: `notebooks/2_3_Knowledge_Extraction_BART.ipynb`
- BART (with Coreference Resolution): `notebooks/2_4_Knowledge_Extraction_BART_Coref.ipynb`
- Attention LSTM (Visual-only): `notebooks/2_5_Knowledge_Extraction_Attention_LSTM_Visual.ipynb`
- Attention LSTM (Multimodal): `notebooks/2_6_Knowledge_Extraction_Attention_LSTM_Multimodal.ipynb`
- Multihead Transformer (Multimodal): `notebooks/2_7_Knowledge_Extraction_Multihead_Transformer.ipynb`

### Scripts
#### Attention LSTM
**Train**
```
cd attn-lstm
# For training Attention LSTM Visual model
python train_visual.py

# For training Attention LSTM Multimodal model
python train_multimodal.py
```

**Evaluate**
```
cd attn-lstm
# For training Attention LSTM Visual model
python evaluate_visual.py

# For training Attention LSTM Multimodal model
python evaluate_multimodal.py
```

#### Multihead Transformer
**Train**
```
cd multihead-transformer
python prepare_dataset.py
python main_multimodal.py
```

**Evaluate**
```
cd multihead-transformer
python evaluate_multimodal.py
```