# plm-augmented-sparse-retrieval
<p align="center">
  <img src="pics/method-v3.jpg" width="650"/>
</p>

## Usage
### Environment
```
conda activate -n sparse python=3.9
conda activate sparse
pip install -r requirements.txt
```

### Data and Checkpoints
Please see the `README.md` file in `zeshel` directory to download the processed data.

### Train the retriever
```
python run_model_keyword.py \
--model retriever_model/keyword_model.pt
--pretrained_model google/electra-base-discriminator/
--data zeshel/data/\[forgotten_realms.json | lego.json | yugioh.json | star_trek.json\] \
--kb zeshel/kb/\[forgotten_realms.json | lego.json | yugioh.json | star_trek.json\]
```
