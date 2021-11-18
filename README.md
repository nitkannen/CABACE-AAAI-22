# BacKGProp-AAAI-22
Code and Pre-Processed data for Acronym Extraction Task in SDU Worskshop

To create BIO tags, use the following  example command

```python data/prep_BIO_tags.py -s data/english/legal/dev.json -t   data/sample.txt```

We support different model architectures to solve AE, and they can be identified using the following **model_ids**

* SimpleBert - 0
* CharacterTransformBert - 1
(More models to be added)

To run the code on English Legal dataset using simple BERT for Sequence labelling(model_id = 0) use:

```
python main.py --src_folder data/
               --trg_folder logs/
               --model_id 0
               --seed_value 42
               --batch_size 8
               --epoch 6
               --tokenizer_checkpoint bert-base-cased
               --model_checkpoint bert-base-cased
               --dataset english/legal
     
 ```


