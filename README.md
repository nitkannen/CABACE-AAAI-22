# BacKGProp-AAAI-22
Code and Pre-Processed data for our paper in SDU **AAAI 2022** titled: **CABACE: Injecting Character Sequence Information and Domain Knowledge for Enhanced Acronym Extraction**

To create BIO tags, use the following  example command

```python data/prep_BIO_tags.py -s data/english/legal/dev.json -t   data/sample.txt```

We support different model architectures to solve AE, and they can be identified using the following **model_ids**

* SimpleBert - 0
* CharacterTransformBert - 1
* Masking and augmented Loss
* Seq2Seq
* Attention staged prediction


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
 
 To run the code on English Legal dataset using CharacterTransformBert for Sequence labelling(model_id = 1) use:
 
 * 1) Download and unzip fastText Word Vectors in the root directory using the following commands:
  ``` 
  wget https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip  
  ```
  ```
  unzip wiki-news-300d-1M.vec.zip
  ```
 * 2) Now use the command to run code:

```
python main.py --src_folder data/
               --trg_folder logs/
               --model_id 1
               --seed_value 42
               --batch_size 8
               --epoch 6
               --tokenizer_checkpoint bert-base-cased
               --model_checkpoint bert-base-cased
               --dataset english/legal
               --max_word_len 16
               --cnn_filter_size 4
     
 ```


