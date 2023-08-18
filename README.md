## Data-Augmentation for Bangla-English Code-Mixed Sentiment Analysis: Enhancing Cross Linguistic Contextual Understanding

This code is the official implementation of the following paper:
* Mohammad Tareq, Md Fokhrul Islam, Swakshar Deb, Sejuti Rahman, Abdullah Al Mahmud, "[Data-augmentation for Bangla-English Code-Mixed Sentiment Analysis: Enhancing Cross Linguistic Contextual Understanding](https://ieeexplore.ieee.org/abstract/document/10129187)," in IEEE Access, 2023.

![intro-1](img/intro.PNG)

Figure 1: Distinct languages are represented by different colors (<font color="blue"> blue </font>: English, <font color="brown"> brown </font>: Bangla, <font color="green"> green </font>: transliterated Bangla) in a shared semantic space for CM sentiment classification. (*Left*) Previous studies have used existing monolingual word embeddings for CM sentiment analysis, and therefore, words from different languages cannot be related. (*Right*) When the proposed data augmentation is paired with existing word embeddings, cross-lingual understanding is developed, which improves CM sentiment classification performance.

![intro-1](img/main.PNG)

Figure 2: (a) Proposed data augmentation process with multiple sampling rates. For simplicity, we only showed sampling rate 1 and 2 in the above diagram. (b) Illustration of word embedding training process. We augment input data with several sampling rate. (c) Training the classifier using learned word embedding.
## Dataset descriptions

* `Dictionary_BN_EN_61208.xlsx`: Dictionary of collected word from different sources. Here we made huge dictionary which contain code-mixed bangla word and their english counter word.
* `final_code_mixed_BN_app_review_dataset_csv.xlsx`: The large collectd dataset on code mixed sentiment analysis.

## Running instructions

To run the baseling model with our proposed data augmentation strategy run the main.py inside the **align** folder. You can select the word embedding method (Fastext, W2V) inside the main.py file. For example, in case of the Fastext embedding with our proposed data augmentation run this command in the terminal   

```shell
python align/main.py --fastext
``` 

Similarly, to run the baseling model without the data augmentation strategy run the main.py inside the **non-align** folder. Run this command for the baseline performance with Fastext word embedding   

```
python non-align/main.py --fastext
```

## Citation
If you use this code and the dataset for your research, please consider to cite our paper:

```
@article{tareq2023data,
  title={Data-augmentation for Bangla-English Code-Mixed Sentiment Analysis: Enhancing Cross Linguistic Contextual Understanding},
  author={Tareq, Mohammad and Islam, Md Fokhrul and Deb, Swakshar and Rahman, Sejuti and Al Mahmud, Abdullah},
  journal={IEEE Access},
  year={2023},
  publisher={IEEE}
}
  ```
## Contact
For any question, feel free to contact @
```
Swakshar Deb     : swakshar.sd@gmail.com
Md Fokhrul Islam : fokhrul.rmedu@gmail.com
```
