# Intrinsic Plagiarism Detection in Armenian Texts Using Stylometric Analysis

This repository contains the models and resources for our stylometry-based intrinsic plagiarism detection research for the Armenian language. We provide the evaluation script, as well as the datasets and a set of manually compiled stylometric feature extraction resources that were used for evaluation.

## Tasks and models

Intrinsic stylometric analysis was studied from the perspective of two PAN @ CLEF shared tasks: Style Change Detection and Style Breach Detection. The first task is about detecting whether there is a style change within the document. For this task, we adapted and studied the clustering-based [Nath et al.](http://ceur-ws.org/Vol-2380/paper_163.pdf) model and [Zlatkova et al.](http://ceur-ws.org/Vol-2125/paper_142.pdf) ensemble of classifiers:

- _Nath et al._ are using threshold clustering on text windows' vectors. Style change is detected if the number of obtained clusters is greater than one.

- _Zlatkova et al._ are using a weighted combination of 4 classifiers(SWM, Random Forest, AdaBoost and Multi-layer Perceptron), applied to the feature vectors. Then the obtained results are analysed by logistic regression to detect a style change.

The second task is about detecting the exact indices where style change occurs in the text. For this task, we studied the model from [Karas et al.](http://ceur-ws.org/Vol-1866/paper_133.pdf) and an agglomerative clustering-based new approach, inspired by the top solutions for the Authorship Clustering task. For both models, we extract and use our own stylometric features set.

- _Karas et al._ model represents a statistic approach using Wilcoxon test to compare neighboring paragraphs' similarity score. Paragraph pairs with the least 30% of test scores are considered to have style differences and a style breach between them is fixed.

- _Agglomerative Clustering_ model is using clustering approach to group the paragraphs by their clusters and indicates a style breach if there is a cluster change between two neighboring paragraphs. 

## Dataset

The dataset was generated automatically by combining fragments from different texts on the same topic. We generated multiple test sets to cover various writing genres:

1. **PhD theses**, collected from [the official website](http://etd.asj-oa.am/).
2. **Encyclopedic articles**, collected from Wikipedia and [Armenian encyclopedia](http://www.encyclopedia.am/).
3. **Academic books**, collected from history [textbooks](https://lib.armedu.am/) for high school 7-9 grades and for universities.
4. **Fiction**, collected from "Pheasant" by Axel Bakunts and the its movie adaptation by Hrant Matevosyan.
5. **News**, collected from [NewsHub](https://newshub.am/) news aggregator.

## Stylometric features

We used a combination of character-, word-, sentence- and paragraph-level features, implemented in `feature_extraction` module:

a) **Character level**: features describing the usage of suffixes, prefixes and punctuation. 

b) **Word level**: this group includes features based on the usage of foreign, informal words, specific n-grams, preferred style of writing abbreviations and numerals.

c) **Sentence level** features consist of morphological and syntactic features as well as general sentence length features.

d) **Paragraph level** readability features: Flesch reading ease, SMOG grade, Flesch-Kincaid grade, Coleman-Liau index, automated readability index, Dale-Chall readability score, difficult words, Linsear write formula and Gunning fog.

For more details about the models and dataset construction, refer to the [paper](https://doi.org/10.15514/ISPRAS-2021-33(1)-14).
