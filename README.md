# Language Identification of Short Text Documents
Jack Kaloger 2017

## Abstract
In this report, I outline a language classifier for twitter documents that uses long and short text documents as training. I ask whether long text data is useful for short text classification and apply a variety ensemble learners, deep learning and support vector machines to twitter language identification. Finally, I demonstrate the effectiveness of three selected classifiers using test data from twitter.

## 1	Introduction
Language identification of twitter documents presents several challenges compared to longer text classification. Among those are the length of the documents, multilingual challenges presented by hashtags and @mentions, as well as short hand internet slang that is used frequently to reduce character count. To combat these, and to make longer text documents useful for training, it may be necessary to preprocess some training documents. 

## 2	Initial dataset analysis
### 2.1	Feature Representation
To compare two instances of data, features must be constructed based on the text. In this project, I use two variations of the bag of words approach.

First is the bag of words approach, which splits text data into unique words as features, with each attribute being the number of times that word appears in the document. This is known as a document-term matrix, and maps word frequencies to instances. The second approach is a variation of this, which instead of mapping word frequencies, maps byte frequencies.

The mentioned approaches are equivalent to a unigram approach, where each feature is a single word or byte long. Other approaches used in this project include Bigrams and Unigrams with Bigrams.

### 2.2	Datasets
One of the main issues with twitter language identification is the number of differences it holds with longer-text publications such as books. Some of these differences include sentence structure, use of internet slang or shortened words as well as length of documents. Training data provided came from 4 sources: JRC-Acquis, Debian documentation, Wikipedia and Twitter. To demonstrate the differences, figure 1 shows a comparison between the Nearest Centroid baseline with character unigrams and bigrams trained on each of the datasets alone, as well as combined. Here, LM represents the length of the document in characters.

| Document | Document Length | Acc | Macro-Precision | Macro-Recall | F1-Score |
|---------|---------|---------|---------|---------|
| JRC-Acquis | 16618 | 0.3566 | 0.1480 | 0.2563 | 0.1876 |
| Debian | 10310 | 0.5920 | 0.6497 | 0.6518 | 0.6507 |
| Wikipedia | 7175 | 0.6086 | 0.6219 | 0.6936 | 0.6558 |
| Twitter | 77 | 0.6523 | 0.6845 | 0.7568 | 0.7189 |
| ALL | 8545 | 0.6541 | 0.5875 | 0.7559 | 0.7201 |

#### Figure 1: document evaluation.

Clearly, the longer the document, the less similar it is to twitter data. I tried a few experiments to remedy this. Among them included splitting instances into sentences and lines, and Tf-idf transformation. Figure 2 shows the results of two experiments over the JRC-Acquis dataset; once again using the baseline nearest centroid classifier with character unigrams and bigrams.

| Method | Acc | Macro-Precision | Macro-Recall | F1-Score |
|---------|---------|---------|---------|---------|
| Sentences | -0.05 | 0.068 | -0.03 | 0.03 |
| Tf-idf | 0.02 | 0.00 | 0.02 | 0.01 |

#### Figure 2: Document cleaning evaluation.

Unfortunately, none of these had a significant effect on the accuracy of classifiers. Most likely, this is due to the actual language and sentence structure of tweets being different to that of longer text documents, especially those from JRC-Acquis. Nonetheless, Tf-idf appears useful, and so should be applied to future experiments.

### 2.3	Feature Selection
In order to improve usefulness of classifiers, feature selection can be an effective tool. In the case of language identification, I hypothesized that it could be used to weed out irrelevant features exclusive to long text classifiers. Figure 3 shows the change in Accuracy and F1-Score compared to the baseline Accuracy and F1-Score. It shows a baseline nearest centroid classifier using word unigrams and a χ2 selection function.

![Figure](https://github.com/jkaloger/langid/raw/master/docs/figure3.png)

#### Figure 3: Feature Selection Evaluation.

Accuracy increases drastically as we increase the number of features until 200-300 features, as does F1-Score. This is likely not an improvement unique to long text training documents. In general, the document-term matrix is very sparse and much of the data, especially byte unigrams, are most likely unnecessary. By using feature selection, we reduce ambiguity/overlap between languages; separating them further in the feature space, and increasing the effectiveness of training.

### 2.4	Baseline comparison
Using dataset cleaning, feature selection and normalization techniques to improve the use of larger datasets, I return to a comparison between dataset training using the Nearest Centroid classifier. Figure 5 refers to training on the different datasets with a Nearest Centroid Classifier using character unigrams and bigrams with a χ2 selection function and languages cleaned from the datasets. It presents the change from the original data in figure 1:

| Document | Acc | Macro-Precision | Macro-Recall | F1-Score |
|---------|---------|---------|---------|---------|
| JRC-Acquis | 0.0290 | 0.0099 | 0.0209 | 0.0136 |
| Debian | 0.0582 | 0.0634 | 0.0954 | 0.0791 |
| Wikipedia | 0.0386 | 0.0564 | 0.0584 | 0.0574 |
| Twitter | 0.0820 | 0.0702 | 0.0865 | 0.0776 |
| ALL | 0.0290 | 0.0099 | 0.0209 | 0.0136 |

#### Figure 5: Re-evaluation.

While the methods outlined have improved the classifier, they do so for all training datasets in proportion to dataset size and thus are unlikely to be improving the use of long-text training data. As mentioned previously, this is likely due to the different structure of longer text documents. They are nonetheless useful inclusions for a short-text training data, as seen in the ~8% increase across all evaluation metrics for twitter data.

## 3	Classifiers
Thus far, only a nearest centroid classifier has been used in testing. In this section, I provide an overview of 5 other classifiers and their usefulness.

### 3.1	SVM
Briefly, Support Vector Machines define a hyperplane along which the dataset is separated. They attempt to maximize the margin between the hyperplane and the closest data points. Figure 6 shows three weak, baseline classifiers trained on all datasets except JRC-Acquis: Nearest Centroid, Multinomial Naïve Bayes, Decision Tree; compared to two stronger SVMs: A Linear Support Vector Machine (SVM) and a gradient descent L-SVM.
Classifier

| Method | Acc | Macro-Precision | Macro-Recall | F1-Score |
|---------|---------|---------|---------|---------|
| NC | 0.7215 | 0.7513 | 0.8361 | 0.7914 |
| MNB | 0.6922 | 0.8369 | 0.7984 | 0.8172 |
| DT | 0.6784 | 0.6983 | 0.7837 | 0.7386 |
| Linear SVM | **0.7872** | **0.8174** | **0.8944** | **0.8542** |
| SGD SVM | 0.7760 | 0.7895 | 0.8828 | 0.8336 |

#### Figure 6: Baseline Classifiers.

The Linear SVM proves to be the strongest classifier in every category. Linear SVMs define the hyperplane to be linear. In a sense, this is a form of regularization, since the complexity of the rules within the classifier are reduced – preventing overfitting. SVMs are also known to work well with sparse features – our dataset is very sparse. The use of SVMs is explored more in section 3.3 

### 3.2	Ensemble Learning
Ensemble learning, or classifier combination involves the aggregation of a set of base classifiers into a single meta-classifier (Tan et al. [2006, p277-80]). By combining several weaker classifiers, a potentially stronger meta-classifier can be constructed.

In the following tests, I demonstrate the usefulness of ensemble learning for short-text language classification by combining several weak classifiers (usually decision trees). I use 5 ensemble learners; bagging, voting, gradient boosting, random forests and AdaBoost.

Voting classifiers involve running base classifiers over all data and selecting the class most predicted for each instance (or using some other voting method) the classifiers used for voting are the five baseline classifiers from figure 6. Figure 7 compares these ensemble learners with the SVM classifier.
Classifier

| Method | Acc | Macro-Precision | Macro-Recall | F1-Score |
|---------|---------|---------|---------|---------|
| L-SVM | 0 | 0 | 0 | 0 |
| Voting | -0.0152 | 0.0066 | -0.0206 | -0.0060 |
| Bagging | -0.0413 | -0.0381 | -0.0460 | -0.0418 |
| Random Forest | -0.0106 | -0.0064 | -0.0155 | -0.0106 |
| AdaBoost | -0.4351 | -0.3415 | -0.4852 | -0.4142 |

#### Figure 7: Ensemble learners.

Ensemble learners are usually as good or nearly as good as strong learners. Their usefulness comes from their ability to combine weak classifiers into a stronger classifier. However, the SVM is a more accurate classifier. This is likely due to the regularization aspect mentioned earlier – moreover, there is a limit to how much ensemble learning can improve classifiers.

### 3.3	The ‘Unknown’ language
An added layer of complexity in this language classification system is the ‘unk’ class, signifying an unknown language – a language outside the basic set of 20 languages specified. Many of the datasets scraped from the web have langauges outside this set, the labels of which can be changed to ‘unk’.

However, the JRC-Acquis dataset is the only one in these experiments to have other languages. Thus, another approach is required. Using the ensemble learner Random Forests and our linear SVM, we assign a threshold – that is, a minimum probability of a class label – for assignment. Any classification that does not exceed this threshold is automatically converted to the unknown class. Figure 8 shows the results of a random forest ensemble learner vs SVM using thresholding over all datasets except JRC-Acquis from 5% to 95% certainty.

![Figure](https://github.com/jkaloger/langid/raw/master/docs/figure8.png)

Figure 8: Thresholding evaluation

Both the classifiers receive a boost in accuracy. In the development set, around 10% of the data is in an unknown language, the SVM classifier increases by ~7% accuracy; consistent with the accuracy of the classifier and the number of unknown labeled data in the development set.

### 3.4	Deep Learning
Deep learning models require large amounts of training data. As such, the neural network presented is not as accurate as it could be. In order to account for the unknown class, I hypothesised increasing the class weight of ‘unk’ would simulate thresholding. Figure 9 shows the results from a neural network classifier.

Method | Acc | Macro-Precision | Macro-Recall | F1-Score
|---------|---------|---------|---------|---------|
| NN | 0.7727 | 0.8060 | 0.8663 | 0.8350 |

#### Figure 9: Neural network evaluation.

Due to time constraints, I was unable to apply thresholding to the neural network. This most likely accounts for the lower accuracy.

## 4	Other Notes
I tried a variety of ideas over the course of the project. The ones that didn’t work are listed here.

### 4.1	Visualization
I attempted to visualize the data using PCA, however the sparsity of the document term matrix – even with word unigrams on twitter data – was too great to view easily. It’s likely we would have seen clusters for each language, with similar languages such as Dutch and English having significant overlap; however more interesting data could have come up such as overlap between less similar languages due to internet slang, hashtags and mentions.

### 4.2	w-shingling – a bag of words alternative
Another document similarity method called w-shingling was attempted, where a set of w unique ‘shingles’ is created from each document, and the similarity is measured as

A 1NN classifier was used, but produced very low accuracy. A similar approach used Booleans instead of counts in the document term matrix, however this produced very similar results. 

## 5	Conclusion
The three classifiers produced are reasonably accurate, but not nearly as accurate as other classifiers (Baldwin and Lui, 2010). Perhaps this is due to the problems mentioned with a lack of short text training data. Certainly, the ‘unknown’ label adds an additional layer of complexity that reduces overall effectiveness of the classifiers.

# References
Pang-Ning Tan, Michael Steinbach, and Vipin Kumar. Introduction to Data Mining. Addison Wesley, 2006.

Tim Baldwin, Marco Lui. Language Identification: The Long and the Short of the Matter. Association for Computational Linguistics, 2010.

Tim Baldwin, Marco Lui. Accurate Language Identification of Twitter Messages. Association for Computational Linguistics, 2014.
