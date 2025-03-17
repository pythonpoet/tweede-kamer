The files bert_train_and_test.py and bert_utils.py contain code adapted from the paper by Goffredo et al. (2023) titled 
Argument-based detection and classification of fallacies in political debates, published in EMNLP 2023. This code, 
originally developed by the authors, is available on their official GitHub repository at this link. The scripts are 
responsible for training and testing a BERT-based model for fallacy detection in political debates, following the approach 
outlined in the paper.

The Jupyter notebook apply_bert.ipynb utilizes these scripts to apply the fallacy detection model to the dataset of Tweede 
Kamer speeches from 2022-2023. The dataset, stored in speeches_2022-2023.csv, is processed using the BERT model to identify 
and classify fallacies within the political discourse. The detected fallacies and their classifications are then saved in the
output file fallacy_detection_results.csv, providing an overview of the model’s performance on the dataset.
