# Clothing-data-NLP-Analysis-and-Recommender
A course project of Analyzing Unstructured Data class at UCSD

## Description
In this project, we conduct analysis on the rating and fit of customers based on the e-commerce clothing data collected from RentTheRunway.com, and then develop a recommender to return similar items given the user inputs. From the model results, we conclude that review texts could largely improve the performance of classifiers which only includes customer demographics and body measurements. Other than that, word embedding models such as Skip-gram and CBOW which focus on learning from context perform best among a series of text mining models.

## Files
`renttherunway_final_data.json`

The raw data collected from RentTheRunway.com, containing 192,544 records from 105,508 users regarding 5,850 items.

`glove.6B.100d.txt`

The corpus which is used to train the Glove model (with the embedding dimensions of 100).

`code.ipynb`

The code containing all the works mentioned in our report.

`report.pdf`

The final report which includes detailed process.

## Reference
Rishabh Misra, Mengting Wan, Julian McAuley (2018). *Decomposing fit semantics for product size recommendation in metric spaces.*

https://cseweb.ucsd.edu//~jmcauley/pdfs/recsys18e.pdf

Hsin-Yu Chen, Huang-Chin Yen, Tian Tian (2020). *Female Clothing E-commerce Review and Rating.*

https://github.com/hsinyuchen1017/Python-Prediction-of-Rating-using-Review-Text/blob/master/Female%20Clothing%20E-commerce%20Review%20and%20Rating.pdf
