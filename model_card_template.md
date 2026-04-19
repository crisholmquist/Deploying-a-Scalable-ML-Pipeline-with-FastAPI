# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

This model is a binary classification model built to predict whether a person's income is greater than \$50,000 per year
based on Census data. For this project, I used a Random Forest Classifier. I trained the model using the provided census dataset,
processed the categroical features with the provided processing function, and saved both the trained model and the encoder for later use in the API.

## Intended Use

This model's purpose is to demonstrate the full machine learning workflow, including data processing, model training, evaluation,
slice-based performance checks, and deployment through FastAPI.

The model predicts one of the two income classes: `>50K` or `<=50K`.

## Training Data

The training data came from the census dataset and includes features such as age, workclass, education, marital status, occupation,
relationship, race, sex, capital gain, capital loss, hours per week, and native country. The target variable is whether a person earns more than \$50,000 per year.

To prepare the data, I used the `process_data()` function from `ml/data.py`. The function handled the categorical encoding and prepared the label column for the model to be trained as a classification model. 

## Evaluation Data

The evaluation data came from the census dataset after it was split into training and test sets. 
I trained the model on the training set and then evaluated it on the test set to measure how well it performed on unseen data. I also evaluated the model on slices of the data by checking performance across distinct values within categorical features. This helped show how the model performed on different groups in the dataset. The slice-based results were saved to `slice_output.txt`.

## Metrics

I evaluated the model using precision, recall and F1 score.

The overall performance of my model was:
- **Precision:** 0.7391
- **Recall:** 0.6384
- **F1:** 0.6851

These results show that the model performed reasonably well overall. Since precision is higher than recall, it is reasonable to say the model is doing better at making correct positive decisions than at catching all actual positive cases.

## Ethical Considerations

The census dataset includes demographic-related features such as sex, race, marital status, and native country. Because of this, the model may reflect biases or patterns that already exist in the data. Even when the model performs reasonable well overall, that does not mean it performs equally well for every group.

Another consideration is that some non-sensitive-looking features, such as education or occupation, may still act as proxy variables for sensitive characteristics. Bias can still affect the model even if it is not obvious from the overall results.

## Caveats and Recommendations

One limitation of this model is that it was built for this project, using a provided dataset. This means it may not generalize well to other populations or newer data. Another limitation is that overall metrics do not tell the full story, which is why slice-based evaluation is important.

If this project were expanded further, one good next step would be to review the slice metrics more closely, to identify whether the model performs better or worse for certain groups. It would also be useful to compare this model against other classification models, to see whether a different model could improve the results. Before this model could be considered for any real-world use, it would need more testing, monitoring, and review.