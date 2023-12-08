# Sentiment_analysis_airline

Using object oriented programming for training the model.different scikit models are trained with just one command.

## pipeline 





![pipeline](https://user-images.githubusercontent.com/82393353/216391530-8b3991f5-897c-4893-9948-870eaf5d4d30.jpg)



- > This repository contains sentiment classification model , which takes text as input and outputs the whether the statement is positive or negative comment.

- > The app.py  is main file of webapp deployed on huggingface, uses streamlit framework as ui.

- > Try the app here - https://huggingface.co/spaces/SSahas/sentiment_classifier_airline

- > The fast_api_swagger.py is code to create restful api using fastapi and uvicorn as server, swagger documentation is integrated with it, takes statement as input and outputs the prediction in the form of json.


# Results 

| mean_fit_time | std_fit_time | mean_score_time | std_score_time | params                                     | mean_test_score | std_test_score | model_name             |
|---------------|--------------|------------------|-----------------|--------------------------------------------|-----------------|----------------|------------------------|
| 75.238871     | 7.649687     | 1.116081         | 0.396170        | {'alpha': 0.9}                             | 0.887878        | 0.010712       | Ridge_classifier       |
| 32.970682     | 19.089618    | 0.537239         | 0.069481        | {'alpha': 1.0}                             | 0.887532        | 0.011108       | Ridge_classifier       |
| 59.446653     | 4.041303     | 0.925700         | 0.067038        | {'alpha': 0.8}                             | 0.887098        | 0.010756       | Ridge_classifier       |
| 4.578112      | 0.228420     | 0.208683         | 0.015610        | {'max_depth': None, 'n_estimators': 10}    | 0.842732        | 0.049028       | Randomforestclassifier |
| 33.682326     | 6.913221     | 0.231637         | 0.023633        | {'alpha': 9.5e-05}                        | 0.838400        | 0.021505       | SGDClassifier          |
| 56.504217     | 3.619727     | 0.524913         | 0.152613        | {'max_depth': None, 'n_estimators': 150}  | 0.833893        | 0.096404       | Randomforestclassifier |
| 1.457530      | 0.724146     | 0.263000         | 0.058040        | {'alpha': 0.5}                             | 0.830604        | 0.016622       | MultinomialNB          |
| 40.424593     | 12.100794    | 0.216021         | 0.011447        | {'alpha': 8e-05}                          | 0.830603        | 0.014611       | SGDClassifier          |
