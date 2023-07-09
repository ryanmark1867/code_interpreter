# Experiments with code intepreter

Result of using [code interpreter](https://openai.com/blog/chatgpt-plugins#code-interpreter) to train a Keras model on the [Airbnb NYC dataset](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data)
1. Upload the CSV file in the ChatGPT UI
2. Input prompt `share code to train a Keras model on this dataset where the trained model would be able to predict whether a given listing would have a price above or below the average price`
3. Upload [resulting Python output code](https://github.com/ryanmark1867/code_interpreter/blob/master/airbnb_model.py) to Cloud Shell along with a local copy of the CSV file
4. Update resulting code to ingest local copy of dataset:
```
import pandas as pd
data = pd.read_csv('AB_NYC_2019.csv')
```


