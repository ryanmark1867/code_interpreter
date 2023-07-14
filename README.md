# Experiments with code intepreter

In Google Cloud Shell, exercise the result of using [code interpreter](https://openai.com/blog/chatgpt-plugins#code-interpreter) to train a Keras model on the [Airbnb NYC dataset](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data):
1. Upload the CSV file in the ChatGPT UI
2. Input prompt `share code to train a Keras model on this dataset where the trained model would be able to predict whether a given listing would have a price above or below the average price`
3. Upload [resulting Python output code](https://github.com/ryanmark1867/code_interpreter/blob/master/airbnb_model.py) to Cloud Shell along with a local copy of the CSV file
4. Install any required libraries: `pip install -U scikit-learn`
5. Update resulting code to ingest local copy of dataset:
```
import pandas as pd
data = pd.read_csv('AB_NYC_2019.csv')
```
Also, results from two other experiments that didn't generate working code:
- Same as above, but asking for a PyTorch model - generates [code that is missing any data handling](https://github.com/ryanmark1867/code_interpreter/blob/master/airbnb_model_pytorch.py)
- Asking for a Keras model for the [Kuala Lumpur real estate dataset](https://www.kaggle.com/datasets/dragonduck/property-listings-in-kuala-lumpur) - generates [code that handles both data preparation and model training](https://github.com/ryanmark1867/code_interpreter/blob/master/kl_model.py), but data preparation code generates an error because it does not handle conversion of currency value strings correctly
- Asking for a Keras model for the [Tokyo Airbnb dataset listing.csv](https://kaggle.com/datasets/449ca86bef12e30b0976e06613e3496a0c1151ae6ef177faa0dfb0536dad8ae5) - generates [code that has import issues](https://github.com/ryanmark1867/code_interpreter/blob/master/airbnb_tokyo_model.py) when run in Cloud Shell

