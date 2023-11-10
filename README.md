![image](https://github.com/PedramPeiro/Active-Customer-Forecasting-Using-Time-Series/assets/102898063/8eddc26d-b8d5-43e1-bd0d-a35ed68a3398)# Active Customer Forecasting Using Time Series
The primary goal of this project was to predict future active customers using the Didar CRM software. The analysis harnessed various features to achieve this objective, including:

| Date       | ActiveCustomer | Churned | DealsAdded | Leads | NewPaid | ClosedDeals |
|------------|----------------|---------|------------|-------|---------|-------------|
| 3/21/2022  | 1183           | 0       | 3565       | 19    | 1       | 1196        |
| 3/22/2022  | 1097           | 2       | 774        | 20    | 1       | 478         |
| 3/23/2022  | 946            | 0       | 997        | 21    | 1       | 458         |
| 3/24/2022  | 918            | 0       | 590        | 19    | 0       | 370         |
| 3/25/2022  | 857            | 5       | 498        | 17    | 0       | 282         |
| 3/26/2022  | 977            | 0       | 2171       | 41    | 2       | 1765        |
| 3/27/2022  | 1019           | 5       | 2792       | 57    | 0       | 1860        |
| 3/28/2022  | 1053           | 4       | 2903       | 47    | 9       | 2033        |


- **Date**: The timestamp indicating when events were recorded.
- **ActiveCustomer**: Our target variable, representing the count of customers currently using the software.
- **Churned**: The number of customers identified as having churned at that timestamp.
- **DealsAdded**:The total count of deals added to the software by different users on a given day.
- **Leads**: The number of potential customers that can be added to the software's subscription through successful sales.
- **NewPaid**: The count of new successful customers, either through subscription plan renewals or purchases after the free trial.
- **ClosedDeals**: The number of deals closed on a specific day, including both successful and unsuccessful closures by our users.


## 1. Preprocessing
You can find this section in the [1. Preprocessing.ipynb](https://github.com/PedramPeiro/Active-Customer-Forecasting-Using-Time-Series/blob/main/1.%20Preprocessing.ipynb) notebook. In machine learning, a crucial step is preprocessing, guided by the principle of avoiding **GIGO**: garbage in, garbage out. To mitigate this, we undertake feature engineering and cleaning. Upon visualizing the features, potential outliers become apparent in the dataset. Addressing these anomalies is crucial for ensuring the success of our multivariate time series forecasting.

![image](https://github.com/PedramPeiro/Active-Customer-Forecasting-Using-Time-Series/assets/102898063/c508612a-780c-481c-9ae1-7258226a3604)

To detect outliers, various methods such as IQR, z-score, and advanced techniques like Autoencoders can be employed. In this project, Autoencoders were utilized with tuned parameters to identify outliers for each feature individually. Unlike considering entire records with multiple features, this approach evaluates each feature in isolation.

The rationale behind this approach is that the encoder attempts to learn the trend and predict the next step. If the reconstruction error, resulting from the disparity between the actual and reconstructed values, is high, it indicates an unexpected pattern that the model couldn't have anticipated. Therefore, it may be flagged as an outlier. By calculating these errors for each feature, a threshold is applied, and values exceeding this threshold are identified as outliers.

Consider the following plot:
![image](https://github.com/PedramPeiro/Active-Customer-Forecasting-Using-Time-Series/assets/102898063/cdf50012-4ebe-47e4-ab40-ca8c5436965a)

Setting the threshold at the 99th percentile, records surpassing this limit are flagged as outliers. Subsequently, these outliers are replaced using polynomial interpolation. This technique helps smooth out irregularities by fitting a polynomial function to the surrounding data points, providing a more continuous and representative replacement for the identified outliers.
![image](https://github.com/PedramPeiro/Active-Customer-Forecasting-Using-Time-Series/assets/102898063/484e6b6f-ced3-4be3-ad39-0eff6c7b0cd3)

The outliers and the result of adressing them:
![image](https://github.com/PedramPeiro/Active-Customer-Forecasting-Using-Time-Series/assets/102898063/cc3e6d8a-758f-479e-b4e4-c24973f0cb1e)

With the completion of the dataset preparation phase, we now proceed to the next step: applying Time Series methods.

Please note that the discussion in this section primarily focuses on univariate models. It's worth mentioning that, despite the implementation of a Conv1-D LSTM multivariate model in the main project, its detailed exploration is deferred due to data limitations. Further insights into project challenges and resolutions will be thoroughly examined in the concluding section.

## Univariate TS Models
As detailed in [2. Univariate TS Models.ipynb](https://github.com/PedramPeiro/Active-Customer-Forecasting-Using-Time-Series/blob/main/2.%20Univariate%20TS%20Models.ipynb), a variety of models, including RNN, LSTM, and 1D-Convolutional LSTM, were deployed. While the procedural aspects are similar across these models, the focus here is on explaining the best-performing model: the **1D-Convolutional LSTM**."

Before delving into the specifics, it's beneficial to familiarize yourself with key terminologies in Time Series. I recommend exploring the concepts covered in the Coursera course [Sequences, Time Series and Prediction](https://www.coursera.org/learn/tensorflow-sequences-time-series-and-prediction/home/info)

1. The optimal window size, determined through multiple trial-and-error iterations, was found to be 50.
2. For training purposes, 60% of the data was utilized, while the remaining 40% was evenly split between validation and test sets.
3. To establish a baseline for comparison, the baseline error was calculated using the *Last Period Demand Forecast Method* and was found to be 13. It's important to note that this method is most effective for forecasting the demand for a single timestep ahead, not for multiple timesteps.
4. **IMPORTANT**: Ensure that you scale your dataset, either through normalization or standardization. In this context, normalization has been chosen.

### Some Important Functions
- ```windowed_dataset```: Transforms the input data into a windowed dataset suitable for time series tasks.
- ```model_forecast```: This function is employed for predicting the next future value.
- ```plot_loss_from_epoch```: Plots the loss function over the course of training epochs.
- ```metric_calculator_val_test```: Calculates metrics such as Mean Squared Error (MSE) and Mean Absolute Error (MAE) for validation and test sets.
- ```plot_mae_mse_based_on_epochs```:  Plots the MSE and MAE throughout the training epochs for both test and validation sets, along with their weighted average in a single plot. This aids in selecting the optimal number of epochs for training.
- ```plot_time_series```: Plots the time series trend.

### Conv1D-LSTM

- ```create_uncompiled_model_ConvLSTM```: This function focuses on structuring the model without compiling it. Key hyperparameters, including the number of neurons in each layer, the quantity of LSTM networks, the number of filters, and more, are crucially determined through trial and error. For a deeper understanding of hyperparameter tuning, you may refer to the course [Improving Deep Neural Networks: Hyperparameter Tuning, Regularization and Optimization](https://www.coursera.org/learn/deep-neural-network?specialization=deep-learning&utm_medium=sem&utm_source=gg&utm_campaign=B2C_EMEA_deep-learning_deeplearning-ai_FTCOF_specializations_country-DE&campaignid=20416373453&adgroupid=155810820630&device=c&keyword=&matchtype=&network=g&devicemodel=&adposition=&creativeid=667829385248&hide_mobile_promo&gclid=CjwKCAiAxreqBhAxEiwAfGfndM4fg0f09E-dBT-MuEnMiRzwXrfuXBWw1m0x1CJNbkWtyG_6yv45QxoCbWcQAvD_BwE).
- ```create_model_ConvLSTM```: This function compiles the previously constructed model.
- After compiling the model, the objective is to determine the optimal number of steps (epochs) for training. To achieve this, after each epoch, the resulting model is saved at a specific path for subsequent analysis. The detailed path and analysis are mentioned below:
```python
save_path_ConvLSTM = '2. Univariate TS Models/ConvLSTM'
os.makedirs(save_path_ConvLSTM , exist_ok = True)
model_ConvLSTM = create_model_ConvLSTM(learning_rate=0.00075)
checkpoint_callback = ModelCheckpoint(save_path_ConvLSTM+"/model_weights_epoch_{epoch:03d}.h5", save_weights_only=True, save_freq=1)
history = model_ConvLSTM.fit(x_train_univariate , validation_data = x_val_univariate,
                        epochs = 500, 
                        callbacks=[checkpoint_callback])
```
- The overall trend in the training process exhibits a steady decrease until the 400th epoch, beyond which it experiences a gradual increase. This observation suggests that selecting 400 epochs is a reasonable choice for the current scenario. The indication that the gradient descent algorithm performs well, coupled with the absence of coding issues, is reassuring. However, it's important to note that the presence of numerous spikes in the learning process is inherent to our problem. Despite efforts, smoothing out these spikes proved challenging and remains a characteristic of the nature of our data and model structure.
![image](https://github.com/PedramPeiro/Active-Customer-Forecasting-Using-Time-Series/assets/102898063/6c9d4e79-7047-4c48-aa95-035f0d63cf69)

- As previously mentioned, the weighted average metrics of Mean Absolute Error (MAE) and Mean Squared Error (MSE) on both validation and test sets are showcased. Through this analysis, the optimal number of training epochs is identified as 310.
![image](https://github.com/PedramPeiro/Active-Customer-Forecasting-Using-Time-Series/assets/102898063/ee1c82f7-f340-43ac-897f-06dac94db7cb)

- Below, the performance of the model in making predictions is illustrated:
![image](https://github.com/PedramPeiro/Active-Customer-Forecasting-Using-Time-Series/assets/102898063/481f758a-8dc8-4cb4-9303-cfa25f723a6d)

- An essential point, to be further elaborated in the conclusion and takeaway section, is that this model is designed for single-step prediction and is not inherently a multistep model. *However*, in practice, it is employed for multi-step prediction by assuming that the predicted value for the next step might serve as an approximation for the actual value of the future step.
- The overall forecasting trend is illustrated below:
![image](https://github.com/PedramPeiro/Active-Customer-Forecasting-Using-Time-Series/assets/102898063/e4c11dce-a82b-4b86-aac2-2180057635ed)
