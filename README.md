# Active Customer Forecasting Using Time Series
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
You can find this section in the *1. Preprocessing.ipynb* notebook. In machine learning, a crucial step is preprocessing, guided by the principle of avoiding **GIGO**: garbage in, garbage out. To mitigate this, we undertake feature engineering and cleaning. Upon visualizing the features, potential outliers become apparent in the dataset. Addressing these anomalies is crucial for ensuring the success of our multivariate time series forecasting.

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
As you can see in 2. Univariate TS Models.ipynb
