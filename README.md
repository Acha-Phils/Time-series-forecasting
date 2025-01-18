# TIME SERIES FORECASTING USING DEEP LEARNING (hourly power consumption data)

## Project objectives

This study compares the performance of Autoregressive Integrated Moving Average with Exogenous Regressors (SARIMAX), 
Prophet, and Long Short-Term Memory (LSTM) models in forecasting hourly power consumption in the Eastern Intercontinental grid of the United States (2002-2018). 
The focus was on predictive accuracy and training time, using metrics like Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), 
and Mean Absolute Percent Error (MAPE). 

### Data source 

The project utilises data obtained from https://www.kaggle.com/datasets/raminhuseyn/energyconsumption-dataset on hourly power consumption, sourced from Pennsylvania-New JerseyMaryland (PJM)'s website, with measurements in Megawatts (MW). PJM is a regional transmission organisation in the United States, operating within the Eastern Interconnection grid. It oversees an electric transmission system that services all or parts of Delaware, Illinois, Indiana, Kentucky, Maryland, Michigan, New Jersey, North Carolina, Ohio, Pennsylvania, Tennessee, Virginia, West Virginia, and the District of Columbia.

### Tools used in the project
- Excel - Data inspection
- Python - Data cleaning, analysis and visualisation
- TensorFlow 2.10.0. for deep learning

### Data preparation

The sample size for the data set is 145366 and cover the period from 2002-2018.
The data was checked for missing values and inconsistencies.
It was then sorted by ‘DateTime’ to ensure correct chronological order of time essential for time series analysis.
I wrote a function to read the data in various formats based on the requirements of the algorithm being modelled. The DataFrame contains two columns: a date stamp in DateTime format (‘ds’ for the Prophet model) and PJME_MW (‘y’ for Prophet), representing hourly power consumption. 
The data was divided into training 70% and test set 30% for SARIMAX and Prophet models and 70% (training), 20% (test), and 10% (validation) sets for deep learning model (LSTM). 

### Evaluation Metrics 

- Root Mean Squared Error (RMSE), 
- Mean Absolute Error (MAE), 
- Mean Absolute Percent Error (MAPE)

### Time series models

#### SARIMAX model

Because of the large data set and limited computing resources and the extensive time required to train such a model, it was not feasible to use the entire training set. Instead, the last 3500 timesteps of the hourly power consumption data were utilised. These 3500 data points were divided into 70% (2450 timesteps) for training and 30% (1050 timesteps) for testing. Auto-correlation and partial auto-correlation plots of the series were generated, and the model was subsequently fitted with a seasonality of 24 hours. Akaike Information Criterion (AIC) was used to automatically select the best model (with least AIC). The model's performance is visualised on a plot and evaluated.

#### Prophet model

The training and test sets have 70% (101756 timesteps) and 30% (43610 timesteps), respectively. First, we trained a model that considers only daily, weekly, and yearly seasonality. Then, we trained another
model that also accounted for monthly seasonality within the year. The results of these models were plotted and evaluated.

#### LSTM model

To ensure the results are reproducible, a random Seed was set. 
The data was scaled using MinMaxScaler to normalise the values between 0 and 1. This reduces the time required to train the model and improves model performance.
A helper function ‘create_dataset’ was defined to generate the input features and target values for the model, based on a specified look-back period. The input data was reshaped to fit the requirements of the LSTM network, which expects the data in a 3D array of [samples, timesteps, features]. An early stopping callback was set up to prevent overfitting by stopping the training process if the validation loss does not improve for 3 consecutive epochs.
LSTM Network: An LSTM network was built using Keras. which consists of an LSTM layer with 32 units followed by a Dense layer with 1 unit. The model was trained on the training data for 50 epochs with
a batch size of 32, using early stopping and validating against the validation set. The results were plotted and evaluated.

### Results/Findings 

- LSTM model demonstrated the best predictive accuracy, although it required the longest training time. Its ability to capture complex temporal dependencies made it particularly suited for power consumption data, which involves complex patterns that simpler models struggle to represent. 
- Prophet model offered a strong balance between accuracy and training efficiency, particularly when incorporating multiple seasonality, making it a practical choice when computational resources are limited.
- The SARIMAX model, while traditionally effective for time series forecasting, showed limitations with this dataset, both in terms of accuracy and the computational effort required, partly due to the challenges in handling large datasets and complex seasonal components.

### Conclusion and Recommendation 
- These findings emphasise the need for careful model selection based on the specific requirements of a forecasting task.
- While deep learning models like LSTM offer superior accuracy, more efficient models like Prophet may be preferable in scenarios where speed and computational resources are critical. 
- Future research could explore the potential of hybrid models, combining the strengths of these approaches to further improve forecasting performance and efficiency.

### Limitations 
A large portion of the data set was eliminated when building the SARIMAX model.

### References

- Abdoli, G. (2020). Comparing the prediction accuracy of LSTM and ARIMA models for time-series with permanent fluctuation. Periódico do Núcleo De Estudos E Pesquisas Sobre Gênero E
DireitovCentro De Ciências Jurídicas-Universidade Federal Da Paraíba, 9
- Al-Nefaie, A. H., & Aldhyani, T. H. (2022). Predicting close price in emerging Saudi Stock Exchange: time series models. Electronics, 11(21), 3443.
- Bala, A., Ismail, I., Ibrahim, R., Sait, S. M., & Oliva, D. (2020). An improved grasshopper optimization algorithm based echo state network for predicting faults in airplane engines. IEEE Access, 8, 159773–159789.
- Binkowski, M., Marti, G., & Donnat, P. (2018). Autoregressive convolutional neural networks for asynchronous time series. Paper presented at the International Conference on Machine Learning, 580–589.
- Böse, J., Flunkert, V., Gasthaus, J., Januschowski, T., Lange, D., Salinas, D., Schelter, S., Seeger, M., &
- Wang, Y. (2017). Probabilistic demand forecasting at scale. Proceedings of the VLDB Endowment, 10(12), 1694–1705.
- Box, G. E., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). Time series analysis: forecasting and control. John Wiley & Sons.
- Brockwell, P. J., & Davis, R. A. (1991). Time series: theory and methods. Springer science & business media.
- Cao, D., Jia, F., Arik, S. O., Pfister, T., Zheng, Y., Ye, W., & Liu, Y. (2023). Tempo: Prompt-based generative pre-trained transformer for time series forecasting. arXiv Preprint arXiv:2310.04948,
- Cao, J., Li, Z., & Li, J. (2019). Financial time series forecasting model based on CEEMDAN and LSTM. Physica A: Statistical Mechanics and its Applications, 519, 127–139.
- Chai, T., & Draxler, R. R. (2014). Root mean square error (RMSE) or mean absolute error (MAE)?– Arguments against avoiding RMSE in the literature. Geoscientific Model Development, 7(3), 1247–1250.
- Chatfield, C., & Xing, H. (2019). The analysis of time series: an introduction with R. Chapman and hall/CRC.
- Chaturvedi, S., Rajasekar, E., Natarajan, S., & McCullen, N. (2022). A comparative assessment of SARIMA, LSTM RNN and Fb Prophet models to forecast total and peak monthly energy demand for India. Energy Policy, 168, 113097.
- Chen, Y., Kang, Y., Chen, Y., & Wang, Z. (2020). Probabilistic forecasting with temporal convolutional neural network. Neurocomputing, 399, 491–501.
- Clements, M. P., & Hendry, D. F. (1998). Forecasting economic processes. International Journal of Forecasting, 14(1), 111–131.
- Dietterich, T. (1995). Overfitting and undercomputing in machine learning. ACM Computing Surveys (CSUR), 27(3), 326–327.
- Dimoulkas, I., Mazidi, P., & Herre, L. (2019). Neural networks for GEFCom2017 probabilistic load forecasting. International Journal of Forecasting, 35(4), 1409–1423.
- Duarte, F. B., Tenreiro Machado, J. A., & Monteiro Duarte, G. (2010). Dynamics of the Dow Jones and the NASDAQ stock indexes. Nonlinear Dynamics, 61, 691–705.
- Feng, T., Zheng, Z., Xu, J., Liu, M., Li, M., Jia, H., & Yu, X. (2022). The comparative analysis of SARIMA, Facebook Prophet, and LSTM for road traffic injury prediction in Northeast China. Frontiers in Public Health, 10, 946563.
- García, F., Guijarro, F., Oliver, J., & Tamošiūnienė, R. (2023). Foreign exchange forecasting models: ARIMA and LSTM comparison. Engineering Proceedings, 39(1), 81.
- Goodfellow, I. (2016). Deep learning.
- Hamilton, J. D. (2020). Time series analysis. Princeton university press.
- Han, M., & Xu, M. (2017). Laplacian echo state network for multivariate time series prediction. IEEE Transactions on Neural Networks and Learning Systems, 29(1), 238–244.
- Hipel, K. W., & McLeod, A. I. (1994). Time series modelling of water resources and environmental systems. Elsevier.
- Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. Neural Computation, 9(8), 1735–1780.
- Hwang, J., Jeong, Y., Park, J. M., Lee, K. H., Hong, J. W., & Choi, J. (2015). Biomimetics: forecasting the future of science, engineering, and medicine. International Journal of Nanomedicine, , 5701– 5713.
- Hyndman, R. J., & Athanasopoulos, G. (2018). Forecasting: principles and practice. OTexts.
- Hyndman, R. J., & Koehler, A. B. (2006). Another look at measures of forecast accuracy. International Journal of Forecasting, 22(4), 679–688.
- Kempelis, A., Narigina, M., Osadcijs, E., Patlins, A., & Romanovs, A. (2023). Machine Learning-based Sensor Data Forecasting for Precision Evaluation of Environmental Sensing. Paper presented at the 2023 IEEE 10th Jubilee Workshop on Advances in Information, Electronic and Electrical Engineering (AIEEE), 1–6.
- Kim, Y. (2023). A Study on the Machine Learning Model for the Financial Performance Prediction of Startups. Asia-Pacific Journal of Convergent Research Interchange, 9, 67–77.10.47116/apjcri.2023.07.06
- Kumar, D., Singh, A., Samui, P., & Jha, R. K. (2019). Forecasting monthly precipitation using sequential modelling. Hydrological Sciences Journal, 64(6), 690–700.
- Laptev, N., Yosinski, J., Li, L. E., & Smyl, S. (2017). Time-series extreme event forecasting with neural networks at uber. Paper presented at the International Conference on Machine Learning, , 341–5.
- Li, Y., Yu, R., Shahabi, C., & Liu, Y. (2017). Diffusion convolutional recurrent neural network: Datadriven traffic forecasting. arXiv Preprint arXiv:1707.01926,
- Lipton, Z. C., Kale, D. C., Elkan, C., & Wetzel, R. (2015). Learning to diagnose with LSTM recurrent neural networks. arXiv Preprint arXiv:1511.03677,
- Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2020). The M4 Competition: 100,000 time series and 61 forecasting methods. International Journal of Forecasting, 36(1), 54–74.
- Nelson, D. M., Pereira, A. C., & De Oliveira, R. A. (2017). Stock market's price movement prediction with LSTM neural networks. Paper presented at the 2017 International Joint Conference on Neural Networks (IJCNN), 1419–1426.
- Paine, C. T., Marthews, T. R., Vogt, D. R., Purves, D., Rees, M., Hector, A., & Turnbull, L. A. (2012). How to fit nonlinear plant growth models and calculate growth rates: an update for ecologists. Methods in Ecology and Evolution, 3(2), 245–256.
- Pang, X., Zhou, Y., Wang, P., Lin, W., & Chang, V. (2020). An innovative neural network approach for stock market prediction. The Journal of Supercomputing, 76, 2098–2118.
- Peixeiro, M. (2022). Time Series Forecasting in Python. Manning Publications Co.
- Salinas, D., Flunkert, V., Gasthaus, J., & Januschowski, T. (2020). DeepAR: Probabilistic forecasting with autoregressive recurrent networks. International Journal of Forecasting, 36(3), 1181–1191.
- Shumway, R. H., & Stoffer, D. S. (2017). Time Series Analysis and Its Applications. Springer Texts in Statistics, 10.1007/978-3-319-52452-8
- Siami-Namini, S., Tavakoli, N., & Namin, A. S. (2018). A comparison of ARIMA and LSTM in forecasting time series. Paper presented at the 2018 17th IEEE International Conference on Machine Learning and Applications (ICMLA), 1394–1401.
- Siami-Namini, S., Tavakoli, N., & Namin, A. S. (2019a). A comparative analysis of forecasting financial time series using arima, lstm, and bilstm. arXiv Preprint arXiv:1911.09512,
- Siami-Namini, S., Tavakoli, N., & Namin, A. S. (2019b). The performance of LSTM and BiLSTM in forecasting time series. Paper presented at the 2019 IEEE International Conference on Big Data (Big Data), 3285–3292.
- Staudemeyer, R., & Omlin, C. (2013). Evaluating performance of long short-term memory recurrent neural networks on intrusion detection data10.1145/2513456.2513490
- Taylor, S. J., & Letham, B. (2018). Forecasting at scale. The American Statistician, 72(1), 37–45.
- Wei, W. (2006). Time Series Analysis: Univariate and Multivariate Methods, 2nd edition, 2006
- Willmott, C. J., & Matsuura, K. (2005). Advantages of the mean absolute error (MAE) over the root mean square error (RMSE) in assessing average model performance. Climate Research, 30(1), 79–82.
- Yamak, P., Li, Y., & Gadosey, P. (2019). A Comparison between ARIMA, LSTM, and GRU for Time Series Forecasting10.1145/3377713.3377722
- Ye, J., Liu, Z., Du, B., Sun, L., Li, W., Fu, Y., & Xiong, H. (2022). Learning the evolutionary and multiscale graph structure for multivariate time series forecasting. Paper presented at the Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, 2296–2306.
- Yu, L., Wang, S., & Lai, K. K. (2008). Forecasting crude oil price with an EMD-based neural network ensemble learning paradigm. Energy Economics, 30(5), 2623–2635.
- Zhang, G. P. (2003). Time series forecasting using a hybrid ARIMA and neural network model. Neurocomputing, 50, 159–175.
- Zhou, K., Wang, W. Y., Hu, T., & Wu, C. H. (2020). Comparison of Time Series Forecasting Based on Statistical ARIMA Model and LSTM with Attention Mechanism. Journal of Physics: Conference Series, 1631(1), 012141. 10.1088/1742-6596/1631/1/012141.
