# DUAL USE IN CHEM: exploring ways to censor chemical data to mitigate dual use risks.
As we explore strageties to mitigate dual use risks in predictive chemistry (DURPC), we present our data-level mitigation strategy: Selective Noise Addition. 
In pursuit of public distribution of chemical data in safe ways, we test adding noise to only selected data in the dataset wiht labels identified as sensitive. 
We test this method with three models: 

1. 1-D Polynomial Regression
2. Multilayer Perceptron (MLP)
3. Graph Convolutional Network (GCN) predicting lipophilicity
