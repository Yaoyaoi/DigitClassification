# DigitClassification
Using ANN to classify digit pictures.

* data set: 
    * [data in website](http://amlbook.com/support.html)  
    * [data in this repo](data/)
    * two features: symmetry and intensity
    * raw features: 256 grayscale values.

### Task1  
Plot a scatter plot of two features for 10 labels.

### Task2
Apply neural network of 1 hidden layer to classify 1 and 5. The features are: symmetry and average intensity. Use 3-fold cross-validation.

### Task3
Apply two-layer neural network for classification of 1 and 5, using the raw features as input. Apply 3-fold cross-validation. Train and test the following structures [256, 6, 2, 1], [256, 3, 2, 1].plot the change of the in-sample error and test-set error for each iteration. 

### Task4
Apply neural network and SVM for classification for all 10 digits, using the raw features as input.

### code
* [code](code/)
* [code instructions](report/readme.md)
### Final report
[report.md](report/report.md)   
[report.pdf](report/report.pdf)