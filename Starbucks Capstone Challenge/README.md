# Starbucks Capstone Challenge
Udacity Data Scientist Nanodegree Capstone Project.

### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Description](#files)
4. [Results](#results)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

This code runs with Python version 3.* and there should be no necessary libraries to run the code here beyond the Anaconda distribution of Python.


## Project Motivation <a name="motivation"></a>

This is part of Udacity Nanodegree Capstone Project. The aim is develop a set of characteristics to determine which customers complete which offer type. To explore the customers of Starbucks and their behavior regarding the offers they receive. Also, to check if we can predict what type of offer is the best for each user based on their demographics.

## File Description <a name="files"></a>

**Starbucks_Capstone_notebook.ipynb**: Notebook containing the data analysis. </br>
**Starbucks_Capstone_notebook.html**: An HTML file of the jupyter notebook. </br>
**data/portfolio.json**: containing offer ids and meta data about each offer (duration, type, etc.). </br>
**data/profile.json**: demographic data for each customer. </br>
**data/transcript.json**: records for transactions, offers received, offers viewed, and offers completed. </br>


## Results <a name="results"></a>
The main findings of the analysis are:</br>
* The largest age group in the customers data are between 48 to 70 as they have above 200 user per age (up to 400). </br>
* Most of the customers are males. </br>
* The customers income is mostly middle class, and the number of low income customers is higher than the high income customers. </br>
* The number of profiles is increasing yearly but it dropped at the last year which may be due to the offer types are not interesting anymore. </br>
* The order at which both genders and all age groups prefer the offers is: BOGO, discount, then informational. </br>
* The most recurring event is the transaction event by approximately 50% more than the other events. </br>
* Females complete offers more than males.</br>
* The models selected were overfitted to the dataset even though the models were simple. Thus, no model were selected to predict the offer type. </br>

For more details, please check the post available [here](https://medium.com/@MaysamF/starbucks-capstone-challenge-9003c9422a76)

## Licensing, Authors, Acknowledgements<a name="licensing"></a>
Must give credit to Starbucks for the data and Udacity for providing the opportunity to perform this project.
