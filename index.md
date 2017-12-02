---
title: Team 51 Will Cho, Haruka Uchida, Jessica Zhao
---
> Alzheimer's Classification - CS 109a Final Project

This website serves to display subsets of both the EDA we've performed, as well as the models we have tested and ultimately decided on for the classification of Alzheimer's Disease.

## Data Description: 
The data we use is from the Alzheimer's Disease Neuroimaging Initiative (ADNI) database, and includes a large collection of patient information including demographics and genetics, lifestyle factors, medical history, lab records such as blood biomarkers, imaging data such as MRI and PET images, and cognitive test scores. In addition, we have the diagnostic categories for each patient (AD, MCI, and Normal, where AD indicates presence of Alzheimer’s disease, and MCI is mild cognitive impairment).

An important characteristic of our data is that there are 94 variables per observation, but there are many missing values. This is an issue that will have to be addressed in our model- we can delete observations with missing values, impute missing values using values such as means, or impute missing values using a model. We will not want to simply delete observations with missing values because most observations do not have all 94 variables, and it is crucial that we do not bias the data that we use to make our model; it is possible that patients who are missing values for certain predictors are similar in some way, such as being from a low income town with a low amenity hospital. Therefore in our model, we will impute missing values.

The 94 variables of the dataset are diverse; some variables capture physiological measures, while some are simply unique ID numbers. In our model, we will not use most of the uniquely identifying variables such as Patient ID number. Further, some of the variables are categorical or binary, or may not be recorded as so but conceptually are; for these variables we created indicator variables. For variables we deem necessary, we included interaction variables in addition to the individual variables themselves.

## Visualizations/Methods:

We have created charts to show trends within the data, shown below. Factors that we specifically analyze include patient demographics, to understand the balance of the dataset. We find that roughly half of the patients in the ADNI study are diagnosed with MCI, a quarter are diagnosed with Alzheimer’s Disease, and the remaining receive a Normal diagnosis. This is not representative of the prevalence of AD in a random population sample, however, is beneficial for our purposes of examining the most important factors in determining whether, and at which age, the onset of AD is likely to occur. 

## Project question:
From our EDA, we can confirm that we must make indicator variables corresponding to certain variables in the data. Additionally, there appears to be a difference in risk of developing Alzheimer’s between demographics; demographic factors are thus important in determining risk of Alzheimer’s.
Our project question is: What factors of a patient are important in predicting the patient’s risk of Alzheimer's?  Specifically, we have limited patient data, and want to develop a model using as little data as possible, for the least expensive early detection of Alzheimer’s.

Thus, our model will predict risk of Alzheimer’s as a probability for observations given the observed traits. Our second project question is: given a patient is expected to develop Alzheimer’s, at what would they be expected to do so? Therefore, our second model will predict the age at which a patient will develop Alzheimer’s, given the observed traits.

