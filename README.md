<br/>
<br/>

# ML-python-studyML 🤖 
: Study and practice various machine learning models

<br/>
<br/>

### 1. Naive Bayes Model

- 1-1 Gaussian Naive Bayes
- 1-2 Multinomial naive bayes


### 2. KNN : k-Nearest neighborhood

- select Distance d(a,b)
  * Categorical variables
     : Hamming distance
     
  * Continuous variables
     : Euclidian distance, Manhattan distance

### 3. LDA : Linear Discriminant Analysis 

 - 3-1. Assumption
     - each group of numbers has a probability distribution in the form of a normal distribution. 
     - each group of numbers has a similar covariance structure.
     
 - 3-2. The characteristics of the decision boundary obtained as a result of LDA
    
     - Axis orthogonal to boundary
          - Consider the shape of the distribution when the data is projected onto this axis.

     - Maximize the difference in means?
         - Use the vector difference vector of the two means.
    
      => Boundary that maximize the difference between variance and mean
 
 - 3-3. Advantage
 
    - Unlike the naïve bayes model, it reflects the covariance structure between the explanatory variables.
    - Relatively **robust** even when assumptions are violated.  
    
 - 3-4. Disadvantages
   
    - The number of smallest samples must be greater than the number of explanatory variables.  
    - Poorly explained if it deviates significantly from the normal distribution assumption.  
    - Fails to reflect cases where the covariance structure is different between categories y . 

 - 3-5. Define and understand QDA
    - QDA removes the assumption of a common covariance structure∑ independent of k.
        - It can be utilized when different categories of Y have different covariance structures.
        
 - 3-6. LDA vs QDA
     <img width="1071" alt="Capture d’écran 2023-04-15 à 15 49 07" src="https://user-images.githubusercontent.com/63314860/232228009-d66c9be1-8f92-48b0-bb2d-ffcd85a97e50.png">
    
    - Relative advantages of QDA
        - y Allows for different covariance structures for different categories.
    - Relative disadvantages of QDA
        - If you have a large number of explanatory variables, there are more parameters to estimate
                
           - Requires a large sample size

<br/>

### 6. Neural Network
