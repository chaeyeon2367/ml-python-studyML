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

### 4. SVM : Support Vector Machine 

- 4-1. Background 

   - When assumptions about the distribution of data are hard to make, how do you split the data below?

     - focus on the boundary
     - determine the boundary that maximizes the margin as shown below. 
   <br/>
   <img width="388" alt="Capture d’écran 2023-04-17 à 02 49 14" src="https://user-images.githubusercontent.com/63314860/232354109-27210622-7360-4539-b0ff-82ed02014c38.png">

   - Problem
     - What if there are cases that are not exactly distinct?

        => Allow a small amount of error and determine the boundary to minimize it
 
   <img width="393" alt="Capture d’écran 2023-04-17 à 02 51 25" src="https://user-images.githubusercontent.com/63314860/232354271-6513431c-c8d1-416b-9840-c17f0e3dd2c6.png">
   

   - The dependent variable is divided into two categories based on the form of the data.

     - Categorical variables
        - Support vector classifier
     - Continuous variables
        - Support vector regression (SVR)
       
   <br/>
   <img width="750" alt="Capture d’écran 2023-04-17 à 02 56 25" src="https://user-images.githubusercontent.com/63314860/232354648-1591620b-8670-44d7-9a37-7e494b45cb1c.png">
   <br/>
   
   
   - Key to SVM, SVR
      - Distinguish between what will and will not affect model cost with margins

         - SVM 
             - Points that fall within the margin, or are categorized in the opposite direction.
         - SVR
             - Points that are outside the margin. 

- 4-2. SVM with Kernel

    - For non-linear relationships
    - The curse of dimensionality
      - When fitting data with a non-linear structure, it is necessary to use a kernel.
      - However, as the dimensionality of the dth-degree polynomial increases, above a certain dimensionality, the number of parameters that need to be estimated increases, resulting in higher test errors.
      
- 4-3. SVM vs. LDA
   - Relative Advantages of SVM
   
      - When the data distribution is difficult, it is inefficient to consider the covariance structure.  
      
         - Only observations near the boundary can be considered.
              
      - Higher prediction accuracy. 
      
   - Relative disadvantages of SVM
   
      - Need to determine C
      - Takes a long time to build the model
   
   
   
<br/>

### 6. Neural Network
