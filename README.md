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

### 5. Decision Tree

- 5-1. Definition 
 
   : A model that creates a criterion of variables and uses them to categorize a sample, and then estimates the properties of the categorized group.

  - Advantages: highly interpretable, intuitive, universal.
  - Disadvantages: high volatility. Can be sensitive to sample. 

- 5-2. Decision tree terminology

<br/>
<img width="315" alt="Capture d’écran 2023-04-17 à 14 45 20" src="https://user-images.githubusercontent.com/63314860/232487938-a21315ce-fbe6-45c2-bb1b-bd9b3ce036cc.png">
<br/>

   - Node - The location of the variable on which the classification is based. Divide the sample based on this.

      - Parent node - a relative concept. Parent node.
      - Child node - Lower node.
      - Root node - The top-level node with no child nodes.
      - Leaf node (Tip) - The lowest node with no children.
      - Internal node - a node that is not a Leaf node.

   - Edge - Where the conditions that categorize the samples are located.
   - Depth - the number of edges that must be traversed to reach a particular node from the Root node.
   
   - Depending on the response variable
   
      - Categorical variables : Classification tree
      - Continuous Variables  : Regression Tree (Estimate the category of y from its mean value)

<br/>
<img width="1027" alt="Capture d’écran 2023-04-17 à 14 46 40" src="https://user-images.githubusercontent.com/63314860/232488226-46f27242-0059-4ded-b715-000d40d5349e.png">
<br/>

- 5-3. Entropy

  - Entropy is often used as a criterion to select the best attribute for splitting a node in the tree.
  
  - The attribute that maximizes the information gain, which is the reduction in entropy achieved by splitting the node according to that attribute, is chosen as the best attribute.
  
  - The entropy of a set S with respect to a binary classification problem is given by the following formula:  
         <img width="345" alt="Capture d’écran 2023-04-17 à 15 14 12" src="https://user-images.githubusercontent.com/63314860/232494687-85ffc913-bd73-44ac-8414-4c6679e22487.png">


- 5-3. Information Gain

  - Entropy difference before and after a particular node in a decision tree.
  
  - A higher information gain indicates that the attribute can split the dataset into more homogeneous subsets, making the classification task easier. Conversely, a lower information gain indicates that the attribute is less useful for classification.
  
      <img width="926" alt="Capture d’écran 2023-04-17 à 15 25 40" src="https://user-images.githubusercontent.com/63314860/232497603-11c937ed-77c2-4f72-a16e-139207bf28d7.png">
  

- 5-4. classification Tree

  - According to the Tree condition. The idea of dividing the area that X can have into blocks.

  - Estimate Y from the attributes of the samples in the blocked region.
  
  <img width="866" alt="Capture d’écran 2023-04-17 à 15 32 42" src="https://user-images.githubusercontent.com/63314860/232499520-6be25d89-1b13-4d24-9efb-d6e95a065e2c.png">

  - For the areas divided, select the variables and criteria that give the best values for the measure below.

     - Entropy

     - Misclassification rate
  
       <img width="493" alt="Capture d’écran 2023-04-17 à 15 36 07" src="https://user-images.githubusercontent.com/63314860/232500457-1e2dcac8-106a-4996-8a67-afacb5d05efd.png">

     - Gini index
  
       <img width="478" alt="Capture d’écran 2023-04-17 à 15 36 37" src="https://user-images.githubusercontent.com/63314860/232500606-e9199e02-fc8e-447c-9196-299cb8257f10.png">
  
  - For a determined R<sub>m</sub>, 
     
      <img width="361" alt="Capture d’écran 2023-04-17 à 15 42 04" src="https://user-images.githubusercontent.com/63314860/232501931-af4b3159-5e78-455b-9c86-5cee0e38ff92.png">
     
     - the category of estimated Y : 
       
       <img width="242" alt="Capture d’écran 2023-04-17 à 15 42 12" src="https://user-images.githubusercontent.com/63314860/232501966-17d29abf-8fdd-4341-a28a-e8fdb0eed249.png">
      
 - 5-5. Regression Tree
          
   <img width="1106" alt="Capture d’écran 2023-04-17 à 15 51 26" src="https://user-images.githubusercontent.com/63314860/232504500-f52cb623-bc10-4ab4-b058-c70b4b8e4008.png">
   
   
   - Estimated value of Y:
   
      <img width="387" alt="Capture d’écran 2023-04-17 à 15 52 37" src="https://user-images.githubusercontent.com/63314860/232504765-e288ebd2-f0f5-4949-a5f2-d9d3ce08aa10.png">

   - For a determined R<sub>m</sub>,  

     <img width="390" alt="Capture d’écran 2023-04-17 à 15 57 11" src="https://user-images.githubusercontent.com/63314860/232506039-068729b8-c21c-4bce-b215-a14e694102db.png">

### 6. Neural Network
