# KNN

## Algorithm principle

### Classification

As the name of K-NN, we classify an object by finding k nearest neighbor “around” the object and decide which class the object belongs to. It sounds very simple and we can obtain three key aspects that influence the result of k-NN:
#### K value:

The value of k determines how many neighbors we will use to classify an input. We can choose k=1 by intuitive--assign the input to the class of its closest neighbor, and it sometime works not bad (and 1-NN can also guarantee a lower bound of the error rate). But that seems so arbitrary and we often need to try other k to find the best performance on the validation set (e.g. cross-validation, but we didn’t use it in our experiment since the validation set was fixed).

#### Similarity metric

Since we need to find a way to measure the similarity between two samples, distance is a good choice for us. There are many distance we can choose (Euclidean distance, Cosine similarity, etc.), we need to find out which one is better according to different dataset.  
I have tried three distance in ![](http://latex.codecogs.com/gif.latex?$L_p$) space (Manhattan distance, Euclidean distance, Chebyshev distance) and cosine similarity.   
First we compare these ![](http://latex.codecogs.com/gif.latex?$L_p$) distances. According to the definition of ![](http://latex.codecogs.com/gif.latex?$L_p$) distance, we can find that when we have different dimension, we can get different result if we use different value of “p” because the dimensions they take into account are different. The larger p it is, the less dimension it will consider and the small the value is. There is a very simple example to explain it. Let us have two point ![](http://latex.codecogs.com/gif.latex?p_1(4,0)) and ![](http://latex.codecogs.com/gif.latex?p_2(4,3)), and we have another point p_3 locate at ![](http://latex.codecogs.com/gif.latex?(0,0)). By intuitive we will say that ![](http://latex.codecogs.com/gif.latex?p_2) is farther to ![](http://latex.codecogs.com/gif.latex?p_3) than ![](http://latex.codecogs.com/gif.latex?p_1) (since the Euclidean distance between ![](http://latex.codecogs.com/gif.latex?p_2) and ![](http://latex.codecogs.com/gif.latex?p_3) is 4.24). But when we choose p larger than 2, we will find that in such scenario, ![](http://latex.codecogs.com/gif.latex?p_2) is closer to ![](http://latex.codecogs.com/gif.latex?p_3) than ![](http://latex.codecogs.com/gif.latex?p_1) (sometimes this is not we want). Then we can generalize it to a common condition,   
![](http://latex.codecogs.com/gif.latex?‖x‖_{p+α}≤‖x‖_p,) for any p≥1 and α≥0.
So if we have many dimension have to be taken, choose a small p to handle more dimension (but we usually do not choose p less than 0 since it will increase the complexity).
<<<<<<< HEAD
And another way to measure the similarity between two samples is cosine similarity. An explicit comparison is shown bellow   
![comparison of L2 distance and cosine similarity](./img/L2_cos_comp.png)  
=======
And another way to measure the similarity between two samples is cosine similarity. An explicit comparison is shown bellow  
![comparison of L2 distance and cosine similarity](./img/L2_cos_comp.png) 
>>>>>>> b1e8909d9198c366f6d7315e6bf4353716270416
As the figure shows, cosine similarity measures the difference between two vector by measuring the angle between them (i.e. inner product space).  
![](http://latex.codecogs.com/gif.latex?cos\\_similarity=\\frac{\\mathbf{x_1}\\mathbf{x_2}}{‖\\mathbf{x_1}‖_2‖\mathbf{x_2}‖_2})  
As we all know that the cosine of 0° is 1 and it’s less than 1 for other angle. And since we only use it in positive space (normalized tf-idf value, with the interval [0, 1]), I transform the cosine similarity via cosine distance via  
![](http://latex.codecogs.com/gif.latex?cos\\_distance=1-cos\\_similarity)  
for calculation convenience.  
The cosine similarity is widely used in text classification owing to its efficiency and it high performance when facing to sparse vectors since it will only consider the non-zero dimensions.   
And another important reason we use it in text classification is that, what we care about sometimes is not the direct distance between two vectors but the similarity between them. To be more clear, we can imagine that we have training data on the unit circle and a test sample locate at the origin, the Euclidean distances are same but we know the vector them form are totally different, and some of them even opposite to each other.

#### Decision rule
When we choose k larger than 1, one thing we need to take into account is how to decide the class of the input according to those k neighbors. A common way is to use a rule called majority voting rule, that means, assign it to the class that most occurs in k neighbors.   
But a drawback of the basic "majority voting" classification is that when the feature distribution is not uniform, that is, a more frequent class tend to dominate the prediction since it has the superiority of quantity. One way to overcome this problem is to weight the contribution of each neighbor using the distance between the input and its neighbor, I also use it in my experiment.

### Regression
In regression problems, we are facing continuous variables. We can calculate the average value of the k nearest neighbors easily. And since we are deal with the probability of 6 emotions the document might be, we then need to normalize the result to make the sum of 6 probabilities to be 1 (dividing by the sum of 6 regression result).
