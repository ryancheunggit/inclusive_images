# inclusive-images-challenge
Repo for 2018 Nips competition  

## Quick summary of my experience with this competition: 

### TRAINING: 
1. Training a vanilla ResNet50 with multi-label classifier head on the training set with binary cross-entropy loss seems overfitting really quickly. 
2. Focal loss is a bit hard to start with. It might be good to warm up the model with BCE and then swap the loss to focal loss. Maybe because the ouput is 7178 class, and for small batches, focal loss here is hard to startwith?
3. F2 loss which I found towards the end seems the best choice. (Well, it so happens the competition metric is also f2). So whenever possible, optimize directly according to the metric. 
4. As a result of 1, much of my time was spent to set up a multi-task setting. My model was trained on classifier output and an label embedding output. I was hopping that, with proper weighting, the cosine proximity loss from embedding output can help with generalization. 
5. I found variants of ResNets are a bit harder to train on this dataset then vanilla ResNet.   
+ To summarize: my ResNets were trained on a two tasks setting towards convergence, then throw out the embedding output layer and fine tuning a bit as a pure classifier model.  

### LABEL EMBEDDING: 
1. Label embedding is calculated on PMI matrix from the training labels.  
2. I tried to directly use the predicted embedding to generate labels by sorting label embeddings with the prediction using cosine similarity and take the top k. It gives better coverage with top k predictions, but worse f2 than the classifier + threshold method. 
+ To summarize: I wasn't able to use label embedding to directly generating good predictions, but it helped during training phase acting a sort of like a regularizer. 

### ABANDONED ADAPTATION / THRESHOLDING: 
1. Very early in the competition I tried to fine tune my trained ResNet on the 1000 tuning set, as a result, the adapted model gives very promising public leaderboard score. But I think it is totally against the spirit of the task and given up on this approach. 
2. I did not try the adapted model on stage 2. But I suspect the adaptation would still gives a lot information to the model about how the labeling is done differently between the training set and testing set.  
3. I also decided not to tune thresholds using the tuning set, due to the same rational, and I believe that I am eating the consequences of my decision.  
+ To summarize: It is very confusing to me that google provided this 1000 image tuning set. An analogy I can think of is: 
> Say, you are taking a course and the professor gives you a bunch exercise to practice with, which are mostly multiple choice questions with 3 or more correct answers. There are two exams for the course: a midterm and an final. Your course grade depends only on the final. During the midterm, the professor wrote a few answers on the blackboard, and you notice they all have 1 to 2 correct answers, and your peers who adjusted their answers did very well on the midterm. The question now is, should you adjust your solutions during the final or not.   

+ If this type of adaptation is useful towards stage 2, I think it would be a fault of the host in the competition setting. It is irony that the competition overview stated that the purpose of the competition is to promote methods that depends not on gathering new images, and in reality the awards goes to people who explicitly used some test images to adapt their model. 
