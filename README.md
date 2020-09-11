# TextAnlaysis

## 1.Introduction
### 1.1.Motivation
In the UK, where social classes are relatively distinct, people in different classes tend to read different different newspapers. In other words, it can be said that people can be 'classified by the papers they read.

It is a well known fact that in Korea, people with different political views read news from different publishers, meaning political articles from different publishers are probably different. So, we wanted to see if the difference in political articles could be detected by a model, and also see if other types of articles could be classified into publishers.

### 1.2.Problem Definition/Goal
In this project, we set out to classify newspapers through machine learning. Newspaper articles are to be classified by their publishers.
Specifically, we divided newspapers by social, political, and technical subsections, and we wanted to figure out which newspaper company they were published from.

We tried various methods like TextCNN, Fasttext, and RNN to solve this problem. And we also tried well known traditional classifiers such as logistic regression, SVM, and naive bayes for comparison.


## 2.Data
### 2.1.News Crawling
We used Selenium to crawl(scrape) news articles from naver.com. '경향신문', '한겨례신문', '동아일보', and '조선일보' were chosen as publishers, the first two being 'liberal', and the last two being 'conservative'.

Each publisher was then divided into 3 subsections, 'Politics', 'Society', and 'Technology'. The subsections were introduced as a heuristic way to indicate 'politicalness'. 'Politics', 'Society', and 'Technology' can be seen as showing high, intermediate, and low politicalness, respectively.

Our initial guess was that the articles in subsection 'Politics' shoulc be classified with the highest accuracy, followed by 'Society', and 'Tech'.

We collected 500 articles per subsection, with the total of 500*3*4=6000 articles. Training:Test ratio was set to 7:3.
<div>
<img src = "https://user-images.githubusercontent.com/33714067/92872469-06a8d180-f441-11ea-99e2-a4ea3810d512.png">
</div>

### 2.2.Data Preprocessing
The twitter tokenizer known as Okt was used to tokenize the documents. We defined general stopwords using 'https://www.ranks.nl/stopwords/Korean' and we additionally defined other stopwords like special characters. As a result of tokenization, each document was transformed into a vector of tokens.

Next, we embedded the documents. Depending on the appropriate model, different embedding techniques were used. For logistic regression, SVM, and naive bayes, each of the tokenized newspaper articles was converted to a numeric vector of length 300 using the doc2vec embedding technique.

In the case of neural networks, each token was embedded a s a 300 dimension vector using word2vc embedding. Each document must be of uniform size to be used as input, so if the document was longer than 350 tokens, the rest were cut off, and if it was shorter, padding was added to fix the size to 350.

In summary, each newspaper article was converted into a 300*350 matrix.


## 3. Methods
### 3.1.Traditional Classifiers
Logistic regression, SVM, and Naive bayes were used before implementing neural networks. However, all the traditional classifiers worked poorly.

### 3.2.CNN
We designed Convolution Neural network using TensorFlow.
Figure 2 shows the overall architecture.
<div>
<img src = "https://user-images.githubusercontent.com/33714067/92873565-21c81100-f442-11ea-8b68-f8aad2bc680b.png">
</div>

### 3.3.RNN
Since one news article is of 350 vectors, n_step was set to 350. n_hidden was set to 32. We tried various n_hidden values, but the performance would not improve. Thus, We chose a small n_hidden to have shorter computation time and also to address overfitting. Figure 3 shows the RNN code.
<div>
<img src = "https://user-images.githubusercontent.com/33714067/92873833-681d7000-f442-11ea-8dbb-55e8b9cdeb3d.png">
</div>

### 3.4.FastText
We used an already existing model. Figure 4 shows how we imported such model.
<div>
<img src = "https://user-images.githubusercontent.com/33714067/92874097-adda3880-f442-11ea-80ba-334c1cf86868.png">
</div>


## 4.Result and Result Analysis
Figure 5 shows the results of various classification methods.
<div>
<img src = "https://user-images.githubusercontent.com/33714067/92874596-23de9f80-f443-11ea-85c8-5edd3cb942eb.png">
</div>


CNN and Fasttext performed very well. However, RNN showed relatively bad performance. In all cases, CNN and Fasttext outperformed RNN. The reason is thought to be caused by the method of tokenization. 

RNN is a suitable algorithm for learning data with sequential structure. Therefore, it can be regarded as a suitable algorithm when learning data including the detailed elements of real sentences such as propositions, sentence symbols, and so on.

Meanwhile, CNN is an algorithm that finds patterns in neighboring elements, rather than learning the sequential order.

In this project, we used tokenization to extract nouns only. As a result, it is thought that the information loss on propositions or sentence symbols was far more critical to RNN than it was to CNN.

Despite the underperformance of RNN, We can say it is  still a meaningful method. At least, RNN was better than the traditional classifiers.

Also, Given that the base probability (probability of choosing randomly) is 0.25, we can see that all method are meaningful.


## 5.Visualization
We visualized the results using wordcloud and principal component analysis.

### 5.1.WordCloud
We visualized the tokenized documents with wordclouds.
<div>
<img src = "https://user-images.githubusercontent.com/33714067/92874894-6acc9500-f443-11ea-97e0-b496a1647bb8.png">
<img src = "https://user-images.githubusercontent.com/33714067/92874959-78821a80-f443-11ea-9ee3-0a5f51117f73.png">
<img src = "https://user-images.githubusercontent.com/33714067/92875006-86d03680-f443-11ea-9c48-b03def061fe7.png">
</div>

### 5.2.PCA
We visualized raw input data and the last hidden representation using PCA. Figure 6 shows the result. We can see that after passing through a model, the data becomes well clustered.
<div>
<img src = "https://user-images.githubusercontent.com/33714067/92875531-183fa880-f444-11ea-8c4a-198c2d81ff0f.png">
</div>


## 6.SOTA
SOTA is shown in figure 7. SOTA measures performance with AG news or IMDB data. However, due to the different datasets, performance comparisons with our model seems pointless.
<div>
<img src = "https://user-images.githubusercontent.com/33714067/92876064-a87ded80-f444-11ea-9841-857c993bd9a7.png">
</div>


## 7.Main Challenges and Our Solution
Data preprocessing(including embedding) was the most challenging step. It was hard to define appropriate stopwords, due to their subjectivity.

Also, there are many types of Korean tokenizers, and tokenizing methods. To choose the best tokenizer and method, we had to try several times. Hannanum, Kkma, Okt were tried out. Okt yielded the best result.


## 8.Evaluation Our Model/Solution
Considering that the base probability (probability of choosing randomly) is 0.25, we can see that all methods are meaningful.

Also, TextCNN, RNN, and Fasttext are all better than wellknown traditional classifier like SVM, logistic regression, Naive Bayes. Thus we can say our models are not bad.


## 9.References
1. Implementing a CNN for Text Classification in TensorFlow, Denny Britz http://www.wildml.com/2015/12/implementing-a-cnn-fortext-classification-in-tensorflow/
2. Tensorflow-Tutorials, 10-RNN 01-MNIST, golbin https://github.com/golbin/TensorFlowTutorials/blob/master/10%20-%|20RNN/01%20- %20MNIST.py
3. Yoon Kim. Convolutional Neural Networks for Sentence Classification. 2014.
