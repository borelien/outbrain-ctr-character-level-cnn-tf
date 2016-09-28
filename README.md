# Text regression for Click-Through Rate prediction using ConvNet

Introduction
===============================
These last years, deep learning has been succesfully applied to Natural Language Processing. 
By their ability to model sequential phenoma, Recurrent Neural Networks-based models have obtained state of the art results on difficult tasks including speech recognition [5], image captioning [6] and machine translation [7].

In less structured tasks like text classification, character-level CNN architectures have also shown promising results.

*Zhang et al.* [1] achieved state-of-the arts results on several classification datasets like Yelp Review Polarity, Yahoo! Answers or Amazon Review. More recently, *Alexis Conneau et al.* [2] improved these results with a deeper char-level CNN using shortcut connections [3] and temporal Batch Normalization [4].

In this post we will present how this kind of architecture can be applied to click-through rate prediction using TenforFlow and using only a few training samples. In particular, we are interested in predicting the click-through rate of Outbrain article recommendations, given only their title (in french language).

Data
===============================
We have collected *46,458* pairs of article title / CTR from production data.

The dataset was then reduced to *2,886* pairs by removing all recommendations displayed less than 100 times: *2,634* for training and *254* for validation. 

Each title correspond to an Outbrain's recommendation, as seen at the end of articles on various medias.

<p align="center">
    <img src="http://cdn.crunchify.com/wp-content/uploads/2012/01/Outbrain.widget.png" width="350">
</p>

Model
===============================
Char-level networks process text as a succession of characters, without any knowledge of words, semantic or syntactic structures of any language.

Each character is quantized using a one-hot-vector representation of dimension *93*, the size of our alphabet:
```
abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}\nêéèàôîïëçù€
```

Titles are then represented by *93* * *sequence_length* embeddings, where *sequence_length* represents the title length.

````python
import tensorflow as tf

class Network(object):
    def __init__(self, alphabet_size):
        self.inputs = tf.placeholder(tf.float32, [None, alphabet_size, None, 1], name="inputs")
        self.labels = tf.placeholder(tf.float32, [None], name="labels")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
````

Our model is a *5*-layers Convolution Neural Networks with: *2* max-pooling layers of kernel size *5* and *3* respectively, *2* dropout layers of probability *0.4*, and ends with a sigmoid.
It was then trained to minimize the squared error loss between groundtruth and predicted CTRs.

<p align="center">

    <img src="https://github.com/borelien/outbrain-ctr-character-level-cnn-tf/blob/master/images/graph.png" width="250">
</p>

Unlike *Zhang et al.* [1] where the last layers are fully connected layers with fixed entry size, we opted for a fully convolutional approach. Our model can thus process batches of different dimensions between *min_size* and *max_size* (here set to *96* and *156* repectively) characters by padding each title with *(0,0,...,0)* vectors to reach the maximum title length within the batch. When the title length was greater than *max_size*, we randomly choose a subset of it.

By using this technique, we were able to train our model more efficiently and reached a better minimum than with fixed entry size models.

````python
def deploy(self):
    conv0 = self.convolution(input=self.inputs, nOut=128, kH=self.inputs.get_shape().dims[1].value, kW=5, strides=[1, 1, 1, 1])
    pool0 = self.pooling(input=conv0, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], pooling='max')

    conv1 = self.convolution(input=pool0, nOut=128, kH=1, kW=3, strides=[1, 1, 1, 1])
    pool1 = self.pooling(input=conv1, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], pooling='max')
    pool1 = tf.nn.dropout(pool1, self.dropout_keep_prob, name='Dropout_{0}'.format(self.layer_key))

    previous_shape = [dim.value or 22 for dim in pool1.get_shape().dims]
    fc2 = self.convolution(input=pool1, nOut=256, kH=previous_shape[1], kW=previous_shape[2], strides=[1, 1, 1, 1])
    fc2 = tf.nn.dropout(fc2, self.dropout_keep_prob, name='Dropout_{0}'.format(self.layer_key))

    previous_shape = [dim.value or 1 for dim in fc2.get_shape().dims]
    fc3 = self.convolution(input=fc2, nOut=256, kH=previous_shape[1], kW=previous_shape[2], strides=[1, 1, 1, 1])
    
    previous_shape = [dim.value or 1 for dim in fc3.get_shape().dims]
    fc4 = self.convolution(input=fc3, nOut=1, kH=previous_shape[1], kW=previous_shape[2], strides=[1, 1, 1, 1], non_linearity=None)
    
    fc4_hist_summary = tf.histogram_summary("fc4", fc4)
    self.scores = tf.squeeze(tf.reduce_mean(tf.sigmoid(fc4), 2), name='scores')
````

We didn't use any weight regularization as it has shown to degrade results. On the other hand, we added 2 dropout layers as shown in the graph figure to avoid too severe overfitting.

Finally, ADAM optimizer was used with a base learning rate equal to *1e-3* and *beta1* was set to *0.5*.


Results
===============================
- squared error loss
<p align="center">
    <img src="https://github.com/borelien/outbrain-ctr-character-level-cnn-tf/blob/master/images/loss.png" width="550">
</p>
*light blue = val, blue = train*.

Because there are many titles with very low CTRs values, it is quite easy to obtained a low squared error loss.

In order to evaluate how our model understood the potential success of a title, we monitored two additional metrics:

- top-k retrieval value which measures the intersection of well ranked top-k CTR titles for k varying from *1* to *259*

````python
from scipy.integrate import quad

def topk(sorted_like_preds, sorted_like_groundtrouth, step, res_dir):
    length = len(sorted_like_preds)
    xs = range(length)
    ys = []
    print(sorted_like_preds)
    for x in xs:
        best_x = sorted_like_groundtrouth[-(x + 1):]
        best_x_preds = sorted_like_preds[-(x + 1):]
        dict_i = {}
        dict_j = {}
        for i, value_preds in enumerate(best_x_preds):
            for j, value_gt in enumerate(best_x):
                if value_preds == value_gt and not (i in dict_i or j in dict_j):
                    dict_i[i] = True
                    dict_j[j] = True

        nb_good = len(dict_i.keys())
        acc_x = float(nb_good) / (x + 1)
        ys.append(acc_x)
    res = quad(lambda x:ys[int(x)], 0, len(ys) - 1, limit=10)[0] / length
````

<p align="center">
    <img src="https://github.com/borelien/outbrain-ctr-character-level-cnn-tf/blob/master/images/topk.jpg" width="500">
</p>

Where a random guess will approximatively give an identity line with score *0.5*. (It will not be exactly an identity line because, according to our scoring function, pairs of same CTR eases global ranking)

- rank2 value which measures the ratio of well-ranked pair-wise comparaisons between titles which have at least a CTR ratio of *200*.

````python
import sklearn.metrics as metrics

def rank2(y_preds, y_val, step, res_dir):
    ratio = 2.
    val_size = len(y_val)
    scores = []
    labels = []
    for i in range(val_size-1):
        for j in range(i + 1, val_size):
            if max(y_val[i], y_val[j]) / max(1e-5, min(y_val[i], y_val[j])) > ratio:
                labels.append((y_val[i] - y_val[j]) * (y_preds[i] - y_preds[j]) > 0)
                score = np.abs(y_preds[i] - y_preds[j])
                scores.append(score)
    mAP = metrics.average_precision_score(labels, scores)
````

<p align="center">
    <img src="https://github.com/borelien/outbrain-ctr-character-level-cnn-tf/blob/master/images/rank2.jpg" width="500">
</p>

Where a random guess will give a constant function equal to *0.5*.

Examples
===============================

In this section we present the *10* most and the *5* less successful articles according to our model as well as their groundtruth rank.

>Highest predicted CTRs:

1) les anges 8 : andréane en dit plus sur son couple avec aurélie *(groundtruth: 1st)*

2) secret story, les 8 plus grosses prises de poids : jessica, nadège, aurélie... *(groundtruth: 2nd)*

3) nabilla et thomas vergara : révélations sur leur vie sexuelle *(groundtruth: 13th)*

4) la folle virée d'elodie frégé et joeystarr... kim kardashian, poupée pour sa fille... *(groundtruth: 72th)*

5) pamela anderson : entièrement nue, à 46 ans, pour une série photo érotique *(groundtruth: 29th)*

6) mort d'isabelle (secret story 2) : une femme généreuse qui avait peur de mourir *(groundtruth: 3th)*

7) patrick poivre d'arvor : claire chazal, la mort de ses trois filles et son fils françois *(groundtruth: 4th)*

8) 10 alcooliques qui ont marqué le monde par leur intelligence *(groundtruth: 48th)*

9) laurence chirac : sa vie hors de l'élysée et ses derniers jours dans l'ombre *(groundtruth: 35th)*

10) sylvie vartan présente sa fille, darina : "elle m'a apporté un coup de jeune" *(groundtruth: 13th)*
<br/>
<br/>

>Lowest predicted CTRs:

250) mauvaise haleine ? ces 7 astuces simples vont y mettre un terme *(groundtruth: 155th)*

251) entre sculpture et photographie, huit artistes modernes au musée rodin *(groundtruth: 230th)*

252) les recettes de tartes aux artichauts *(groundtruth: 223th)*

253) pintadeau de la drôme sur canapé par alain ducasse *(groundtruth: 242th)*

254) recette de cuisses de grenouilles par alain ducasse: *(groundtruth: 208th)*

References
===============================
[1] Xiang Zhang, Junbo Zhao and Yann Le Cun [Character-level convolutional networks for text classification](http://arxiv.org/pdf/1509.01626v3.pdf), 2015.

[2] Alexis Conneau, Holger Schwenk and Yann Le Cun [Very Deep Convolutional Networks for Natural Language Processing](https://arxiv.org/pdf/1606.01781v1.pdf), 2016.

[3] Kaiming He, Xiangyu Zhang, Shaoqing Ren and Jian Sun [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385v1), 2015.

[4] Sergey Ioffe and Christian Szegedy [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/pdf/1502.03167v3), 2015.

[5] Alex Graves, Abdel-rahman Mohamed and Geoffrey Hinton [Speech Recognition with Deep Recurrent Neural Networks](https://arxiv.org/pdf/1303.5778v1.pdf), 2013.

[6] Andrej Karpathy and Li Fei-Fei [Deep Visual-Semantic Alignments for Generating Image Descriptions](https://arxiv.org/pdf/1409.0473v7), 2014.

[7] Dzmitry Bahdanau, Kyunghyun Cho and Yoshua Bengio [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/pdf/1409.0473v7), 2014.
