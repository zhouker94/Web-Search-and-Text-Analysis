# COMP90042 Web Search and Text Analysis

## Overview

In this project, I build an end-to-end QA system, which takes the plain text of question and context as inputs, and outputs an answer span from the context. I also compare my deep learning based QA system with another IR-based QA system using chunk and NER techniques on developing and test set, the results show that this deep learning based model performs much better than traditional IR approach on the given data set without more manual effort.

## DEEP LEARNING MODEL 

The model architecture is derived from 2 3 4. The figure shows the architecture of the neural network, which is implemented in Python using Tensorflow.

![nn_architecture](res/nn_architecture.jpg)

Every RNN block is similar throughout the model. A block consists of a norm layer, a bi-directional RNN, and a self- attention layer. After both question and context are encoding by an RNN block respectively, they pass through a coat- tention block, which is used to calculate the similarity on each pair of words from question and context. Self-attention mechanism is also applied inside an RNN block for increasing the weight of important words and decrease those’s which are less important. Then the same block are used to decode the coattention results, and finally they pass to a softmax layer to calculate the loss of cross entropy from start probability and end probability respectively.

After 3 hours training on a virtual machine with a four cores CPU without any GPU resource, the training loss is shown in following plots. Validate loss on developing set is also provided in order to apply early stopping strategy and avoid overfitting.

<figure class="half"> 
    <img src="res/training_loss.jpg">
    <img src="res/eval_loss.jpg">
</figure>

