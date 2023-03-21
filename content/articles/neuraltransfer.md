title: Neural Style Transfer
author: Heather Ann Dye
date: 3/19/23
category: data science, art
tags: Pytorch

## Neural Style Transfer

What is neural style transfer and how does it involve artificial intelligence?  
Neural style transfer is a technique that takes a content image and style image and combines the two images so that the final output is in the style of the *style image*.  

So for example, I can take single input image:
BEE WITH FLOWER IMAGE 
and combine it with a style image: 
YELLOWFLOWERWATER PIXEL 
and produce a third image with the imputed style:
STYLE IMAGE. 

Alternatively, I can take the same photo and combine it with the 
style of Picasso and produce a stylized image:
PICASSO STYLE IMAGE. 

This technique utlizes a convolutional neural network (CNN) such as VGG-29 which has the capability to recognize XX image classes.  We will be utilitzing the ability of the CNN to recognize patterns in order to perform style transfer. 

More importantly, from an artist's perspective, this technique utilizes AI in a manner that sidesteps the issue of copy right theft. 

### How is Neural Style Transfer Performed

We successifvely apply the layers of the CNN to the content and style images; harvesting the  to the max pooling - essentially, we can view these layers as matrices as the same dimension as the 
image file. 

We'll also consider a generated file that will initially be set equal to the content file. 

Let $C^k _{ij} $ denote the output of the content, 
$G^k _{ij}$ denote the output of the generated image, and $A^k_{ij}$ denotes the output of the style file.
In the selected model, there are five possible layers to harvest occuring before max pooling.  

For the content layers, we let $S_c$ denote the layers selected to optimize the content. For the style layers, we let $S_t$ denote the layers selected from the selected from the style image. 

We denote the content loss function as $\mathit{L}_c $ and 
$$ \mathit{L}_c (S_c)= \sum_{k \in S_c} (C^k _{ij} - G^k _{ij})^2 .$$
Next, the style loss function is modeled as 
$$\mathit{L}_s (S_t) = \sum_{k \in S_t}
[(A^k _{ij})^T (A^k _{ij})  -  (G^k _{ij})^T (G^k _{ij})]^2 .$$  
We sum these two functions to obtain the total loss:
$$\mathit{L} = \mathit{L}_s + \mathit{L}_c$$. 
This is the function that we'll optimize. 
Notice the choice of $A^T A$ in the style loss function. This ensures that the loss function has some flexibility as explained below. 

##### Why? 
Let's consider the simplest version of our problem. 
We have a *picture* consisting of a single digit. 
Our content image is represented by $(a)$ and our style image is represented by $(b)$.  Remember that our generated image is initially  the content image, so we'll represent the content image as $(a-h)$.

Then our loss function is 
$L = (a - (a-h))^2 + (b -(b-h))^2$. Rewriting, we find that 
$L = (a-b)^2 + 2h^2 + 2h(b-a)$, meaning that for this simplistic case the minimum is at the halfway point. 


Then, we can write the loss function in two parts. 
We consider the content loss:

Using this loss function, we optimize $G^k_{ij}$ moving the total loss towards zero.   

To set up the model, we need to select the layers to be part of the loss function. 