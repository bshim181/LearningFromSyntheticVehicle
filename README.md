#Learning From Synthetic Vehicles 
https://openaccess.thecvf.com/content/WACV2022W/RWS/papers/Kim_Learning_From_Synthetic_Vehicles_WACVW_2022_paper.pdf

Model

UNet Model (Encoder connected with a Decoder) 

<img width="346" alt="Screen Shot 2022-04-05 at 10 28 42 PM" src="https://user-images.githubusercontent.com/53489568/161764649-e95fcdd4-f81c-44f6-8127-5c01644c3084.png">

  a. Implemented Attention Map to see how the model makes its decisions.

____________________________________________________________________

SAVED (Simulated Articulated Vehicle Dataset)

1. Utilized an unreal engine generated images with geometric signals to pretrain our model 
2. Performed a classification task on DIVA dataset to determine whether a door of the car is open or not. 
3. Tested to see if the incorporation of geometric signals (Surface Normal, Depth) into the loss function can improve the performance. 
4. Hypothesized based on the attention map projection that a multilabel classification task (Predicting opened or closed state of Front, Back, Left, Right)
   doors independently) will improve the performance on a simple binary classification task(Predicting whether any door is opened or not). 

____________________________________________________________________

Results

<img width="345" alt="Screen Shot 2022-04-05 at 10 51 45 PM" src="https://user-images.githubusercontent.com/53489568/161769278-30e74183-1763-42ff-aec3-246df8902379.png">

1. Demonstrates increase in 5% accuracy when pretrained with SAVED dataset compared to our control(ImageNet pretrained)
2. Incorporation of geometric signals lead to further 5% increase in accuracy compared to the model pretrained without geometric information. 


____________________________________________________________________

Improvements.

1. Need for an improved control experiment: Rather than employing an ImageNet pretrained model, V-KITTI pretrained model will serve as a better control. Showing an improved performance compared to V-KITTI will strongly support the benefit of this dataset. 

2. Normalization of background noise for SAVED dataset: unrealistic demonstration of the background noisy for simulated images can perturb domain adaptation and result in hindered performance on the real data. 

