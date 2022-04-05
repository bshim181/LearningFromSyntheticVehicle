#Learning From Synthetic Vehicles 

Model

Attention Models 

UNet Model (Encoder connected with a Decoder) 

<img width="346" alt="Screen Shot 2022-04-05 at 10 28 42 PM" src="https://user-images.githubusercontent.com/53489568/161764649-e95fcdd4-f81c-44f6-8127-5c01644c3084.png">




____________________________________________________________________

Dataloaders 

Simulation  Data 

Binary Classification 
  During initialization, after reading in meta['angles'] 
  set trunkState = 0 then, uncomment  
  if trunkOpen[0] > 5 or trunkOpen[1] > 5 or trunkOpen[2] > 5 or trunkOpen[3] > 5 :
    trunkState =1 

MultiLabel Classification
  trunkState = []
  FL, FR, RL, RR, Trunk = 0, 0, 0, 0, 0
  if trunkOpen[0] > 5:
    FL = 1
  if trunkOpen[1] > 5:
    FR =1
  if trunkOpen[2] > 5:
    RL =1
  if trunkOpen[3] > 5:
    RR =1
  if trunkOpen[5]> 5:
    Trunk = 1
  trunkState.append((FL, FR, RL, RR, Trunk))
  
______________________________________________________________________

Real Data (Dataloader) 

Binary Classification
  Initialize with DIVA_DoorDataset 

MultiLabel Classification
  Intialize with DIVA_DoorDataset_MultiLabel
    
_______________________________________________________________________

Models

UNet_N2 
  Models.py 
  Model Name: ResNet101_depth_with_multi

UNet_N1 
  Models_N1.py
  Model Name: ResNet101_depth_with_multi
  
ResNet + Attention
  ResNet_attention_models.py
  
UNet_N2 + attention
  UNet_attention_models.py
  
ResNet 
  UNet_N1 -> serves as a baseline without normal loss 
  
_________________________________________________________________________

Sim2Real 

  Binary Classification Task (Sim2Real)
    File: Sim2Real_Binary_Classification_Model.py 
    a. Also, Commment out or Uncomment calculation of normal loss based on needs (Line Number 177)
      normal_loss = criterion_l1(normal_pred, normal)
      loss = 0.5*loss_cls + 0.5*normal_loss
    b. Line number 146, 114 (Normal Visualization) 

  MultiLabel Classification Task(Sim2Real)
    File: Sim2Real_MultiLabel_Classification.py
      a. normal_loss = criterion_l1(normal_pred, normal) (Line Number 208)
      loss = 0.5*loss_cls + 0.5*normal_loss (Comment out or Uncomment depending on whether you want to incorporate normal loss or not) 
      b. For Normal Visualization: if batch_count % 1000 == 0 or batch_count == 0: 
                                      visualize_and_log_normal('Normal_Pred_Sim', model, inputs[0], epoch, writer)(Line Number 228) 
                                      a. for both train and validation 

  
  Binary Classification + Attention Model
    File: Sim2Real_Attention_Model.py
      a. Initialized for ResNet + attention 
      b. can be altered for UNet by altering the outputs received from the model 
      output,_,_,_,_ = model(images) 
    c. Must change this for visualize attention functions as well. (Line 41, 56)
    
    
  MultiLabel Classification + Attention Model
    File: Sim2Real_Attn_MultiLabel_Classification.py
    a.Change Visualize Attention Functions according to the outputs your model returns.
    b. Initialized for UNet with attention that returns 4 attn layers at different periods of model.
    c. change output,__,__,__,__ = model(inputs) according to the model's output. 
    
   
    
__________________________________________________________________________
    
Real2Real 

  MultiLabel Classification Task(Real2Real) 
    Real2Real_MultiLabel_Classification.py
      a. when running a pre-trained model with normal, uncomment 
      visualize_and_log_normal('normal_pred_real', model, images[0], epoch, writer)
      b. load pretrained MultiLabel Classification Models. 
      
  MultiLabel Classification Task with Attention (Real2Real)
    a. Executed from Real2Real_MultiLabel_Classification.py
    b. Change normal, pred = model(images) to pred,__,__,__ = model(images) 
    c. Also uncomment attn visualization function (Line Number 179)
        
   
  Binary Classification Task(Real2Real) 
    Binary_Classification_Model.py
      a. Load Binary Classification Model and Run 
      
  Binary Classification Task with Attention(Real2Real)  
    a. Executed from Binary_Classification_Model.py
    b. change depth_pred, outputs = model(inputs) to pred,_,_,_ = model(inputs) 

  
__________________________________________________________________________

Things to Note

Real2Real train without simulation pretrain causes loss explosion during multi classification task. 

I also think that when trained with normal loss during the simulation pre-training phase, this information should be utilized in some way during the real2real phase (either through attention map projection, or integrated into the model similar to how UNet_N2 passed the input through the decoder layer which was initially used to predict normal values). 

Challenges from the MultiLabel Classification Task is the Real2Real training phase. Even with simulation pre-trained model, it shows a quite a range of precision, recall, f1,  and loss flunctuations. Also, it does not present an increasing trend of precision, f1, and recall after real data training. 


