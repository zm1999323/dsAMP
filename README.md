## dsAMP and dsAMPGAN-- AMP  predictor and AMP generator
![image](https://github.com/zm1999323/dsAMP/blob/master/figure1.png)
  The overall architecture of dsAMP and dsAMPGAN

## What about dsAMP and dsAMPGAN？

* AMP?
    *  Antimicrobial peptides (AMPs) represent a crucial arm of the innate immune system, offering a compelling alternative to conventional antibiotics
    *  Their unique structure, characterized by hydrophobic and amphiphilic residues, enables effective interaction with microbial membranes, leading to disruption and subsequent cell death. 
* dsAMP, which enhances existing AMP classification models in two ways:
    *  by embedding AMP sequences using a protein pretraining model trained on large datasets
    *  by integrating a performance-enhancing module based on CNNs, attention mechanisms, and Bi-directional Long Short-Term Memory (BiLSTM) to predict AMP sequences
* dsAMPGAN:
    *  The generator consists of a SplitAttention module 
    *  The discriminator consists of 4 CNN blocks and a CirssCrossAttention module

## Feedback
If you have any problems, please feel free to give me feedback, you can use the following contact information to communicate with me!

* Email:zhaomin21a@mails.ucas.ac.cn
