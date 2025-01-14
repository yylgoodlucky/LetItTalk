# LETITTALK: Multi-pro Talking Face Generation for Real-world Videos

TODO:
-  Demo videos
-  pre-trained model
-  code for testing
-  code for training
-  code for preprocess dataset
-  guideline 
-  arxiv paper release


LetItTalk-Net Will Come Soon!

UPDATE:
- 2025-01-08: Train VQVAE , GAN Inversion pipeline!
- 2025-01-10: Train Motsync , Pretrained model!
- 2025-01-13: Train Stylesync , Avatar model!


Training:

    python train.py --cfg ./config/motsync_params.yaml --mode motsync

    python train.py --cfg ./config/stylesync_params.yaml --mode stylesync