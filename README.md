# BotMAR
Implementation for the paper: **BotMAR: Aligned Mask-View Pretraining and Robust Dual-Brandch Tuning for Social Bot Detection**

### Dataset Preparation

**MGTAB**: The original dataset provides processed files in .pt format. Simply place them in the corresponding folder.

**Twibot-20**: The original dataset does not provide processed files. You need to generate them using the code in [TwiBot-22/src/BotRGCN](https://github.com/LuoUndergradXJTU/TwiBot-22/tree/master/src/BotRGCN)
 and then place the generated files in the corresponding folder.

### Train and Test
You can run the following commands to train and test the model:

```bash
# For Twibot-20
python main_transductive.py --dataset Twibot-20

# For MGTAB
python main_transductive.py --dataset MGTAB-FF
```
