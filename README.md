# nlp-ad-in-game
Identify  advertisements in game dialogue  using Natural Language Processing and Deep Learning.

## Task description
Resource advertisements seriously affect the player's experience. In this task, I am asked to identify whether a text is an advertisemnet or not.

### Dataset
This is a supervised learning task, with labeled data in the following format:
```text
label	text
1	最后一千八百本书加三百三钻石30元,可以先货
1	正规渠道200元有5万点卷加5万钻石要的加q859945704先给
0	老子700钻石
1	27127.海豚YB阁,海豚小班助你战力狂飙,求微374583830331
```

## Solution

### Data cleaning
I adopt rule based data cleaning methods:
1. Noting that advertisements usually contain contact methods like QQ/WeChat/Phone number for follow-up communication.
2. Exrtact frequently used words in sentences contains advertisements in training set.

### Model selection
I fine tune the Bert (bert-base-chinese) model to do the identification task, considering its strong semantic understanding ability, and the accuracy in validation sets turns out good.

## Training

Mini-batch stochastic gradient descent is adopted with Adam to be the optimizer. Tensorboard is used for recording training loss and accuracy.
Train the model by runnig:
```shell
python train.py
```

## Testing
Try the following command to test the trained model.
```shell
python main_test.py --model_file adv.model --data_file dataset/clean_validation.csv --out_put output.txt
```

PS: This is a possible solution for the problem, and there are a lot to improve. If you got anything to share, like new ways to clean the data, advice for modification the model, etc. please feel free to leave a issue :D
