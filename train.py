# 导入必要的包和定义超参数
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
import pandas as pd
from torch.utils.tensorboard import SummaryWriter

# 定义超参数
MAX_LEN = 40
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 2e-5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# 创建SummaryWriter对象
writer = SummaryWriter()


def getDataloader(validname,bs,tokenizer):
    df=pd.read_csv(validname, sep='\t')
    text = df['text'].tolist()
    label = df['label'].tolist()
    valid_dataset = SentimentDataset(text, label, tokenizer, MAX_LEN)
    valid_loader = DataLoader(valid_dataset, batch_size=bs, shuffle=False)
    return valid_loader,text




# 定义数据集和数据加载器
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):

        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]


        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

        return {

            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }




# 定义SentimentClassifier模型
class SentimentClassifier(nn.Module):
    def __init__(self):
        super(SentimentClassifier, self).__init__()
        # 采用预训练的Bert模型
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        # 因为数据比较少，采用dropout防止过拟合
        self.dropout = nn.Dropout(0.3)
        # 二分类问题，直接用线性层使得输出维度为 2
        self.linear = nn.Linear(self.bert.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        hidden_layer = outputs[1]
        hidden_layer = self.dropout(hidden_layer)
        logits = self.linear(hidden_layer)
        # 最后这个logits，经过torch.max(logits, 1) 就可以得到所分的类别
        return logits

def train(train_texts,train_labels,test_texts,test_labels,tokenizer):

    train_dataset = SentimentDataset(train_texts, train_labels, tokenizer, MAX_LEN)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    test_dataset = SentimentDataset(test_texts, test_labels, tokenizer, MAX_LEN)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = SentimentClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 定义保存模型的路径和文件名
    PATH = "./adv.model"

    # improve=True
    best_test_acc=-1
    consist=0

    # 训练模型
    for epoch in range(EPOCHS):
        # print(f'epoch:{epoch}')
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += len(labels)

            loss = criterion(outputs, labels)
            running_loss += loss.item()

            loss.backward()
            optimizer.step()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct_predictions / total_predictions

        # 记录loss和accuracy
        writer.add_scalar('Train/Loss', epoch_loss, epoch+1)
        writer.add_scalar('Train/Accuracy', epoch_acc, epoch+1)

        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.4f}")

        # 测试集
        model.eval()
        with torch.no_grad():
            correct_predictions = 0
            total_predictions = 0

            for batch in test_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                _, predicted = torch.max(outputs, 1)

                correct_predictions += (predicted == labels).sum().item()
                total_predictions += len(labels)

            test_acc = correct_predictions / total_predictions

            # 记录loss和accuracy
            # writer.add_scalar('Train/Loss', epoch_loss, epoch + 1)
            writer.add_scalar('Train/Accuracy_test', test_acc, epoch + 1)

            print(f"Test Accuracy: {test_acc:.4f}")


            # 如果模型效果变好保存模型
            if (test_acc>best_test_acc):
                consist=0
                torch.save(model.state_dict(), PATH)
                best_test_acc=test_acc
            # best_test_acc=max(best_test_acc, test_acc)
            consist+=1

            # 如果模型效果连续30个epoch还没有提升的话就退出模型了
            if consist>30:
                break

    # 关闭SummaryWriter对象
    writer.close()

if __name__ == '__main__':
    # 测试集
    df_train = pd.read_csv('dataset/clean_train.csv', sep='\t')
    train_texts = df_train['text'].tolist()
    train_labels = df_train['label'].tolist()

    # 验证集
    df_valid = pd.read_csv('dataset/clean_validation.csv', sep='\t')
    test_texts = df_train['text'].tolist()
    test_labels = df_train['label'].tolist()

    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    print("train start.....")
    train(train_texts,train_labels,test_texts,test_labels,tokenizer)


