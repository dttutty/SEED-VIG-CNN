import torch
import pickle as pk
import numpy as np
from torchvision.models.detection import ssd300_vgg16
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn
from torchvision.models import vgg16
import torch.optim as optim
from torch.utils.data import random_split
import wandb

inputs_file = open('inputs.pickle', 'rb')
inputs = pk.load(inputs_file)
inputs_file.close() 
outputs_file = open('outputs.pickle', 'rb')
outputs = pk.load(outputs_file)
outputs_file.close() 

# 这个是包装
class SeedDataset(Dataset):
    def __init__(self, inputs, outputs):
        self.data = inputs
        
    def __getitem__(self, index) -> tuple:
        return_value = {}
        for key, item in self.data.items():
            return_value[key] = item[index]
        return return_value, outputs[index]
    
    def __len__(self):
        return len(self.data['EEG_Feature_2Hz_psd_movingAve'])

seedDataset = SeedDataset(inputs, outputs)

train_set, val_set = random_split(seedDataset,[885*21, 885*2])
trainLoader = DataLoader(train_set, 4, shuffle=True)
valLoader = DataLoader(val_set, 2, shuffle=True)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

class MyModel(nn.Module):
    
    def __init__(self) -> None:
        super().__init__()
        self.conv1a, self.conv1b, self.conv1c, self.fc1a, self.fc1b, self.fc1c = nn.Conv2d(4,8,3,1,1), nn.Conv2d(8,16,3,1,1), nn.Conv2d(16,4,3,1,1), nn.Linear(17*25*4, 17*5*4), nn.Linear(17*5*4, 4*5*4), nn.Linear(4*5*4, 36*3)
        self.conv2a, self.conv2b, self.conv2c, self.fc2a, self.fc2b = nn.Conv2d(4,8,3,1,1), nn.Conv2d(8,16,3,1,1), nn.Conv2d(16,4,3,1,1), nn.Linear(17*5*4, 4*5*4), nn.Linear(4*5*4, 36*3)
        
        self.conv3a, self.conv3b, self.conv3c, self.fc3a, self.fc3b = nn.Conv2d(4,8,3,1,1), nn.Conv2d(8,16,3,1,1), nn.Conv2d(16,4,3,1,1), nn.Linear(4*25*4, 4*5*4), nn.Linear(4*5*4, 36*3)
        self.conv4a, self.conv4b, self.conv4c, self.fc4a = nn.Conv2d(4,8,3,1,1), nn.Conv2d(8,16,3,1,1), nn.Conv2d(16,4,3,1,1), nn.Linear(4*5*4, 36*3)

        self.fc5 = nn.Linear(36*3*5, 36*3)
        self.fc6 = nn.Linear(36*3, 36)
        self.fc7 = nn.Linear(36, 1)
        
    def forward(self, input):
        
        EEG_Feature_2Hz = np.stack([input['EEG_Feature_2Hz_psd_movingAve'], input['EEG_Feature_2Hz_psd_LDS'], input['EEG_Feature_2Hz_de_movingAve'], input['EEG_Feature_2Hz_de_LDS']],1)
        EEG_Feature_2Hz = torch.from_numpy(EEG_Feature_2Hz)
        EEG_Feature_2Hz = EEG_Feature_2Hz.to(device, dtype=torch.float)
        EEG_Feature_2Hz = self.conv1a(EEG_Feature_2Hz)
        EEG_Feature_2Hz = nn.Sigmoid()(EEG_Feature_2Hz)
        EEG_Feature_2Hz = self.conv1b(EEG_Feature_2Hz)
        EEG_Feature_2Hz = nn.Sigmoid()(EEG_Feature_2Hz)
        EEG_Feature_2Hz = self.conv1c(EEG_Feature_2Hz)
        EEG_Feature_2Hz = nn.Sigmoid()(EEG_Feature_2Hz)
        EEG_Feature_2Hz = nn.Flatten()(EEG_Feature_2Hz)
        EEG_Feature_2Hz = self.fc1a(EEG_Feature_2Hz)
        EEG_Feature_2Hz = nn.Sigmoid()(EEG_Feature_2Hz)
        EEG_Feature_2Hz = self.fc1b(EEG_Feature_2Hz)
        EEG_Feature_2Hz = nn.Sigmoid()(EEG_Feature_2Hz)
        EEG_Feature_2Hz = self.fc1c(EEG_Feature_2Hz)
        EEG_Feature_2Hz = nn.Sigmoid()(EEG_Feature_2Hz)
        
        EEG_Feature_5Bands = np.stack([input['EEG_Feature_5Bands_psd_movingAve'], input['EEG_Feature_5Bands_psd_LDS'], input['EEG_Feature_5Bands_de_movingAve'], input['EEG_Feature_5Bands_de_LDS']],1)
        EEG_Feature_5Bands = torch.from_numpy(EEG_Feature_5Bands)
        EEG_Feature_5Bands = EEG_Feature_5Bands.to(device, dtype=torch.float)
        EEG_Feature_5Bands = self.conv2a(EEG_Feature_5Bands)
        EEG_Feature_5Bands = nn.Sigmoid()(EEG_Feature_5Bands)
        EEG_Feature_5Bands = self.conv2b(EEG_Feature_5Bands)
        EEG_Feature_5Bands = nn.Sigmoid()(EEG_Feature_5Bands)
        EEG_Feature_5Bands = self.conv2c(EEG_Feature_5Bands)
        EEG_Feature_5Bands = nn.Sigmoid()(EEG_Feature_5Bands)
        EEG_Feature_5Bands = nn.Flatten()(EEG_Feature_5Bands)
        EEG_Feature_5Bands = self.fc2a(EEG_Feature_5Bands)
        EEG_Feature_5Bands = nn.Sigmoid()(EEG_Feature_5Bands)
        EEG_Feature_5Bands = self.fc2b(EEG_Feature_5Bands)
        EEG_Feature_5Bands = nn.Sigmoid()(EEG_Feature_5Bands)
        
        Forehead_EEG_Feature_2Hz = np.stack([input['Forehead_EEG_Feature_2Hz_psd_movingAve'], input['Forehead_EEG_Feature_2Hz_psd_LDS'], input['Forehead_EEG_Feature_2Hz_de_movingAve'], input['Forehead_EEG_Feature_2Hz_de_LDS']],1)
        Forehead_EEG_Feature_2Hz = torch.from_numpy(Forehead_EEG_Feature_2Hz)
        Forehead_EEG_Feature_2Hz = Forehead_EEG_Feature_2Hz.to(device, dtype=torch.float)
        Forehead_EEG_Feature_2Hz = self.conv3a(Forehead_EEG_Feature_2Hz)
        Forehead_EEG_Feature_2Hz = nn.Sigmoid()(Forehead_EEG_Feature_2Hz)
        Forehead_EEG_Feature_2Hz = self.conv3b(Forehead_EEG_Feature_2Hz)
        Forehead_EEG_Feature_2Hz = nn.Sigmoid()(Forehead_EEG_Feature_2Hz)
        Forehead_EEG_Feature_2Hz = self.conv3c(Forehead_EEG_Feature_2Hz)
        Forehead_EEG_Feature_2Hz = nn.Sigmoid()(Forehead_EEG_Feature_2Hz)
        Forehead_EEG_Feature_2Hz = nn.Flatten()(Forehead_EEG_Feature_2Hz)
        Forehead_EEG_Feature_2Hz = self.fc3a(Forehead_EEG_Feature_2Hz)
        Forehead_EEG_Feature_2Hz = nn.Sigmoid()(Forehead_EEG_Feature_2Hz)
        Forehead_EEG_Feature_2Hz = self.fc3b(Forehead_EEG_Feature_2Hz)
        Forehead_EEG_Feature_2Hz = nn.Sigmoid()(Forehead_EEG_Feature_2Hz)
        
        Forehead_EEG_Feature_5Bands = np.stack([input['Forehead_EEG_Feature_5Bands_psd_movingAve'], input['Forehead_EEG_Feature_5Bands_psd_LDS'], input['Forehead_EEG_Feature_5Bands_de_movingAve'], input['Forehead_EEG_Feature_5Bands_de_LDS']],1)
        Forehead_EEG_Feature_5Bands = torch.from_numpy(Forehead_EEG_Feature_5Bands)
        Forehead_EEG_Feature_5Bands = Forehead_EEG_Feature_5Bands.to(device, dtype=torch.float)
        Forehead_EEG_Feature_5Bands = self.conv4a(Forehead_EEG_Feature_5Bands)
        Forehead_EEG_Feature_5Bands = nn.Sigmoid()(Forehead_EEG_Feature_5Bands)
        Forehead_EEG_Feature_5Bands = self.conv4b(Forehead_EEG_Feature_5Bands)
        Forehead_EEG_Feature_5Bands = nn.Sigmoid()(Forehead_EEG_Feature_5Bands)
        Forehead_EEG_Feature_5Bands = self.conv4c(Forehead_EEG_Feature_5Bands)
        Forehead_EEG_Feature_5Bands = nn.Sigmoid()(Forehead_EEG_Feature_5Bands)
        Forehead_EEG_Feature_5Bands = nn.Flatten()(Forehead_EEG_Feature_5Bands)
        Forehead_EEG_Feature_5Bands = self.fc4a(Forehead_EEG_Feature_5Bands)
        Forehead_EEG_Feature_5Bands = nn.Sigmoid()(Forehead_EEG_Feature_5Bands)
        
        EOG_Feature = np.stack([input['EOG_Feature_features_table_ica'], input['EOG_Feature_features_table_minus'], input['EOG_Feature_features_table_icav_minh']],1)
        EOG_Feature = torch.from_numpy(EOG_Feature)
        EOG_Feature = EOG_Feature.to(device, dtype=torch.float)
        EOG_Feature = nn.Flatten()(EOG_Feature)
        
        All_Features = torch.stack((EEG_Feature_2Hz, EEG_Feature_5Bands, Forehead_EEG_Feature_2Hz, Forehead_EEG_Feature_5Bands, EOG_Feature),1)
        output = nn.Flatten()(All_Features)
        output = nn.Sigmoid()(output)
        output = self.fc5(output)
        output = nn.Sigmoid()(output)
        output = self.fc6(output)
        output = nn.Sigmoid()(output)
        output = self.fc7(output)
        output = nn.Sigmoid()(output)
        
        return output
    
myModel = MyModel()
myModel.to(device)

run = wandb.init(project="seed_simple_cnn")
config = run.config
config.momentum = 0.9
config.lr = 0.001



criterion = nn.MSELoss()
optimizer = optim.SGD(myModel.parameters(), **config)
for epoch in range(300):
    running_loss = 0.0
    for i, data in enumerate(trainLoader, 0):
        batch_inputs, batch_outputs = data
        batch_outputs = batch_outputs.to(device, dtype=torch.float)
        optimizer.zero_grad()
        
        myModel.train()
        infer_result = myModel(batch_inputs)
        loss = criterion(infer_result, batch_outputs)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if (i%1000 == 999):
            print('训练集上测得的损失函数[第%d个epoch, 第%5d个iteration] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 1000))
            running_loss = 0.0
            
            val_loss = 0.0
            for j, data in enumerate(valLoader, 0):
                batch_inputs, batch_outputs = data
                batch_outputs = batch_outputs.to(device, dtype=torch.float)
                myModel.eval()
                with torch.no_grad():
                    infer_result = myModel(batch_inputs)
                    loss = criterion(infer_result, batch_outputs)
                    val_loss += loss.item()
                    if (j%885 == 884):
                        print('测试集上测得的损失函数: %.3f' %
                            (val_loss / 885))
                        if (i==3999):
                            run.log({"loss": val_loss / 885})
                        val_loss = 0.0

        
        

torch.save(myModel.state_dict(), './model.pth')


print('Finished Training')