import torch
import torch.nn as nn

class PurchaseClassifier(nn.Module):
    def __init__(self,num_classes=100):
        super(PurchaseClassifier, self).__init__()
        self.features = nn.Sequential(
            nn.Linear(600,1024),
            nn.Tanh(),
            nn.Linear(1024,512),
            nn.Tanh(),
            nn.Linear(512,256),
            nn.Tanh(),
            nn.Linear(256,128),
            nn.Tanh(),
        )
        self.classifier = nn.Linear(128,num_classes)
        
    def forward(self,inp):
        
        outputs=[]
        x=inp
        module_list =list(self.features.modules())[1:]
        for l in module_list:
            
            x = l(x)
            outputs.append(x)
        
        y = x.view(inp.size(0), -1)
        o = self.classifier(y)
        
        return o