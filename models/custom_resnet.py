import torch.nn as nn
import torch
from torchvision.models import resnet18

class CustomResNet18(nn.Module):
    def __init__(self, num_classes=10, channels=3, freeze_method='from_scratch', layers_to_unfreeze=['layer3', 'layer4']):
        super(CustomResNet18, self).__init__()

        if freeze_method == 'from_scratch':
            self.model = resnet18(pretrained=False)
        else:
            self.model = resnet18(pretrained=True)

        # Adjust for different number of input channels
        if channels != 3:
            self.model.conv1 = nn.Conv2d(channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        if freeze_method == 'pre_trained_unfreeze_top_layer':
            self._freeze_all()
            for param in self.model.fc.parameters():
                param.requires_grad = True

        elif freeze_method == 'pre_trained_unfreeze_partial_last_layers':
            self._freeze_all()
            for layer_name in layers_to_unfreeze:
                for param in getattr(self.model, layer_name).parameters():
                    param.requires_grad = True
            for param in self.model.fc.parameters():
                param.requires_grad = True

        elif freeze_method == 'pre_trained_unfreeze_all_layers':
            # All layers will be unfrozen by default
            pass

        # Modify the final layer
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def _freeze_all(self):
        for param in self.model.parameters():
            param.requires_grad = False


    def forward(self, x, last=False, freeze=False):
        if freeze:
            with torch.no_grad():
                out = F.relu(self.bn1(self.conv1(x)))
                out = self.layer1(out)
                out = self.layer2(out)
                out = self.layer3(out)
                out = self.layer4(out)
                out = F.avg_pool2d(out, 4)
                e = out.view(out.size(0), -1)
        else:
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = F.avg_pool2d(out, 4)
            e = out.view(out.size(0), -1)
        out = self.linear(e)
        if last:
            return out, e
        else:
            return out

    def get_embedding_dim(self):
        return self.embDim


# # Example usage:
# model = CustomResNet18(num_classes=10, channels=3, freeze_method='pre_trained_unfreeze_top_layer', layers_to_unfreeze=['layer3', 'layer4'])
# print(model)



