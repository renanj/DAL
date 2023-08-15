import torch
import torch.nn as nn
import torchvision.models as models

class CustomVGG(nn.Module):
    def __init__(self, num_classes=10, channels=3, architecture='vgg11', freeze_method=None):
        super(CustomVGG, self).__init__()

        self.embDim = 512

        if architecture == 'vgg11':
            self.base_model = models.vgg11(pretrained=True)
        elif architecture == 'vgg16':
            self.base_model = models.vgg16(pretrained=True)
        else:
            raise ValueError("Unknown architecture")

        # Adjust for different number of input channels
        if channels != 3:
            self.base_model.features[0] = nn.Conv2d(channels, 64, kernel_size=3, stride=1, padding=1)

        self.adjust_parameters(freeze_method)

        # Modify the final layer
        self.base_model.classifier[-1] = nn.Linear(self.base_model.classifier[-1].in_features, num_classes)
        
        # Add a dummy attribute for compatibility with batch_bald.py
        self.linear = self.base_model.classifier

    def adjust_parameters(self, freeze_method):
        if freeze_method == 'from_scratch':
            self.base_model = self.base_model.apply(self._reset_weights)
        elif freeze_method == 'pre_trained_unfreeze_top_layer':
            self._freeze_all()
            for param in self.base_model.classifier[-1].parameters():
                param.requires_grad = True
        elif freeze_method == 'pre_trained_unfreeze_partial_last_layers':
            self._freeze_all()
            for param in self.base_model.classifier[-3:].parameters():
                param.requires_grad = True
        elif freeze_method == 'pre_trained_unfreeze_all_layers':
            pass
        else:
            raise ValueError("Unknown freeze_method")

    def _reset_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            m.reset_parameters()

    def _freeze_all(self):
        for param in self.base_model.parameters():
            param.requires_grad = False

    def forward(self, x, last=False, freeze=False):
        if freeze:
            with torch.no_grad():
                out = self.base_model.features(x)
                e = out.view(out.size(0), -1)
        else:
            out = self.base_model.features(x)
            e = out.view(out.size(0), -1)
        out = self.base_model.classifier(e)
        if last:
            return out, e
        else:
            return out

    def get_embedding_dim(self):
        return self.embDim
