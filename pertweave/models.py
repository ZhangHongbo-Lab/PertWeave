# model_architectures.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# --- GLU Layer Implementations ---

class GLU(nn.Module):
    """Gated Linear Unit."""
    def __init__(self, input_dim, output_dim, activation='relu', reduction=1):
        super(GLU, self).__init__()
        self.fc = nn.Linear(input_dim, input_dim // reduction)
        self.fc_gate = nn.Linear(input_dim, input_dim // reduction)
        self.fc_out = nn.Linear(input_dim // reduction, output_dim)
        self.activation = activation

    def forward(self, x):
        x1 = self.fc(x)
        # Using eval here to exactly match original code's flexibility.
        # A safer implementation would use a dictionary lookup for activation functions.
        gate = eval(f'F.{self.activation}')(self.fc_gate(x))
        return self.fc_out(x1 * gate)

class AM_GLU(nn.Module):
    """Efficient GLU with multiple parallel branches of decreasing dimensions."""
    def __init__(self, input_dim, output_dim, activation='relu', reduction=1, efficient_num=2):
        super(AM_GLU, self).__init__()
        self.fc = nn.ModuleList([])
        self.fc_gate = nn.ModuleList([])
        output_count = 0
        inner_dim = input_dim
        for i in range(efficient_num):
            if i != 0:
                inner_dim = inner_dim // 2
            self.fc.append(nn.Linear(input_dim, inner_dim))
            self.fc_gate.append(nn.Linear(input_dim, inner_dim))
            output_count += inner_dim
        self.fc_out = nn.Linear(output_count, output_dim)
        self.activation = activation

    def forward(self, x):
        outputs = []
        for i in range(len(self.fc)):
            tmp_x = self.fc[i](x)
            gate = eval(f'F.{self.activation}')(self.fc_gate[i](x))
            outputs.append(tmp_x * gate)
        x = torch.cat(outputs, dim=1)
        return self.fc_out(x)

# --- Main Model Architecture ---

class PertWeave(nn.Module):
    def __init__(self, input_dim_scfound, input_dim_gene, input_dim_raw, hidden_dim, output_dim, group_output_dim, feat_dim, 
                 layer_num=2, layer_list=None, dropout=0.5, dropout_num=8, linear=False, activation='relu', 
                 reduction_factor=2, GLU_reduction=1, data_mask="111", efficient_num=2):
        super(PertWeave, self).__init__()
        self.data_mask = data_mask
        self.activation = activation
        self.linear = linear

        # Input projections
        self.preprocess_scfound = nn.Linear(input_dim_scfound, hidden_dim)
        self.preprocess_gene = nn.Linear(input_dim_gene, hidden_dim)
        self.preprocess_raw = nn.Linear(input_dim_raw, hidden_dim)
        
        current_hidden_dim = sum([int(mask_bit) for mask_bit in data_mask]) * hidden_dim
        print(f"Initial concatenated hidden dimension: {current_hidden_dim}")

        # Core layers construction
        self.layers = nn.ModuleList()
        for i in range(layer_num):
            layer_count = layer_list[i] if layer_list and i < len(layer_list) else 1
            for j in range(layer_count):
                output_layer_dim = current_hidden_dim // reduction_factor if j == layer_count - 1 else current_hidden_dim
                
                if i == 0:
                    self.layers.append(GLU(current_hidden_dim, output_layer_dim, activation, GLU_reduction))
                else:
                    self.layers.append(AM_GLU(current_hidden_dim, output_layer_dim, activation, GLU_reduction, efficient_num))
                
                self.layers.append(nn.LayerNorm(output_layer_dim))
                if j == layer_count - 1:
                    current_hidden_dim = output_layer_dim

            if i != layer_num - 1:
                if dropout_num == 1:
                    self.layers.append(nn.Dropout(dropout))
                else:
                    self.layers.append(nn.ModuleList([nn.Dropout(dropout) for _ in range(dropout_num)]))

        self.inner_proj = nn.Linear(current_hidden_dim, current_hidden_dim)

        # Output heads
        self.output_fc = nn.Sequential(
            nn.Linear(current_hidden_dim, feat_dim),
            nn.Linear(feat_dim, output_dim)
        )
        self.output_group_fc = nn.Sequential(
            nn.Linear(current_hidden_dim, feat_dim),
            nn.Linear(feat_dim, group_output_dim)
        )

        # Fixed modality scaling coefficients; all are set to 1.
        # self.alpha = nn.Parameter(torch.ones(3))
        self.alpha = torch.ones(3)
        
        self.init_weights()

    def init_weights(self, mode='kaiming', type_='normal', fan='fan_in', act=None):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if mode == 'kaiming':
                    nn.init.kaiming_normal_(module.weight, mode=fan, nonlinearity=act if act else 'leaky_relu')
                else: # Default fallback to xavier normal
                    nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, scfound, scgene, raw, return_embedding=False, data_mask=None):
        if data_mask is None:
            data_mask = self.data_mask
        
        inputs = []
        if data_mask[0] == '1':
            inputs.append(self.preprocess_scfound(scfound) * self.alpha[0])
        if data_mask[1] == '1':
            inputs.append(self.preprocess_gene(scgene) * self.alpha[1])
        if data_mask[2] == '1':
            inputs.append(self.preprocess_raw(raw) * self.alpha[2])
        
        x = torch.cat(inputs, dim=1)

        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.ModuleList): # MC Dropout simulation
                x = torch.stack([dropout_layer(x) for dropout_layer in layer], dim=0).mean(dim=0)
            else:
                x = layer(x)

            # if not self.linear and i != len(self.layers)-1 and not isinstance(layer, (nn.LayerNorm, nn.ModuleList)):
            if not self.linear and i != len(self.layers)-1:
                x = eval(f'F.{self.activation}')(x)
        
        x = self.inner_proj(x)
        feat_raw = x
        
        output_class = self.output_fc(feat_raw)
        output_group = self.output_group_fc(feat_raw)

        feat_cls = self.output_fc[0](feat_raw)
        feat_group = self.output_group_fc[0](feat_raw)
            
        return output_class, output_group, feat_raw, feat_cls, feat_group

# --- Custom Loss Functions ---

class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, device):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.device = device
        self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).to(self.device))

    def forward(self, x, labels):
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(x, self.centers.t(), beta=1, alpha=-2)
        
        classes = torch.arange(self.num_classes).long().to(self.device)
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12).sum() / (batch_size * self.feat_dim) # Avoid division by zero issues
        return loss

def custom_cross_entropy(logits1, logits2):
    target_distribution = F.softmax(logits2, dim=1)
    log_probabilities = F.log_softmax(logits1, dim=1)
    cross_entropy_loss = -torch.sum(target_distribution * log_probabilities, dim=1).mean()
    return cross_entropy_loss

class MCCLoss(nn.Module):
    """MCC loss implementation using cross-entropy formulation."""
    def __init__(self, temperature=1.0):
        super(MCCLoss, self).__init__()
        self.temperature = temperature # Temperature not used in CE formulation, but kept for consistency with original naming

    def forward(self, logits1, logits2):
        loss = (custom_cross_entropy(logits1, logits2) + custom_cross_entropy(logits2, logits1)) / 2
        return loss


class UnknownPertWeave(PertWeave):
    """Use two dropout views and FP32 output projections for Unknown training."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.first_stochastic_layer = next(
            (
                index
                for index, layer in enumerate(self.layers)
                if isinstance(layer, (nn.Dropout, nn.ModuleList))
            ),
            len(self.layers),
        )

    def _prepare_inputs(self, scfound, scgene, raw, data_mask):
        inputs = []
        if data_mask[0] == '1':
            inputs.append(self.preprocess_scfound(scfound) * self.alpha[0])
        if data_mask[1] == '1':
            inputs.append(self.preprocess_gene(scgene) * self.alpha[1])
        if data_mask[2] == '1':
            inputs.append(self.preprocess_raw(raw) * self.alpha[2])
        
        return torch.cat(inputs, dim=1)

    def _run_layers(self, x, start=0, end=None):
        end = len(self.layers) if end is None else end
        for i in range(start, end):
            layer = self.layers[i]
            if isinstance(layer, nn.ModuleList): # MC Dropout simulation
                x = torch.stack([dropout_layer(x) for dropout_layer in layer], dim=0).mean(dim=0)
            else:
                x = layer(x)

            # if not self.linear and i != len(self.layers)-1 and not isinstance(layer, (nn.LayerNorm, nn.ModuleList)):
            if not self.linear and i != len(self.layers)-1:
                x = eval(f'F.{self.activation}')(x)
        return x

    def _build_outputs(self, x):
        # Keep the unnormalized representation and logits in FP32. Casting
        # after these projections would be too late if FP16/BF16 had already
        # produced Inf, which then turns Center/MCC losses into NaN.
        with torch.cuda.amp.autocast(enabled=False):
            x = x.float()
            feat_raw = self.inner_proj(x)
            feat_cls = self.output_fc[0](feat_raw)
            feat_group = self.output_group_fc[0](feat_raw)
            output_class = self.output_fc[1](feat_cls)
            output_group = self.output_group_fc[1](feat_group)
        return output_class, output_group, feat_raw, feat_cls, feat_group

    def _forward_from_layer(self, x, start):
        return self._build_outputs(self._run_layers(x, start=start))

    def forward(self, scfound, scgene, raw, return_embedding=False, data_mask=None):
        if data_mask is None:
            data_mask = self.data_mask
        x = self._prepare_inputs(scfound, scgene, raw, data_mask)
        return self._forward_from_layer(x, start=0)

    def forward_two_views(self, scfound, scgene, raw, data_mask=None):
        """Builds two dropout views while sharing their deterministic prefix."""
        if data_mask is None:
            data_mask = self.data_mask

        x = self._prepare_inputs(scfound, scgene, raw, data_mask)
        shared = self._run_layers(x, end=self.first_stochastic_layer)

        if self.first_stochastic_layer == len(self.layers):
            outputs = self._build_outputs(shared)
            return outputs, outputs

        # Run complete stochastic suffixes sequentially so dropout consumes RNG
        # in the same view-1-then-view-2 order as two ordinary forward calls.
        outputs1 = self._forward_from_layer(shared, self.first_stochastic_layer)
        outputs2 = self._forward_from_layer(shared, self.first_stochastic_layer)
        return outputs1, outputs2

