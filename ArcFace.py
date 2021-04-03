from lib import *

class ArcFace(nn.Module):
    def __init__(self, in_features, out_classes, m=0, s=10, easy_margin=False):
        super().__init__()
        self.out_features = out_classes
        self.in_features = in_features

        self.easy_margin = easy_margin
        self.s = s  # scaler for the size of input features
        self.m = m
        self.cos_m = math.cos(m)
        self.sin_m = math.cos(m)

        self.threshold = math.cos(math.pi - self.m)  # Threshold is 180 - m
        self.mm = math.sin(math.pi - self.m) * self.m  # To keep function

        self.weight = nn.Parameter(torch.FloatTensor(out_classes, in_features))  # Weights are (out_dim, in_dim) shaped
        self.initialize_parameters()

    def initialize_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, features, labels):
        # cosine of the angle between embeddings and the weights
        cos_theta = F.linear(F.normalize(features), F.normalize(
            self.weight))  # To normalize the weights, nn.Parameter is used with nn.Functional
        cos_theta = cos_theta.clamp(-1, 1)

        sin_theta = torch.sqrt(1 - torch.pow(cos_theta, 2))
        phi = cos_theta * self.cos_m - sin_theta * self.sin_m  # cos(x+y) = cosx cosy - sinx siny

        # To keep the function monotonous and avoid theta + m >= pi
        if self.easy_margin:
            phi = torch.where(cos_theta > 0, phi, cos_theta)

        else:
            phi = torch.where(cos_theta > self.threshold, phi, cos_theta - self.mm)

        theta = torch.acos(cos_theta)

        onehot = torch.zeros(cos_theta.shape, device=device)
        onehot = onehot.scatter_(1, torch.unsqueeze(labels, 1).long(), 1)  # labels should have same shape with onehot

        out = onehot * phi + (1 - onehot) * cos_theta
        out *= self.s
        return out