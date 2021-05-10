import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dropout=0.2):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        out = self.conv(x)
        out = self.relu(out)
        out = self.dropout(out)

        return out


class Wav2Letter(nn.Module):

    def __init__(self, num_classes, num_features=40):
        super(Wav2Letter, self).__init__()

        model = nn.Sequential(
            ConvBlock(in_channels=num_features, out_channels=250, kernel_size=48, stride=2, padding=23, dropout=0.2),

            ConvBlock(in_channels=250, out_channels=250, kernel_size=7, stride=1, padding=3, dropout=0.2),
            ConvBlock(in_channels=250, out_channels=250, kernel_size=7, stride=1, padding=3, dropout=0.2),
            ConvBlock(in_channels=250, out_channels=250, kernel_size=7, stride=1, padding=3, dropout=0.2),
            ConvBlock(in_channels=250, out_channels=250, kernel_size=7, stride=1, padding=3, dropout=0.2),
            ConvBlock(in_channels=250, out_channels=250, kernel_size=7, stride=1, padding=3, dropout=0.2),
            ConvBlock(in_channels=250, out_channels=250, kernel_size=7, stride=1, padding=3, dropout=0.2),
            ConvBlock(in_channels=250, out_channels=250, kernel_size=7, stride=1, padding=3, dropout=0.2),

            ConvBlock(in_channels=250, out_channels=2000, kernel_size=32, stride=1, padding=16, dropout=0.2),
            ConvBlock(in_channels=2000, out_channels=2000, kernel_size=1, stride=1, padding=0, dropout=0.2),
            nn.Conv1d(in_channels=2000, out_channels=num_classes, kernel_size=1, stride=1, padding=0)
        )

        self.model = model
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # Input - (batch_size, num_features, input_length)
        out = self.model(x)
        out = self.log_softmax(out)

        return out.transpose(0, 1)