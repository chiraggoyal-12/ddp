"""
cnn/model_a.py - "Model A", the small CNN from Phase 1, rewritten in PyTorch.

Layer by layer it is the same network as the original Keras model:

    input 1 x 64 x 64 (grayscale)
    conv 3x3, 16 filters, "same" padding, ReLU -> max-pool 2x2  -> 16 x 32 x 32
    conv 3x3, 32 filters, "same" padding, ReLU -> max-pool 2x2  -> 32 x 16 x 16
    conv 3x3, 64 filters, "same" padding, ReLU -> max-pool 2x2  -> 64 x  8 x  8
    flatten                                                     -> 4096
    dense 64, ReLU -> dropout 0.5 -> dense 11                   -> 11 class scores

Parameter count (weights + biases):
    conv1   1*16*9 + 16  =     160
    conv2  16*32*9 + 32  =   4,640
    conv3  32*64*9 + 64  =  18,496
    dense1 4096*64 + 64  = 262,208
    output   64*11 + 11  =     715
    total                = 286,219   (same as Keras)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ModelA(nn.Module):
    def __init__(self, num_classes=11):
        super().__init__()
        # padding=1 on a 3x3 kernel keeps the image size unchanged = Keras padding="same"
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2)  # max-pooling has no weights, so one is reused
        self.dense1 = nn.Linear(64 * 8 * 8, 64)
        self.dropout = nn.Dropout(0.5)
        self.output = nn.Linear(64, num_classes)
        self.use_keras_initialization()

    def use_keras_initialization(self):
        """Start from the same kind of random weights Keras uses.

        PyTorch and Keras initialize layers differently by default. Keras uses
        Glorot-uniform weights and zero biases; using the same here means
        "same recipe" really means the same starting distribution.
        """
        for layer in [self.conv1, self.conv2, self.conv3, self.dense1, self.output]:
            nn.init.xavier_uniform_(layer.weight)  # "xavier" is PyTorch's name for Glorot
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.dense1(x))
        x = self.dropout(x)
        # Return raw scores (logits). Keras put softmax inside the model; in PyTorch
        # the softmax is inside nn.CrossEntropyLoss, which is numerically safer.
        return self.output(x)


def use_full_float32():
    """Make GPU convolutions use full float32 precision, like Keras did on the CPU.

    On newer NVIDIA GPUs PyTorch by default runs convolutions in "TF32", a faster
    mode that keeps only ~3 significant digits inside the multiply. Measured here:
    with TF32 the Keras weights gave probabilities up to 4e-4 away from Keras;
    with it off, 1.4e-6 (plain float32 rounding). Off is the faithful choice.
    """
    torch.backends.cudnn.allow_tf32 = False


def count_parameters(model):
    total = 0
    for p in model.parameters():
        total += p.numel()
    return total


def images_to_tensor(images):
    """uint8 (N, 64, 64) -> float32 (N, 1, 64, 64) in [0, 1], same as the Keras notebook.

    PyTorch wants the channel axis first (N, C, H, W); Keras had it last.
    """
    x = torch.from_numpy(images).float() / 255.0
    return x.unsqueeze(1)
