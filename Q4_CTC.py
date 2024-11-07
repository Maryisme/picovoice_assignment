import torch
import torch.nn as nn


class CTCLoss(nn.Module):
    def __init__(self, blank=0):
        super().__init__()
        self.blank = blank

    def forward(self, log_probs, targets, input_lengths, target_lengths):
        # log_probs: (T, N, C)
        # targets: (N, S)
        # input_lengths: (N,)
        # target_lengths: (N,)

        T, N, C = log_probs.size()
        S = targets.size(1)

        # Extend targets with blanks
        extended_targets = torch.full(
            (N, 2*S + 1), self.blank, dtype=torch.long, device=log_probs.device)
        extended_targets[:, 1::2] = targets

        # Initialize forward and backward variables
        alphas = torch.zeros(T, N, 2*S + 1, device=log_probs.device)
        betas = torch.zeros(T, N, 2*S + 1, device=log_probs.device)

        # Initialize alphas
        alphas[0, :, 0] = log_probs[0, :, self.blank]
        alphas[0, :, 1] = log_probs[0, :, extended_targets[:, 1]]

        # Forward pass
        for t in range(1, T):
            for s in range(2*S + 1):
                label = extended_targets[:, s]

                a = alphas[t-1, :, s]
                if s > 0:
                    a = torch.logsumexp(torch.stack(
                        [a, alphas[t-1, :, s-1]]), dim=0)
                if s > 1 and label != extended_targets[:, s-2]:
                    a = torch.logsumexp(torch.stack(
                        [a, alphas[t-1, :, s-2]]), dim=0)

                alphas[t, :, s] = a + log_probs[t, :, label]

        # Initialize betas
        betas[-1, :, -1] = log_probs[-1, :, self.blank]
        betas[-1, :, -2] = log_probs[-1, :, extended_targets[:, -1]]

        # Backward pass
        for t in range(T-2, -1, -1):
            for s in range(2*S, -1, -1):
                label = extended_targets[:, s]

                b = betas[t+1, :, s]
                if s < 2*S:
                    b = torch.logsumexp(torch.stack(
                        [b, betas[t+1, :, s+1]]), dim=0)
                if s < 2*S-1 and label != extended_targets[:, s+2]:
                    b = torch.logsumexp(torch.stack(
                        [b, betas[t+1, :, s+2]]), dim=0)

                betas[t, :, s] = b + log_probs[t, :, label]

        # Compute loss
        loss = -torch.logsumexp(alphas[input_lengths-1,
                                torch.arange(N), 2*target_lengths-1], dim=0)

        return loss.mean()


class CTCNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(CTCNetwork, self).__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers=2,
                           bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        h, _ = self.rnn(x)
        y_hat = self.fc(h)
        return nn.functional.log_softmax(y_hat, dim=-1)


# Hyperparameters
input_dim = 26
hidden_dim = 100
output_dim = 62  # 61 phoneme categories + blank

# Initialize model
model = CTCNetwork(input_dim, hidden_dim, output_dim)

# Initialize weights


def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.LSTM):
        nn.init.uniform_(m.weight, -0.1, 0.1)
        if m.bias is not None:
            nn.init.uniform_(m.bias, -0.1, 0.1)


model.apply(init_weights)

# Training setup
criterion = CTCLoss(blank=61)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)

# Training loop
for epoch in range(num_epochs):
    for x, y in dataloader:
        optimizer.zero_grad()

        # Add Gaussian noise to inputs
        noise = torch.randn_like(x) * 0.6
        x = x + noise

        y_hat = model(x)
        input_lengths = torch.full((x.size(0),), x.size(1), dtype=torch.long)
        target_lengths = torch.full((y.size(0),), y.size(1), dtype=torch.long)

        loss = criterion(y_hat.transpose(0, 1), y,
                         input_lengths, target_lengths)
        loss.backward()
        optimizer.step()

# Decoding function


def prefix_search_decode(y_hat, blank=61, threshold=0.999):
    # Convert to probabilities
    probs = torch.exp(y_hat)

    # Find blank sections
    blank_probs = probs[:, blank]
    mask = blank_probs > threshold

    # Split into sections
    splits = torch.where(mask[1:] != mask[:-1])[0] + 1
    sections = torch.split(probs, splits.tolist())

    result = []
    for section in sections:
        # Best path decoding for each section
        best_path = torch.argmax(section, dim=1)
        decoded = torch.unique_consecutive(best_path)
        decoded = decoded[decoded != blank]
        result.extend(decoded.tolist())

    return result
