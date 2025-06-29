import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

import matplotlib.pyplot as plt
from torch.nn.parameter import Parameter

class Net(nn.Module):
    def __init__(self, expansive=False):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.e1 = nn.Linear(128, 128, bias=False)
        self.e2 = nn.Linear(128, 128, bias=False)
        self.e3 = nn.Linear(128, 128, bias=False)
        self.e4 = nn.Linear(128, 128, bias=False)
        self.e5 = nn.Linear(128, 128, bias=False)
        self.e6 = nn.Linear(128, 128, bias=False)
        self.e7 = nn.Linear(128, 128, bias=False)
        self.e8 = nn.Linear(128, 128, bias=False)
        self.e9 = nn.Linear(128, 128, bias=False)
        self.e10 = nn.Linear(128, 128, bias=False)
        self.e11 = nn.Linear(128, 128, bias=False)
        self.e12 = nn.Linear(128, 128, bias=False)
        self.e13 = nn.Linear(128, 128, bias=False)
        self.e14 = nn.Linear(128, 128, bias=False)
        self.e15 = nn.Linear(128, 128, bias=False)
        self.e16 = nn.Linear(128, 128, bias=False)
        self.e17 = nn.Linear(128, 128, bias=False)
        self.e18 = nn.Linear(128, 128, bias=False)
        self.e19 = nn.Linear(128, 128, bias=False)
        self.e20 = nn.Linear(128, 128, bias=False)
        self.fc2 = nn.Linear(128, 10)
        self.expansive = expansive

    def forward(self, x, test_collapse=False):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        if self.expansive:
            if self.training or test_collapse:
                res = x
                x = self.e1(x)
                x = self.e2(x)
                x = self.e3(x)
                x = self.e4(x)
                x = self.e5(x)
                x = self.e6(x)
                x = self.e7(x)
                x = self.e8(x)
                x = self.e9(x)
                x = self.e10(x)
                x = self.e11(x)
                x = self.e12(x)
                x = self.e13(x)
                x = self.e14(x)
                x = self.e15(x)
                x = self.e16(x)
                x = self.e17(x)
                x = self.e18(x)
                x = self.e19(x)
                x = self.e20(x)    
                x = x + res
            else:
                res = x
                c = nn.Linear(128, 128, bias=False)
                with torch.no_grad():
                    c.weight = Parameter(
                        self.e20.weight @ self.e19.weight @ self.e18.weight @ self.e17.weight @ self.e16.weight @ \
                        self.e15.weight @ self.e14.weight @ self.e13.weight @ self.e12.weight @ self.e11.weight @ \
                        self.e10.weight @ self.e9.weight @ self.e8.weight @ self.e7.weight @ self.e6.weight @ \
                        self.e5.weight @ self.e4.weight @ self.e3.weight @ self.e2.weight @ self.e1.weight)
                x = c(x) 
                x = x + res
        x = F.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            output_test_collapse = model(data, test_collapse=True)
            print("collapsed - expanded:", torch.sum(torch.abs(torch.sub(output, output_test_collapse))))
            # assert torch.equal(output, output_test_collapse)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return correct


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-accel', action='store_true',
                        help='disables accelerator')
    parser.add_argument('--dry-run', action='store_true',
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', 
                        help='For Saving the current Model')
    args = parser.parse_args()

    use_accel = not args.no_accel and torch.accelerator.is_available()

    torch.manual_seed(args.seed)

    if use_accel:
        device = torch.accelerator.current_accelerator()
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_accel:
        accel_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(accel_kwargs)
        test_kwargs.update(accel_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    score = {False: [], True: []}

    init_model = Net().to(device)

    for expansive in [False, True]:

        model = Net(expansive=expansive).to(device)
        model.load_state_dict(init_model.state_dict())
        if not expansive:
            for name, param in model.named_parameters():
                if name[0] == 'e':
                    with torch.no_grad():
                        param.copy_(float('nan'))
                        # print(param)
        # optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
        optimizer = optim.AdamW(model.parameters(), lr=0.001, betas=(0.95, 0.95))
        # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

        scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
        for epoch in range(1, args.epochs + 1):
            # score[expansive].append(test(model, device, test_loader))
            train(args, model, device, train_loader, optimizer, epoch)
            score[expansive].append(test(model, device, test_loader))
            scheduler.step()

        if args.save_model:
            torch.save(model.state_dict(), "mnist_cnn.pt")

        plt.plot(score[expansive], label=str(expansive)+" "+str(score[expansive][-1])+" "+str(score[expansive][-1]/100)+"%")
    plt.legend(title="Expansive")
    plt.xlabel("Epoch")
    plt.ylabel("Number of Correct Test Classifications on MNIST")
    plt.title("Expansive Versus Baseline on MNIST")
    plt.savefig("score.png")


if __name__ == '__main__':
    main()