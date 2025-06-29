# Expansive Networks
The hypothesis is that, while mathematically equivalent and contrary to modern wisdom, two consecutive matrices of sizes 100x10000 and 10000x100 with no non-linearity inbetween them actually learn more easily than one 100x100 matrix. We call the act of replacing the 100x100 matrix by the former two matrices "expansion". We propose Expansive Networks -- a family of neural networks that are trained with expansions which are then collapsed at test-time via matrix multiplication. Only one layer is expanded at a time for computational memory savings.

![image](https://github.com/user-attachments/assets/f802bd01-57a1-4ce7-a277-091c9639a9bd)

## Usage

```
python3.10 expansive.py --epochs 10
```

## Code Summary

```py
class Net(nn.Module):
    def __init__(self, expansive=False):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.e1 = nn.Linear(128, 128, bias=False)
        self.e2 = nn.Linear(128, 128, bias=False)
        self.e3 = nn.Linear(128, 128, bias=False)
        self.e4 = nn.Linear(128, 128, bias=False)
        self.e5 = nn.Linear(128, 128, bias=False)
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
                x = self.e5(x) + res
            else:
                res = x
                c = nn.Linear(128, 128, bias=False)
                with torch.no_grad():
                    c.weight = Parameter(self.e5.weight @ self.e4.weight @ self.e3.weight @ self.e2.weight @ self.e1.weight)
                x = c(x) + res
        x = F.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
```
