import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader

from utils.dataset import Dataset
from utils.model import select_model
from utils.options import parse_args_function


def main():
    args = parse_args_function()

    """# Load Dataset"""

    root = args.input_file

    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor()])

    if args.train:
        print("Loading training files...")
        trainset = Dataset(root=root, load_set='train', transform=transform)
        trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=10)

        print('Training files loaded')

    if args.val:
        print("Loading validation files...")

        valset = Dataset(root=root, load_set='val', transform=transform)
        valloader = DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=8)

        print('Validation files loaded')

    if args.test:
        print("Loading test files...")

        testset = Dataset(root=root, load_set='test', transform=transform)
        testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=8)

        print('Test files loaded')

    """# Model"""

    model = select_model(args.model_def)
    model = nn.DataParallel(model)

    """# Load Snapshot"""

    if args.pretrained_model != '':
        model.load_state_dict(torch.load(args.pretrained_model, map_location=torch.device('cpu')))
        losses = np.load(args.pretrained_model[:-4] + '-losses.npy').tolist()
        start = len(losses)
    else:
        losses = []
        start = 0

    """# Optimizer"""

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_step_gamma)
    scheduler.last_epoch = start

    """# Train"""

    if args.train:
        print('Begin training the network...')
        print(f"Looping {args.num_iterations} times...")
        print(f"Starting with epoch {start}...")

        for epoch in range(start, args.num_iterations):  # loop over the dataset multiple times
            print(f"Running epoch {epoch}...")
            model.train()
            running_loss = 0.0
            train_loss = 0.0
            for i, tr_data in enumerate(trainloader):
                # get the inputs
                _, labels2d, labels3d = tr_data

                # wrap them in Variable
                labels2d = Variable(labels2d)
                labels3d = Variable(labels3d)

                labels2d = labels2d.float()
                labels3d = labels3d.float()

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs3d = model(labels2d)
                loss = criterion(outputs3d, labels3d)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.data
                train_loss += loss.data
                if (i + 1) % args.log_batch == 0:  # print every log_iter mini-batches
                    print(f'[{epoch + 1:d}, {i + 1:5d}] loss: {running_loss / args.log_batch:.5f}')
                    running_loss = 0.0

            if args.val and (epoch + 1) % args.val_epoch == 0:
                model.eval()
                val_loss = 0.0
                for v, val_data in enumerate(valloader):
                    # get the inputs
                    _, labels2d, labels3d = val_data

                    # wrap them in Variable
                    labels2d = Variable(labels2d)
                    labels3d = Variable(labels3d)

                    labels2d = labels2d.float()
                    labels3d = labels3d.float()

                    outputs3d = model(labels2d)
                    loss = criterion(outputs3d, labels3d)
                    val_loss += loss.data
                print(f'val error: {val_loss / (v + 1):.5f}')
            losses.append((train_loss / (i + 1)).cpu().numpy())

            if (epoch + 1) % args.snapshot_epoch == 0:
                torch.save(model.state_dict(), args.output_file + str(epoch + 1) + '.pkl')
                np.save(args.output_file + str(epoch + 1) + '-losses.npy', np.array(losses))

            # Decay Learning Rate
            scheduler.step()

        print('Finished Training')

    """# Test"""

    if args.test:
        print('Begin testing the network...')
        model.eval()
        running_loss = 0.0
        for i, ts_data in enumerate(testloader):
            # get the inputs
            _, labels2d, labels3d = ts_data

            # wrap them in Variable
            labels2d = Variable(labels2d)
            labels3d = Variable(labels3d)

            labels2d = labels2d.float()
            labels3d = labels3d.float()

            outputs3d = model(labels2d)
            loss = criterion(outputs3d, labels3d)
            running_loss += loss.data
        print(f'test error: {running_loss / (i + 1):.5f}')


if __name__ == "__main__":
    main()
