import numpy as np
from pathlib import Path
from collections import OrderedDict
from models.hopenet import HopeNet
from PIL import Image
import torch
from torchvision import transforms
import matplotlib.pyplot as plt

applied_transform = transforms.Compose([transforms.Resize((224, 224)),
                                        transforms.ToTensor()])


def imports():
    import numpy as np
    from pathlib import Path
    from collections import OrderedDict
    from models.hopenet import HopeNet
    from PIL import Image
    import torch
    from torchvision import transforms
    import matplotlib.pyplot as plt
    return None


def loaded_model():
    hope = HopeNet()
    p = Path("/Users/aenguscrowley/PycharmProjects/REAL_HOPE/checkpoints/Feb_26.pkl375.pkl")
    bad_dict = torch.load(p, map_location=torch.device('cpu'))

    good_dict = OrderedDict({key[len('module.'):]: value for key, value in bad_dict.items()})
    hope.load_state_dict(good_dict)
    return hope


def prep_single_image(image_path):
    """
    Note
    _________
    default:
     * (JpegImageFile, Parameter, NoneType, tuple, tuple, tuple, int),
    required:
     * (Tensor input, Tensor weight, Tensor bias, tuple of ints stride, tuple of ints padding, tuple of ints dilation, int groups)

    Parameters
    ----------
    image_path : path
    """
    if image_path.is_file():
        pic = np.load(image_path)

    return None


def classify(model, image: Path, model_transform):
    model = model.eval()
    image = Image.open(image)
    image = model_transform(image).float()
    image = image.unsqueeze(0)
    try:
        return model(image)
    except Exception as e:
        print("#" * 10 + "image" + "#" * 10)
        print(image)
        print("#" * 10 + "error" + "#" * 10)


def plot_from_single_image(classified_result):
    #figures
    fig = plt.figure()
    twoD = fig.add_subplot(3, 1, 1)
    twoDQ = fig.add_subplot(3, 1, 2)
    threeD = fig.add_subplot(3, 1, 3)
    #split data
    datatwoD = classified_result[0].detach.numpy()
    datatwoDQ = classified_result[1].detach.numpy()
    dataThreeD = classified_result[2].detach.numpy()

    #2d plots
    datatwoDx = [d[0] for d in datatwoD]
    datatwoDy = [d[1] for d in datatwoD]
    datatwoDz = [d[2] for d in datatwoD]

    datatwoDqx = [d[0] for d in datatwoD]
    datatwoDqy = [d[1] for d in datatwoD]
    datatwoDqz = [d[2] for d in datatwoD]

    twoD.scatter3D(datatwoDx, datatwoDy, datatwoDz, c='r')
    twoDQ.scatter3D(datatwoDqx, datatwoDqy, datatwoDqz, c='r')

    return fig

