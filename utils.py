import torch
from torchvision import transforms
import numpy as np
import torch.nn as nn
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def mask_patch(image, patch_index, patch_size):
    # 获取图像通道数、图像高度和图像宽度
    batch_size, channels, height, width = image.shape

    patch_num_per_row = height // patch_size

    # 计算patch在图像中的位置
    patch_row = patch_index // patch_num_per_row
    patch_col = patch_index % patch_num_per_row

    # 计算patch的起始和结束位置
    start_row = patch_row * patch_size
    end_row = start_row + patch_size
    start_col = patch_col * patch_size
    end_col = start_col + patch_size

    # 将patch区域的像素值置为零
    image[:, :, start_row:end_row, start_col:end_col] = 0

    return image

def my_norm(image):

    mean = (0.48145466, 0.4578275, 0.40821073)
    std = (0.26862954, 0.26130258, 0.27577711)
    normalize = transforms.Normalize(mean, std)
    image_norm = normalize(image)

    return image_norm


def denorm(image):

    mean = (0.48145466, 0.4578275, 0.40821073)
    std = (0.26862954, 0.26130258, 0.27577711)
    mean = torch.tensor(mean).to(device)
    std = torch.tensor(std).to(device)

    image_denorm = image*std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)

    return image_denorm


def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

'''
@Parameter atten_grad, ce_grad: should be 2D tensor with shape [batch_size, -2]
'''
def PCGrad(atten_grad, ce_grad, sim, shape):
    pcgrad = atten_grad[sim < 0]
    temp_ce_grad = ce_grad[sim < 0]
    dot_prod = torch.mul(pcgrad, temp_ce_grad).sum(dim=-1)
    dot_prod = dot_prod / torch.norm(temp_ce_grad, dim=-1)
    pcgrad = pcgrad - dot_prod.view(-1, 1) * temp_ce_grad
    atten_grad[sim < 0] = pcgrad
    atten_grad = atten_grad.view(shape)
    return atten_grad



def image_precessing(image):
    # mu = [0.485, 0.456, 0.406]
    # std = [0.229, 0.224, 0.225]
    mu = (0.48145466, 0.4578275, 0.40821073)
    std = (0.26862954, 0.26130258, 0.27577711)
    transform = transforms.Compose([transforms.Resize(224),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=mu, std=std)
                        ])
    return transform(image)


def save_image(image_array: np.ndarray, f_name: str) -> None:
    """
    Saves image into a file inside `ART_DATA_PATH` with the name `f_name`.

    :param image_array: Image to be saved.
    :param f_name: File name containing extension e.g., my_img.jpg, my_img.png, my_images/my_img.png.
    """
    from PIL import Image
    image = Image.fromarray(image_array)
    image.save(f_name)



class PGD(nn.Module):
    r"""

    Examples::
        >> attack = torchattacks.PGD(model, eps=8/255, alpha=2/255, steps=10, random_start=True)
        >> adv_images = attack(images, labels)

    """

    # def __init__(self, device, model, eps=8 / 255, alpha=3 / 255, steps=10):
    #     super(BIM, self).__init__()

    def __init__(self, device, model, eps=8 / 255, alpha=2 / 255, steps=10, random_start=True):
        super(PGD, self).__init__()
        print("attack sb")
        self.eps = eps
        self.alpha = alpha
        self.model = model
        self.steps = steps
        self.random_start = random_start
        self.device = device

    def forward(self, images):
        r"""
        Overridden.
        """
        print("attack start")
        images = images.clone().detach().to(torch.float32).to(self.device)


        adv_images = images.clone().detach()
        images = denorm(images)
        adv_images = denorm(adv_images)

        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(
                -self.eps, self.eps
            )
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True

            adv_images = my_norm(adv_images)
            loss = self.model(adv_images)


            # Update adversarial images
            grad = torch.autograd.grad(
                loss, adv_images, retain_graph=False, create_graph=False
            )[0]

            adv_images = denorm(adv_images)
            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()
            print('step: {}: {}'.format(_, loss))
            print("over")
        #print(images)
        #adv_images = my_norm(adv_images)
        #loss = self.model(adv_images)
        print("over")
        return adv_images
