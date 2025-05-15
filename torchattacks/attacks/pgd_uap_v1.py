import torch
import torch.nn as nn
import time
from ..attack import Attack


class PGD(Attack):
    r"""
    PGD in the paper 'Towards Deep Learning Models Resistant to Adversarial Attacks'
    [https://arxiv.org/abs/1706.06083]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 2/255)
        steps (int): number of steps. (Default: 10)
        random_start (bool): using random initialization of delta. (Default: True)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.PGD(model, eps=8/255, alpha=1/255, steps=10, random_start=True)
        >>> adv_images = attack(images, labels)

    """

    def __init__(self, model, eps=8 / 255, alpha=2 / 255, steps=10, nprompt=1, random_start=True, universal=1):
        super().__init__("PGD", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.supported_mode = ["default", "targeted"]
        self.nprompt = nprompt
        self.universal = universal

        # ImageNet标准化参数
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)

        # 计算归一化后的最小值和最大值
        self.min_values = (torch.tensor(0.) - self.mean) / self.std
        self.max_values = (torch.tensor(1.) - self.mean) / self.std

    def forward(self, images, labels = None):
        r"""
        Overridden.
        """

        # images = images.clone().detach().to(self.device)
        images_ = []
        adv_images_ = []
        for image in images:
            if image.dim() == 3:  # [C,H,W] -> [1,C,H,W]
                image = image.unsqueeze(0)
            image = image.clone().detach().to(self.device)
            adv_image = image.clone().detach().to(self.device)
            images_.append(image)
            adv_images_.append(adv_image)

        if self.random_start:
            # Starting at a uniformly random point
            for i in range(len(adv_images_)):
                # 对每个通道使用相应的范围生成随机扰动
                random_noise = torch.zeros_like(adv_images_[i])
                for c in range(3):
                    eps_c = self.eps  # 可以为每个通道设置不同的eps
                    random_noise[:, c:c + 1] = torch.empty_like(adv_images_[i][:, c:c + 1]).uniform_(-eps_c, eps_c)
                adv_images_[i] = adv_images_[i] + random_noise

                # 确保在归一化范围内
                for c in range(3):
                    adv_images_[i][:, c:c + 1] = torch.clamp(
                        adv_images_[i][:, c:c + 1],
                        min=self.min_values[0, c],
                        max=self.max_values[0, c]
                    )

        if self.universal == 1:
            noise = torch.zeros(1, 3, 448, 448).to(self.device)

        for _ in range(self.steps):
            # print('step: '+str(_))
            start = time.time()
            cost_step = 0
            for k in range(len(images_)):
                image = images_[k]

                image_ = image.clone()
                for p in range(self.nprompt):

                    image_.requires_grad = True
                    inp = []

                    if self.universal == 1:
                        adv_image = image_ + noise  # universal noise

                    inp.append(adv_image)
                    inp.append(p)
                    cost = self.get_logits(inp)
                    # Update adversarial images
                    grad = torch.autograd.grad(
                        cost, adv_image, retain_graph=False, create_graph=False
                    )[0]

                    cost_step += cost.clone().detach()

                    adv_image = adv_image.detach() + self.alpha * grad.sign()

                    # 对每个通道分别裁剪delta和adv_image
                    delta = torch.zeros_like(adv_image - image)
                    for c in range(3):
                        delta[:, c:c + 1] = torch.clamp(
                            adv_image[:, c:c + 1] - image[:, c:c + 1],
                            min=-self.eps,
                            max=self.eps
                        )

                    adv_image = image + delta

                    # 确保adv_image在ImageNet归一化后的有效范围内
                    for c in range(3):
                        adv_image[:, c:c + 1] = torch.clamp(
                            adv_image[:, c:c + 1],
                            min=self.min_values[0, c],
                            max=self.max_values[0, c]
                        )
                    adv_image = adv_image.detach()

                    if self.universal == 1:
                        noise = adv_image - image

            runtime = time.time() - start
            print('step: {}: {}'.format(_, cost_step), 'Runtime: {}: {}'.format(_, runtime))

        if self.universal == 1:  # universal noise
            images_outputs_ = []
            for k in range(len(images_)):
                # 对每个通道分别裁剪噪声
                delta = torch.zeros_like(noise)
                for c in range(3):
                    delta[:, c:c + 1] = torch.clamp(
                        noise[:, c:c + 1],
                        min=-self.eps,
                        max=self.eps
                    )

                adv_image = images_[k] + delta

                # 确保adv_image在ImageNet归一化后的有效范围内
                for c in range(3):
                    adv_image[:, c:c + 1] = torch.clamp(
                        adv_image[:, c:c + 1],
                        min=self.min_values[0, c],
                        max=self.max_values[0, c]
                    )

                images_outputs_.append(adv_image.detach())

            return images_outputs_