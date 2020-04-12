import torch
import torch.nn as nn
import torch.autograd as autograd
import torchvision.models as models
from torch.autograd import Variable


###############################################################################
#                             Loss Functions                                  #
###############################################################################

def multi_scale_loss(n_scale, x, gt, h, w):
    MSE_LossFn = nn.MSELoss()
    loss = 0
    for i in range(n_scale):
        scale = 0.5 ** (n_scale - i - 1)
        hi = int(round(h * scale))
        wi = int(round(w * scale))  # 88
        frameGts = nn.functional.interpolate(gt, (hi, wi), mode='bilinear', align_corners=False)
        loss += MSE_LossFn(x[i], frameGts) / n_scale
    return loss


def charbonier_loss(x, epsilon):
    loss = torch.mean(torch.sqrt(x * x + epsilon * epsilon))
    return loss


def negPSNR_loss(x, epsilon):
    loss = torch.mean(torch.mean(torch.mean(torch.sqrt(x * x + epsilon * epsilon), dim=1), dim=1), dim=1)
    return torch.mean(-torch.log(1.0 / loss) / 100.0)


def tv_loss(x, epsilon):
    loss = torch.mean(torch.sqrt(
        (x[:, :, :-1, :-1] - x[:, :, 1:, :-1]) ** 2 +
        (x[:, :, :-1, :-1] - x[:, :, :-1, 1:]) ** 2 + epsilon * epsilon
    )
    )
    return loss


def gra_adap_tv_loss(flow, image, epsilon):
    w = torch.exp(- torch.sum(torch.abs(image[:, :, :-1, :-1] - image[:, :, 1:, :-1]) +
                              torch.abs(image[:, :, :-1, :-1] - image[:, :, :-1, 1:]), dim=1))
    tv = torch.sum(torch.sqrt((flow[:, :, :-1, :-1] - flow[:, :, 1:, :-1]) ** 2 + (
                flow[:, :, :-1, :-1] - flow[:, :, :-1, 1:]) ** 2 + epsilon * epsilon), dim=1)
    loss = torch.mean(w * tv)
    return loss


def smooth_loss(x, epsilon):
    loss = torch.mean(
        torch.sqrt(
            (x[:, :, :-1, :-1] - x[:, :, 1:, :-1]) ** 2 +
            (x[:, :, :-1, :-1] - x[:, :, :-1, 1:]) ** 2 + epsilon ** 2
        )
    )
    return loss


def motion_sym_loss(offset, epsilon, occlusion=None):
    if occlusion == None:
        # return torch.mean(torch.sqrt( (offset[:,:2,...] + offset[:,2:,...])**2 + epsilon **2))
        return torch.mean(torch.sqrt((offset[0] + offset[1]) ** 2 + epsilon ** 2))
    else:
        # TODO: how to design the occlusion aware offset symmetric loss?
        # return torch.mean(torch.sqrt((offset[:,:2,...] + offset[:,2:,...])**2 + epsilon **2))
        return torch.mean(torch.sqrt((offset[0] + offset[1]) ** 2 + epsilon ** 2))


def df_loss_func(flows, depths):
    # if the gradient of flow is less than 2 pixel or less than 5% of the current flow
    # we believe this is a smooth region, and impose constraint on depth gradient.
    # print("what to do") #TOdo
    loss = 0
    return loss


def part_loss(diffs, offsets, occlusions, images, epsilon, use_negPSNR=False):
    if use_negPSNR:
        pixel_loss = [negPSNR_loss(diff, epsilon) for diff in diffs]
    else:
        pixel_loss = [charbonier_loss(diff, epsilon) for diff in diffs]
    # offset_loss = [tv_loss(offset[0], epsilon) + tv_loss(offset[1], epsilon) for offset in
    #               offsets]

    if offsets[0][0] is not None:
        offset_loss = [gra_adap_tv_loss(offset[0], images[0], epsilon) + gra_adap_tv_loss(offset[1], images[1], epsilon)
                       for offset in
                       offsets]
    else:
        offset_loss = [Variable(torch.zeros(1).cuda())]
    # print(torch.max(occlusions[0]))
    # print(torch.min(occlusions[0]))
    # print(torch.mean(occlusions[0]))

    # occlusion_loss = [smooth_loss(occlusion, epsilon) + charbonier_loss(occlusion - 0.5, epsilon) for occlusion in occlusions]
    # occlusion_loss = [smooth_loss(occlusion, epsilon) + charbonier_loss(occlusion[:, 0, ...] - occlusion[:, 1, ...], epsilon) for occlusion in occlusions]

    if occlusions[0][0] is not None:
        occlusion_loss = [
            # smooth_loss(occlusion[0], epsilon) + smooth_loss(occlusion[1], epsilon) +
            charbonier_loss(occlusion[0] + occlusion[1] - 1.0, epsilon)
            if not occlusion == None else 0 for occlusion in occlusions]
    else:
        occlusion_loss = [Variable(torch.zeros(1).cuda())]

    sym_loss = [motion_sym_loss(offset, epsilon=epsilon) for offset in offsets]
    # sym_loss = [ motion_sym_loss(offset,occlusion) for offset,occlusion in zip(offsets,occlusions)]
    return pixel_loss, offset_loss, occlusion_loss, sym_loss


# def part_loss(diffs,offsets,occlusions):
#     pixel_loss = [charbonier_loss(diff, epsilon) for diff in diffs]
#     offset_loss = [tv_loss(offset[:, :2, ...], epsilon) + tv_loss(offset[:, 2:, ...], epsilon) for offset in offsets]
#     # print(torch.max(occlusions[0]))
#     # print(torch.min(occlusions[0]))
#     # print(torch.mean(occlusions[0]))
#
#     # occlusion_loss = [smooth_loss(occlusion, epsilon) + charbonier_loss(occlusion - 0.5, epsilon) for occlusion in occlusions]
#     # occlusion_loss = [smooth_loss(occlusion, epsilon) + charbonier_loss(occlusion[:, 0, ...] - occlusion[:, 1, ...], epsilon) for occlusion in occlusions]
#     occlusion_loss = [smooth_loss(occlusion, epsilon) + charbonier_loss(occlusion[:, 0, ...] + occlusion[:, 1, ...] - 1.0, epsilon) for occlusion in occlusions]
#
#     sym_loss = [ motion_sym_loss(offset) for offset in offsets]
#     # sym_loss = [ motion_sym_loss(offset,occlusion) for offset,occlusion in zip(offsets,occlusions)]
#     return pixel_loss, offset_loss, occlusion_loss, sym_loss


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt(diff * diff + self.eps))
        return loss

class ContentLoss:
    def __init__(self, loss):
        self.criterion = loss

    def get_loss(self, fakeIm, realIm):
        return self.criterion(fakeIm, realIm)


class PerceptualLoss():

    def contentFunc(self, gpu_id):
        conv_3_3_layer = 14
        cnn = models.vgg19(pretrained=True).features
        cnn = cnn.cuda(gpu_id)
        model = nn.Sequential()
        model = model.cuda(gpu_id)
        for i, layer in enumerate(list(cnn)):
            model.add_module(str(i), layer)
            if i == conv_3_3_layer:
                break
        return model

    def __init__(self, loss, gpu_id=0):
        self.criterion = loss
        self.contentFunc = self.contentFunc(gpu_id)
        self.gpu_id = gpu_id

    def get_loss(self, fakeIm, realIm):
        f_fake = self.contentFunc.forward(fakeIm)
        f_real = self.contentFunc.forward(realIm)
        f_real_no_grad = f_real.detach()
        loss = self.criterion(f_fake, f_real_no_grad)
        return loss


# Define GAN loss: [vanilla | lsgan | wgan-gp]
class GANLoss(nn.Module):
    def __init__(self, gan_type, real_label_val=1.0, fake_label_val=0.0):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type.lower()
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        if self.gan_type == 'gan' or self.gan_type == 'ragan':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'wgan-gp':

            def wgan_loss(input, target):
                # target is boolean
                return -1 * input.mean() if target else input.mean()

            self.loss = wgan_loss
        else:
            raise NotImplementedError('GAN type [{:s}] is not found'.format(self.gan_type))

    def get_target_label(self, input, target_is_real):
        if self.gan_type == 'wgan-gp':
            return target_is_real
        if target_is_real:
            return torch.empty_like(input).fill_(self.real_label_val)
        else:
            return torch.empty_like(input).fill_(self.fake_label_val)

    def forward(self, input, target_is_real):
        target_label = self.get_target_label(input, target_is_real)
        loss = self.loss(input, target_label)
        return loss



class GradientPenaltyLoss(nn.Module):
    def __init__(self, device=torch.device('cpu')):
        super(GradientPenaltyLoss, self).__init__()
        self.register_buffer('grad_outputs', torch.Tensor())
        self.grad_outputs = self.grad_outputs.to(device)

    def get_grad_outputs(self, input):
        if self.grad_outputs.size() != input.size():
            self.grad_outputs.resize_(input.size()).fill_(1.0)
        return self.grad_outputs

    def forward(self, interp, interp_crit):
        grad_outputs = self.get_grad_outputs(interp_crit)
        grad_interp = torch.autograd.grad(outputs=interp_crit, inputs=interp,
                                          grad_outputs=grad_outputs, create_graph=True,
                                          retain_graph=True, only_inputs=True)[0]
        grad_interp = grad_interp.view(grad_interp.size(0), -1)
        grad_interp_norm = grad_interp.norm(2, dim=1)

        loss = ((grad_interp_norm - 1)**2).mean()
        return loss


class DiscLoss:
    def name(self):
        return 'DiscLoss'

    def __init__(self, tensor, gpu_id=0):
        self.criterionGAN = GANLoss(use_l1=False, tensor=tensor, gpu_id=gpu_id)

        # self.fake_AB_pool = ImagePool(opt.pool_size)

    def get_g_loss(self, net, realA, fakeB):
        # First, G(A) should fake the discriminator
        pred_fake = net.forward(fakeB)
        return self.criterionGAN(pred_fake, 1)

    def get_loss(self, net, realA, fakeB, realB):
        # Fake
        # stop backprop to the generator by detaching fake_B
        # Generated Image Disc Output should be close to zero
        self.pred_fake = net.forward(fakeB.detach())
        self.loss_D_fake = self.criterionGAN(self.pred_fake, 0)

        # Real
        self.pred_real = net.forward(realB)
        self.loss_D_real = self.criterionGAN(self.pred_real, 1)

        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        return self.loss_D


class DiscLossLS(DiscLoss):
    def name(self):
        return 'DiscLossLS'

    def __init__(self, tensor, gpu_id):
        super(DiscLoss, self).__init__( tensor, gpu_id)
        # DiscLoss.initialize(self, opt, tensor)
        self.criterionGAN = GANLoss(use_l1=True, tensor=tensor, gpu_id=gpu_id)

    def get_g_loss(self, net, realA, fakeB):
        return DiscLoss.get_g_loss(self, net, realA, fakeB)

    def get_loss(self, net, realA, fakeB, realB):
        return DiscLoss.get_loss(self, net, realA, fakeB, realB)


class DiscLossWGANGP(DiscLossLS):
    def name(self):
        return 'DiscLossWGAN-GP'

    def __init__(self, tensor, gpu_id=0):
        super(DiscLossWGANGP, self).__init__(tensor, gpu_id)
        # DiscLossLS.initialize(self, opt, tensor)
        self.LAMBDA = 10
        self.gpu_id = gpu_id

    def get_g_loss(self, net, realA, fakeB):
        # First, G(A) should fake the discriminator
        self.D_fake = net.forward(fakeB)
        return -self.D_fake.mean()

    def calc_gradient_penalty(self, netD, real_data, fake_data):
        alpha = torch.rand(1, 1)
        alpha = alpha.expand(real_data.size())
        alpha = alpha.cuda(self.gpu_id)

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        interpolates = interpolates.cuda(self.gpu_id)
        interpolates = Variable(interpolates, requires_grad=True)

        disc_interpolates = netD.forward(interpolates)

        gradients = autograd.grad(
            outputs=disc_interpolates, inputs=interpolates, grad_outputs=torch.ones(disc_interpolates.size()).cuda(self.gpu_id),
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.LAMBDA
        return gradient_penalty

    def get_loss(self, net, realA, fakeB, realB):
        self.D_fake = net.forward(fakeB.detach())
        self.D_fake = self.D_fake.mean()

        # Real
        self.D_real = net.forward(realB)
        self.D_real = self.D_real.mean()
        # Combined loss
        self.loss_D = self.D_fake - self.D_real
        gradient_penalty = self.calc_gradient_penalty(net, realB.data, fakeB.data)
        return self.loss_D + gradient_penalty


def init_loss(model, gan_type, tensor, gpu_id =0):
    # disc_loss = None
    # content_loss = None

    if model == 'content_gan':
        content_loss = PerceptualLoss(nn.MSELoss(), gpu_id)
    # content_loss.initialize(nn.MSELoss())
    elif model == 'pix2pix':
        content_loss = ContentLoss(nn.L1Loss())
    # content_loss.initialize(nn.L1Loss())
    else:
        raise ValueError("Model [%s] not recognized." % model)

    if gan_type == 'wgan-gp':
        disc_loss = DiscLossWGANGP(tensor, gpu_id)
    elif gan_type == 'lsgan':
        disc_loss = DiscLossLS(tensor, gpu_id=gpu_id)
    elif gan_type == 'gan':
        disc_loss = DiscLoss(tensor, gpu_id=gpu_id)
    else:
        raise ValueError("GAN [%s] not recognized." % gan_type)
    # disc_loss.initialize(opt, tensor)
    return disc_loss, content_loss
