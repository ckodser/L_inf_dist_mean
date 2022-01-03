import copy

import torch


# TODO: what is group?
#### forward functions

def inf_dist_forward_nograd(x, weight, output: torch.Tensor, groups):
    with torch.no_grad():
        output.data = x.view(x.size(0), groups, 1, -1, x.size(2)) - weight.view(groups, -1, weight.size(-1), 1)
        output.data = torch.norm(output.data, dim=3, p=float('inf'), keepdim=True)
        output.data = output.data.view(output.size(0), -1, output.data.size(-1))


def inf_dist_forward(x, weight, output, pos, groups):
    with torch.no_grad():
        raise NotImplemented
        output.data = x.view(x.size(0), groups, 1, -1, x.size(2)) - weight.view(groups, -1, weight.size(-1), 1)
        output.data = torch.norm(output.data, dim=3, p=float('inf'), keepdim=True)
        output.data = output.data.view(output.size(0), -1, output.data.size(-1))


def norm_dist_forward(x, weight, output, groups, p):
    with torch.no_grad():
        output.data = x.view(x.size(0), groups, 1, -1, x.size(2)) - weight.view(groups, -1, weight.size(-1), 1)
        normalize = torch.norm(output.data, dim=3, p=float('inf'), keepdim=True)
        output.data = torch.norm(output.data / normalize, dim=3, p=p, keepdim=True) * normalize
        output.data = output.data.view(output.size(0), -1, output.data.size(-1))


#### backward functions

def inf_dist_backward(grad_output, pos, grad_input, grad_weight, groups):
    with torch.no_grad():
        raise NotImplemented


def norm_dist_backward(grad_output, x, weight, output, grad_input, grad_weight, groups, p):
    with torch.no_grad():
        # print("BACK S", grad_weight.shape[0] * grad_weight.shape[0] * grad_output.shape[0])
        o = torch.pow(((x.data.view(x.size(0), groups, 1, -1, x.size(2)) - weight.data.view(groups, -1, weight.size(-1),
                                                                                            1)) / output.data.view(
            x.size(0), groups, weight.size(0), 1, -1)).data, p - 1) * grad_output.data.view(grad_output.size(0), 1,
                                                                                            grad_output.size(1), 1, 1)
        grad_weight.data = -torch.sum(o, dim=0).view(weight.shape)
        grad_input.data = torch.sum(o, dim=2).view(grad_input.shape)

        # for i in range(8):
        #     for j in range(8):
        #         eps = 10 ** (-3)
        #         x[i, j, 0] += eps
        #         new_output = torch.zeros_like(output)
        #         norm_dist_forward(x, weight, new_output, groups, p)
        #         grad = torch.sum((new_output - output) * grad_output) / eps
        #         x[i, j, 0] -= eps
        #         error = grad - grad_input[i, j, 0]
        #         if abs(error)>abs(grad)/10 and abs(error-grad)>10**(-8)*3:
        #             print("FUCK")
        #         print("grad check", x[i, j, 0], grad, error)
        #
        #         eps = 10 ** (-3)
        #         weight[i, j] += eps
        #         new_output = torch.zeros_like(output)
        #         norm_dist_forward(x, weight, new_output, groups, p)
        #         grad = torch.sum((new_output - output) * grad_output) / eps
        #         weight[i, j] -= eps
        #         error = grad - grad_weight[i, j]
        #         if abs(error)>abs(grad)/10 and abs(error-grad)>10**(-8)*3:
        #             print("FUCK")
        #         print("grad check", weight[i, j], grad, error)

        # print("grad output", grad_output)
        # print("grad input", grad_input[0])
        # print("grad weight", grad_weight[0, 1])
        # print("x[0]", x[0])
        # print("weight[:,0]", weight[:, 0])
        # raise NotImplemented
