import torch
import tqdm
import numpy as np

# 欧拉-丸山采样器/求解器（Euler-Maruyama sampler）
def Euler_Maruyama_sampler(score_model,
                           channel,
                           size,
                           marginal_prob_std,
                           diffusion_coeff,
                           batch_size=64,
                           num_steps=1000,
                           device='cuda',
                           eps=1e-3):
    t = torch.ones(batch_size, device=device)
    init_x = torch.randn(batch_size, channel, size, size, device=device) * marginal_prob_std(t)[:, None, None, None]
    time_steps = torch.linspace(1., eps, num_steps, device=device)
    step_size = time_steps[0] - time_steps[1]
    x = init_x
    imgs = [x.cpu().numpy()]
    with torch.no_grad():
        for time_step in tqdm.tqdm(time_steps):
            batch_time_step = torch.ones(batch_size, device=device) * time_step
            g = diffusion_coeff(batch_time_step)
            mean_x = x + (g ** 2)[:, None, None, None] * score_model(x, batch_time_step) * step_size
            x = mean_x + torch.sqrt(step_size) * g[:, None, None, None] * torch.randn_like(x)
            imgs.append(mean_x.cpu().numpy())
            # Do not include any noise in the last sampling step.
    return imgs



# 预测-检验采样器
signal_to_noise_ratio = 0.16
def pc_sampler(score_model,
               channel,
               size,
               marginal_prob_std,
               diffusion_coeff,
               batch_size=64,
               num_steps=1000,
               snr=signal_to_noise_ratio,
               M = 10,
               device='cuda',
               eps=1e-3):
    t = torch.ones(batch_size, device=device)
    init_x = torch.randn(batch_size, channel, size, size, device=device) * marginal_prob_std(t)[:, None, None, None]
    time_steps = np.linspace(1., eps, num_steps)
    step_size = time_steps[0] - time_steps[1]
    x = init_x
    imgs = [x.cpu().numpy()]
    with torch.no_grad():
        for time_step in tqdm.tqdm(time_steps):
            batch_time_step = torch.ones(batch_size, device=device) * time_step
            # 检验器 step (Langevin MCMC)
            grad = score_model(x, batch_time_step)
            grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
            noise_norm = np.sqrt(np.prod(x.shape[1:]))
            langevin_step_size = 2 * (snr * noise_norm / grad_norm) ** 2

            for _ in range(M):
                x = x + langevin_step_size * grad + torch.sqrt(2 * langevin_step_size) * torch.randn_like(x)
                grad = score_model(x, batch_time_step)
                grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
                noise_norm = np.sqrt(np.prod(x.shape[1:]))
                langevin_step_size = 2 * (snr * noise_norm / grad_norm) ** 2

            # 预测器 step (Euler-Maruyama)
            g = diffusion_coeff(batch_time_step)
            x_mean = x + (g ** 2)[:, None, None, None] * score_model(x, batch_time_step) * step_size
            x = x_mean + torch.sqrt(g ** 2 * step_size)[:, None, None, None] * torch.randn_like(x)
            imgs.append(x_mean.cpu().numpy())

        return imgs



# ODE数值求解器
from scipy import integrate
error_tolerance = 1e-5
def ode_sampler(score_model,
                channel,
                size,
                marginal_prob_std,
                diffusion_coeff,
                batch_size=64,
                atol=error_tolerance,
                rtol=error_tolerance,
                device='cuda',
                z=None,
                eps=1e-3):
    t = torch.ones(batch_size, device=device)
    # Create the latent code
    if z is None:
        init_x = torch.randn(batch_size, channel, size, size, device=device) * marginal_prob_std(t)[:, None, None, None]
    else:
        init_x = z

    shape = init_x.shape

    def score_eval_wrapper(sample, time_steps):
        """A wrapper of the score-based model for use by the ODE solver."""
        sample = torch.tensor(sample, device=device, dtype=torch.float32).reshape(shape)
        time_steps = torch.tensor(time_steps, device=device, dtype=torch.float32).reshape((sample.shape[0],))
        with torch.no_grad():
            score = score_model(sample, time_steps)
        return score.cpu().numpy().reshape((-1,)).astype(np.float64)

    def ode_func(t, x):
        """The ODE function for use by the ODE solver."""
        time_steps = np.ones((shape[0],)) * t
        g = diffusion_coeff(torch.tensor(t)).cpu().numpy()
        return -0.5 * (g ** 2) * score_eval_wrapper(x, time_steps)

    # Run the black-box ODE solver.
    res = integrate.solve_ivp(ode_func, (1., eps), init_x.reshape(-1).cpu().numpy(), rtol=rtol, atol=atol,
                              method='RK45')
    print(f"Number of function evaluations: {res.nfev}")
    x = torch.tensor(res.y[:, -1], device=device).reshape(shape)

    return x


# 欧拉折线法 ode
def Euler_ode_sampler(score_model,
                      channel,
                      size,
                      marginal_prob_std,
                      diffusion_coeff,
                      batch_size=64,
                      num_steps=1000,
                      device='cuda',
                      eps=1e-3):
    t = torch.ones(batch_size, device=device)
    init_x = torch.randn(batch_size, channel, size, size, device=device) * marginal_prob_std(t)[:, None, None, None]
    time_steps = torch.linspace(1., eps, num_steps, device=device)
    step_size = time_steps[0] - time_steps[1]
    x = init_x
    imgs = [x.cpu().numpy()]
    with torch.no_grad():
        for time_step in tqdm.tqdm(time_steps):
            batch_time_step = torch.ones(batch_size, device=device) * time_step
            g = diffusion_coeff(batch_time_step)
            mean_x = x - 0.5 * (g ** 2)[:, None, None, None] * score_model(x, batch_time_step) * step_size
            x = mean_x
            imgs.append(mean_x.cpu().numpy())
            # Do not include any noise in the last sampling step.
    return imgs


def marginal_prob_std(t, sigma):
    # 计算p (x(t) | x(0))的标准差
    t = torch.as_tensor(t, device=device)
    return torch.sqrt((sigma ** (2 * t) - 1.) / 2. / np.log(sigma))

def diffusion_coeff(t, sigma):
    # 计算SDE的扩散系数.
    return torch.as_tensor(sigma ** t, device=device)


from Unet_ddpm import UNetModel
import matplotlib.pyplot as plt
import os
import functools
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

if __name__ == '__main__':
    sigma = 25.0
    marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
    diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma)

    num_steps = 500
    device = 'cuda'
    score_model = UNetModel(
        in_channels=1,
        model_channels=128,
        out_channels=1,
        channel_mult=(1, 2, 2),
        attention_resolutions=(2,),
        dropout=0.1
    )
    score_model.to(device)
    score_model.load_state_dict(
        torch.load('model/mnist_score_model.pth', map_location=lambda storage, location: storage)['state_dict'])
    sample_batch_size = 64

    sampler = Euler_Maruyama_sampler
    samples = sampler(score_model,
                      1,
                      28,
                      marginal_prob_std_fn,
                      diffusion_coeff_fn,
                      sample_batch_size,
                      num_steps,
                      device=device)
    imgs = samples[-1].reshape(8, 8, 1, 28, 28)

    # sampler = Euler_ode_sampler
    # samples = sampler(score_model,
    #                   1,
    #                   28,
    #                   marginal_prob_std_fn,
    #                   diffusion_coeff_fn,
    #                   sample_batch_size,
    #                   num_steps,
    #                   device=device)
    # imgs = samples[-1].reshape(8, 8, 1, 28, 28)

    # sampler = ode_sampler
    # samples = ode_sampler(score_model,
    #                       1,
    #                       28,
    #                       marginal_prob_std_fn,
    #                       diffusion_coeff_fn,
    #                       sample_batch_size,
    #                       atol=error_tolerance,
    #                       rtol=error_tolerance,
    #                       device=device,
    #                       z=None,
    #                       eps=1e-3)
    # imgs = samples.reshape(8, 8, 1, 28, 28)

    # generate new images

    fig = plt.figure(figsize=(12, 12), constrained_layout=True)
    gs = fig.add_gridspec(8, 8)
    for n_row in range(8):
        for n_col in range(8):
            f_ax = fig.add_subplot(gs[n_row, n_col])
            img = np.array(imgs[n_row, n_col].transpose([1, 2, 0]), dtype=np.uint8)
            # img = np.array((imgs[n_row, n_col].transpose([1, 2, 0]) + 1.0) * 255 / 2, dtype=np.uint8)
            f_ax.imshow(img)
            f_ax.axis("off")
    fig.savefig('output1.png')  # 将图像保存为文件


    # save picture
    # imgs = generated_images[-1].reshape(64, 3, 32, 32)
    # for i in range(64):
    #     img = imgs[i].transpose([1, 2, 0])
    #     img = np.array((img + 1.0) * 255 / 2, dtype=np.uint8)
    #     # 将数组转换为PIL图像对象
    #     img = Image.fromarray(img.astype(np.uint8))
    #     # 保存图像到文件
    #     img.save('./picture/image_'+str(i+1)+'.png')  # 将图像保存为文件











# def prior_likelihood(z, sigma):
#     """The likelihood of a Gaussian distribution with mean zero and
#         standard deviation sigma."""
#     shape = z.shape
#     N = np.prod(shape[1:])
#     return -N / 2. * torch.log(2 * np.pi * sigma ** 2) - torch.sum(z ** 2, dim=(1, 2, 3)) / (2 * sigma ** 2)
#
#
# def ode_likelihood(x,
#                    score_model,
#                    marginal_prob_std,
#                    diffusion_coeff,
#                    batch_size=64,
#                    device='cuda',
#                    eps=1e-5):
#     """Compute the likelihood with probability flow ODE.
#
#     Args:
#       x: Input data.
#       score_model: A PyTorch model representing the score-based model.
#       marginal_prob_std: A function that gives the standard deviation of the
#         perturbation kernel.
#       diffusion_coeff: A function that gives the diffusion coefficient of the
#         forward SDE.
#       batch_size: The batch size. Equals to the leading dimension of `x`.
#       device: 'cuda' for evaluation on GPUs, and 'cpu' for evaluation on CPUs.
#       eps: A `float` number. The smallest time step for numerical stability.
#     Returns:
#       z: The latent code for `x`.
#       bpd: The log-likelihoods in bits/dim.
#     """
#
#     # Draw the random Gaussian sample for Skilling-Hutchinson's estimator.
#     epsilon = torch.randn_like(x)
#
#     def divergence_eval(sample, time_steps, epsilon):
#         """Compute the divergence of the score-based model with Skilling-Hutchinson."""
#         with torch.enable_grad():
#             sample.requires_grad_(True)
#             score_e = torch.sum(score_model(sample, time_steps) * epsilon)
#             grad_score_e = torch.autograd.grad(score_e, sample)[0]
#         return torch.sum(grad_score_e * epsilon, dim=(1, 2, 3))
#
#     shape = x.shape
#
#     def score_eval_wrapper(sample, time_steps):
#         """A wrapper for evaluating the score-based model for the black-box ODE solver."""
#         sample = torch.tensor(sample, device=device, dtype=torch.float32).reshape(shape)
#         time_steps = torch.tensor(time_steps, device=device, dtype=torch.float32).reshape((sample.shape[0],))
#         with torch.no_grad():
#             score = score_model(sample, time_steps)
#         return score.cpu().numpy().reshape((-1,)).astype(np.float64)
#
#     def divergence_eval_wrapper(sample, time_steps):
#         """A wrapper for evaluating the divergence of score for the black-box ODE solver."""
#         with torch.no_grad():
#             # Obtain x(t) by solving the probability flow ODE.
#             sample = torch.tensor(sample, device=device, dtype=torch.float32).reshape(shape)
#             time_steps = torch.tensor(time_steps, device=device, dtype=torch.float32).reshape((sample.shape[0],))
#             # Compute likelihood.
#             div = divergence_eval(sample, time_steps, epsilon)
#             return div.cpu().numpy().reshape((-1,)).astype(np.float64)
#
#     def ode_func(t, x):
#         """The ODE function for the black-box solver."""
#         time_steps = np.ones((shape[0],)) * t
#         sample = x[:-shape[0]]
#         logp = x[-shape[0]:]
#         g = diffusion_coeff(torch.tensor(t)).cpu().numpy()
#         sample_grad = -0.5 * g ** 2 * score_eval_wrapper(sample, time_steps)
#         logp_grad = -0.5 * g ** 2 * divergence_eval_wrapper(sample, time_steps)
#         return np.concatenate([sample_grad, logp_grad], axis=0)
#
#     init = np.concatenate([x.cpu().numpy().reshape((-1,)), np.zeros((shape[0],))], axis=0)
#     # Black-box ODE solver
#     res = integrate.solve_ivp(ode_func, (eps, 1.), init, rtol=1e-5, atol=1e-5, method='RK45')
#     zp = torch.tensor(res.y[:, -1], device=device)
#     z = zp[:-shape[0]].reshape(shape)
#     delta_logp = zp[-shape[0]:].reshape(shape[0])
#     sigma_max = marginal_prob_std(1.)
#     prior_logp = prior_likelihood(z, sigma_max)
#     bpd = -(prior_logp + delta_logp) / np.log(2)
#     N = np.prod(shape[1:])
#     bpd = bpd / N + 8.
#     return z, bpd
#
#
# batch_size = 32  # @param {'type':'integer'}
#
# dataset = MNIST('.', train=False, transform=transforms.ToTensor(), download=True)
# data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
#
# ckpt = torch.load('ckpt.pth', map_location=device)
# score_model.load_state_dict(ckpt)
#
# all_bpds = 0.
# all_items = 0
# try:
#     tqdm_data = tqdm.notebook.tqdm(data_loader)
#     for x, _ in tqdm_data:
#         x = x.to(device)
#         # uniform dequantization
#         x = (x * 255. + torch.rand_like(x)) / 256.
#         _, bpd = ode_likelihood(x, score_model, marginal_prob_std_fn,
#                                 diffusion_coeff_fn,
#                                 x.shape[0], device=device, eps=1e-5)
#         all_bpds += bpd.sum()
#         all_items += bpd.shape[0]
#         tqdm_data.set_description("Average bits/dim: {:5f}".format(all_bpds / all_items))
#
# except KeyboardInterrupt:
#     # Remove the error message when interuptted by keyboard or GUI.
#     pass