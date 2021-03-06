import networks
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np

class DivCo_DRIT(nn.Module):
  def __init__(self, opts):
    super(DivCo_DRIT, self).__init__()

    # parameters
    lr = 0.0001
    lr_dcontent = lr / 2.5
    self.opt = opts
    self.nz = 8
    self.half_size = self.opt.batch_size // 2

    # discriminators
    if opts.dis_scale > 1:
      self.disA = networks.MultiScaleDis(opts.input_dim_a, opts.dis_scale, norm=opts.dis_norm, sn=opts.dis_spectral_norm)
      self.disB = networks.MultiScaleDis(opts.input_dim_b, opts.dis_scale, norm=opts.dis_norm, sn=opts.dis_spectral_norm)
      self.disA2 = networks.MultiScaleDis(opts.input_dim_a, opts.dis_scale, norm=opts.dis_norm, sn=opts.dis_spectral_norm)
      self.disB2 = networks.MultiScaleDis(opts.input_dim_b, opts.dis_scale, norm=opts.dis_norm, sn=opts.dis_spectral_norm)
    else:
      self.disA = networks.Dis(opts.input_dim_a, norm=opts.dis_norm, sn=opts.dis_spectral_norm)
      self.disB = networks.Dis(opts.input_dim_b, norm=opts.dis_norm, sn=opts.dis_spectral_norm)
      self.disA2 = networks.Dis(opts.input_dim_a, norm=opts.dis_norm, sn=opts.dis_spectral_norm)
      self.disB2 = networks.Dis(opts.input_dim_b, norm=opts.dis_norm, sn=opts.dis_spectral_norm)
    self.disContent = networks.Dis_content()

    # encoders
    self.enc_c = networks.E_content(opts.input_dim_a, opts.input_dim_b)
    if self.opt.concat:
      self.enc_a = networks.E_attr_concat(opts.input_dim_a, opts.input_dim_b, self.nz, \
          norm_layer=None, nl_layer=networks.get_non_linearity(layer_type='lrelu'))
    else:
      self.enc_a = networks.E_attr(opts.input_dim_a, opts.input_dim_b, self.nz)

    # generator
    if self.opt.concat:
      self.gen = networks.G_concat(opts.input_dim_a, opts.input_dim_b, nz=self.nz)
    else:
      self.gen = networks.G(opts.input_dim_a, opts.input_dim_b, nz=self.nz)

    # optimizers
    self.disA_opt = torch.optim.Adam(self.disA.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
    self.disB_opt = torch.optim.Adam(self.disB.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
    self.disA2_opt = torch.optim.Adam(self.disA2.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
    self.disB2_opt = torch.optim.Adam(self.disB2.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
    self.disContent_opt = torch.optim.Adam(self.disContent.parameters(), lr=lr_dcontent, betas=(0.5, 0.999), weight_decay=0.0001)
    self.enc_c_opt = torch.optim.Adam(self.enc_c.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
    self.enc_a_opt = torch.optim.Adam(self.enc_a.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
    self.gen_opt = torch.optim.Adam(self.gen.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)

    # Setup the loss function for training
    self.criterionL1 = torch.nn.L1Loss()
    self.cross_entropy_loss = torch.nn.CrossEntropyLoss()

  def initialize(self):
    self.disA.apply(networks.gaussian_weights_init)
    self.disB.apply(networks.gaussian_weights_init)
    self.disA2.apply(networks.gaussian_weights_init)
    self.disB2.apply(networks.gaussian_weights_init)
    self.disContent.apply(networks.gaussian_weights_init)
    self.gen.apply(networks.gaussian_weights_init)
    self.enc_c.apply(networks.gaussian_weights_init)
    self.enc_a.apply(networks.gaussian_weights_init)

  def set_scheduler(self, opts, last_ep=0):
    self.disA_sch = networks.get_scheduler(self.disA_opt, opts, last_ep)
    self.disB_sch = networks.get_scheduler(self.disB_opt, opts, last_ep)
    self.disA2_sch = networks.get_scheduler(self.disA2_opt, opts, last_ep)
    self.disB2_sch = networks.get_scheduler(self.disB2_opt, opts, last_ep)
    self.disContent_sch = networks.get_scheduler(self.disContent_opt, opts, last_ep)
    self.enc_c_sch = networks.get_scheduler(self.enc_c_opt, opts, last_ep)
    self.enc_a_sch = networks.get_scheduler(self.enc_a_opt, opts, last_ep)
    self.gen_sch = networks.get_scheduler(self.gen_opt, opts, last_ep)

  def setgpu(self, gpu):
    self.gpu = gpu
    self.gpu = torch.cuda.device_count()
    print(self.gpu)
    if self.gpu > 1:
      self.disA = nn.DataParallel(self.disA)
      self.disB = nn.DataParallel(self.disB)
      self.disA2 = nn.DataParallel(self.disA2)
      self.disB2 = nn.DataParallel(self.disB2)
      self.disContent = nn.DataParallel(self.disContent)
      self.enc_c = nn.DataParallel(self.enc_c)
      self.enc_a = nn.DataParallel(self.enc_a)
      self.gen = nn.DataParallel(self.gen)
    self.disA.cuda()
    self.disB.cuda()
    self.disA2.cuda()
    self.disB2.cuda()
    self.disContent.cuda()
    self.enc_c.cuda()
    self.enc_a.cuda()
    self.gen.cuda()

  def get_z_random(self, batchSize, nz, random_type='gauss'):
    z = torch.cuda.FloatTensor(batchSize, nz)
    z.copy_(torch.randn(batchSize, nz))
    return z

  def forward(self):
    # input images
    real_A = self.input_A
    real_B = self.input_B
    self.real_A_encoded = real_A[0:self.half_size]
    self.real_A_random = real_A[self.half_size:]
    self.real_B_encoded = real_B[0:self.half_size]
    self.real_B_random = real_B[self.half_size:]

    # get encoded z_c
    self.z_content_a, self.z_content_b = self.enc_c.forward(self.real_A_encoded, self.real_B_encoded)

    # get encoded z_a
    if self.opt.concat:
      self.mu_a, self.logvar_a, self.mu_b, self.logvar_b = self.enc_a.forward(self.real_A_encoded, self.real_B_encoded)
      std_a = self.logvar_a.mul(0.5).exp_()
      eps_a = self.get_z_random(std_a.size(0), std_a.size(1), 'gauss')
      self.z_attr_a = eps_a.mul(std_a).add_(self.mu_a)
      std_b = self.logvar_b.mul(0.5).exp_()
      eps_b = self.get_z_random(std_b.size(0), std_b.size(1), 'gauss')
      self.z_attr_b = eps_b.mul(std_b).add_(self.mu_b)
    else:
      self.z_attr_a, self.z_attr_b = self.enc_a.forward(self.real_A_encoded, self.real_B_encoded)

    # get two random codes, z_random2 is used for the calculation for Mode seeking loss
    self.z_random = self.get_z_random(self.real_A_encoded.size(0), self.nz, 'guass')

    # first cross translation
    input_content_forA = torch.cat([self.z_content_b, self.z_content_a, self.z_content_b], 0)
    input_content_forB = torch.cat([self.z_content_a, self.z_content_b, self.z_content_a], 0)
    input_attr_forA = torch.cat([self.z_attr_a, self.z_attr_a, self.z_random], 0)
    input_attr_forB = torch.cat([self.z_attr_b, self.z_attr_b, self.z_random], 0)
    output_fakeA = self.gen(input_content_forA, input_attr_forA, True)
    output_fakeB = self.gen(input_content_forB, input_attr_forB, False)
    self.fake_A_encoded, self.fake_AA_encoded, self.fake_A_random = torch.split(output_fakeA, self.z_content_a.size(0), dim=0)
    self.fake_B_encoded, self.fake_BB_encoded, self.fake_B_random = torch.split(output_fakeB, self.z_content_a.size(0), dim=0)

    # get reconstructed encoded z_c
    self.z_content_recon_b, self.z_content_recon_a = self.enc_c.forward(self.fake_A_encoded, self.fake_B_encoded)

    # get reconstructed encoded z_a
    if self.opt.concat:
      self.mu_recon_a, self.logvar_recon_a, self.mu_recon_b, self.logvar_recon_b = self.enc_a.forward(self.fake_A_encoded, self.fake_B_encoded)
      std_a = self.logvar_recon_a.mul(0.5).exp_()
      eps_a = self.get_z_random(std_a.size(0), std_a.size(1), 'gauss')
      self.z_attr_recon_a = eps_a.mul(std_a).add_(self.mu_recon_a)
      std_b = self.logvar_recon_b.mul(0.5).exp_()
      eps_b = self.get_z_random(std_b.size(0), std_b.size(1), 'gauss')
      self.z_attr_recon_b = eps_b.mul(std_b).add_(self.mu_recon_b)
    else:
      self.z_attr_recon_a, self.z_attr_recon_b = self.enc_a.forward(self.fake_A_encoded, self.fake_B_encoded)

    # second cross translation
    self.fake_A_recon = self.gen(self.z_content_recon_a, self.z_attr_recon_a, for_a=True)
    self.fake_B_recon = self.gen(self.z_content_recon_b, self.z_attr_recon_b, for_a=False)

    # for display
    if not self.opt.no_display_img:
      self.image_display = torch.cat((self.real_A_encoded[0:1].detach().cpu(), self.fake_B_encoded[0:1].detach().cpu(), self.fake_B_random[0:1].detach().cpu(), \
                                      self.fake_AA_encoded[0:1].detach().cpu(), self.fake_A_recon[0:1].detach().cpu(), \
                                      self.real_B_encoded[0:1].detach().cpu(), self.fake_A_encoded[0:1].detach().cpu(), self.fake_A_random[0:1].detach().cpu(), \
                                      self.fake_BB_encoded[0:1].detach().cpu(), self.fake_B_recon[0:1].detach().cpu()), dim=0)

  def latent_augmented_sampling(self):
    query = self.get_z_random(self.real_A_encoded.size(0), self.nz, 'gauss')
    pos = torch.cuda.FloatTensor(query.shape).uniform_(-self.opt.radius, self.opt.radius).add_(query)
    negs = []
    for k in range(self.opt.num_negative):
      neg = self.get_z_random(self.real_A_encoded.size(0), self.nz, 'gauss')
      while (neg-query).abs().min() < self.opt.radius:
        neg = self.get_z_random(self.real_A_encoded.size(0), self.nz, 'gauss')
      negs.append(neg)
    return query, pos, negs

  def forward_alone(self):
    # get encoded z_c
    self.z_content_a, self.z_content_b = self.enc_c.forward(self.real_A_encoded, self.real_B_encoded)
    # get random code
    query, pos, negs = self.latent_augmented_sampling()
    self.z_random = torch.cat([query, pos] + negs, 0)
    # random forward
    input_content_forA = self.z_content_b.repeat([self.opt.num_negative+2]+list(np.ones(len(self.z_content_b.shape[1:]), dtype=np.uint8)) )
    input_content_forB = self.z_content_a.repeat([self.opt.num_negative+2]+list(np.ones(len(self.z_content_a.shape[1:]), dtype=np.uint8)) )
    self.fake_A_random = self.gen(input_content_forA, self.z_random, for_a=True)
    self.fake_B_random = self.gen(input_content_forB, self.z_random, for_a=False)
    # extracting features
    if self.opt.concat:
      self.mu2_a, _, self.mu2_b, _ = self.enc_a.forward(self.fake_A_random, self.fake_B_random)
    else:
      self.z_attr_random_a, self.z_attr_random_b = self.enc_a.forward(self.fake_A_random, self.fake_B_random)

  def forward_content(self):
    self.real_A_encoded = self.input_A[0:self.half_size]
    self.real_B_encoded = self.input_B[0:self.half_size]
    # get encoded z_c
    self.z_content_a, self.z_content_b = self.enc_c.forward(self.real_A_encoded, self.real_B_encoded)

  def update_D_content(self, image_a, image_b):
    self.input_A = image_a
    self.input_B = image_b
    self.forward_content()
    self.disContent_opt.zero_grad()
    loss_D_Content = self.backward_contentD(self.z_content_a, self.z_content_b)
    self.disContent_loss = loss_D_Content.item()
    nn.utils.clip_grad_norm_(self.disContent.parameters(), 5)
    self.disContent_opt.step()

  def update_D(self, image_a, image_b):
    self.input_A = image_a
    self.input_B = image_b
    self.forward()

    # update disA
    self.disA_opt.zero_grad()
    loss_D1_A = self.backward_D(self.disA, self.real_A_encoded, self.fake_A_encoded)
    self.disA_loss = loss_D1_A.item()
    self.disA_opt.step()

    # update disA2
    self.disA2_opt.zero_grad()
    loss_D2_A = self.backward_D(self.disA2, self.real_A_random, self.fake_A_random)
    self.disA2_loss = loss_D2_A.item()
    self.disA2_opt.step()

    # update disB
    self.disB_opt.zero_grad()
    loss_D1_B = self.backward_D(self.disB, self.real_B_encoded, self.fake_B_encoded)
    self.disB_loss = loss_D1_B.item()
    self.disB_opt.step()

    # update disB2
    self.disB2_opt.zero_grad()
    loss_D2_B = self.backward_D(self.disB2, self.real_B_random, self.fake_B_random)
    self.disB2_loss = loss_D2_B.item()
    self.disB2_opt.step()

    # update disContent
    self.disContent_opt.zero_grad()
    loss_D_Content = self.backward_contentD(self.z_content_a, self.z_content_b)
    self.disContent_loss = loss_D_Content.item()
    nn.utils.clip_grad_norm_(self.disContent.parameters(), 5)
    self.disContent_opt.step()

  def backward_D(self, netD, real, fake):
    pred_fake = netD.forward(fake.detach())
    pred_real = netD.forward(real)
    loss_D = 0
    for it, (out_a, out_b) in enumerate(zip(pred_fake, pred_real)):
      out_fake = nn.functional.sigmoid(out_a)
      out_real = nn.functional.sigmoid(out_b)
      all0 = torch.zeros_like(out_fake).cuda()
      all1 = torch.ones_like(out_real).cuda()
      ad_fake_loss = nn.functional.binary_cross_entropy(out_fake, all0)
      ad_true_loss = nn.functional.binary_cross_entropy(out_real, all1)
      loss_D += ad_true_loss + ad_fake_loss
    loss_D.backward()
    return loss_D

  def backward_contentD(self, imageA, imageB):
    pred_fake = self.disContent.forward(imageA.detach())
    pred_real = self.disContent.forward(imageB.detach())
    for it, (out_a, out_b) in enumerate(zip(pred_fake, pred_real)):
      out_fake = nn.functional.sigmoid(out_a)
      out_real = nn.functional.sigmoid(out_b)
      all1 = torch.ones((out_real.size(0))).cuda()
      all0 = torch.zeros((out_fake.size(0))).cuda()
      ad_true_loss = nn.functional.binary_cross_entropy(out_real, all1)
      ad_fake_loss = nn.functional.binary_cross_entropy(out_fake, all0)
    loss_D = ad_true_loss + ad_fake_loss
    loss_D.backward()
    return loss_D

  def update_EG(self):
    # update G, Ec, Ea
    self.enc_c_opt.zero_grad()
    self.enc_a_opt.zero_grad()
    self.gen_opt.zero_grad()
    self.backward_EG()
    self.enc_c_opt.step()
    self.enc_a_opt.step()
    self.gen_opt.step()

    # update G, Ec
    self.forward_alone()
    self.enc_c_opt.zero_grad()
    self.gen_opt.zero_grad()
    self.backward_G_alone()
    self.enc_c_opt.step()
    self.gen_opt.step()

  def backward_EG(self):
    # content Ladv for generator
    loss_G_GAN_Acontent = self.backward_G_GAN_content(self.z_content_a)
    loss_G_GAN_Bcontent = self.backward_G_GAN_content(self.z_content_b)

    # Ladv for generator
    loss_G_GAN_A = self.backward_G_GAN(self.fake_A_encoded, self.disA)
    loss_G_GAN_B = self.backward_G_GAN(self.fake_B_encoded, self.disB)

    # KL loss - z_a
    if self.opt.concat:
      kl_element_a = self.mu_a.pow(2).add_(self.logvar_a.exp()).mul_(-1).add_(1).add_(self.logvar_a)
      loss_kl_za_a = torch.sum(kl_element_a).mul_(-0.5) * 0.01
      kl_element_b = self.mu_b.pow(2).add_(self.logvar_b.exp()).mul_(-1).add_(1).add_(self.logvar_b)
      loss_kl_za_b = torch.sum(kl_element_b).mul_(-0.5) * 0.01
    else:
      loss_kl_za_a = self._compute_kl(self.z_attr_a) * 0.01
      loss_kl_za_b = self._compute_kl(self.z_attr_b) * 0.01

    # KL loss - z_c
    loss_kl_zc_a = self._compute_kl(self.z_content_a) * 0.01
    loss_kl_zc_b = self._compute_kl(self.z_content_b) * 0.01

    # cross cycle consistency loss
    loss_G_L1_A = self.criterionL1(self.fake_A_recon, self.real_A_encoded) * 10
    loss_G_L1_B = self.criterionL1(self.fake_B_recon, self.real_B_encoded) * 10
    loss_G_L1_AA = self.criterionL1(self.fake_AA_encoded, self.real_A_encoded) * 10
    loss_G_L1_BB = self.criterionL1(self.fake_BB_encoded, self.real_B_encoded) * 10

    loss_G = loss_G_GAN_A + loss_G_GAN_B + \
             loss_G_GAN_Acontent + loss_G_GAN_Bcontent + \
             loss_G_L1_AA + loss_G_L1_BB + \
             loss_G_L1_A + loss_G_L1_B + \
             loss_kl_zc_a + loss_kl_zc_b + \
             loss_kl_za_a + loss_kl_za_b

    loss_G.backward()

    self.gan_loss_a = loss_G_GAN_A.item()
    self.gan_loss_b = loss_G_GAN_B.item()
    self.gan_loss_acontent = loss_G_GAN_Acontent.item()
    self.gan_loss_bcontent = loss_G_GAN_Bcontent.item()
    self.kl_loss_za_a = loss_kl_za_a.item()
    self.kl_loss_za_b = loss_kl_za_b.item()
    self.kl_loss_zc_a = loss_kl_zc_a.item()
    self.kl_loss_zc_b = loss_kl_zc_b.item()
    self.l1_recon_A_loss = loss_G_L1_A.item()
    self.l1_recon_B_loss = loss_G_L1_B.item()
    self.l1_recon_AA_loss = loss_G_L1_AA.item()
    self.l1_recon_BB_loss = loss_G_L1_BB.item()
    self.G_loss = loss_G.item()

  def backward_G_GAN_content(self, data):
    outs = self.disContent.forward(data)
    for out in outs:
      outputs_fake = nn.functional.sigmoid(out)
      all_half = 0.5*torch.ones((outputs_fake.size(0))).cuda()
      ad_loss = nn.functional.binary_cross_entropy(outputs_fake, all_half)
    return ad_loss

  def backward_G_GAN(self, fake, netD=None):
    outs_fake = netD.forward(fake)
    loss_G = 0
    for out_a in outs_fake:
      outputs_fake = nn.functional.sigmoid(out_a)
      all_ones = torch.ones_like(outputs_fake).cuda()
      loss_G += nn.functional.binary_cross_entropy(outputs_fake, all_ones)
    return loss_G

  def backward_G_alone(self):
    # Ladv for generator
    loss_G_GAN2_A = self.backward_G_GAN(self.fake_A_random[0:1], self.disA2)
    loss_G_GAN2_A2 = self.backward_G_GAN(self.fake_A_random[2:3], self.disA2)
    loss_G_GAN2_B = self.backward_G_GAN(self.fake_B_random[0:1], self.disB2)
    loss_G_GAN2_B2 = self.backward_G_GAN(self.fake_B_random[2:3], self.disB2)

    loss_contra_a, loss_contra_b = 0.0, 0.0
    for i in range(self.half_size):
      if self.opt.concat:
        mu2_a = self.mu2_a[i:self.mu2_a.shape[0]:self.half_size]
        mu2_b = self.mu2_b[i:self.mu2_b.shape[0]:self.half_size]
      else:
        mu2_a = self.z_attr_random_a[i:self.z_attr_random_a.shape[0]:self.half_size]
        mu2_b = self.z_attr_random_b[i:self.z_attr_random_b.shape[0]:self.half_size]
      if self.opt.featnorm:
        mu2_a = mu2_a / torch.norm(mu2_a, p=2, dim=1, keepdim=True)
        mu2_b = mu2_b / torch.norm(mu2_b, p=2, dim=1, keepdim=True)
      loss_contra_a += self.compute_contrastive_loss(mu2_a[0:1], mu2_a[1:]) 
      loss_contra_b += self.compute_contrastive_loss(mu2_b[0:1], mu2_b[1:])

    # latent regression loss
    if self.opt.with_latent_reg:
      if self.opt.concat:
        loss_z_L1_a = torch.mean(torch.abs(self.mu2_a - torch.cat(self.z_random, 0) )) * 10
        loss_z_L1_b = torch.mean(torch.abs(self.mu2_b - torch.cat(self.z_random, 0) )) * 10
      else:
        loss_z_L1_a = torch.mean(torch.abs(self.z_attr_random_a - torch.cat(self.z_random, 0) )) * 10
        loss_z_L1_b = torch.mean(torch.abs(self.z_attr_random_b - torch.cat(self.z_random, 0) )) * 10

    loss_z_L1 = loss_G_GAN2_A + loss_G_GAN2_A2 + loss_G_GAN2_B + loss_G_GAN2_B2 + self.opt.lambda_contra*(loss_contra_a + loss_contra_b)
    if self.opt.with_latent_reg:
      loss_z_L1 += (loss_z_L1_a + loss_z_L1_b)
    loss_z_L1.backward()

    self.gan2_loss_a = loss_G_GAN2_A.item() + loss_G_GAN2_A2.item()
    self.gan2_loss_b = loss_G_GAN2_B.item() + loss_G_GAN2_B2.item()
    self.contra_loss_a = loss_contra_a.item()
    self.contra_loss_b = loss_contra_b.item()
    if self.opt.with_latent_reg:
      self.l1_recon_z_loss_a = loss_z_L1_a.item()
      self.l1_recon_z_loss_b = loss_z_L1_b.item()

  def update_lr(self):
    self.disA_sch.step()
    self.disB_sch.step()
    self.disA2_sch.step()
    self.disB2_sch.step()
    self.disContent_sch.step()
    self.enc_c_sch.step()
    self.enc_a_sch.step()
    self.gen_sch.step()

  def compute_contrastive_loss(self, feat_q, feat_k):
    out = torch.mm(feat_q, feat_k.transpose(1,0)) / self.opt.tau
    loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long,
                                                    device=feat_q.device))
    return loss

  def normalize_image(self, x):
    return x[:,0:3,:,:]

  def _compute_kl(self, mu):
    mu_2 = torch.pow(mu, 2)
    encoding_loss = torch.mean(mu_2)
    return encoding_loss

  def resume(self, model_dir, train=True):
    checkpoint = torch.load(model_dir)
    # weight
    if train:
      self.disA.load_state_dict(checkpoint['disA'])
      self.disA2.load_state_dict(checkpoint['disA2'])
      self.disB.load_state_dict(checkpoint['disB'])
      self.disB2.load_state_dict(checkpoint['disB2'])
      self.disContent.load_state_dict(checkpoint['disContent'])
    self.enc_c.load_state_dict(checkpoint['enc_c'])
    self.enc_a.load_state_dict(checkpoint['enc_a'])
    self.gen.load_state_dict(checkpoint['gen'])
    # optimizer
    if train:
      self.disA_opt.load_state_dict(checkpoint['disA_opt'])
      self.disA2_opt.load_state_dict(checkpoint['disA2_opt'])
      self.disB_opt.load_state_dict(checkpoint['disB_opt'])
      self.disB2_opt.load_state_dict(checkpoint['disB2_opt'])
      self.disContent_opt.load_state_dict(checkpoint['disContent_opt'])
      self.enc_c_opt.load_state_dict(checkpoint['enc_c_opt'])
      self.enc_a_opt.load_state_dict(checkpoint['enc_a_opt'])
      self.gen_opt.load_state_dict(checkpoint['gen_opt'])
    return checkpoint['ep'], checkpoint['total_it']

  def save(self, filename, ep, total_it):
    if self.gpu > 1:
      state = {
             'disA': self.disA.module.state_dict(),
             'disA2': self.disA2.module.state_dict(),
             'disB': self.disB.module.state_dict(),
             'disB2': self.disB2.module.state_dict(),
             'disContent': self.disContent.module.state_dict(),
             'enc_c': self.enc_c.module.state_dict(),
             'enc_a': self.enc_a.module.state_dict(),
             'gen': self.gen.module.state_dict(),
             'disA_opt': self.disA_opt.state_dict(),
             'disA2_opt': self.disA2_opt.state_dict(),
             'disB_opt': self.disB_opt.state_dict(),
             'disB2_opt': self.disB2_opt.state_dict(),
             'disContent_opt': self.disContent_opt.state_dict(),
             'enc_c_opt': self.enc_c_opt.state_dict(),
             'enc_a_opt': self.enc_a_opt.state_dict(),
             'gen_opt': self.gen_opt.state_dict(),
             'ep': ep,
             'total_it': total_it
              }
    else:
      state = {
             'disA': self.disA.state_dict(),
             'disA2': self.disA2.state_dict(),
             'disB': self.disB.state_dict(),
             'disB2': self.disB2.state_dict(),
             'disContent': self.disContent.state_dict(),
             'enc_c': self.enc_c.state_dict(),
             'enc_a': self.enc_a.state_dict(),
             'gen': self.gen.state_dict(),
             'disA_opt': self.disA_opt.state_dict(),
             'disA2_opt': self.disA2_opt.state_dict(),
             'disB_opt': self.disB_opt.state_dict(),
             'disB2_opt': self.disB2_opt.state_dict(),
             'disContent_opt': self.disContent_opt.state_dict(),
             'enc_c_opt': self.enc_c_opt.state_dict(),
             'enc_a_opt': self.enc_a_opt.state_dict(),
             'gen_opt': self.gen_opt.state_dict(),
             'ep': ep,
             'total_it': total_it
              }
    torch.save(state, filename)
    return

  def assemble_outputs(self):
    images_a = self.normalize_image(self.real_A_encoded).detach()
    images_b = self.normalize_image(self.real_B_encoded).detach()
    images_a1 = self.normalize_image(self.fake_A_encoded).detach()
    images_a2 = self.normalize_image(self.fake_A_random[0:1]).detach()
    images_a2x = self.normalize_image(self.fake_A_random[1:2]).detach()
    images_a3 = self.normalize_image(self.fake_A_random[2:3]).detach()
    images_a4 = self.normalize_image(self.fake_A_recon).detach()
    images_a5 = self.normalize_image(self.fake_AA_encoded).detach()
    images_b1 = self.normalize_image(self.fake_B_encoded).detach()
    images_b2 = self.normalize_image(self.fake_B_random[0:1]).detach()
    images_b2x = self.normalize_image(self.fake_B_random[1:2]).detach()
    images_b3 = self.normalize_image(self.fake_B_random[2:3]).detach()
    images_b4 = self.normalize_image(self.fake_B_recon).detach()
    images_b5 = self.normalize_image(self.fake_BB_encoded).detach()
    row1 = torch.cat((images_a[0:1, ::], images_b1[0:1, ::], images_b2[0:1, ::], images_b2x[0:1, ::], images_b3[0:1, ::], images_a5[0:1, ::], images_a4[0:1, ::]),3)
    row2 = torch.cat((images_b[0:1, ::], images_a1[0:1, ::], images_a2[0:1, ::], images_a2x[0:1, ::], images_a3[0:1, ::], images_b5[0:1, ::], images_b4[0:1, ::]),3)
    return torch.cat((row1,row2),2)

  def test_forward(self, image, a2b=True):
    self.z_random = self.get_z_random(image.size(0), self.nz, 'gauss')
    if a2b:
        self.z_content = self.enc_c.forward_a(image)
        output = self.gen.forward_b(self.z_content, self.z_random)
    else:
        self.z_content = self.enc_c.forward_b(image)
        output = self.gen.forward_a(self.z_content, self.z_random)
    return output

  def test_givenz_forward(self, image, latent_code, a2b=True):
    self.z_random = self.get_z_random(image.size(0), self.nz, 'gauss')
    self.z_random = torch.ones_like(self.z_random) * latent_code
    if a2b:
        self.z_content = self.enc_c.forward_a(image)
        output = self.gen.forward_b(self.z_content, self.z_random)
    else:
        self.z_content = self.enc_c.forward_b(image)
        output = self.gen.forward_a(self.z_content, self.z_random)
    return output

  def test_forward_transfer(self, image_a, image_b, a2b=True):
    self.z_content_a, self.z_content_b = self.enc_c.forward(image_a, image_b)
    if self.opt.concat:
      self.mu_a, self.logvar_a, self.mu_b, self.logvar_b = self.enc_a.forward(image_a, image_b)
      std_a = self.logvar_a.mul(0.5).exp_()
      eps = self.get_z_random(std_a.size(0), std_a.size(1), 'gauss')
      self.z_attr_a = eps.mul(std_a).add_(self.mu_a)
      std_b = self.logvar_b.mul(0.5).exp_()
      eps = self.get_z_random(std_a.size(0), std_a.size(1), 'gauss')
      self.z_attr_b = eps.mul(std_b).add_(self.mu_b)
    else:
      self.z_attr_a, self.z_attr_b = self.enc_a.forward(image_a, image_b)
    if a2b:
      output = self.gen.forward_b(self.z_content_a, self.z_attr_b)
    else:
      output = self.gen.forward_a(self.z_content_b, self.z_attr_a)
    return output
