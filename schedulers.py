class DecayingCosineAnnealingLR(CosineAnnealingLR):
  def get_lr(self) -> float:
    lr = super().get_lr()
    decay_factor = np.exp([-0.05 * (self.last_epoch // (2 * self.T_max))])[0]
    lr[0] *= decay_factor
    return lr