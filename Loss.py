class MaskedDepthLoss(nn.Module):
  def __init__(self,mask_val=0):
    super(MaskedDepthLoss, self).__init__()
    self.mask_val = mask_val
  # masked L1 norm
  def forward(self, depth_out, depth_gt):
    loss = torch.abs(depth_gt - depth_out)
    if self.mask_val is not None:
      mask_indices=torch.where(depth_gt == self.mask_val)
      loss[mask_indices] = 0 
    return loss.mean()
