
import torch
import pdb
from torch import nn
from torch.nn import functional as F
from einops import rearrange, repeat
from models.common.pos_embed import FeedForward

# ======================================
#   Global Relation Attention Module
# ======================================

class Global_Relation_Attention(nn.Module):
	def __init__(self, 
				in_channel,
				in_spatial, 
				use_spatial, 
				use_channel,
				cha_ratio, 
				spa_ratio, 
				down_ratio):
		super(Global_Relation_Attention, self).__init__()

		self.in_channel = in_channel
		self.in_spatial = in_spatial
		
		self.use_spatial = use_spatial

		# print ('Use_Spatial_Att: {};\tUse_Channel_Att: {}.'.format(self.use_spatial, self.use_channel))

		self.inter_channel = in_channel // cha_ratio  # 512//4 = 128
		self.inter_spatial = in_spatial // spa_ratio  # 60//8 = 7
		

		if self.use_spatial:
			self.gx_spatial = nn.Sequential(
				nn.Conv2d(in_channels=self.in_channel, out_channels=self.inter_channel,
						kernel_size=1, stride=1, padding=0, bias=False),
				nn.BatchNorm2d(self.inter_channel),
				nn.ReLU()
			)


		if self.use_spatial:
			self.gg_spatial = nn.Sequential(
				nn.Conv2d(in_channels=self.in_spatial * 2, out_channels=self.inter_spatial,
						kernel_size=1, stride=1, padding=0, bias=False),
				nn.BatchNorm2d(self.inter_spatial),
				nn.ReLU()
			)


		if self.use_spatial:
			num_channel_s = 1 + self.inter_spatial
			self.W_spatial = nn.Sequential(
				nn.Conv2d(in_channels=num_channel_s, out_channels=num_channel_s//down_ratio,
						kernel_size=1, stride=1, padding=0, bias=False),
				nn.BatchNorm2d(num_channel_s//down_ratio),
				nn.ReLU(),
				nn.Conv2d(in_channels=num_channel_s//down_ratio, out_channels=1,
						kernel_size=1, stride=1, padding=0, bias=False),
				nn.BatchNorm2d(1)
			)


		if self.use_spatial:
            # theta_spatial and phi_spatial:
            # embedding functions implemented by a 1 x 1 spatial convolutional layer 
            # followed by batch normalization (BN) and ReLU activation
			self.theta_spatial = nn.Sequential(
				nn.Conv2d(in_channels=self.in_channel, out_channels=self.inter_channel,
								kernel_size=1, stride=1, padding=0, bias=False),
				nn.BatchNorm2d(self.inter_channel),
				nn.ReLU()
			)
			self.phi_spatial = nn.Sequential(
				nn.Conv2d(in_channels=self.in_channel, out_channels=self.inter_channel,
							kernel_size=1, stride=1, padding=0, bias=False),
				nn.BatchNorm2d(self.inter_channel),
				nn.ReLU()
			)

				
	def forward(self, x):
		b, c, h, w = x.size()
		
		if self.use_spatial:
			# spatial attention
			theta_xs = self.theta_spatial(x)	
			phi_xs = self.phi_spatial(x)
			theta_xs = theta_xs.view(b, self.inter_channel, -1)   
			theta_xs = theta_xs.permute(0, 2, 1)                  
			phi_xs = phi_xs.view(b, self.inter_channel, -1)       # [b,8,hw]
			Glob_spa = torch.matmul(theta_xs, phi_xs)             
			Gs_in = Glob_spa.permute(0, 2, 1).view(b, h*w, h, w)  
			Gs_out = Glob_spa.view(b, h*w, h, w)                  
			Gs_joint = torch.cat((Gs_in, Gs_out), 1)              #  [b, hw + hw, h, w]
			# Relation Feature R
			Gs_joint = self.gg_spatial(Gs_joint)                  #  [b, 128, h, w]   denote the embedding functions for  the global relations 
		
			g_xs = self.gx_spatial(x)                      # [b,128,h,w]<---[b,c,h,w]  denote the embedding functions for the feature itself 
			g_xs = torch.mean(g_xs, dim=1, keepdim=True)   # [b,1,h,w]<---[b,128,h,w]  denotes global average pooling operation along the channel dimension to further reduce the dimension to be 1.
			ys = torch.cat((g_xs, Gs_joint), 1)            # [b,129,h,w]

			W_ys = self.W_spatial(ys)                      # [b,1,h,w] <--- [b,8(=129//8),h,w] <---[b,129,h,w]
			if not self.use_channel:
				out = F.sigmoid(W_ys.expand_as(x)) * x     # spatial attention value * feature
				return out
			else:
				x = F.sigmoid(W_ys.expand_as(x)) * x

		if self.use_channel:
			# channel attention
			xc = x.view(b, c, -1).permute(0, 2, 1).unsqueeze(-1)
			theta_xc = self.theta_channel(xc).squeeze(-1).permute(0, 2, 1)
			phi_xc = self.phi_channel(xc).squeeze(-1)
			Gc = torch.matmul(theta_xc, phi_xc)
			Gc_in = Gc.permute(0, 2, 1).unsqueeze(-1)
			Gc_out = Gc.unsqueeze(-1)
			Gc_joint = torch.cat((Gc_in, Gc_out), 1)
			Gc_joint = self.gg_channel(Gc_joint)

			g_xc = self.gx_channel(xc)
			g_xc = torch.mean(g_xc, dim=1, keepdim=True)
			yc = torch.cat((g_xc, Gc_joint), 1)

			W_yc = self.W_channel(yc).transpose(1, 2)
			out = F.sigmoid(W_yc) * x

			return out

class GRA_Module(nn.Module):
	def __init__(self, 
				in_channel,
				in_spatial, 
				use_spatial=True, 
				use_channel=True,
				cha_ratio=4, 
				spa_ratio=8, 
				down_ratio=8,
				d_model=512,
				d_ff=2048, 
				dropout=.1):
		super(GRA_Module, self).__init__()

		self.gratt= Global_Relation_Attention(in_channel, in_spatial, use_spatial, use_channel, cha_ratio, spa_ratio, down_ratio)
		self.pwff = FeedForward(d_model, d_ff, dropout)
	
	def forward(self, q):
		b, c, h, w = q.size()
		out = self.gratt(q)   #  b, d_model, h, w <---  b, d_model, h, w
		out = out.permute(0, 2, 3, 1).reshape(b, h*w, c) # b (h w) d_model <---  b, d_model, h, w
		out = self.pwff(out)  #  b (h w) d_model  <--- b (h w) d_model
		out = out.permute(0, 2, 1).reshape(b, c, h, w)  #   b, d_model, h, w  <---  b (h w) d_model 

		return out  # b, d_model, h, w 


class GlobFeatureNetwork(nn.Module):

	def __init__(self, 
				n_layers, 
				in_channel=512, 
				in_spatial = 60,  
				use_spatial = True, 
				use_channel = False,
				cha_ratio = 4,
				spa_ratio = 8,
				down_ratio = 8,
				d_in=1024, 
				d_model=512,  
				dropout=0.1):
		super().__init__()

		self.d_model = d_model 
		self.fc = nn.Linear(d_in, d_model) 
		self.dropout = nn.Dropout(p=dropout)
		self.layer_norm = nn.LayerNorm(d_model)

		self.layers = nn.ModuleList(
			[GRA_Module(in_channel,
				in_spatial, 
				use_spatial, 
				use_channel,
				cha_ratio, 
				spa_ratio, 
				down_ratio,
				d_model=d_model,
				d_ff=2048, 
				dropout=dropout) for _ in range(n_layers)])  # 3层

	# input:  [16, 1024, 6, 10]   glob_feat_dim                                                     
	# mask:  outputs['gri_mask'] = [b 1 1 (h w)]  torch.Size([16, 1, 1, 60])                        
	
	# [b, 4, (h w), 512]  d_model   <---   [16, 1024, 6, 10]   glob_feat_dim
	def forward(self, input, mask=None):

		b, c, h, w  = input.shape

		# check size of tensor
		expected_size = (b, 1024, 6, 10)
		if input.shape != expected_size:
			# raise ValueError(f"Tensor1111 size is not {expected_size}, actual size is {input.size()}")
			# [16, 1024, 6, 10] <---  b, c, h, w
			input  = F.interpolate(input, size=(6, 10), mode='nearest')
			# [16, 6, 10] <---  b h w 
			mask =  F.interpolate(mask[None].float(), size=input.shape[-2:]).to(torch.bool)[0]
			b, c, h, w  = input.shape
		
		# unsqueeze mask
		mask = repeat(mask, 'b h w -> b 1 1 (h w)')     # b 1 1 h w     [16, 1, 1, 60]
		
		d_model = self.d_model
		# b (h w) d_model <--- b, h*w ,c <--- b, h, w ,c <--- b, c, h, w
		out = self.layer_norm(self.dropout(F.relu(self.fc(input.permute(0, 2, 3, 1).reshape(b, h*w , c))))) # [b (h w) d_model]   1024---->512 
		out = out.reshape(b,h,w,d_model).permute(0, 3, 1, 2)  # b, d_model, h, w

		if input.shape != expected_size:
			raise ValueError(f"Tensor  size is not {expected_size}, actual size is {out.size()}")

		outs = []
		for layer in self.layers:
			expected_size = (b, 512, 6, 10)
			if out.shape != expected_size:
				raise ValueError(f" Tensor  size is not {expected_size}, actual size is {out.size()}")
			
			out = layer(out)   # b, d_model, h, w <---  b, d_model, h, w
			outs.append(out.permute(0, 2, 3, 1).reshape(b, h*w, d_model).unsqueeze(1))     # b 1 (h w) d_model <--- b,（h w）, d_model <--- b, h, w, d_model <--- b, d_model, h, w 

        # torch.Size([16, 3, 60, 512])
		outs = torch.cat(outs, 1)  #  [[b 1 (h w) d_model],......] ---> [b, 3, (h w), d_model]

		#    [b, 3, (h w), d_model]     [b 1 1 (h w)]
		return outs, mask 

if __name__ == "__main__":          
 

	in_channel=48        # c= 3*n
	in_spatial = 64*64
	use_spatial = True
	# use_channel = False
	# use_spatial = False
	use_channel = False

	cha_ratio = 4
	spa_ratio = 8
	down_ratio = 8


 
	img_path = '/gemini/data-1/COCO2014/train2014/COCO_train2014_000000004508.jpg'   
	image = Image.open(img_path)
 
	transform = transforms.Compose([
		transforms.Resize((64, 64)),   
		transforms.ToTensor(),   
	])
	tensor_image = transform(image).unsqueeze(0)   
	tensor_image = torch.cat([tensor_image] * 16, dim=1)  # c= 3*n
	tensor_image = torch.cat([tensor_image] * 16, dim=0)  # b= n
	print(tensor_image.shape)       # torch.Size([16, 48, 64, 64])
	module = GRA_Module(in_channel, in_spatial, use_spatial, use_channel, cha_ratio, spa_ratio, down_ratio)
	output = module(tensor_image)

 
	pooled_image = output[0,:3,:,:]  # torch.Size([3, 64, 64])
	print(pooled_image.shape)
 
	pooled_image = pooled_image.permute(1, 2, 0).detach().numpy()
 
	pooled_image = (pooled_image - pooled_image.min()) / (pooled_image.max() - pooled_image.min())
 
	pooled_image_pil = Image.fromarray((pooled_image * 255).astype(np.uint8))

 
	plt.imshow(pooled_image_pil)
	plt.axis('off')   
	plt.show()