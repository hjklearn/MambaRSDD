import torch
import torch.nn as nn
import torch.nn.functional as F
from VMamba.classification.models.vmamba import vmamba_tiny_s1l8, vmamba_small_s2l15
# from MambaVision.mambavision.models.mamba_vision import mamba_vision_S
# import sys
# sys.path.append('/root/autodl-tmp/MambaRSDD/MambaRSDD/')

class ASPP(nn.Module):
    def __init__(self, in_channel=512, depth=256):
        super(ASPP, self).__init__()
        self.atrous_block1 = nn.Sequential(nn.Conv2d(in_channel, depth, 1, 1),
                                           nn.BatchNorm2d(depth),
                                           nn.ReLU())
        self.atrous_block6 = nn.Sequential(nn.Conv2d(in_channel, depth, 3, 1, padding=6, dilation=6),
                                           # nn.Conv2d(in_channel, depth, kernel_size=(1, 3), padding=(0, 6), dilation=(1, 6)),
                                           # nn.Conv2d(depth, depth, kernel_size=(3, 1), padding=(6, 0), dilation=(6, 1)),
                                           nn.BatchNorm2d(depth),
                                           nn.ReLU())
        self.atrous_block12 = nn.Sequential(nn.Conv2d(in_channel, depth, 3, 1, padding=12, dilation=12),
                                            # nn.Conv2d(in_channel, depth, kernel_size=(1, 3), padding=(0, 12), dilation=(1, 12)),
                                            # nn.Conv2d(depth, depth, kernel_size=(3, 1), padding=(12, 0), dilation=(12, 1)),
                                            nn.BatchNorm2d(depth),
                                            nn.ReLU())
        self.atrous_block18 = nn.Sequential(nn.Conv2d(in_channel, depth, 3, 1, padding=18, dilation=18),
                                            # nn.Conv2d(in_channel, depth, kernel_size=(1, 3), padding=(0, 18), dilation=(1, 18)),
                                            # nn.Conv2d(depth, depth, kernel_size=(3, 1), padding=(18, 0), dilation=(18, 1)),
                                            nn.BatchNorm2d(depth),
                                            nn.ReLU())
        self.conv_1x1_output = nn.Sequential(nn.Conv2d(depth * 4, 4*depth, 1),
                                             nn.BatchNorm2d(4*depth),
                                             nn.ReLU())

    def forward(self, x):

        atrous_block1 = self.atrous_block1(x)
        atrous_block6 = self.atrous_block6(x)
        atrous_block12 = self.atrous_block12(x)
        atrous_block18 = self.atrous_block18(x)

        net = self.conv_1x1_output(torch.cat([atrous_block1, atrous_block6,
                                              atrous_block12, atrous_block18], dim=1))
        net = net + x

        return net

    


class Decoer(nn.Module):
    def __init__(self):
        super().__init__()

        # self.aspp5 = ASPP(in_channel=768, depth=192)
        self.aspp4 = ASPP(in_channel=768, depth=192)
        self.aspp3 = ASPP(in_channel=384, depth=96)
        self.aspp2 = ASPP(in_channel=192, depth=48)
        self.aspp1 = ASPP(in_channel=96, depth=24)


        self.up2 = nn.Upsample(scale_factor=2, align_corners=False, mode='bilinear')
        # self.up4 = nn.Sequential(nn.Conv2d(768, 192, 1),
        #                          nn.BatchNorm2d(192),
        #                          nn.ReLU(),
        #                          nn.Upsample(scale_factor=4, align_corners=False, mode='bilinear'))
        # self.up8 = nn.Sequential(nn.Conv2d(768, 96, 1),
        #                          nn.BatchNorm2d(96),
        #                          nn.ReLU(),
        #                          nn.Upsample(scale_factor=8, align_corners=False, mode='bilinear'))

        self.conv4 = nn.Sequential(nn.Conv2d(1536, 768, 1),
                                   nn.BatchNorm2d(768),
                                   nn.ReLU(),
                                   nn.Upsample(scale_factor=2, align_corners=False, mode='bilinear'))
        self.conv3 = nn.Sequential(nn.Conv2d(1152, 384, 1),
                                   nn.BatchNorm2d(384),
                                   nn.ReLU(),
                                   nn.Upsample(scale_factor=2, align_corners=False, mode='bilinear'))
        self.conv2 = nn.Sequential(nn.Conv2d(576, 192, 1),
                                   nn.BatchNorm2d(192),
                                   nn.ReLU(),
                                   nn.Upsample(scale_factor=2, align_corners=False, mode='bilinear'))
        self.conv1 = nn.Sequential(nn.Conv2d(288, 96, 1),
                                   nn.BatchNorm2d(96),
                                   nn.ReLU(),
                                   nn.Upsample(scale_factor=2, align_corners=False, mode='bilinear'))

    def forward(self, f1, f2, f3, f4):
        # f5_aspp = self.aspp5(f5)

        # cat4 = torch.cat((f5_aspp, f4), dim=1)
        cat4 = self.aspp4(self.up2(f4))
        cat3 = torch.cat((f3, cat4), dim=1)
        cat3 = self.aspp3(self.conv3(cat3))
        cat2 = torch.cat((f2, cat3), dim=1)
        cat2 = self.aspp2(self.conv2(cat2))
        cat1 = torch.cat((cat2, f1), dim=1)
        cat1 = self.aspp1(self.conv1(cat1))

        return cat1



class Adfusion(nn.Module):
    def __init__(self, input_channel1, input_channel2):
        super().__init__()

        self.conv = nn.Sequential( nn.Conv2d(input_channel1, input_channel2, 1),
                                   nn.BatchNorm2d(input_channel2),
                                   nn.ReLU())

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, f0, f1):

        f1 = self.conv(f1)
        b1, c1, h1, w1 = f1.size()
        b0, c0, h0, w0 = f0.size()
        f1 = F.interpolate(input=f1, scale_factor=h0//h1, mode='bilinear', align_corners=True)

        max_0,_ = torch.max(f0, dim=1, keepdim=True)
        max_1,_ = torch.max(f1, dim=1, keepdim=True)
        add_max = max_0 + max_1
        add_max = self.sigmoid(add_max)
        f0 = f0 * add_max
        f1 = f1 * add_max

        mul_f = f0 * f1
#         f1_new = self.conv31(self.relu(f1 - mul_f))
#         f0_new = self.conv32(self.relu(f0 - mul_f))

#         out = self.conv33(f1_new + f0_new + mul_f)
        out = mul_f + f0 + f1

        return out



class Depth_Mamba(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.rgb = vmamba_tiny_s1l8()     
        for name, parameter in self.rgb.named_parameters():
            if 'fitune' in name:
                parameter.requires_grad = True
            # if 'mlp' in name:
            #     pass
            else:
                parameter.requires_grad = False
         
               
        self.dep = vmamba_tiny_s1l8()
        for name, parameter in self.dep.named_parameters():
            if 'fitune' in name:
                parameter.requires_grad = True
            # if 'mlp' in name:
            #     pass
            else:
                parameter.requires_grad = False

                

        self.conv1 = nn.Sequential(nn.Conv2d(96, 48, 1),
                                   nn.Upsample(scale_factor=2, align_corners=False, mode='bilinear'),
                                   nn.Conv2d(48, 1, 1))
        self.decoder_f = Decoer()
#         self.conv3r = nn.Sequential(nn.Conv2d(768, 384, 1),
#                                     nn.BatchNorm2d(384),
#                                     nn.ReLU(),
#                                    nn.Upsample(scale_factor=2, align_corners=False, mode='bilinear'))
#         self.conv2r = nn.Sequential(nn.Conv2d(384, 192, 1),
#                                     nn.BatchNorm2d(192),
#                                     nn.ReLU(),
#                                     nn.Upsample(scale_factor=2, align_corners=False, mode='bilinear'))
#         self.conv1r = nn.Sequential(nn.Conv2d(192, 96, 1),
#                                     nn.BatchNorm2d(96),
#                                     nn.ReLU(),
#                                     nn.Upsample(scale_factor=2, align_corners=False, mode='bilinear'))
        
#         self.conv3d = nn.Sequential(nn.Conv2d(768, 384, 1),
#                                     nn.BatchNorm2d(384),
#                                     nn.ReLU(),
#                                     nn.Upsample(scale_factor=2, align_corners=False, mode='bilinear'))
#         self.conv2d = nn.Sequential(nn.Conv2d(384, 192, 1),
#                                     nn.BatchNorm2d(192),
#                                     nn.ReLU(),
#                                     nn.Upsample(scale_factor=2, align_corners=False, mode='bilinear'))
#         self.conv1d = nn.Sequential(nn.Conv2d(192, 96, 1),
#                                     nn.BatchNorm2d(96),
#                                     nn.ReLU(),
#                                     nn.Upsample(scale_factor=2, align_corners=False, mode='bilinear'))


        self.adfusionr4 = Adfusion(768, 768)
        self.adfusionr3 = Adfusion(768, 384)
        self.adfusionr2 = Adfusion(384, 192)
        self.adfusionr1 = Adfusion(192, 96)

        self.adfusiond4 = Adfusion(768, 768)
        self.adfusiond3 = Adfusion(768, 384)
        self.adfusiond2 = Adfusion(384, 192)
        self.adfusiond1 = Adfusion(192, 96)
        
                                    

    def forward(self, rgb, dep):
       
        # dep = torch.cat((dep, dep, dep), dim=1)
        rgb1, rgb2, rgb3, rgb4, rgb5 = self.rgb(rgb)
        dep1, dep2, dep3, dep4, dep5 = self.dep(dep)

        add4r = self.adfusionr4(rgb4, rgb5)
        add3r = self.adfusionr3(rgb3, rgb4)
        add2r = self.adfusionr2(rgb2, rgb3)
        add1r = self.adfusionr1(rgb1, rgb2)

        add4d = self.adfusiond4(dep4, dep5)
        add3d = self.adfusiond3(dep3, dep4)
        add2d = self.adfusiond2(dep2, dep3)
        add1d = self.adfusiond1(dep1, dep2)

#         add4r = rgb4 + rgb5
#         add3r = rgb3 + self.conv3r(rgb4)
#         add2r = rgb2 + self.conv2r(rgb3)
#         add1r = rgb1 + self.conv1r(rgb2)
        
#         add4d = dep4 + dep5
#         add3d = dep3 + self.conv3d(dep4)
#         add2d = dep2 + self.conv2d(dep3)
#         add1d = dep1 + self.conv1d(dep2)
        
        
        add4 = add4d + add4r
        add3 = add3d + add3r
        add2 = add2d + add2r
        add1 = add1d + add1r


        out_f = self.decoder_f(add1, add2, add3, add4)
        sal_finally = self.conv1(out_f)


        return sal_finally

if __name__ == '__main__':
    r = torch.rand(2, 3, 320, 320).cuda()
    d = torch.rand(2, 3, 320, 320).cuda()

    net = Depth_Mamba().cuda()
    sal_finally = net(r, d)

    a = torch.randn(1, 3, 256, 256).cuda()
    b = torch.randn(1, 3, 256, 256).cuda()
    from RSDD_Tool.FLOP import CalParams

    CalParams(net, a, b)
    print('Total params % .2fM' % (sum(p.numel() for p in net.parameters()) / 1000000.0))




