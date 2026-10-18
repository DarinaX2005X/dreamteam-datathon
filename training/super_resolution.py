"""Offline Real-ESRGAN general-x4v3 comparison for detector input crops.

Architecture adapted from xinntao/Real-ESRGAN, realesrgan/archs/srvgg_arch.py.
BSD 3-Clause License
Copyright (c) 2021, Xintao Wang. All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software without
   specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
OF THE POSSIBILITY OF SUCH DAMAGE.
"""
from __future__ import annotations


class CropUpscaler:
    """Use bounded private RGB crops; output is model input, never evidence."""
    def __init__(self, path, device="cuda"):
        import torch
        from torch import nn

        class CompactVGG(nn.Module):
            def __init__(self):
                super().__init__()
                layers = [nn.Conv2d(3, 64, 3, padding=1), nn.PReLU(64)]
                for _ in range(32):
                    layers.extend((nn.Conv2d(64, 64, 3, padding=1), nn.PReLU(64)))
                layers.append(nn.Conv2d(64, 48, 3, padding=1))
                self.body = nn.ModuleList(layers)
                self.upsampler = nn.PixelShuffle(4)

            def forward(self, value):
                output = value
                for layer in self.body:
                    output = layer(output)
                return self.upsampler(output) + nn.functional.interpolate(value, scale_factor=4, mode="nearest")

        self.device = device
        self.model = CompactVGG()
        weights = torch.load(path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(weights.get("params_ema", weights.get("params", weights)), strict=True)
        self.model.to(device).eval()

    def __call__(self, image):
        import numpy as np
        import torch
        from PIL import Image

        if image.width * image.height > 1_000_000:
            raise ValueError("Upscale comparison crop exceeds its pixel budget")
        pixels = np.asarray(image.convert("RGB")).copy()
        tensor = torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0).to(self.device, dtype=torch.float32) / 255.
        with torch.inference_mode():
            output = self.model(tensor).clamp(0, 1)
        result = output.squeeze(0).permute(1, 2, 0).mul(255).round().to(torch.uint8).cpu().numpy()
        return Image.fromarray(result)
