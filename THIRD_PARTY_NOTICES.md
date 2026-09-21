# Third-Party Notices

This repository contains original SyDRA code together with adapted components
from the projects listed below. The original SyDRA code is released under the
MIT License. Third-party components retain their upstream licenses.

## DRAEM

- Source: https://github.com/VitjanZ/DRAEM
- License: MIT License
- Components: reconstruction, SSIM-related utilities, and synthetic-anomaly
  training structure adapted from the DRAEM codebase.
- The applicable MIT license text and copyright notice are retained in
  `LICENSE`.

## Loss_ToolBox-PyTorch

- Source: https://github.com/Hsuxu/Loss_ToolBox-PyTorch
- License: Apache License 2.0
- Component: the `FocalLoss` class in `loss.py`, adapted and integrated for
  SyDRA.
- The Apache-2.0 license text is included in
  `LICENSES/Apache-2.0.txt`.

The upstream project identifies the following additional source for its Focal
Loss implementation:

## RetinaNet

- Source: https://github.com/c0nn3r/RetinaNet
- License: BSD 3-Clause License
- Relevance: credited upstream source for the Focal Loss implementation used by
  `Loss_ToolBox-PyTorch`.
- The BSD-3-Clause license text is included in
  `LICENSES/BSD-3-Clause.txt`.

The Focal Loss code in `loss.py` has been modified from the upstream version
and integrated with the other loss utilities used by SyDRA.
