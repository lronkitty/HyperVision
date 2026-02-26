# HyperVision: Channel-Adaptive Ground-Based Hyperspectral Vision Foundation Models

*This is the official repository for the paper "HyperVision: Channel-Adaptive Ground-Based Hyperspectral Vision Foundation Models".*

---

🚀 **News:**
- **[Coming Soon]** We will soon open-source the official implementation and pre-trained foundation models! Please stay tuned!

---

## 📖 Abstract

Ground-based hyperspectral foundation models are still lacking, constrained by varying spectral configurations across sensors, the widespread lack of pixel-level labels, and the limited scale and scene diversity of existing datasets. To address these challenges and enable universal perception, we propose HyperVision, the first ground-based hyperspectral foundation model.

- First, to handle varying spectral configurations, HyperVision employs a channel-adaptive dynamic embedding mechanism to map heterogeneous inputs into a unified token space. 
- Second, to overcome label scarcity, we introduce an unsupervised pseudo-label training method that fuses spatial priors from SAM2 with material-aware representations from HyperFree. 
- Finally, to enrich scene diversity and understanding capabilities, a cross-modal knowledge distillation mechanism is utilized to transfer rich semantic representations from pre-trained RGB models to our spectral framework. 

Experimental results confirm that HyperVision can accept various ground-based hyperspectral images as input, enabling efficient head-only adaptation without adjusting backbone parameters. It achieves state-of-the-art performance on hyperspectral semantic segmentation, object tracking, and salient object detection.
