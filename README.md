# HyperVision: Channel-Adaptive Ground-Based Hyperspectral Vision Foundation Models

*This is the official repository for the paper "HyperVision: Channel-Adaptive Ground-Based Hyperspectral Vision Foundation Models".*

---

🚀 **News:**
- **[Coming Soon]** We will soon open-source the official implementation and pre-trained foundation models! Please stay tuned!

---

## 📖 Abstract

Ground-based hyperspectral foundation models remain absent, as their development is severely hindered by annotation scarcity, limited scene diversity, and hardware fragmentation across sensors with varying band numbers. To address these challenges, we propose **HyperVision**, a channel-adaptive foundation model specifically designed for ground-based hyperspectral perception. 

- First, to overcome structural constraints, HyperVision employs a dynamic embedding mechanism to map heterogeneous hyperspectral inputs of arbitrary bands into a unified token space. 
- Second, to address the manual annotation bottleneck, we introduce an automatic mask generation strategy that fuses spatial priors from SAM2 with material-aware representations from HyperFree, establishing a robust unsupervised self-training paradigm. 
- Finally, to alleviate the limited scene diversity in hyperspectral datasets, we propose a cross-modal knowledge distillation mechanism that transfers robust object understanding capabilities from pre-trained RGB models to our spectral framework. 

Extensive experiments demonstrate that via efficient head-only adaptation, HyperVision achieves state-of-the-art performance on hyperspectral semantic segmentation, object tracking, and salient object detection.
