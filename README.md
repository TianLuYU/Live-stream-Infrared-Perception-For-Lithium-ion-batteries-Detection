# Live-stream-Infrared-Perception
This is the public code related to “Live-stream Infrared Perception: Two-stage Thermal Characterization and Localization for Lithium-ion Batteries.”

To guarantee the safe functioning of batteries, it is essential to monitor their operational status. Nevertheless, there is mutual obstruction between battery cells because of the close arrangement inside the battery pack, which makes it extremely difficult to localize thermal faults. Therefore, this article proposes a two-stage thermal characterization and localization for lithium-ion batteries, which achieves accurate end-to-end positioning of thermal fault cells. Camera rotation allows for multiple perspectives of the battery pack's surface thermal in real time, avoiding the blockage issue that single-perspective cameras present. Stage 1 is based on Detection Transformer (DETR), which achieves coarse characterization of targets by video instance query and frame target tracking. The 3D incoherent region detector set in stage 2 achieved mask fine-adjustment through a point quadtree, which precisely localizes faulty battery cells. Testing the battery model created with Ansys Fluent revealed that the diagnostic system's mAP and mAR values were higher than 0.7.

![image](https://github.com/user-attachments/assets/24db28c3-3f91-4015-bd46-e29177ebec00)

This article primarily contributes by:
1. A two-stage thermal characterization and localization model for lithium-ion batteries is developed based on the Detection Transformer (DETR) for end-to-end fault diagnosis and localization. 
2. Two-stage mask optimization refines edges using the initial segmentation mask. The 3D distinct region detector and point quadtree set-up combine multi-scale feature maps (FMs) to reconstruct mask boundary information, ensuring segmentation accuracy under tight battery cell disassembly settings.
3. A rotating camera covers the battery pack's 3D location information to monitor its global surface temperature in real-time from numerous perspectives, which avoids battery cell obstruction in single-point cameras and offers better fault localization.
4. To simulate heat generation from internal short circuit faults in 1P3S lithium-ion batteries, Ansys Fluent software is used to perform finite volume analysis and acquire surface thermal images. Image processing verifies the algorithm's capability.

