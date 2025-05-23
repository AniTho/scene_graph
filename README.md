## Scene Graph Generation & Subgraph Retrieval on Satellite Imagery
![alt text](scene_graph.png)
![alt text](scene_graph_over_image.png)

### Project Objective
---
- Primary objective of this research project is to develop an efficient and accurate scene graph generation method to generate a semantic structural scene graph for satellite imagery analysis.
- Secondary objective is to propose a subgraph retrieval approach that can effectively extract specific subgraphs from the generated scene graph, which will be useful for various remote-sensing & GIS applications.

### Dataset Used
---
- We are using `FloodNet-Supervised Dataset` from here: https://github.com/BinaLab/FloodNet-Supervised_v1.0 

### Project Checklist
---
- [x] Data set selection
- [x] Create Segmentation Model using `UNet` Architecture with `ResNet34` base model
- [x] Integrate wandb 
- [x] **Generate Scene Graph:** Building connectivity graph using `NetworkX` 
- [x] Plotted Scene Graph over Image 
- [x] Generate Query Graph
- [ ] Apply maximum common subgraph matching algorithm 
- [ ] **Subgraph Isomorphism** between scene graph and query graph
- [ ] **Subgraph Retrievel** using GCN
- [ ] Try other architectures instead of UNet
- [ ] Implement `learning rate scheduler` in UNet
  
