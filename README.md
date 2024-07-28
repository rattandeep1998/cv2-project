# Enhanced Data Table Extraction from Charts

## Integrating Type-Aware Finetuning and Hide-and-Seek Strategies

### Abstract

Charts are essential in data analysis, offering insights and supporting data-driven decision-making. Accurate chart comprehension requires specialized skills, especially in converting chart information into tabular and textual formats. In this paper, we introduce two novel fine-tuning strategies to enhance the accuracy of chart-to-table (CTT) conversion models, thereby improving performance across all related downstream tasks. Our first strategy, hide-and-
seek method diversifies model attention by randomly masking image patches and the second strategy, type-aware fine-tuning, customizes the model’s input prompt to specific chart types, optimizing alignment with different chart layouts. These approaches significantly improve model generalization and accuracy, facilitating effective adaptation to charts from unfamiliar rendering engines. In addition, we have written the refined mathematical definition of Relative Mapping Similarity (RMS) [6] which is a key metric for CTT tasks. Our results indicate enhanced accuracy and robustness in the CTT task, underscoring the effectiveness of our fine-tuning strategies in practical applications. Despite these advances, challenges in building generalized models highlight ongoing research opportunities in this field.

### Pipeline

![Pipeline](/diagrams/CV2-Proj-Pipeline.png)