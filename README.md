# Cognition-MLLM: Visual Cognition in Multimodal LLMs

This repository hosts the project pages for our research on visual cognition capabilities in Multimodal Large Language Models (MLLMs).

---

## 📄 Paper 1: What is the Visual Cognition Gap between Humans and Multimodal LLMs?

**[COLM 2025]**

<p align="center">
  <a href="https://arxiv.org/abs/2406.10424">📝 arXiv</a> •
  <a href="https://arxiv.org/pdf/2406.10424">📄 PDF</a> •
  <a href="https://huggingface.co/datasets/IrohXu/VCog-Bench">🤗 Benchmark</a> •
  <a href="https://pediamedai.github.io/Cognition-MLLM/cognition-mllm/">🌐 Project Page</a>
</p>

### Authors
[Xu Cao](https://www.irohxucao.com/)<sup>1*</sup>, 
[Yifan Shen](https://shenyifans.github.io/)<sup>1*</sup>, 
[Bolin Lai](https://bolinlai.github.io/)<sup>2</sup>, 
[Wenqian Ye](https://wenqian-ye.github.io/)<sup>3</sup>, 
[Yunsheng Ma](https://ysma.me/)<sup>4</sup>, 
[Joerg Heintz](https://neuroscience.illinois.edu/directory/profile/jheintz)<sup>1</sup>, 
[Jintai Chen](https://whatashot.github.io/)<sup>5</sup>, 
[Meihuan Huang](https://scholar.google.com/citations?user=sM_jWNkAAAAJ&hl=en)<sup>6</sup>, 
[Jianguo Cao](https://scholar.google.com/citations?user=dPPOpHUAAAAJ&hl=en)<sup>6</sup>, 
[Aidong Zhang](https://www.cs.virginia.edu/~az9eg/website/home.html)<sup>3</sup>, 
[James M. Rehg](https://rehg.org/)<sup>1</sup>

<sup>1</sup>University of Illinois at Urbana-Champaign, <sup>2</sup>Georgia Institute of Technology, <sup>3</sup>University of Virginia, <sup>4</sup>Purdue University, <sup>5</sup>HKUST (Guangzhou), <sup>6</sup>Shenzhen Children's Hospital

<sup>*</sup>Equal Contribution

### Abstract
Recently, Multimodal Large Language Models (MLLMs) and Vision Language Models (VLMs) have shown great promise in language-guided perceptual tasks such as recognition, segmentation, and object detection. However, their effectiveness in addressing visual cognition problems that require high-level multi-image reasoning and visual working memory is not well-established. One such challenge is **matrix reasoning** - the cognitive ability to discern relationships among patterns in a set of images and extrapolate to predict subsequent patterns. This skill is crucial during the early neurodevelopmental stages of children.

Inspired by the matrix reasoning tasks in Raven's Progressive Matrices (RPM) and Wechsler Intelligence Scale for Children (WISC), we propose a new dataset **MaRs-VQA** to evaluate the visual cognition capability of MLLMs and compare their performance with existing human visual cognition studies. Based on the training data of MaRs-VQA, we also finetune a baseline model **Qwen2-VCog** with multi-stage cognition reasoning annotations.

---

## 📄 Paper 2: Toward Cognitive Supersensing in Multimodal Large Language Models

**[2026]**

<p align="center">
  <a href="https://arxiv.org/abs/2602.01541">📝 arXiv</a> •
  <a href="https://arxiv.org/pdf/2602.01541">📄 PDF</a> •
  <a href="https://huggingface.co/datasets/PediaMedAI/CogSense-Bench">🤗 Benchmark</a> •
  <a href="https://huggingface.co/PediaMedAI/CogSense-8B">🤗 Model</a> •
  <a href="https://pediamedai.com/Cognition-MLLM/cogsense/">🌐 Project Page</a>
</p>

### Authors
[Boyi Li](https://resurgamm.github.io/)<sup>1*</sup>,
[Yifan Shen](https://shenyifans.github.io/)<sup>1*§</sup>,
[Yuanzhe Liu](https://www.linkedin.com/in/yuanzhe-liu-upenn-ds/)<sup>1*</sup>,
[Yifan Xu](https://yfxu.org/)<sup>1</sup>,
[Jiateng Liu](https://lumos-jiateng.github.io/)<sup>1</sup>,
[Xinzhuo Li](https://xinzhuo20.github.io/)<sup>1</sup>,
[Zhengyuan Li](https://scholar.google.com/citations?user=HXN7DNoAAAAJ&hl=en)<sup>1</sup>,
[Jingyuan Zhu](https://www.linkedin.com/in/jingyuan-zhu-6a0ab3322/)<sup>2</sup>,
[Yunhan Zhong](#)<sup>1</sup>,
[Fangzhou Lan](#)<sup>2</sup>,
[Jianguo Cao](https://scholar.google.com/citations?user=dPPOpHUAAAAJ&hl=en)<sup>2</sup>,
[James M. Rehg](https://rehg.org/)<sup>1</sup>,
[Heng Ji](https://blender.cs.illinois.edu/hengji.html)<sup>1</sup>,
[Ismini Lourentzou](https://isminoula.github.io/)<sup>1†</sup>,
[Xu Cao](https://www.irohxucao.com/)<sup>1,2†</sup>

<sup>1</sup>University of Illinois at Urbana-Champaign, <sup>2</sup>PediaMed AI

<sup>*</sup>Equal Contribution, <sup>§</sup>Project Lead, <sup>†</sup>Corresponding Author

### Abstract
Multimodal Large Language Models (MLLMs) have achieved remarkable success in open-vocabulary perceptual tasks, yet their ability to solve complex cognitive problems remains limited, especially when visual details are abstract and require visual memory. Current approaches primarily scale Chain-of-Thought (CoT) reasoning in the text space, even when language alone is insufficient for clear and structured reasoning, and largely neglect visual reasoning mechanisms analogous to the human visuo-spatial sketchpad and visual imagery.

To mitigate this deficiency, we introduce **Cognitive Supersensing**, a novel training paradigm that endows MLLMs with human-like visual imagery capabilities by integrating a **Latent Visual Imagery Prediction (LVIP)** head that jointly learns sequences of visual cognitive latent embeddings and aligns them with the answer, thereby forming vision-based internal reasoning chains. We further introduce a reinforcement learning stage that optimizes text reasoning paths based on this grounded visual latent.

To evaluate the cognitive capabilities of MLLMs, we present **CogSense-Bench**, a comprehensive visual question answering (VQA) benchmark assessing five cognitive dimensions: fluid intelligence, crystallized intelligence, visuospatial cognition, mental simulation, and visual routines.

---

## 📄 Paper 3: ChronoVision: Temporal Reasoning via Latent State Reconstruction

**[2026]**

<p align="center">
  <a href="https://pediamedai.com/Cognition-MLLM/ChronoVision/">🌐 Project Page</a>
</p>

### Authors
[Yifan Shen](https://shenyifans.github.io/)<sup>1,2</sup>,
Jian Xu<sup>1</sup>,
[Boyi Li](https://resurgamm.github.io/)<sup>1</sup>,
Yuner Zhang<sup>3</sup>,
Tianjiao Yu<sup>1</sup>,
Bingxuan Li<sup>1</sup>,
Houze Yang<sup>1</sup>,
Rushi Wang<sup>1</sup>,
[Xu Cao](https://www.irohxucao.com/)<sup>1,2</sup>

<sup>1</sup>University of Illinois Urbana-Champaign, <sup>2</sup>PediaMed AI, <sup>3</sup>University of Pennsylvania

### Abstract
Multimodal large language models excel at passive perception but struggle with complex visual cognitive tasks requiring multi-step temporal reasoning. This degradation largely stems from the ambiguity of language-based reasoning, which often fails to accurately articulate continuous visual transformations. We propose **ChronoVision**, a multimodal framework that aligns visual logic with latent imagery. During supervised fine-tuning, a **Reconstructive Visual Head** predicts the latent representation of the final transformed state, while an **ROI Attention Locating** module focuses the model on key visual evidence. In post-training, reinforcement learning with implicit process grounding jointly rewards outcome correctness, latent process alignment, and visual focus.

We further introduce **Vbvr-VQA**, which evaluates temporal tracking by reformulating video reasoning as a strict image-ordering task. ChronoVision achieves state-of-the-art performance with **73.2% in-domain overall exact-match accuracy** and **71.6% out-of-domain accuracy**, alongside **55.0% accuracy on IntPhys2**.

### Citation
```bibtex
@misc{cao2024visualcognitiongaphumans,
      title={What is the Visual Cognition Gap between Humans and Multimodal LLMs?}, 
      author={Xu Cao and Bolin Lai and Wenqian Ye and Yunsheng Ma and Joerg Heintz and Jintai Chen and Jianguo Cao and James M. Rehg},
      year={2024},
      eprint={2406.10424},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2406.10424}, 
}

@misc{li2026cognitivesupersensingmultimodallarge,
      title={Toward Cognitive Supersensing in Multimodal Large Language Model}, 
      author={Boyi Li and Yifan Shen and Yuanzhe Liu and Yifan Xu and Jiateng Liu and Xinzhuo Li and Zhengyuan Li and Jingyuan Zhu and Yunhan Zhong and Fangzhou Lan and Jianguo Cao and James M. Rehg and Heng Ji and Ismini Lourentzou and Xu Cao},
      year={2026},
      eprint={2602.01541},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2602.01541}, 
}

@misc{shen2026chronovision,
      title={ChronoVision: Temporal Reasoning via Latent State Reconstruction},
      author={Yifan Shen and Jian Xu and Boyi Li and Yuner Zhang and Tianjiao Yu and Bingxuan Li and Houze Yang and Rushi Wang and Xu Cao},
      year={2026}
}
```

---

<p align="center">
  <b>PediaMed AI</b> • University of Illinois at Urbana-Champaign
</p>
