# Research papers I've read recently

> [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
>> - Introduced a method for efficiently fine-tuning LLMs by significantly reducing the number of trainable parameters
>> - Key finding was that the delta weights matrix of a large language model has a low intrinsic dimensionality
>> - LoRA uses low-rank decomposition to adapt the model's weights, capturing most of the benefits of full fine-tuning
>> - Allows for creating small, task-specific adaptations that can be easily switched or combined

> [Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning](https://arxiv.org/abs/2012.13255v1)
>> The authors show that pre-trained language models have a low intrinsic dimensionality. This means that fine-tuning can be done effectively by optimizing only a small subset of parameters. For instance, they demonstrate that tuning only around 200 parameters can yield 90% of the performance of fine-tuning the full model

> [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
>> - QLoRA enables memory-efficient fine-tuning of large language models, such as 65B parameter models, on a single 48GB GPU. It maintains performance similar to full 16-bit precision fine-tuning.
>> - Introduced NF4, a new data type optimized for normally distributed weights, helps reduce memory usage by quantizing models to 4-bit precision without significant performance degradation

> [The case for 4-bit precision: k-bit Inference Scaling Laws](https://arxiv.org/abs/2212.09720)
>> - Through extensive experimentation with models ranging from 19M to 176B parameters, the authors demonstrate that 4-bit precision strikes the best balance between model size and performance, particularly for zero-shot tasks
>> - Introduced scaling laws that guide how performance (in terms of accuracy and efficiency) scales with model size and bit precision. These laws help determine the bit-precision and model size combinations that maximize performance for zero-shot inference tasks

> [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
>> - Demonstrated the effectiveness of applying Transformer models to image recognition tasks
>> - Vision Transformers (ViT) are trained on sequences of image patches rather than full images, where each patch is treated like a token in a sequence, similar to words in text models

> [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752)
>> The key idea proposed in this paper is to model the diffusion process in a lower-dimensional latent space, rather than directly in the high-dimensional pixel space. This latent diffusion model (LDM) approach involves training an autoencoder to map images to a lower-dimensional latent representation. The diffusion process is then performed in this latent space, gradually adding noise to the latent representations and then learning to remove it

> [Deep Unsupervised Learning using Nonequilibrium Thermodynamics](https://arxiv.org/abs/1503.03585)
>> The original paper from 2015 which introduced Diffusion technique to the field of machine learning coming originally from statistical physics

> [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
>> Second influential paper for diffusion models published in 2020 which introduced few groundbreaking changes which led to huge jump in quality

> [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672)
>> First paper on diffusion models by OpenAI, finds that learning the variance of the conditional distribution (besides the mean) helps in improving performance

> [Deep Double Descent: Where Bigger Models and More Data Hurt](https://arxiv.org/abs/1912.02292)
>> - Describes the double descent phenomenon, shows that it is a function of not just the model size but the number of training epochs as well
>> - Introduced a generalized double descent hypothesis: models and training procedures exhibit atypical behavior when their Effective Model Complexity is comparable to the number of train samples
>> - Also shows that the double descent phenomenon can lead to a regime where training on more data leads to worse test performance

> [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
>> Introduced the Transformer architecture which is prevalent in the field of AI now

> [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
>> Google's LLM where the objective isn't next word prediction but masked LM (cloze task) and next sentence prediction

> [Neural Machine Translation with Byte-Level Subwords](https://arxiv.org/abs/1909.03341)
>> Proposed the use of Byte-level Byte Pair Encoding algorithm for tokenization

> [Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288)
>> META's LLM with a decoder only architecture

> [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)
>> Proposed that re-centering invariance in LayerNorm is dispensible and re-scaling invariance alone is enough

> [Self-Attention with Relative Position Representations](https://arxiv.org/abs/1803.02155)
>> Introduced the concept of relative position embeddings

> [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
>> Introduced Rotary Positional Embeddings

> [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150)
>> Introduced Multi-Query Attention

> [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245)
>> Introduced Grouped-Query Attention

> [RouteLLM: Learning to Route LLMs with Preference Data](https://arxiv.org/abs/2406.18665)

> [GLU Variants Improve Transformer](https://arxiv.org/pdf/2002.05202)
>> Proposed using GLU activation functions for Transformers

> [Large Language Models: A Survey](https://arxiv.org/abs/2402.06196)
>> Survery of popular LLM families (GPT, LLaMA, PaLM), popular datasets for LLM training, fine-tuning and evaluation and more

> [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)
>> GPT-3 paper

> [Mistral 7B](https://arxiv.org/pdf/2310.06825)
>> MistralAI's LLM

> [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781)
>> Word2Vec Paper 1

> [Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546)
>> Word2Vec Paper 2

> [word2vec Explained: deriving Mikolov et al.'s negative-sampling word-embedding method](https://arxiv.org/abs/1402.3722)
>> Word2Vec Paper 3