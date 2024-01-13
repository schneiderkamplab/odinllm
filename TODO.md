* fix quantization problem with c4 dataset (realnewslike?)
* add bnb quantization (save_pretrained 4bit and 8bit from transformers 4.37)
* test multi-GPU multi-node with tranformers 4.37
* test chat templates
* infinite is deprecated in newest trl - remove?
* update README with different scenarios (full training, lora, pseudo qlora, pre-training)
* play with neftune?
* update OdinTrainer with next trl release
* update OdinTrainer with next transformers release that incorporates the changes
* evaluate mistral 7b sft beta
* add replay/from-metadata functionality
* train directly from adapter with automated detection and merging of base model and previous adapters
* print and compare tensors for base models, quantized models, (q)lora adapters, and merged models
* fix training with gptq model (qlora)
* quantize adapter weights and merge with quantized model (real qlora)
* quantized adapter and merge with quantized model (qalora)
* quantization-aware training (llm-qat)
* figure out device maps such that CUDA_VISIBLE_DEVICES is not needed and compatible with accelerate
* more parameters (e.g. LoRA parameters, training parameters)
* implement test suite
