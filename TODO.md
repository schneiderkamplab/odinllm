* update README with different scenarios (full training, lora, pseudo qlora, pre-training)
* add replay/from-metadata functionality
* adopt dataset loading and other insightful stuff from alignment-handbook
* train directly from adapter with automated detection and merging of base model and previous adapters
* print and compare tensors for base models, quantized models, (q)lora adapters, and merged models
* fix training with gptq model (qlora)
* quantize adapter weights and merge with quantized model (real qlora)
* quantized adapter and merge with quantized model (qalora)
* quantization-aware training (llm-qat)
* figure out device maps such that CUDA_VISIBLE_DEVICES is not needed and compatible with accelerate
* more parameters (e.g. LoRA parameters, training parameters)
* implement test suite
