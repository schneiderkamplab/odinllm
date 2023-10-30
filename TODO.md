* update README with different scenarios (lora, pseudo qlora)
* print and compare tensors for base models, quantized models, (q)lora adapters, and merged models
* fix training with gptq model (qlora)
* see if other layers can be meaningfully LoRA'ed
* quantize adapter weights and merge with quantized model (real qlora)
* quantized adapter and merge with quantized model (qalora)
* quantization-aware training (llm-qat)
* figure out device maps such that CUDA_VISIBLE_DEVICES is not needed
* use datacollatorforlanguagemodelling instead of concatenator
* more parameters (e.g. LoRA parameters, training parameters)
* implement test suite
