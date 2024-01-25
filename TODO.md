* how to train mixtral-2x?
  - better parameters for training?
  - distribute to 2 GPUs via FSDP
  - LoRA? Q-LoRA?
  - partial LoRA?
  - partially quantize mixtral-2x (the frozen parts)
  - mixtral-2x of smaller models (either smaller intermediate for new expert or both models smaller)
* add bnb quantization (save_pretrained 4bit and 8bit from transformers 4.37)
* test multi-GPU multi-node with tranformers 4.37
* integrate and test chat templates via tokenizer
* infinite is deprecated in newest trl - remove?
* update README with different scenarios (full training, lora, pseudo qlora, pre-training)
* play with neftune?
* update OdinTrainer with next trl release
* update OdinTrainer with next transformers release that incorporates the changes
* train directly from adapter with automated detection and merging of base model and previous adapters
* print and compare tensors for base models, quantized models, (q)lora adapters, and merged models
* fix training with gptq model (qlora)
* quantize adapter weights and merge with quantized model (real qlora)
* quantized adapter and merge with quantized model (qalora)
* quantization-aware training (llm-qat)
* figure out device maps such that CUDA_VISIBLE_DEVICES is not needed and compatible with accelerate
