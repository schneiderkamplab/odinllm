* test chat templates
* move corpora into yaml file for init (eg ft-1000.jsonl to tiny.yaml)
* check that all config.CMD params can be overriden on the command line (e.g. init)
* how to override other params from command line (training_args.bf16=true)
* no infinite configuration but depending on eval/train?
* move all commands to subdirectory
* update README with different scenarios (full training, lora, pseudo qlora, pre-training)
* add launch command that takes configs and runs the preconfigured commands (list of commands as an argument, command arguments from yaml)
* play with neftune
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
