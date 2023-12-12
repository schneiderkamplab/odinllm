* gitlab ci
* abstract load_datasets into shared with list of eval and train datasets (with proportion/size given)
* adapt finalize, infer, snapshot, train
* move corpora into yaml file for init
* how to override from command line (training_args.bf16=true)
* check that all config.CMD params can be overriden on the command line (e.g. init)
* no infinite configuration but depending on eval/train?
* move all commands to subdirectory
* multiple training datasets
* update README with different scenarios (full training, lora, pseudo qlora, pre-training)
* pass in yaml with trainer args, training args, model.from_pretrained args, tokenizere.from_pretrained args, lora args
* add launch command that takes configs and runs the preconfigured commands (list of commands as an argument, command arguments from yaml)
* play with neftune
* update OdinTrainer with next trl release
* update OdinTrainer with next transformers release that incorporates the changes
* evaluate mistral 7b sft beta
* add replay/from-metadata functionality
* adopt dataset loading incl. shuffling and mixing and other insightful stuff from alignment-handbook such as chat templates
* train directly from adapter with automated detection and merging of base model and previous adapters
* print and compare tensors for base models, quantized models, (q)lora adapters, and merged models
* fix training with gptq model (qlora)
* quantize adapter weights and merge with quantized model (real qlora)
* quantized adapter and merge with quantized model (qalora)
* quantization-aware training (llm-qat)
* figure out device maps such that CUDA_VISIBLE_DEVICES is not needed and compatible with accelerate
* more parameters (e.g. LoRA parameters, training parameters)
* implement test suite
