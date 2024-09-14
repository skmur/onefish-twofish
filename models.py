import torch
import transformers

class Model:
    def __init__(self, cache_dir):
        self.cache_dir = cache_dir
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def initialize_model(self, model_name, model_path):
        if model_name.lower() == "openchat":
            self._initialize_openchat(model_path)
        elif model_name.lower() == "starling":
            self._initialize_starling(model_path)
        elif model_name.lower() == "mistral-instruct":
            self._initialize_mistral(model_path)
        elif model_name.lower() == "zephyr-mistral":
            self._initialize_zephyr(model_path)
        # ADD ANOTHER RLHF MODEL HERE
        elif model_name.lower() == "llama-chat":
            self._initialize_llama(model_path)
        elif model_name.lower() == "gemma-instruct":
            self._initialize_gemma(model_path)
        elif model_name.lower() == "zephyr-gemma":
            self._initialize_zephyr(model_path)
        else:
            raise ValueError(f"Unknown model type: {model_name}")

    def _initialize_openchat(self, model_path):
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_path,
                                                                    cache_dir=self.cache_dir)
        self.model = transformers.AutoModelForCausalLM.from_pretrained(model_path,
                                                                       cache_dir=self.cache_dir)
        self.model.generation_config.cache_implementation = "static"
        self.model.forward = torch.compile(self.model.forward,
                                           mode="reduce-overhead",
                                           fullgraph=True)

    def _initialize_starling(self, model_path):
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_path,
                                                                    cache_dir=self.cache_dir)
        self.model = transformers.AutoModelForCausalLM.from_pretrained(model_path,
                                                                       cache_dir=self.cache_dir)
        self.model.generation_config.cache_implementation = "static"
        self.model.forward = torch.compile(self.model.forward,
                                           mode="reduce-overhead",
                                           fullgraph=True)

    def _initialize_mistral(self, model_path):
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_path,
                                                                    cache_dir=self.cache_dir)
        self.model = transformers.AutoModelForCausalLM.from_pretrained(model_path,
                                                                       cache_dir=self.cache_dir, device_map="auto")

    def _initialize_gemma(self, model_path):
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, cache_dir=self.cache_dir)
        self.model = transformers.AutoModelForCausalLM.from_pretrained(model_path, 
                                                          device_map="auto",
                                                          torch_dtype=torch.bfloat16,
                                                          cache_dir=self.cache_dir)

    def _initialize_llama(self, model_path):
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_path,
                                                               cache_dir=self.cache_dir)
        self.pipeline = transformers.pipeline("text-generation", 
                             model=model_path, 
                             torch_dtype=torch.bfloat16, 
                             device_map="auto")
        
    def _initialize_zephyr(self, model_path):
        self.pipeline = transformers.pipeline("text-generation",
                                              model=model_path, 
                                              torch_dtype=torch.bfloat16, 
                                              device_map="auto")
    
    def format_prompt(self, model_name, context, prompt):
        if model_name.lower() == "openchat":
            return self._format_openchat_prompt(context, prompt)
        elif model_name.lower() == "starling":
            return self._format_starling_prompt(context, prompt)
        elif model_name.lower() == "mistral-instruct":
            return self._format_mistral_prompt(context, prompt)
        elif model_name.lower() == "zephyr-mistral":
            return self._format_zephyr_prompt(context, prompt)
        # ADD ANOTHER RLHF MODEL HERE
        elif model_name.lower() == "llama-chat":
            return self._format_llama_prompt(context, prompt)
        elif model_name.lower() == "gemma-instruct":
            return self._format_gemma_prompt(context, prompt)
        elif model_name.lower() == "zephyr-gemma":
            return self._format_zephyr_prompt(context, prompt)
        else:
            raise ValueError(f"Unknown model type: {model_name}")
        
    def _format_openchat_prompt(self, context, prompt):
        if context == "":
            return f"GPT4 Correct User: {prompt}.<|end_of_turn|>GPT4 Correct Assistant:"
        else:
            return f"GPT4 Correct User: {context}. {prompt}<|end_of_turn|>GPT4 Correct Assistant:"
    
    def _format_starling_prompt(self, context, prompt):
        # same as openchat
        return self._format_openchat_prompt(context, prompt)

    def _format_llama_prompt(self, context, prompt):
        if context == "":
            return f"<s>[INST] {{{prompt}}} [/INST]"
        else:
            return f"<s>[INST] <<SYS>>\n {{{context}}}<</SYS>>\n\n{{{prompt}}} [/INST]"

    def _format_gemma_prompt(self, context, prompt):
        messages = [
            {"role": "user", "content": context + prompt},
        ]
        return self.tokenizer.apply_chat_template(messages, 
                                                  tokenize=False,
                                                  add_generation_prompt=True)
    
    def _format_mistral_prompt(self, context, prompt):
        messages = [
            {"role": "user", "content": context + prompt}
            ]
        
        return self.tokenizer.apply_chat_template(messages, return_tensors="pt")
    
    def _format_zephyr_prompt(self, context, prompt):
        messages = [
            {"role": "system", "content": context},
            {"role": "user", "content": prompt},
            ]

        return self.pipeline.tokenizer.apply_chat_template(messages,
                                                tokenize=False,
                                                add_generation_prompt=True,
                                                # temperature=temp
                                                )

    def generate(self, model_name, messages, temp):
        if model_name.lower() == "openchat":
            return self._generate_openchat(messages, temp)
        elif model_name.lower() == "starling":
            return self._generate_starling(messages, temp)
        elif model_name.lower() == "mistral-instruct":
            return self._generate_mistral(messages, temp)
        elif model_name.lower() == "zephyr-mistral":
            return self._generate_zephyr(messages, temp)
        # ADD ANOTHER RLHF MODEL HERE
        elif model_name.lower() == "llama-chat":
            return self._generate_llama(messages, temp)
        elif model_name.lower() == "gemma-instruct":
            return self._generate_gemma(messages, temp)
        elif model_name.lower() == "zephyr-gemma":
            return self._generate_zephyr(messages, temp)
        else:
            raise ValueError(f"Unknown model type: {model_name}")
    
    def _generate_llama(self, messages, temp):
        output = self.pipeline(
            messages,
            do_sample=True,
            max_length=100,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            temperature=float(temp) if temp != "default" else None,
        )
        return output[0]['generated_text']
    
    def _generate_gemma(self, messages, temp):
        inputs = self.tokenizer.encode(messages, 
                              add_special_tokens=False, 
                              return_tensors="pt")
        outputs = self.model.generate(input_ids=inputs.to(self.device),
                                      temperature=float(temp) if temp != "default" else None,max_new_tokens=150)
        return self.tokenizer.decode(outputs[0])
    
    def _generate_zephyr(self, messages, temp):
        outputs = self.pipeline(messages,
                                do_sample=True,
                                max_new_tokens=100,
                                temperature=float(temp) if temp != "default" else None,
                                )
        return outputs[0]["generated_text"]
    
    def _generate_mistral(self, messages, temp):
        generated_ids = self.model.generate(messages.to(self.device), 
                                            do_sample=True,
                                            max_new_tokens=150, 
                                            pad_token_id=self.tokenizer.eos_token_id,
                                            temperature=float(temp) if temp != "default" else None)
        
        outputs = self.tokenizer.batch_decode(generated_ids)
        return outputs[0]
    
    def _generate_openchat(self, messages, temp):
        inputs = self.tokenizer.encode(messages, return_tensors="pt")
        output = self.model.generate(inputs.to(self.device), 
                            max_length=200,
                            pad_token_id=self.tokenizer.pad_token_id,
                            eos_token_id=self.tokenizer.eos_token_id,
                            do_sample=True,
                            temperature=float(temp) if temp != "default" else None)
        output = self.tokenizer.decode(output[0], skip_special_tokens=True)

        return output
    
    def _generate_starling(self, messages, temp):
        # same as openchat
        return self._generate_openchat(messages, temp)
    



    
