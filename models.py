import torch
import transformers
from huggingface_hub import login
from datasets import Dataset
from transformers.pipelines.pt_utils import KeyDataset
from tqdm import tqdm

class Model:
    def __init__(self, cache_dir, hf_token, batch_size):
        self.cache_dir = cache_dir
        print(self.cache_dir)
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hf_token = hf_token
        self.batch_size = batch_size
        self.max_new_tokens = 200

        login(token=self.hf_token)

    def initialize_model(self, model_name, model_path):
        if model_name.lower() == "openchat":
            self._initialize_openchat(model_path)
        elif model_name.lower() == "starling":
            self._initialize_starling(model_path)
        elif model_name.lower() == "mistral-instruct":
            self._initialize_mistral(model_path)
        elif model_name.lower() in ["zephyr-mistral", "zephyr-gemma"]:
            self._initialize_zephyr(model_path)
        # ADD ANOTHER RLHF MODEL HERE
        elif model_name.lower() in ["llama2", "llama2-chat"]:
            self._initialize_llama(model_path)
        elif model_name.lower() == "gemma-instruct":
            self._initialize_gemma(model_path)
        else:
            raise ValueError(f"initialize_model() received unknown model type: {model_name}")

    def _initialize_openchat(self, model_path):
        torch.set_float32_matmul_precision('high')
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_path,
                                                                    cache_dir=self.cache_dir)
        self.model = transformers.AutoModelForCausalLM.from_pretrained(model_path,
                                                                       cache_dir=self.cache_dir,
                                                                       device_map="auto",
                                                                       torch_dtype=torch.bfloat16)
        
    def _initialize_starling(self, model_path):
        #same as openchat
        return self._initialize_openchat(model_path)

    def _initialize_mistral(self, model_path):
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_path,
                                                                    cache_dir=self.cache_dir)
        self.model = transformers.AutoModelForCausalLM.from_pretrained(model_path,
                                                                       cache_dir=self.cache_dir,
                                                                       device_map="auto",
                                                                       torch_dtype=torch.bfloat16)

    def _initialize_gemma(self, model_path):
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, cache_dir=self.cache_dir)
        self.model = transformers.AutoModelForCausalLM.from_pretrained(model_path, 
                                                          device_map="auto",
                                                          torch_dtype=torch.bfloat16,
                                                          cache_dir=self.cache_dir)
        
    def _initialize_zephyr(self, model_path):
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, 
                                                                     cache_dir=self.cache_dir,
                                                                     padding_side='left')
        self.model = transformers.AutoModelForCausalLM.from_pretrained(model_path, 
                                                          device_map="auto",
                                                          torch_dtype=torch.bfloat16,
                                                          cache_dir=self.cache_dir)
        
        # self.pipeline = transformers.pipeline("text-generation",
        #                                       model=model_path, 
        #                                       torch_dtype=torch.bfloat16, 
        #                                       device_map="auto", 
        #                                       batch_size=self.batch_size)
    
    def _initialize_llama(self, model_path):
        self.tokenizer = transformers.LlamaTokenizer.from_pretrained(model_path, 
                                                                     cache_dir=self.cache_dir,
                                                                     padding_side='left')
        self.model = transformers.LlamaForCausalLM.from_pretrained(model_path, 
                                                          device_map="auto",
                                                          torch_dtype=torch.bfloat16,
                                                          cache_dir=self.cache_dir)

        # self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_path,
        #                                                        cache_dir=self.cache_dir)
        # self.pipeline = transformers.pipeline("text-generation", 
        #                      model=model_path, 
        #                      torch_dtype=torch.bfloat16, 
        #                      device_map="auto", 
        #                      batch_size=self.batch_size)
    
    def shard_model(self):
        if self.model is not None:
            num_gpus = torch.cuda.device_count()
            device_ids = list(range(num_gpus))
            self.model = torch.nn.DataParallel(self.model, 
                                               device_ids=device_ids)
        elif self.pipeline is not None:
            pass
        else:
            raise ValueError("Model not initialized")
        
    def format_prompt(self, model_name, context, prompt):
        if model_name.lower() == "openchat":
            return self._format_openchat_prompt(context, prompt)
        elif model_name.lower() == "starling":
            return self._format_starling_prompt(context, prompt)
        elif model_name.lower() == "mistral-instruct":
            return self._format_mistral_prompt(context, prompt)
        elif model_name.lower() == "zephyr-mistral":
            return self._format_zephyrMistral_prompt(context, prompt)
        elif model_name.lower() == "zephyr-gemma":
            return self._format_zephyrGemma_prompt(context, prompt)
        # ADD ANOTHER RLHF MODEL HERE
        elif model_name.lower() == "llama2":
            return self._format_llama_prompt(context, prompt)
        elif model_name.lower() == "llama2-chat":
            return self._format_llamaChat_prompt(context, prompt)
        elif model_name.lower() == "gemma-instruct":
            return self._format_gemma_prompt(context, prompt)
        else:
            raise ValueError(f"format_prompt() received unknown model type: {model_name}")
        
    def _format_openchat_prompt(self, context, prompt):
        if context == "":
            messages = [
                {"role": "user", "content": prompt},
            ]
        else:
            messages = [
                {"role": "user", "content": context + " " + prompt},
            ]

        return self.tokenizer.apply_chat_template(messages, 
                                                  tokenize=False,
                                                  add_generation_prompt=True)
    
    def _format_starling_prompt(self, context, prompt):
        # same as openchat
        return self._format_openchat_prompt(context, prompt)

    def _format_llama_prompt(self, context, prompt):
        # https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-2/
        if context == "":
            return f"<s>{prompt}"
        else:
            return f"<s>{context} {prompt}"

    def _format_llamaChat_prompt(self, context, prompt):
        # https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-2/
        if context == "":
            return f"<s>[INST] {prompt} [/INST]"
        else:
            return f"<s>[INST] <<SYS>>\n {context}<</SYS>>\n\n{prompt} [/INST]"

    def _format_gemma_prompt(self, context, prompt):
        if context == "":
            messages = [
                {"role": "user", "content": prompt},
            ]
        else:
            messages = [
                {"role": "user", "content": context + " " + prompt},
            ]

        return self.tokenizer.apply_chat_template(messages, 
                                                  tokenize=False,
                                                  add_generation_prompt=True)
    
    def _format_mistral_prompt(self, context, prompt):
        if context == "":
            messages = [
                {"role": "user", "content": prompt},
            ]
        else:
            messages = [
                {"role": "system", "content": context},
                {"role": "user", "content": prompt},
            ]
        
        return self.tokenizer.apply_chat_template(messages, tokenize=False)
    
    def _format_zephyrMistral_prompt(self, context, prompt):
        messages = [
            {"role": "system", "content": context},
            {"role": "user", "content": prompt},
            ]

        return self.tokenizer.apply_chat_template(messages,
                                                tokenize=False,
                                                add_generation_prompt=True,
                                                )
    
    def _format_zephyrGemma_prompt(self, context, prompt):
        # according to model card, zephyrGemma doesn't support the system role, 
        # so add context to the user role
        if context == "":
            messages = [
                {"role": "user", "content": prompt},
            ]
        else:
            messages = [
                {"role": "user", "content": context + " " + prompt},
            ]

        return self.tokenizer.apply_chat_template(messages,
                                                tokenize=False,
                                                add_generation_prompt=True,
                                                )
    
    def generate(self, model_name, messages, temp):
        if model_name.lower() == "openchat":
            return self._generate_openchat(messages, temp)
        elif model_name.lower() == "starling":
            return self._generate_starling(messages, temp)
        elif model_name.lower() == "mistral-instruct":
            return self._generate_mistral(messages, temp)
        elif model_name.lower() == "zephyr-mistral":
            return self._generate_zephyrMistral(messages, temp)
        elif model_name.lower() == "zephyr-gemma":
            return self._generate_zephyrGemma(messages, temp)
        # ADD ANOTHER RLHF MODEL HERE
        elif model_name.lower() in ["llama2", "llama2-chat"]:
            return self._generate_llama(messages, temp)
        elif model_name.lower() == "gemma-instruct":
            return self._generate_gemma(messages, temp)
        else:
            raise ValueError(f"generate() received unknown model type: {model_name}")
    
    # HANDLES BATCHING
    def _generate_llama(self, messages, temp):
        self.tokenizer.pad_token = self.tokenizer.eos_token

        all_outputs = []
        for i in tqdm(range(0, len(messages), self.batch_size)):
            batch = messages[i:i+self.batch_size]
            inputs = self.tokenizer(batch, 
                                    return_tensors="pt",
                                    padding=True,
                                    truncation=True).to(self.device)
            
            input_lengths = inputs.input_ids.shape[1]    

            outputs = self.model.module.generate(**inputs,
                                            do_sample=True,
                                            max_new_tokens=self.max_new_tokens, 
                                            temperature=float(temp) if temp != "default" else None,
                                            pad_token_id=self.tokenizer.eos_token_id,
                                            )
            
            # Slice off the input tokens
            generated_tokens = outputs[:, input_lengths:]

            decoded_outputs = self.tokenizer.batch_decode(generated_tokens,skip_special_tokens=True)
            
            all_outputs.extend(decoded_outputs)
        return all_outputs
    
    # HANDLES BATCHING
    def _generate_gemma(self, messages, temp):
        all_outputs = []
        for i in tqdm(range(0, len(messages), self.batch_size)):
            batch = messages[i:i+self.batch_size]
            inputs = self.tokenizer(batch,
                                    add_special_tokens=False,
                                    return_tensors="pt",
                                    padding=True, 
                                    truncation=True).to(self.device)
            
            input_lengths = inputs.input_ids.shape[1]

            outputs = self.model.module.generate(**inputs,
                                                do_sample=True,
                                                temperature=float(temp) if temp != "default" else None,
                                                max_new_tokens=self.max_new_tokens)
            
            # Slice off the input tokens
            generated_tokens = outputs[:, input_lengths:]

            decoded_outputs = self.tokenizer.batch_decode(generated_tokens,
                                                          skip_special_tokens=True)
            
            all_outputs.extend(decoded_outputs)
        return all_outputs
    
    # HANDLES BATCHING
    def _generate_zephyrGemma(self, messages, temp):
        self.tokenizer.pad_token = self.tokenizer.eos_token

        all_outputs = []
        for i in tqdm(range(0, len(messages), self.batch_size)):
            batch = messages[i:i+self.batch_size]
            inputs = self.tokenizer(batch, 
                                    return_tensors="pt",
                                    padding=True,
                                    truncation=True).to(self.device)
            
            input_lengths = inputs.input_ids.shape[1]    

            outputs = self.model.module.generate(**inputs,
                                            do_sample=True,
                                            max_new_tokens=self.max_new_tokens, 
                                            temperature=float(temp) if temp != "default" else None,
                                            pad_token_id=self.tokenizer.eos_token_id,
                                            )
            
            # Slice off the input tokens
            generated_tokens = outputs[:, input_lengths:]

            decoded_outputs = self.tokenizer.batch_decode(generated_tokens,skip_special_tokens=True)
            
            all_outputs.extend(decoded_outputs)
        return all_outputs
    
    # HANDLES BATCHING
    def _generate_zephyrMistral(self, messages, temp):
        self.tokenizer.pad_token = self.tokenizer.eos_token

        all_outputs = []
        for i in tqdm(range(0, len(messages), self.batch_size)):
            batch = messages[i:i+self.batch_size]
            inputs = self.tokenizer(batch, 
                                    return_tensors="pt",
                                    padding=True,
                                    truncation=True).to(self.device)
            
            input_lengths = inputs.input_ids.shape[1]    

            outputs = self.model.module.generate(**inputs,
                                            do_sample=True,
                                            max_new_tokens=self.max_new_tokens, 
                                            temperature=float(temp) if temp != "default" else None,
                                            pad_token_id=self.tokenizer.eos_token_id,
                                            )
            
            # Slice off the input tokens
            generated_tokens = outputs[:, input_lengths:]

            decoded_outputs = self.tokenizer.batch_decode(generated_tokens,skip_special_tokens=True)
            
            all_outputs.extend(decoded_outputs)
        return all_outputs

    # HANDLES BATCHING
    def _generate_mistral(self, messages, temp):
        self.tokenizer.pad_token = self.tokenizer.eos_token

        all_outputs = []
        for i in tqdm(range(0, len(messages), self.batch_size)):
            batch = messages[i:i+self.batch_size]
            inputs = self.tokenizer(batch, 
                                    return_tensors="pt",
                                    padding=True,
                                    truncation=True).to(self.device)
            
            input_lengths = inputs.input_ids.shape[1]    

            outputs = self.model.module.generate(**inputs,
                                            do_sample=True,
                                            max_new_tokens=self.max_new_tokens, 
                                            temperature=float(temp) if temp != "default" else None,
                                            pad_token_id=self.tokenizer.eos_token_id,
                                            )
            
            # Slice off the input tokens
            generated_tokens = outputs[:, input_lengths:]

            decoded_outputs = self.tokenizer.batch_decode(generated_tokens,skip_special_tokens=True)
            
            all_outputs.extend(decoded_outputs)
        return all_outputs

    # HANDLES BATCHING
    def _generate_openchat(self, messages, temp):
        self.tokenizer.pad_token = self.tokenizer.eos_token

        all_outputs = []
        for i in tqdm(range(0, len(messages), self.batch_size)):
            batch = messages[i:i+self.batch_size]
            inputs = self.tokenizer(batch, 
                                    return_tensors="pt",
                                    padding=True,
                                    truncation=True).to(self.device)
            
            input_lengths = inputs.input_ids.shape[1]

            outputs = self.model.module.generate(**inputs, 
                                                 max_new_tokens=self.max_new_tokens,
                                                 do_sample=True,
                                                 temperature=float(temp) if temp != "default" else None)
            
            # Slice off the input tokens
            generated_tokens = outputs[:, input_lengths:]
            
            # Decode only the generated part
            decoded_outputs = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            # outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            all_outputs.extend(decoded_outputs)
        return all_outputs
    
    # HANDLES BATCHING
    def _generate_starling(self, messages, temp):
        # same as openchat
        return self._generate_openchat(messages, temp)
    



    
