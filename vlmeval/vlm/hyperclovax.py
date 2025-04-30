import torch
from .base import BaseModel
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

class HyperCLOVAX(BaseModel):

    def __init__(self, model_name='/home/jiyeon/바탕화면/yjha/hyperclovax_seed_3b', **kwargs):
        assert model_name is not None, "Model name must be provided."
        self.model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(device="cuda")
        self.preprocessor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                
    def generate_inner(self, message, dataset=None):
        prompt, image_path = self.message_to_promptimg(message)

        vlm_chat = [
            {"role": "user", "content": {"type": "text", "text": prompt}},
            {
                "role": "user",
                "content": {
                    "type": "image",
                    "filename": "test.png",
                    "image": image_path,
                }
            }
        ]

        new_vlm_chat, all_images, is_video_list = self.preprocessor.load_images_videos(vlm_chat)
        preprocessed = self.preprocessor(all_images, is_video_list=is_video_list)
        input_ids = self.tokenizer.apply_chat_template(
                new_vlm_chat, return_tensors="pt", tokenize=True, add_generation_prompt=True,
        ).to(device="cuda")

        with torch.amp.autocast(device_type="cuda"):
            output_ids = self.model.generate(
                    input_ids=input_ids,
                    max_new_tokens=8192,
                    do_sample=True,
                    top_p=0.6,
                    temperature=0.5,
                    repetition_penalty=1.0,
                    **preprocessed,
            )
        return self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]