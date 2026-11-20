import torch
from transformers import AutoModelForCausalLM, AutoProcessor, AutoModelForMultimodalLM
from safetensors.torch import save_file

MODEL_ID = "google/gemma-4-E2B-it"

# Load model
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForMultimodalLM.from_pretrained(
    MODEL_ID,
    dtype="auto",
    device_map="auto"
)


messages = [
    {
        "role": "user", "content": [
            {"type": "audio", "url": "/path/to/FastFlowLM/FastFlowLM/src/tb_files/output_voice_clone.wav"},
            {"type": "text", "text": "Transcribe the following speech segment in its original language. Follow these specific instructions for formatting the answer:\n* Only output the transcription, with no newlines.\n* When transcribing numbers, write the digits, i.e. write 1.7 and not one point seven, and write 3 instead of three."}
        ]
    }
]


# # Process input
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    add_generation_prompt=True
).to(model.device)


# out, ref = model(**inputs)
# print(ref)
# ref["input_ids"] = inputs["input_ids"].to(torch.int32)
# Ensure ref contains tensors and save it
# if isinstance(ref, dict):
#     save_file({k: v.cpu() for k, v in ref.items()}, "gemma4_ref.safetensors")
# else:
#     raise ValueError("Expected ref to be a dictionary of tensors.")

input_len = inputs["input_ids"].shape[-1]
# Generate output
outputs = model.generate(**inputs, max_new_tokens=1024, temperature = 1, do_sample = True, top_k = 20, top_p = 0.95)
response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)

# Parse output
print(processor.parse_response(response))

# print(model.model.language_model.layers[0].input_layernorm.weight)
