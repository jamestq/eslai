import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


def run_whisper():
    # Accounts for Apple silicon chip https://developer.apple.com/metal/pytorch/
    device = "cpu"
    if torch.backends.mps.is_available():
        mps_device = torch.device("mps")
        x = torch.ones(1, device=mps_device)
        device = x.device
    if torch.cuda.is_available():
        device = "cuda:0"        
    torch_dtype = torch.float16 if torch.cuda.is_available() or torch.backends.mps.is_available() else torch.float32
    model_id = "openai/whisper-large-v3"
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)
    processor = AutoProcessor.from_pretrained(model_id)
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )
    result = pipe("example-4-general-en.wav")
    print(result["text"])