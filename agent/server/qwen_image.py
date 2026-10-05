import os
import uuid
import time
import torch
import torch.multiprocessing as mp
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from diffusers import DiffusionPipeline
import io
import uvicorn
import asyncio

NUM_GPUS = int(os.getenv("QWEN_IMAGE_NUM_GPUS", "1"))
RESOLUTION = int(os.getenv("QWEN_IMAGE_RESOLUTION", "1328"))
STEPS = int(os.getenv("QWEN_IMAGE_STEPS", "50"))
CFG_SCALE = float(os.getenv("QWEN_IMAGE_CFG_SCALE", "4.0"))
MODEL_PATH = os.getenv("QWEN_IMAGE_MODEL_PATH", "").strip()
HOST = os.getenv("QWEN_IMAGE_HOST", "0.0.0.0")
PORT = int(os.getenv("QWEN_IMAGE_PORT", "8000"))
TIMEOUT_SECONDS = int(os.getenv("QWEN_IMAGE_TIMEOUT_SECONDS", "600"))

app = FastAPI(title="Qwen-Image Generator API")

manager = None
input_queue = None
result_dict = None
worker_status = None

def worker(rank, in_queue, res_dict, status_dict):
    device = f"cuda:{rank}"
    status_dict[rank] = {"state": "loading"}
    try:
        torch.cuda.set_device(device)
        pipe = DiffusionPipeline.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
        ).to(device)
    except BaseException as error:
        status_dict[rank] = {
            "state": "error",
            "error_type": type(error).__name__,
            "message": str(error)[:1000],
        }
        return
    status_dict[rank] = {"state": "ready"}
    
    while True:
        task_id = None
        try:
            task_id, prompt, seed = in_queue.get()
        
            resolved_seed = int(seed) if seed is not None else torch.seed() % 10000
            generator = torch.Generator(device=device).manual_seed(resolved_seed)
            
            image = pipe(
                prompt=prompt,
                num_inference_steps=STEPS,
                guidance_scale=CFG_SCALE,
                width=RESOLUTION,
                height=RESOLUTION,
                generator=generator
            ).images[0]
            
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            res_dict[task_id] = img_byte_arr.getvalue()
            
        except Exception as error:
            print(f"Qwen-Image task failed: {type(error).__name__}: {error}", flush=True)
            if task_id is not None:
                res_dict[task_id] = "ERROR"

@app.post("/generate")
async def generate_image(prompt: str, seed: int | None = None):
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is empty")

    task_id = str(uuid.uuid4())
    
    input_queue.put((task_id, prompt, seed))
    
    start_time = time.time()
    while task_id not in result_dict:
        if worker_status is not None and all(
            worker_status.get(rank, {}).get("state") == "error"
            for rank in range(NUM_GPUS)
        ):
            raise HTTPException(status_code=503, detail="Image workers failed to load")
        if time.time() - start_time > TIMEOUT_SECONDS:
            raise HTTPException(status_code=504, detail="Processing timeout")
        await asyncio.sleep(0.1)
    
    result = result_dict.pop(task_id)
    
    if result == "ERROR":
        raise HTTPException(status_code=500, detail="Image generation failed")
    
    return Response(content=result, media_type="image/png")

@app.on_event("startup")
def startup_event():
    global manager, input_queue, result_dict, worker_status

    if NUM_GPUS <= 0:
        raise RuntimeError("QWEN_IMAGE_NUM_GPUS must be positive")
    if not MODEL_PATH:
        raise RuntimeError("Set QWEN_IMAGE_MODEL_PATH to the local generator checkpoint")
    if not os.path.isdir(MODEL_PATH):
        raise RuntimeError(f"Qwen-Image model directory is missing: {MODEL_PATH}")
    visible_gpus = torch.cuda.device_count()
    if NUM_GPUS > visible_gpus:
        raise RuntimeError(
            f"QWEN_IMAGE_NUM_GPUS={NUM_GPUS}, but only {visible_gpus} GPUs are visible"
        )

    mp.set_start_method('spawn', force=True)
    
    manager = mp.Manager()
    input_queue = manager.Queue()
    result_dict = manager.dict()
    worker_status = manager.dict()
    
    for rank in range(NUM_GPUS):
        p = mp.Process(target=worker, args=(rank, input_queue, result_dict, worker_status))
        p.daemon = True
        p.start()


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
