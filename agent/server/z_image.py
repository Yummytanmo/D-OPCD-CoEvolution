import os
import uuid
import time
import hashlib
from ipaddress import ip_address
import queue
import torch
import torch.multiprocessing as mp
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from diffusers import DiffusionPipeline
import io
import uvicorn
import asyncio

NUM_GPUS = int(os.getenv("Z_IMAGE_NUM_GPUS", "1"))
WORKERS_PER_GPU = int(os.getenv("Z_IMAGE_WORKERS_PER_GPU", "1"))
NUM_WORKERS = NUM_GPUS * WORKERS_PER_GPU
RESOLUTION = int(os.getenv("Z_IMAGE_RESOLUTION", "1024"))
STEPS = int(os.getenv("Z_IMAGE_STEPS", "9"))
CFG_SCALE = float(os.getenv("Z_IMAGE_CFG_SCALE", "0.0"))
MODEL_PATH = os.getenv("Z_IMAGE_MODEL_PATH", "").strip()
LORA_PATH = os.getenv("Z_IMAGE_LORA_PATH", "").strip()
EXPECTED_LORA_SHA256 = os.getenv("Z_IMAGE_LORA_SHA256", "").strip().lower()
LORA_SCALE = float(os.getenv("Z_IMAGE_LORA_SCALE", "1.0"))
HOST = os.getenv("Z_IMAGE_HOST", "0.0.0.0")
PORT = int(os.getenv("Z_IMAGE_PORT", "8001"))
TIMEOUT_SECONDS = int(os.getenv("Z_IMAGE_TIMEOUT_SECONDS", "600"))
MAX_SEQUENCE_LENGTH = int(os.getenv("Z_IMAGE_MAX_SEQUENCE_LENGTH", "512"))
REJECT_TRUNCATION = os.getenv("Z_IMAGE_REJECT_TRUNCATION", "0") == "1"

app = FastAPI(title="Z-Image Generator API")

manager = None
interactive_queue = None
calibration_queue = None
result_dict = None
worker_status = None
lora_sha256 = None

def worker(rank, main_queue, load_queue, res_dict, status_dict):
    device = f"cuda:{rank % NUM_GPUS}"

    status_dict[rank] = {"state": "loading"}
    try:
        torch.cuda.set_device(device)

        pipe = DiffusionPipeline.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
        )
        if LORA_PATH:
            pipe.load_lora_weights(
                os.path.dirname(LORA_PATH),
                weight_name=os.path.basename(LORA_PATH),
                adapter_name="dopcd_student",
            )
            pipe.set_adapters("dopcd_student", adapter_weights=LORA_SCALE)
        pipe = pipe.to(device)
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
            try:
                task_id, prompt, seed = main_queue.get_nowait()
            except queue.Empty:
                try:
                    task_id, prompt, seed = load_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
            
            resolved_seed = int(seed) if seed is not None else torch.seed() % 10000
            generator = torch.Generator(device=device).manual_seed(resolved_seed)

            if REJECT_TRUNCATION:
                rendered = pipe.tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}], tokenize=False,
                    add_generation_prompt=True, enable_thinking=True,
                )
                token_count = len(pipe.tokenizer(rendered)["input_ids"])
                if token_count > MAX_SEQUENCE_LENGTH:
                    raise ValueError(f"prompt has {token_count} tokens; limit={MAX_SEQUENCE_LENGTH}; refusing truncation")
            
            image = pipe(
                prompt=prompt,
                num_inference_steps=STEPS,
                guidance_scale=CFG_SCALE,
                width=RESOLUTION,
                height=RESOLUTION,
                generator=generator,
                max_sequence_length=MAX_SEQUENCE_LENGTH
            ).images[0]
            
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            res_dict[task_id] = img_byte_arr.getvalue()
            
        except Exception as error:
            print(f"Z-Image task failed: {type(error).__name__}: {error}", flush=True)
            if task_id is not None:
                res_dict[task_id] = "ERROR"

async def _generate(prompt: str, seed: int | None, *, calibration: bool) -> bytes:
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is empty")

    task_id = str(uuid.uuid4())
    target_queue = calibration_queue if calibration else interactive_queue
    target_queue.put((task_id, prompt, seed))
    
    start_time = time.time()
    while task_id not in result_dict:
        if time.time() - start_time > TIMEOUT_SECONDS:
            raise HTTPException(status_code=504, detail="Processing timeout")
        await asyncio.sleep(0.1)

    result = result_dict.pop(task_id)
    
    if result == "ERROR":
        raise HTTPException(status_code=500, detail="Image generation failed")

    return result


@app.post("/generate")
async def generate_image(prompt: str, seed: int | None = None):
    result = await _generate(prompt, seed, calibration=False)
    return Response(content=result, media_type="image/png")


@app.post("/calibration")
async def generate_calibration(
    request: Request,
    prompt: str,
    seed: int | None = None,
):
    client_host = request.client.host if request.client is not None else ""
    try:
        is_loopback = ip_address(client_host).is_loopback
    except ValueError:
        is_loopback = client_host.casefold() == "localhost"
    if not is_loopback:
        raise HTTPException(status_code=403, detail="Calibration endpoint is local-only")
    result = await _generate(prompt, seed, calibration=True)
    return {
        "status": "completed",
        "image_bytes": len(result),
        "image_sha256": hashlib.sha256(result).hexdigest(),
    }


@app.get("/ready")
def ready():
    statuses = {
        str(rank): worker_status.get(rank, {"state": "starting"})
        for rank in range(NUM_WORKERS)
    } if worker_status is not None else {}
    ready_workers = (
        sum(
            1
            for value in statuses.values()
            if isinstance(value, dict) and value.get("state") == "ready"
        )
        if worker_status is not None
        else 0
    )
    value = {
        "status": "ready" if ready_workers == NUM_WORKERS else "loading",
        "generator": "z-image",
        "lora_enabled": bool(LORA_PATH),
        "lora_weight": os.path.basename(LORA_PATH) if LORA_PATH else None,
        "lora_path": LORA_PATH or None,
        "lora_sha256": lora_sha256,
        "lora_scale": LORA_SCALE if LORA_PATH else None,
        "ready_workers": ready_workers,
        "expected_workers": NUM_WORKERS,
        "workers": statuses,
    }
    if ready_workers != NUM_WORKERS:
        return JSONResponse(status_code=503, content=value)
    return value

@app.on_event("startup")
def startup_event():
    global manager, interactive_queue, calibration_queue, result_dict, worker_status, lora_sha256

    if NUM_GPUS <= 0:
        raise RuntimeError("Z_IMAGE_NUM_GPUS must be positive")
    if not MODEL_PATH:
        raise RuntimeError("Set Z_IMAGE_MODEL_PATH to the local generator checkpoint")
    if not os.path.isdir(MODEL_PATH):
        raise RuntimeError(f"Z-Image model directory is missing: {MODEL_PATH}")
    if LORA_PATH and not os.path.isfile(LORA_PATH):
        raise RuntimeError(f"Z-Image LoRA is missing: {LORA_PATH}")
    if EXPECTED_LORA_SHA256 and not LORA_PATH:
        raise RuntimeError("Z_IMAGE_LORA_SHA256 requires Z_IMAGE_LORA_PATH")
    if LORA_PATH:
        checksum = hashlib.sha256()
        with open(LORA_PATH, "rb") as weights:
            for block in iter(lambda: weights.read(1024 * 1024), b""):
                checksum.update(block)
        lora_sha256 = checksum.hexdigest()
        if EXPECTED_LORA_SHA256 and lora_sha256 != EXPECTED_LORA_SHA256:
            raise RuntimeError("Z-Image LoRA SHA256 mismatch")
    visible_gpus = torch.cuda.device_count()
    if NUM_GPUS > visible_gpus:
        raise RuntimeError(
            f"Z_IMAGE_NUM_GPUS={NUM_GPUS}, but only {visible_gpus} GPUs are visible"
        )

    mp.set_start_method('spawn', force=True)
    
    manager = mp.Manager()
    interactive_queue = manager.Queue()
    calibration_queue = manager.Queue()
    result_dict = manager.dict()
    worker_status = manager.dict()
    
    for rank in range(NUM_WORKERS):
        p = mp.Process(
            target=worker,
            args=(
                rank,
                interactive_queue,
                calibration_queue,
                result_dict,
                worker_status,
            ),
        )
        p.daemon = True
        p.start()


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
