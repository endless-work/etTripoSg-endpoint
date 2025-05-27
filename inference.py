from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
import os
import uuid
import torch
from inference_triposg import run_triposg
from triposg.pipelines.pipeline_triposg import TripoSGPipeline
from briarmbg import BriaRMBG
import zipfile
from huggingface_hub import hf_hub_download

# Настройка устройства
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32



# Загружаем архив весов из HF Datasets
zip_path = hf_hub_download(
    repo_id="endlesstools/pretrained-assets",
    filename="pretrained_models.zip",
    repo_type="dataset"
)

# Распаковываем только если нужно
extracted_dir = os.path.abspath("./pretrained_models")

# Указываем пути к весам
triposg_weights_dir = os.path.join(extracted_dir, "TripoSG")
rmbg_weights_dir = os.path.join(extracted_dir, "RMBG-1.4")


if not os.path.isdir(triposg_weights_dir) or not os.path.isdir(rmbg_weights_dir):
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extracted_dir)



try:
    pipe: TripoSGPipeline = TripoSGPipeline.from_pretrained(triposg_weights_dir).to(device, dtype)
    rmbg_net = BriaRMBG.from_pretrained(rmbg_weights_dir).to(device)
    rmbg_net.eval()
except Exception as e:
    print("[ERROR] Failed to load models:", str(e))
    raise

app = FastAPI()

@app.get("/")
def healthcheck():
    return {"status": "ok"}

@app.post("/generate")
async def generate_model(
    file: UploadFile = File(...),
    seed: int = Form(42),
    num_inference_steps: int = Form(50),
    guidance_scale: float = Form(7.0),
    faces: int = Form(-1),
    octree_depth: int = Form(9),
):
    # Сохраняем входной файл
    uid = str(uuid.uuid4())
    input_path = f"/tmp/input_{uid}.png"
    output_path = f"/tmp/output_{uid}.glb"

    with open(input_path, "wb") as f:
        f.write(await file.read())

    # Запускаем пайплайн
    mesh = run_triposg(
        pipe=pipe,
        image_input=input_path,
        rmbg_net=rmbg_net,
        seed=seed,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        faces=faces,
        octree_depth=octree_depth,
    )

    # Сохраняем результат
    mesh.export(output_path)
    print(f"[DEBUG] Saved output to: {output_path}")

    return FileResponse(output_path, media_type="model/gltf-binary", filename="output.glb")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("inference:app", host="0.0.0.0", port=7860)
