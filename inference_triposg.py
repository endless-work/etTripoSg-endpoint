import argparse
import os
import sys
from glob import glob
from typing import Any, Union

import numpy as np
import torch
import trimesh
from huggingface_hub import snapshot_download
from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from triposg.pipelines.pipeline_triposg import TripoSGPipeline
from image_process import prepare_image
from briarmbg import BriaRMBG

import pymeshlab


@torch.no_grad()
def run_triposg(
    pipe: Any,
    image_input: Union[str, Image.Image],
    rmbg_net: Any,
    seed: int,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.0,
    faces: int = -1,
    octree_depth: int = 9, # 👈 добавлено_et
) -> trimesh.Scene:
    print("[DEBUG] Preparing image...")
    img_pil = prepare_image(image_input, bg_color=np.array([1.0, 1.0, 1.0]), rmbg_net=rmbg_net)

    print("[DEBUG] Running TripoSG pipeline...")
    outputs = pipe(
        image=img_pil,
        generator=torch.Generator(device=pipe.device).manual_seed(seed),
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        flash_octree_depth=octree_depth, # 👈 добавлено_et
    ).samples[0]

    print("[DEBUG] TripoSG output keys:", type(outputs), outputs[0].shape, outputs[1].shape)

    mesh = trimesh.Trimesh(outputs[0].astype(np.float32), np.ascontiguousarray(outputs[1]))
    print(f"[DEBUG] Mesh created: {mesh.vertices.shape[0]} verts / {mesh.faces.shape[0]} faces")

    if faces > 0:
        print(f"[DEBUG] Simplifying mesh to {faces} faces")
        mesh = simplify_mesh(mesh, faces) # 👈 добавлено_et

    return mesh


def mesh_to_pymesh(vertices, faces):
    mesh = pymeshlab.Mesh(vertex_matrix=vertices, face_matrix=faces)
    ms = pymeshlab.MeshSet()
    ms.add_mesh(mesh)
    return ms

def pymesh_to_trimesh(mesh):
    verts = mesh.vertex_matrix()#.tolist()
    faces = mesh.face_matrix()#.tolist()
    return trimesh.Trimesh(vertices=verts, faces=faces)  #, vID, fID

# old version
# def simplify_mesh(mesh: trimesh.Trimesh, n_faces):
#     if mesh.faces.shape[0] > n_faces:
#         ms = mesh_to_pymesh(mesh.vertices, mesh.faces)
#         ms.meshing_merge_close_vertices()
#         ms.meshing_decimation_quadric_edge_collapse(targetfacenum = n_faces)
#         return pymesh_to_trimesh(ms.current_mesh())
#     else:
#         return mesh

# new version
# def simplify_mesh(mesh: trimesh.Trimesh, n_faces):
#     if mesh.faces.shape[0] > n_faces:
#         ms = mesh_to_pymesh(mesh.vertices, mesh.faces)
#         ms.meshing_merge_close_vertices()
#         ms.meshing_decimation_quadric_edge_collapse(targetfacenum=n_faces)
#         simplified = ms.current_mesh()
#         if simplified is None or simplified.face_number() == 0:
#             return None
#         return pymesh_to_trimesh(simplified)
#     return mesh

# new version demo
def simplify_mesh(mesh: trimesh.Trimesh, n_faces):
    original_faces = mesh.faces.shape[0]  # 👈 сохраняем исходное количество

    if original_faces > n_faces:
        ms = mesh_to_pymesh(mesh.vertices, mesh.faces)
        ms.meshing_merge_close_vertices()
        ms.meshing_decimation_quadric_edge_collapse(targetfacenum=n_faces)
        simplified = ms.current_mesh()
        if simplified is None or simplified.face_number() == 0:
            return None

        simplified_faces = simplified.face_number()
        print(f"[DEBUG] Simplified mesh: {original_faces} → {simplified_faces} faces")  # 👈 лог здесь

        return pymesh_to_trimesh(simplified)

    return mesh


if __name__ == "__main__":
    device = "cuda"
    dtype = torch.float16

    parser = argparse.ArgumentParser()
    parser.add_argument("--image-input", type=str, required=True)
    parser.add_argument("--output-path", type=str, default="./output.glb")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=7.0)
    parser.add_argument("--faces", type=int, default=-1)
    args = parser.parse_args()

    # download pretrained weights
    triposg_weights_dir = "pretrained_weights/TripoSG"
    rmbg_weights_dir = "pretrained_weights/RMBG-1.4"
    snapshot_download(repo_id="VAST-AI/TripoSG", local_dir=triposg_weights_dir)
    snapshot_download(repo_id="briaai/RMBG-1.4", local_dir=rmbg_weights_dir)

    # init rmbg model for background removal
    rmbg_net = BriaRMBG.from_pretrained(rmbg_weights_dir).to(device)
    rmbg_net.eval() 

    # init tripoSG pipeline
    pipe: TripoSGPipeline = TripoSGPipeline.from_pretrained(triposg_weights_dir).to(device, dtype)

    # run inference
    run_triposg(
        pipe,
        image_input=args.image_input,
        rmbg_net=rmbg_net,
        seed=args.seed,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        faces=args.faces,
        octree_depth=octree_depth,
    ).export(args.output_path)
    print(f"Mesh saved to {args.output_path}")
