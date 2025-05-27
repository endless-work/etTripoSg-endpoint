# EtTripoSg Endpoint

FastAPI-based inference endpoint for 3D generation using TripoSG + RMBG.

This model is deployed via a custom Docker container on Hugging Face Inference Endpoints.

## Usage

Send a `POST` request to `/generate` with an image file and optional parameters:

- `seed`: int = 42
- `num_inference_steps`: int = 50
- `guidance_scale`: float = 7.0
- `faces`: int = -1
- `octree_depth`: int = 9
