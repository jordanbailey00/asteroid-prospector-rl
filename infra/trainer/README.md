# Trainer Image

This image provides the Linux runtime required for `training/train_puffer.py --trainer-backend puffer_ppo`.

## Image naming

The compose service is configured with an explicit image tag:
- default: `jordanbailey00/rl-puffer-base:py311-puffercore3.0.17`
- override via env var: `TRAINER_IMAGE`

Example override:

```powershell
$env:TRAINER_IMAGE='yourdockeruser/rl-puffer-base:py311-puffercore3.0.17'
```

## Published reference image (2026-02-28)

- Image: `jordanbailey00/rl-puffer-base:py311-puffercore3.0.17`
- Digest: `sha256:723c58843d9ed563fa66c0927da975bdbab5355c913ec965dbea25a2af67bb71`
- Pin by digest for strict reproducibility:

```dockerfile
FROM jordanbailey00/rl-puffer-base@sha256:723c58843d9ed563fa66c0927da975bdbab5355c913ec965dbea25a2af67bb71
```

## Build (cache-friendly)

```powershell
$env:DOCKER_BUILDKIT='1'
docker compose -f infra/docker-compose.yml build --progress=plain trainer
```

## Smoke test

```powershell
docker compose -f infra/docker-compose.yml run --rm trainer python -c "import importlib.metadata as im, pufferlib; print(im.version('pufferlib-core')); print(pufferlib.__version__)"
```

## Push for reuse in other projects

```powershell
$env:TRAINER_IMAGE='yourdockeruser/rl-puffer-base:py311-puffercore3.0.17'
$env:DOCKER_BUILDKIT='1'
docker compose -f infra/docker-compose.yml build trainer
docker compose -f infra/docker-compose.yml push trainer
```

## Consume from another project

Use the pushed base image in that project's Dockerfile:

```dockerfile
FROM yourdockeruser/rl-puffer-base:py311-puffercore3.0.17
```

Add only project-specific dependencies/layers on top.

Notes:
- `pufferlib-core==3.0.17` is installed from PyPI on Python 3.11 for this image (`import pufferlib` runtime reports `3.0.3`).
- BuildKit cache mount (`/root/.cache/pip`) + Docker layers are used so subsequent builds are fast and avoid repeated downloads.
- For strict reproducibility across repos, keep explicit dependency pins in `infra/trainer/requirements.txt`.
