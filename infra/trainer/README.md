# Trainer Image

This image provides the Linux runtime required for `training/train_puffer.py --trainer-backend puffer_ppo`.

Build:

```powershell
$env:DOCKER_BUILDKIT='1'
docker compose -f infra/docker-compose.yml build --progress=plain trainer
```

Smoke test:

```powershell
docker compose -f infra/docker-compose.yml run --rm trainer python -c "import pufferlib; print('ok')"
```

Notes:
- `pufferlib==2.0.6` is installed from source on Python 3.11 for this image.
- BuildKit cache mount (`/root/.cache/pip`) + Docker layers are used so subsequent builds are fast and avoid repeated downloads.
