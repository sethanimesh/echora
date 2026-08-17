# Dedicated RunPod Pod inference

Use this path when you want a persistent GPU service with predictable latency. Training-sized L40S capacity is unnecessary for inference. Start with an RTX A5000, then L4, A40, or RTX 3090 depending on availability.

## 1. Configure the Mac terminal

```bash
export ECHORA_POD_HOST="your-pod-host"
export ECHORA_POD_PORT="your-ssh-port"
export ECHORA_POD_USER="root"
export ECHORA_POD_KEY="$HOME/.ssh/id_ed25519"
```

Use a Pod template with CUDA, at least 24 GB RAM, 30 GB container disk, and port `8000` exposed if you want the RunPod HTTPS proxy.

## 2. Transfer the production bundle

From the project root:

```bash
./deploy/runpod/pod/deploy_to_pod.sh
```

This copies the backend and the complete verified inference bundle. Research datasets and training checkpoints are not transferred.

## 3. Install and start

```bash
ssh -o IdentitiesOnly=yes -i "$ECHORA_POD_KEY" \
  -p "$ECHORA_POD_PORT" "$ECHORA_POD_USER@$ECHORA_POD_HOST"

/workspace/echora/deploy/runpod/pod/setup_pod.sh
export ECHORA_WORKER_TOKEN="replace-with-a-long-random-value"
/workspace/echora/deploy/runpod/pod/start_service.sh
```

The service loads CUDA weights once and writes logs to `/workspace/echora/runtime/pod-service.log`.

## 4. Test without exposing a public port

Keep this tunnel open on the Mac:

```bash
ssh -N -L 8001:127.0.0.1:8000 \
  -o IdentitiesOnly=yes -i "$ECHORA_POD_KEY" \
  -p "$ECHORA_POD_PORT" "$ECHORA_POD_USER@$ECHORA_POD_HOST"
```

In another Mac terminal:

```bash
export ECHORA_WORKER_URL="http://127.0.0.1:8001"
export ECHORA_WORKER_TOKEN="the-same-token"
.venv/bin/python deploy/runpod/pod/smoke_test.py
```

After the smoke test, configure the main `.env`:

```dotenv
ECHORA_ASR_BACKEND=pod
ECHORA_POD_URL=http://127.0.0.1:8001
ECHORA_POD_TOKEN=the-same-token
```

## Stop the service

```bash
/workspace/echora/deploy/runpod/pod/stop_service.sh
```

This stops only the inference process. It never stops or deletes the paid Pod; do that yourself in RunPod after copying anything you need.
