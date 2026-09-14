# Deploying to Google Cloud Run

Cloud Run builds the image remotely, so Docker is not needed locally — only the
`gcloud` CLI, or the browser-based Cloud Shell, which needs nothing installed.

## One-time setup

```bash
gcloud auth login
gcloud config set project <YOUR_PROJECT_ID>
gcloud services enable run.googleapis.com cloudbuild.googleapis.com
```

## Deploy

```bash
python deploy/prepare_deploy.py        # assembles deploy/bundle/ (~56 MB)
cd deploy/bundle

gcloud run deploy dormdeals \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --concurrency 8 \
  --timeout 120 \
  --max-instances 3 \
  --set-env-vars "JWT_SECRET_KEY=$(openssl rand -hex 32)"
```

The command prints a `https://dormdeals-*.run.app` URL when it finishes. First
build takes 5–10 minutes; later deploys are faster.

## Why those flags

| flag | reason |
|---|---|
| `--memory 2Gi` | SigLIP weights plus torch need well over the 512 MiB default; the container OOMs without this |
| `--cpu 2` | inference is CPU-bound, and `main.py` pins one thread per request |
| `--concurrency 8` | the default of 80 would queue 80 CPU-bound inferences onto 2 cores. The latency benchmark measured p95 climbing steeply past 8 concurrent |
| `--timeout 120` | cold starts load a 44 MB index and the model |
| `--max-instances 3` | caps spend if the URL ever gets traffic |
| `JWT_SECRET_KEY` | `auth.py` raises on import without it, so the container never starts |

For a real secret rather than an env var, use Secret Manager:

```bash
echo -n "$(openssl rand -hex 32)" | gcloud secrets create jwt-secret --data-file=-
gcloud run services update dormdeals --region us-central1 \
  --set-secrets "JWT_SECRET_KEY=jwt-secret:latest"
```

## Cost

The free tier covers 2M requests, 360k GiB-seconds and 180k vCPU-seconds per
month. With `--max-instances 3` and scale-to-zero, an idle portfolio demo costs
nothing; the billing account exists only to enable the service.

Scale-to-zero means the first request after an idle period pays a cold start —
roughly 20–30 seconds while the model loads. Setting `--min-instances 1` removes
that but runs an instance continuously, which does cost money.

## Checks after deploying

```bash
URL=$(gcloud run services describe dormdeals --region us-central1 --format='value(status.url)')
curl -s $URL/health                  # {"status":"ok","indexed":14873}
curl -s -o /dev/null -w '%{http_code}\n' $URL/   # 200, serves the frontend
```

If `/health` reports `indexed: 0`, the data files were excluded from the build —
check that `deploy/bundle/.gcloudignore` exists, since gcloud otherwise falls
back to `.gitignore`, which excludes `backend/data/`.
