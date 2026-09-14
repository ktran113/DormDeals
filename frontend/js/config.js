// Where the API lives. The frontend is hosted separately (Vercel) from the
// backend, so this must be absolute in production — a relative URL would
// resolve against the static host, which serves no API.
//
// Empty string = same origin, which is what happens when FastAPI serves these
// files itself (uvicorn main:app, then open :8000).
const API_URL = "";
