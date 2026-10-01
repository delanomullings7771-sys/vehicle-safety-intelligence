// Base URL of the model API. Local development uses the uvicorn server on port 8000;
// the deployed site uses the Render service (update this after the first Render deploy).
window.API_BASE = ["localhost", "127.0.0.1"].includes(location.hostname)
  ? "http://127.0.0.1:8000"
  : "https://vehicle-safety-api.onrender.com";
