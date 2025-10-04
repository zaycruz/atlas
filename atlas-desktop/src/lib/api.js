const DEFAULT_BASE = 'http://127.0.0.1:5175';

function baseUrl() {
  return import.meta.env.VITE_ATLAS_API || DEFAULT_BASE;
}

async function handleResponse(response) {
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || 'Request failed');
  }
  return response.json();
}

export async function sendMessage(message) {
  const response = await fetch(`${baseUrl()}/chat`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ message }),
  });
  return handleResponse(response);
}

export async function fetchMetrics() {
  const response = await fetch(`${baseUrl()}/metrics`);
  return handleResponse(response);
}

export async function resetSession() {
  const response = await fetch(`${baseUrl()}/session/reset`, {
    method: 'POST',
  });
  return handleResponse(response);
}

export async function checkHealth() {
  const response = await fetch(`${baseUrl()}/health`);
  return handleResponse(response);
}
