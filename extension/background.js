const BACKEND_URL = 'https://ai-podcast-editor-production.up.railway.app';

let lastSentToken = null;

chrome.webRequest.onSendHeaders.addListener(
  (details) => {
    const authHeader = details.requestHeaders?.find(
      (h) => h.name.toLowerCase() === 'authorization'
    );
    if (!authHeader?.value) return;

    // Strip "Bearer " prefix if present
    const token = authHeader.value.startsWith('Bearer ')
      ? authHeader.value.slice(7)
      : authHeader.value;

    if (!token || token === lastSentToken) return;

    lastSentToken = token;
    console.log('[AdobeTokenSync] New token detected, syncing to backend...');

    fetch(`${BACKEND_URL}/api/set-adobe-token`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ token }),
    })
      .then((resp) => {
        if (resp.ok) {
          console.log('[AdobeTokenSync] Token synced to backend');
        } else {
          console.error(`[AdobeTokenSync] Backend returned ${resp.status}`);
          lastSentToken = null; // retry next time
        }
      })
      .catch((err) => {
        console.error('[AdobeTokenSync] Sync failed:', err);
        lastSentToken = null; // retry next time
      });
  },
  { urls: ['https://phonos-server-flex.adobe.io/*'] },
  ['requestHeaders', 'extraHeaders']
);
