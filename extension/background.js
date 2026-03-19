const BACKEND_URL = 'https://ai-podcast-editor-production.up.railway.app';

// Inject token interceptor into Adobe Podcast tabs
chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
  if (changeInfo.status === 'complete' && tab.url?.startsWith('https://podcast.adobe.com')) {
    chrome.scripting.executeScript({
      target: { tabId },
      world: 'MAIN',
      func: injectTokenInterceptor,
    });
    console.log('[AdobeTokenSync] Injected into tab', tabId);
  }
});

// Poll backend every 60s — refresh Adobe tab only when the token was just used.
// Flow: pipeline uses token → backend sets refresh_needed → extension sees it →
// reloads Adobe tab → fresh API calls → interceptor captures new token →
// syncs to backend (which clears refresh_needed).
chrome.alarms.create('checkTokenStatus', { periodInMinutes: 1 });

chrome.alarms.onAlarm.addListener((alarm) => {
  if (alarm.name !== 'checkTokenStatus') return;
  // Check both token-status and /api/status to detect missing tokens (e.g. after redeploy)
  Promise.all([
    fetch(`${BACKEND_URL}/api/token-status`).then(r => r.json()).catch(() => null),
    fetch(`${BACKEND_URL}/api/status`).then(r => r.json()).catch(() => null),
  ]).then(([tokenData, statusData]) => {
    const needsRefresh = tokenData?.refresh_needed;
    const tokenMissing = statusData && !statusData.adobe_enhance_configured;

    if (!needsRefresh && !tokenMissing) return;

    const reason = tokenMissing ? 'token missing (redeploy?)' : 'token was used';
    chrome.tabs.query({ url: 'https://podcast.adobe.com/*' }, (tabs) => {
      if (tabs.length === 0) {
        console.log('[AdobeTokenSync] No Adobe tab open — cannot refresh token');
        return;
      }
      for (const tab of tabs) {
        chrome.tabs.reload(tab.id);
        console.log(`[AdobeTokenSync] Refreshed Adobe tab ${tab.id} (${reason})`);
      }
    });
  });
});

function injectTokenInterceptor() {
  if (window.__adobeTokenSyncInjected) return;
  window.__adobeTokenSyncInjected = true;

  const BACKEND_URL = 'https://ai-podcast-editor-production.up.railway.app';
  let lastSentToken = null;
  const _fetch = window.fetch.bind(window);

  window.fetch = function (input, init) {
    const url = typeof input === 'string' ? input : input?.url;
    if (url && url.includes('phonos-server-flex.adobe.io')) {
      let authValue = null;
      const headers = init?.headers;
      if (headers instanceof Headers) {
        authValue = headers.get('authorization');
      } else if (headers) {
        authValue = headers['Authorization'] || headers['authorization'];
      }
      if (authValue) {
        const token = authValue.startsWith('Bearer ')
          ? authValue.slice(7)
          : authValue;
        if (token && token !== lastSentToken) {
          lastSentToken = token;
          console.log('[AdobeTokenSync] Token captured, syncing to backend...');
          _fetch(`${BACKEND_URL}/api/set-adobe-token`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ token }),
          })
            .then((r) => console.log('[AdobeTokenSync] Token synced:', r.ok))
            .catch((e) => {
              console.error('[AdobeTokenSync] Sync failed:', e);
              lastSentToken = null;
            });
        }
      }
    }
    return _fetch(input, init);
  };

  console.log('[AdobeTokenSync] Monitoring Adobe API requests');
}
