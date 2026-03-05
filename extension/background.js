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

// Reload Adobe tabs every 10 minutes to keep the token fresh.
// Adobe tokens expire after a single pipeline run; reloading the page
// triggers fresh API calls whose tokens the interceptor captures.
chrome.alarms.create('refreshAdobeToken', { periodInMinutes: 10 });

chrome.alarms.onAlarm.addListener((alarm) => {
  if (alarm.name !== 'refreshAdobeToken') return;
  chrome.tabs.query({ url: 'https://podcast.adobe.com/*' }, (tabs) => {
    for (const tab of tabs) {
      chrome.tabs.reload(tab.id);
      console.log('[AdobeTokenSync] Refreshed Adobe tab', tab.id);
    }
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
