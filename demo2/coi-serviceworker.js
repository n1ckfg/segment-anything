/*! coi-serviceworker - enables Cross-Origin Isolation (COOP/COEP) on static servers
 *  so onnxruntime-web can use SharedArrayBuffer / WASM multithreading without a custom backend.
 */
if (typeof window === "undefined") {
  self.addEventListener("install", () => self.skipWaiting());
  self.addEventListener("activate", (event) => event.waitUntil(self.clients.claim()));

  self.addEventListener("fetch", (event) => {
    const req = event.request;
    if (req.cache === "only-if-cached" && req.mode !== "same-origin") return;

    event.respondWith(
      fetch(req)
        .then((response) => {
          if (response.status === 0) return response;
          const headers = new Headers(response.headers);
          headers.set("Cross-Origin-Embedder-Policy", "credentialless");
          headers.set("Cross-Origin-Opener-Policy", "same-origin");
          return new Response(response.body, {
            status: response.status,
            statusText: response.statusText,
            headers,
          });
        })
        .catch((e) => {
          console.error(e);
          throw e;
        })
    );
  });
} else {
  (() => {
    if (window.crossOriginIsolated) return;
    if (!("serviceWorker" in navigator) || !window.isSecureContext) return;
    const reloadedKey = "coi-sw-reloaded";
    navigator.serviceWorker
      .register(new URL("./coi-serviceworker.js", window.location.href).href)
      .then((reg) => {
        reg.addEventListener("updatefound", () => {
          const worker = reg.installing;
          if (!worker) return;
          worker.addEventListener("statechange", () => {
            if (worker.state === "activated" && !sessionStorage.getItem(reloadedKey)) {
              sessionStorage.setItem(reloadedKey, "1");
              window.location.reload();
            }
          });
        });
        if (reg.active && !navigator.serviceWorker.controller && !sessionStorage.getItem(reloadedKey)) {
          sessionStorage.setItem(reloadedKey, "1");
          window.location.reload();
        }
      })
      .catch((err) => console.warn("COI service worker registration failed:", err));
  })();
}
