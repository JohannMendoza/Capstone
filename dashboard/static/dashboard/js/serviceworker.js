var staticCacheName = "lanzofields-pwa-v" + new Date().getTime();
var filesToCache = [
    '/',
    '/static/dashboard/img/192x192.png',
    '/static/dashboard/img/512x512.png',
    '/static/dashboard/img/640x1136.png',
    '/admin_dashboard/',
    '/detector/',
    '/pest-detector/',
    '/history/',
    '/inventory/',
    '/offline/',
    '/login/',
    '/register/',
    '/static/dashboard/css/style.css',  // Add CSS files
    '/static/dashboard/js/main.js',     // Add JS files
];

// Cache on install
self.addEventListener("install", event => {
    self.skipWaiting();
    event.waitUntil(
        caches.open(staticCacheName)
            .then(cache => {
                return cache.addAll(filesToCache);
            })
    );
});

// Clear cache on activate
self.addEventListener('activate', event => {
    event.waitUntil(
        caches.keys().then(cacheNames => {
            return Promise.all(
                cacheNames
                    .filter(cacheName => (cacheName.startsWith("lanzofields-pwa-")))
                    .filter(cacheName => (cacheName !== staticCacheName))
                    .map(cacheName => caches.delete(cacheName))
            );
        })
    );
    return self.clients.claim();
});

// Serve from Cache
self.addEventListener("fetch", event => {
    event.respondWith(
        caches.match(event.request)
            .then(response => {
                if (response) {
                    return response;
                }
                
                // For navigation requests, try network first
                if (event.request.mode === 'navigate') {
                    return fetch(event.request)
                        .catch(() => {
                            return caches.match('/offline/');
                        });
                }
                
                // For other requests, try network first then cache
                return fetch(event.request)
                    .then(response => {
                        // Don't cache non-successful responses
                        if (!response || response.status !== 200) {
                            return response;
                        }
                        
                        // Clone the response to cache it
                        var responseToCache = response.clone();
                        caches.open(staticCacheName)
                            .then(cache => {
                                cache.put(event.request, responseToCache);
                            });
                        
                        return response;
                    })
                    .catch(() => {
                        // If offline and not a navigation request, show offline page
                        if (event.request.mode === 'navigate') {
                            return caches.match('/offline/');
                        }
                        return null;
                    });
            })
    );
});