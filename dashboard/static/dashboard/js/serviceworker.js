// ============================================
// LANZOFIELDS PWA SERVICE WORKER
// Version: 2.0
// ============================================

const CACHE_NAME = 'lanzofields-pwa-v2';
const OFFLINE_URL = '/offline/';

// App Shell - Files that make up the basic app
const APP_SHELL_FILES = [
  '/',
  OFFLINE_URL,
  '/login/',
  '/register/',
  '/static/dashboard/manifest.json'
];

// Static Assets - Images, CSS, JS
const STATIC_ASSETS = [
  '/static/dashboard/img/192x192.png',
  '/static/dashboard/img/512x512.png',
  '/static/dashboard/style.css',
  '/static/dashboard/js/jquery/jquery-2.2.4.min.js',
  '/static/dashboard/js/bootstrap/popper.min.js',
  '/static/dashboard/js/bootstrap/bootstrap.min.js',
  '/static/dashboard/js/plugins/plugins.js',
  '/static/dashboard/js/active.js'
];

console.log('[SW] Service Worker loading...');

// INSTALL EVENT - Cache App Shell
self.addEventListener('install', event => {
  console.log('[SW] Installing...');
  
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(cache => {
        console.log('[SW] Caching app shell and offline page');
        // Prioritize offline page
        return cache.addAll([OFFLINE_URL, ...APP_SHELL_FILES, ...STATIC_ASSETS]);
      })
      .then(() => {
        console.log('[SW] App Shell cached successfully');
        return self.skipWaiting();
      })
      .catch(error => {
        console.error('[SW] Cache error during install:', error);
      })
  );
});

// ACTIVATE EVENT - Clean up old caches
self.addEventListener('activate', event => {
  console.log('[SW] Activating...');
  
  event.waitUntil(
    caches.keys().then(cacheNames => {
      return Promise.all(
        cacheNames.map(cacheName => {
          // Delete old caches
          if (cacheName !== CACHE_NAME) {
            console.log('[SW] Deleting old cache:', cacheName);
            return caches.delete(cacheName);
          }
        })
      );
    })
    .then(() => {
      console.log('[SW] Service Worker activated');
      return self.clients.claim();
    })
  );
});

// FETCH EVENT - Handle network requests
self.addEventListener('fetch', event => {
  const url = new URL(event.request.url);
  
  // Skip non-GET requests
  if (event.request.method !== 'GET') return;
  
  // For HTML pages
  if (event.request.headers.get('accept').includes('text/html')) {
    event.respondWith(
      fetch(event.request)
        .then(response => {
          // Cache successful responses
          if (response.status === 200) {
            const responseClone = response.clone();
            caches.open(CACHE_NAME)
              .then(cache => {
                cache.put(event.request, responseClone);
              });
          }
          return response;
        })
        .catch(() => {
          // If offline, try cache
          return caches.match(event.request)
            .then(cachedResponse => {
              // Return cached page if found
              if (cachedResponse) {
                console.log('[SW] Serving from cache:', event.request.url);
                return cachedResponse;
              }
              
              // If no cache, show offline page
              console.log('[SW] Showing offline page for:', event.request.url);
              return caches.match(OFFLINE_URL);
            });
        })
    );
    return;
  }
  
  // For static assets, cache first
  if (url.pathname.includes('/static/')) {
    event.respondWith(
      caches.match(event.request)
        .then(cachedResponse => {
          if (cachedResponse) {
            console.log('[SW] Serving static from cache:', url.pathname);
            return cachedResponse;
          }
          
          return fetch(event.request)
            .then(response => {
              if (response.status === 200) {
                const responseClone = response.clone();
                caches.open(CACHE_NAME)
                  .then(cache => {
                    cache.put(event.request, responseClone);
                  });
              }
              return response;
            })
            .catch(() => {
              // For images, return a placeholder if offline
              if (url.pathname.includes('.png') || url.pathname.includes('.jpg')) {
                return caches.match('/static/dashboard/img/192x192.png');
              }
              return new Response('Offline', { status: 408 });
            });
        })
    );
    return;
  }
  
  // For other requests, network first with cache fallback
  event.respondWith(
    fetch(event.request)
      .catch(() => {
        return caches.match(event.request)
          .then(cachedResponse => {
            return cachedResponse || new Response('Offline', { 
              status: 408,
              headers: { 'Content-Type': 'text/plain' }
            });
          });
      })
  );
});

// MESSAGE HANDLING
self.addEventListener('message', event => {
  if (event.data && event.data.type === 'SKIP_WAITING') {
    self.skipWaiting();
  }
});

console.log('[SW] Service Worker loaded successfully');