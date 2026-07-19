const express = require('express');
const { createProxyMiddleware } = require('http-proxy-middleware');

const requestLogger = require('./middlewares/requestLogger');
const authGuard = require('./middlewares/authGuard');

const app = express();
const PORT = process.env.PORT || 3000;

const AUTH_SERVICE_URL = process.env.AUTH_SERVICE_URL || 'http://localhost:4001';
const PRODUCT_SERVICE_URL = process.env.PRODUCT_SERVICE_URL || 'http://localhost:4002';

// 1) Tüm isteklerde çalışan Request Logger middleware'i
app.use(requestLogger);

// 2) Health check endpoint'i - kimlik doğrulama gerektirmez
app.get('/health', (req, res) => {
  res.json({ status: 'ok', service: 'api-gateway' });
});

// 3) Servislere gitmeden önce çalışan Auth Guard middleware'i
app.use(authGuard);

// 4) /auth -> auth-service (port 4001) yönlendirmesi
app.use(
  '/auth',
  createProxyMiddleware({
    target: AUTH_SERVICE_URL,
    changeOrigin: true,
    pathRewrite: { '^/auth': '' },
    onProxyReq: (proxyReq, req) => {
      console.log(`[PROXY] /auth -> ${AUTH_SERVICE_URL}${req.url}`);
    },
    onError: (err, req, res) => {
      console.error(`[PROXY ERROR] auth-service: ${err.message}`);
      res.status(502).json({ error: 'Bad Gateway', message: 'auth-service şu anda erişilemez.' });
    }
  })
);

// 5) /products -> product-service (port 4002) yönlendirmesi
app.use(
  '/products',
  createProxyMiddleware({
    target: PRODUCT_SERVICE_URL,
    changeOrigin: true,
    pathRewrite: { '^/products': '' },
    onProxyReq: (proxyReq, req) => {
      console.log(`[PROXY] /products -> ${PRODUCT_SERVICE_URL}${req.url}`);
    },
    onError: (err, req, res) => {
      console.error(`[PROXY ERROR] product-service: ${err.message}`);
      res.status(502).json({ error: 'Bad Gateway', message: 'product-service şu anda erişilemez.' });
    }
  })
);

// 6) Tanımsız route'lar için 404
app.use((req, res) => {
  res.status(404).json({ error: 'Not Found', message: `${req.originalUrl} bulunamadı.` });
});

app.listen(PORT, () => {
  console.log(`🚀 API Gateway http://localhost:${PORT} adresinde çalışıyor`);
  console.log(`   /auth      -> ${AUTH_SERVICE_URL}`);
  console.log(`   /products  -> ${PRODUCT_SERVICE_URL}`);
});
