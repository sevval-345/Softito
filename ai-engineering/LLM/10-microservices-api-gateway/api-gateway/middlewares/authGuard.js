/**
 * Auth Guard Middleware
 * Gelen isteğin header'ında 'x-api-key' değerini kontrol eder.
 * Anahtar doğru değilse isteği servislere iletmeden 401 Unauthorized
 * hatasıyla reddeder.
 */
const VALID_API_KEY = process.env.API_KEY || 'secret-123';

function authGuard(req, res, next) {
  const apiKey = req.headers['x-api-key'];

  if (!apiKey || apiKey !== VALID_API_KEY) {
    console.warn(`[AUTH GUARD] Reddedildi -> ${req.method} ${req.originalUrl} (geçersiz anahtar: ${apiKey || 'yok'})`);
    return res.status(401).json({
      error: 'Unauthorized',
      message: 'Geçersiz veya eksik x-api-key header değeri.'
    });
  }

  next();
}

module.exports = authGuard;
