/**
 * Request Logger Middleware
 * Gateway'e gelen her isteğin HTTP metodunu, URL'ini ve zaman damgasını
 * konsola yazdırır.
 */
function requestLogger(req, res, next) {
  const timestamp = new Date().toISOString();
  console.log(`[LOG] ${timestamp} | ${req.method} ${req.originalUrl}`);
  next();
}

module.exports = requestLogger;
