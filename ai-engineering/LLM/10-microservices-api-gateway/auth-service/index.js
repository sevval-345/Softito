const express = require('express');

const app = express();
app.use(express.json());

const PORT = process.env.PORT || 4001;

app.get('/health', (req, res) => {
  res.json({ status: 'ok', service: 'auth-service' });
});

app.post('/login', (req, res) => {
  const { username, password } = req.body || {};
  res.json({
    message: 'Giriş başarılı (örnek yanıt)',
    user: username || 'anonymous',
    token: 'mock-jwt-token-12345'
  });
});

app.post('/register', (req, res) => {
  const { username } = req.body || {};
  res.status(201).json({
    message: 'Kullanıcı oluşturuldu (örnek yanıt)',
    user: username || 'anonymous'
  });
});

app.get('/', (req, res) => {
  res.json({ service: 'auth-service', endpoints: ['/health', 'POST /login', 'POST /register'] });
});

app.listen(PORT, () => {
  console.log(`🔐 Auth Service http://localhost:${PORT} adresinde çalışıyor`);
});
