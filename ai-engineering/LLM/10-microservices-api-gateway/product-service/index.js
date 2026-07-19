const express = require('express');

const app = express();
app.use(express.json());

const PORT = process.env.PORT || 4002;

const products = [
  { id: 1, name: 'Kablosuz Kulaklık', price: 799.9 },
  { id: 2, name: 'Mekanik Klavye', price: 1299.5 },
  { id: 3, name: 'USB-C Hub', price: 349.0 }
];

app.get('/health', (req, res) => {
  res.json({ status: 'ok', service: 'product-service' });
});

app.get('/', (req, res) => {
  res.json({ products });
});

app.get('/:id', (req, res) => {
  const product = products.find((p) => p.id === Number(req.params.id));
  if (!product) {
    return res.status(404).json({ error: 'Not Found', message: 'Ürün bulunamadı.' });
  }
  res.json({ product });
});

app.listen(PORT, () => {
  console.log(`📦 Product Service http://localhost:${PORT} adresinde çalışıyor`);
});
