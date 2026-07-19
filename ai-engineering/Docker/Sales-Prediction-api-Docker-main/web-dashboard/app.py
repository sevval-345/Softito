from flask import Flask, render_template_string, jsonify
import psycopg2

app = Flask(__name__)

HTML_TEMPLATE = """
<!doctype html>
<html lang="tr">
<head>
    <meta charset="utf-8">
    <title>Bitcoin Fiyat Tahmin Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 30px; }
        h1 { margin-bottom: 10px; }
        table { border-collapse: collapse; width: 100%; max-width: 900px; }
        th, td { padding: 10px; border: 1px solid #ccc; text-align: left; }
        th { background: #f3f3f3; }
        tr:nth-child(even) { background: #fafafa; }
        .status { margin-bottom: 15px; color: #333; }
    </style>
</head>
<body>
    <h1>Bitcoin Fiyat Tahmin Dashboard</h1>
    <div class="status">Son güncelleme: <span id="updatedAt">Yükleniyor...</span></div>
    <table>
        <thead>
            <tr><th>Zaman</th><th>Fiyat</th><th>Tahmin</th></tr>
        </thead>
                <tbody id="resultsTable"></tbody>
    </table>
        <div style="margin-top:20px; max-width:900px;">
            <canvas id="predictionChart" height="150"></canvas>
        </div>
        <div style="margin-top:10px;">
            <button id="prevBtn">Önceki</button>
            <span id="pageInfo"></span>
            <button id="nextBtn">Sonraki</button>
        </div>
    <script>
        // Load Chart.js from CDN
        const script = document.createElement('script');
        script.src = 'https://cdn.jsdelivr.net/npm/chart.js';
        document.head.appendChild(script);

        let allItems = [];
        let currentPage = 1;
        const pageSize = 10;
        let currentSort = { key: 'time', dir: -1 };
        let chart = null;
        async function loadData() {
            const status = document.getElementById('updatedAt');
            try {
                const response = await fetch('/data');
                if (!response.ok) throw new Error('Veri yüklenemedi.');
                allItems = await response.json();
                renderTable();
                renderChart();

                status.textContent = new Date().toLocaleTimeString('tr-TR');
            } catch (error) {
                status.textContent = 'Hata: ' + error.message;
            }
        }

        function sortItems(key) {
            if (currentSort.key === key) currentSort.dir *= -1;
            else { currentSort.key = key; currentSort.dir = 1; }
            allItems.sort((a,b)=>{
                if (a[key] == null) return 1;
                if (b[key] == null) return -1;
                if (a[key] < b[key]) return -1 * currentSort.dir;
                if (a[key] > b[key]) return 1 * currentSort.dir;
                return 0;
            });
            currentPage = 1;
            renderTable();
        }

        function renderTable(){
            const tbody = document.getElementById('resultsTable');
            tbody.innerHTML = '';
            if (allItems.length === 0) {
                tbody.innerHTML = '<tr><td colspan="3">Henüz veri yok.</td></tr>';
                document.getElementById('pageInfo').textContent = '';
                return;
            }
            const start = (currentPage-1)*pageSize;
            const page = allItems.slice(start, start+pageSize);
            for (const item of page) {
                const row = document.createElement('tr');
                row.innerHTML = `<td>${item.time || ''}</td><td>${item.price}</td><td>${item.prediction ?? ''}</td>`;
                tbody.appendChild(row);
            }
            const totalPages = Math.max(1, Math.ceil(allItems.length / pageSize));
            document.getElementById('pageInfo').textContent = `${currentPage}/${totalPages}`;
        }

        document.getElementById('prevBtn').addEventListener('click', ()=>{ if (currentPage>1){currentPage--; renderTable();}});
        document.getElementById('nextBtn').addEventListener('click', ()=>{ const totalPages=Math.max(1,Math.ceil(allItems.length/pageSize)); if (currentPage<totalPages){currentPage++; renderTable();}});

        // Header click sorting
        document.querySelector('thead tr').addEventListener('click', (e)=>{
            if (e.target.tagName === 'TH'){
                const keyMap = { 'Zaman':'time','Fiyat':'price','Tahmin':'prediction' };
                sortItems(keyMap[e.target.textContent.trim()] || 'time');
            }
        });

        function renderChart(){
            if (!window.Chart) return; // Chart.js not loaded yet
            const ctx = document.getElementById('predictionChart').getContext('2d');
            const items = allItems.slice(0,50).reverse();
            const labels = items.map(i=>i.time ? new Date(i.time).toLocaleTimeString('tr-TR') : '');
            const dataPrices = items.map(i=>i.price);
            const dataPred = items.map(i=>i.prediction);
            if (chart) { chart.data.labels = labels; chart.data.datasets[0].data = dataPrices; chart.data.datasets[1].data = dataPred; chart.update(); return; }
            chart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels,
                    datasets: [
                        { label: 'Fiyat', data: dataPrices, borderColor: '#007bff', fill:false },
                        { label: 'Tahmin', data: dataPred, borderColor: '#ff6600', fill:false }
                    ]
                },
                options: { responsive:true, maintainAspectRatio:false }
            });
        }

        loadData();
        setInterval(loadData, 5000);
    </script>
</body>
</html>
"""

def create_db_connection():
        return psycopg2.connect(
                dbname="pricedb",
                user="postgres",
                password="password",
                host="db",
                port="5432"
        )

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/data')
def data():
    try:
        conn = create_db_connection()
        cur = conn.cursor()
        cur.execute(
            "SELECT price, prediction, created_at FROM price_history ORDER BY created_at DESC LIMIT 100"
        )
        rows = cur.fetchall()
        cur.close()
        conn.close()
    except Exception as e:
        return jsonify({'error': f'Veritabanı bağlantı hatası: {e}'}), 500

    return jsonify([
        {
            'price': float(row[0]),
            'prediction': float(row[1]) if row[1] is not None else None,
            'time': row[2].isoformat() if row[2] is not None else ''
        }
        for row in rows
    ])

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)