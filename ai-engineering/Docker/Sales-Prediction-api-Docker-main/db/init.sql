-- Initialize price_history table for dashboard and ML service
CREATE TABLE IF NOT EXISTS price_history (
	id SERIAL PRIMARY KEY,
	price NUMERIC NOT NULL,
	prediction NUMERIC,
	created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
);

-- Index for faster recent queries
CREATE INDEX IF NOT EXISTS idx_price_history_created_at ON price_history (created_at DESC);
