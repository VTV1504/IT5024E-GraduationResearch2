CREATE TABLE IF NOT EXISTS news_raw (
  id BIGINT PRIMARY KEY,
  source TEXT,
  source_country VARCHAR(2),
  language VARCHAR(5),
  category TEXT,
  publish_date TIMESTAMPTZ,
  title TEXT,
  summary TEXT,
  text TEXT,
  url TEXT,
  image TEXT,
  sentiment DOUBLE PRECISION,
  fetched_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_news_raw_publish_date ON news_raw (publish_date);
CREATE INDEX IF NOT EXISTS idx_news_raw_lang_country ON news_raw (language, source_country);

CREATE TABLE IF NOT EXISTS prices (
  symbol VARCHAR(10) NOT NULL,
  date DATE NOT NULL,
  open DOUBLE PRECISION,
  high DOUBLE PRECISION,
  low DOUBLE PRECISION,
  close DOUBLE PRECISION,
  volume DOUBLE PRECISION,
  PRIMARY KEY (symbol, date)
);
CREATE INDEX IF NOT EXISTS idx_prices_symbol_date ON prices (symbol, date);

CREATE TABLE IF NOT EXISTS news_events (
  event_id TEXT PRIMARY KEY,
  symbol VARCHAR(10) NOT NULL,
  publish_date TIMESTAMPTZ,
  title TEXT,
  url TEXT,
  event_type VARCHAR(32),
  sentiment DOUBLE PRECISION,
  impact_hint DOUBLE PRECISION,
  evidence_json JSONB
);
CREATE INDEX IF NOT EXISTS idx_news_events_symbol_date ON news_events (symbol, publish_date);

CREATE TABLE IF NOT EXISTS market_reactions (
  reaction_id TEXT PRIMARY KEY,
  event_id TEXT REFERENCES news_events(event_id),
  symbol VARCHAR(10) NOT NULL,
  t0 DATE NOT NULL,
  horizon_days INTEGER NOT NULL,
  ret_1d DOUBLE PRECISION,
  ret_5d DOUBLE PRECISION,
  ret_h DOUBLE PRECISION,
  vol_5d DOUBLE PRECISION,
  dd_h DOUBLE PRECISION,
  label_up INTEGER,
  meta_json JSONB
);
CREATE INDEX IF NOT EXISTS idx_market_reactions_symbol_t0 ON market_reactions (symbol, t0);

CREATE TABLE IF NOT EXISTS collector_state (
  source TEXT NOT NULL,
  symbol TEXT NOT NULL,
  chunk_start DATE NOT NULL,
  chunk_end DATE NOT NULL,
  offset INTEGER NOT NULL DEFAULT 0,
  done BOOLEAN NOT NULL DEFAULT FALSE,
  updated_at TIMESTAMPTZ DEFAULT now(),
  PRIMARY KEY (source, symbol, chunk_start, chunk_end)
);

CREATE TABLE IF NOT EXISTS features_joint (
  event_id TEXT PRIMARY KEY REFERENCES news_events(event_id),
  symbol VARCHAR(10) NOT NULL,
  t0 DATE NOT NULL,
  horizon_days INTEGER NOT NULL,
  feature_json JSONB NOT NULL,
  created_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_features_joint_symbol_t0 ON features_joint (symbol, t0);

CREATE TABLE IF NOT EXISTS models (
  model_id TEXT PRIMARY KEY,
  created_at TIMESTAMPTZ DEFAULT now(),
  meta_json JSONB
);

CREATE TABLE IF NOT EXISTS predictions (
  pred_id TEXT PRIMARY KEY,
  model_id TEXT REFERENCES models(model_id),
  event_id TEXT REFERENCES news_events(event_id),
  symbol VARCHAR(10),
  t0 DATE,
  horizon_days INTEGER,
  proba_up DOUBLE PRECISION,
  created_at TIMESTAMPTZ DEFAULT now(),
  meta_json JSONB
);
