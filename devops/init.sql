-- ---------------------------------------------------------------------------
-- Schema bootstrap + default user seed
--
-- This script runs once when the postgres container initialises a fresh
-- data directory (docker-entrypoint-initdb.d).  It mirrors the SQLAlchemy
-- ORM models exactly so that `Base.metadata.create_all()` on app startup
-- is a safe no-op (all tables / indexes already exist).
-- ---------------------------------------------------------------------------

-- ── Users ──────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS users (
    user_id    VARCHAR(36)  PRIMARY KEY,
    email      VARCHAR(255) NOT NULL,
    password   VARCHAR(255) NOT NULL,
    created_at TIMESTAMP    DEFAULT NOW(),
    updated_at TIMESTAMP    DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS ix_users_email ON users (email);

-- ── Conversations ──────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS conversations (
    conv_id    VARCHAR(36)  PRIMARY KEY,
    user_id    VARCHAR(36)  NOT NULL REFERENCES users (user_id) ON DELETE CASCADE,
    title      VARCHAR(255),
    summary    TEXT,
    created_at TIMESTAMP    DEFAULT NOW(),
    updated_at TIMESTAMP    DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ix_conversations_user_id ON conversations (user_id);

-- ── Messages ───────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS messages (
    msg_id     VARCHAR(36) PRIMARY KEY,
    conv_id    VARCHAR(36) NOT NULL REFERENCES conversations (conv_id) ON DELETE CASCADE,
    question   TEXT        NOT NULL,
    answer     TEXT        NOT NULL,
    created_at TIMESTAMP   DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ix_messages_conv_id ON messages (conv_id);

-- ── Default application user ───────────────────────────────────────────────
INSERT INTO users (user_id, email, password, created_at, updated_at)
VALUES (
    '00000000-0000-0000-0000-000000000001',
    'admin@chatwithdoc.local',
    'not-used',
    NOW(),
    NOW()
) ON CONFLICT DO NOTHING;
