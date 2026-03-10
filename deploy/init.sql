-- CellType-Agent Database Initialization
-- PostgreSQL schema for session logging and feedback

-- Sessions table
CREATE TABLE IF NOT EXISTS sessions (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(255) UNIQUE NOT NULL,
    query TEXT NOT NULL,
    response TEXT,
    mode VARCHAR(50) DEFAULT 'single',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    user_id VARCHAR(255),
    metadata JSONB DEFAULT '{}'
);

-- Tool calls table
CREATE TABLE IF NOT EXISTS tool_calls (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(255) REFERENCES sessions(session_id),
    tool_name VARCHAR(100) NOT NULL,
    parameters JSONB DEFAULT '{}',
    result JSONB DEFAULT '{}',
    latency_ms INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Feedback table
CREATE TABLE IF NOT EXISTS feedback (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(255) REFERENCES sessions(session_id),
    rating INTEGER CHECK (rating >= 1 AND rating <= 5),
    comments TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    user_id VARCHAR(255)
);

-- Training data view
CREATE OR REPLACE VIEW training_data AS
SELECT
    s.session_id,
    s.query,
    s.response,
    s.mode,
    f.rating,
    f.comments,
    s.created_at,
    json_agg(
        json_build_object(
            'tool', tc.tool_name,
            'parameters', tc.parameters,
            'result', tc.result
        )
    ) AS tool_calls
FROM sessions s
LEFT JOIN feedback f ON s.session_id = f.session_id
LEFT JOIN tool_calls tc ON s.session_id = tc.session_id
WHERE f.rating IS NOT NULL
GROUP BY s.session_id, s.query, s.response, s.mode, f.rating, f.comments, s.created_at;

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_sessions_created_at ON sessions(created_at);
CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_feedback_rating ON feedback(rating);
CREATE INDEX IF NOT EXISTS idx_tool_calls_session_id ON tool_calls(session_id);

-- Statistics view
CREATE OR REPLACE VIEW session_stats AS
SELECT
    COUNT(*) as total_sessions,
    COUNT(CASE WHEN f.rating IS NOT NULL THEN 1 END) as sessions_with_feedback,
    AVG(f.rating) as avg_rating,
    COUNT(DISTINCT s.user_id) as unique_users
FROM sessions s
LEFT JOIN feedback f ON s.session_id = f.session_id;