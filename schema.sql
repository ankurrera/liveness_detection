-- setting up the database from scratch
CREATE DATABASE IF NOT EXISTS employee_activity_db;
USE employee_activity_db;

-- 1. table for all the folks we are tracking
CREATE TABLE IF NOT EXISTS employees (
    employee_id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    department VARCHAR(255) DEFAULT 'Engineering',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- nuke the old tables first so we can apply changes cleanly
DROP TABLE IF EXISTS daily_summary;
DROP TABLE IF EXISTS activity_logs;

-- 2. the meat of the app, storing chunks of activity along with the computer vision debug metrics
CREATE TABLE activity_logs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    employee_id INT NOT NULL,
    state ENUM('WORKING', 'IDLE', 'ABSENT') NOT NULL,
    start_time DATETIME NOT NULL,
    end_time DATETIME DEFAULT NULL,
    duration_seconds INT DEFAULT 0,
    confidence FLOAT NOT NULL,
    raw_score FLOAT DEFAULT 0.0,
    smoothed_score FLOAT DEFAULT 0.0,
    transition_reason VARCHAR(255) DEFAULT NULL,
    notes VARCHAR(255) DEFAULT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (employee_id) REFERENCES employees(employee_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- slap some indexes on so our analytics queries run fast
CREATE INDEX idx_logs_employee_start ON activity_logs (employee_id, start_time);
CREATE INDEX idx_logs_employee_state ON activity_logs (employee_id, state);
CREATE INDEX idx_logs_open_session ON activity_logs (employee_id, end_time);

-- 3. the daily rolled up stats (just seconds to keep it simple)
CREATE TABLE daily_summary (
    id INT AUTO_INCREMENT PRIMARY KEY,
    employee_id INT NOT NULL,
    date DATE NOT NULL,
    working_seconds INT DEFAULT 0,
    idle_seconds INT DEFAULT 0,
    absent_seconds INT DEFAULT 0,
    productivity_score FLOAT DEFAULT 0.0,
    UNIQUE KEY uq_summary_employee_date (employee_id, date),
    FOREIGN KEY (employee_id) REFERENCES employees(employee_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- make loading the stats page snappy
CREATE INDEX idx_summary_employee_date ON daily_summary (employee_id, date);
